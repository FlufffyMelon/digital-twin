"""Serve a trained adapter over HTTP for the Telegram bot (runs on a GPU node).

    python -m digital_twin.server outputs/<run>/final --load-in-4bit

The cluster cannot reach Telegram, so the bot runs on the laptop and reaches
this server through an SSH tunnel; the node and port are written to
logs/serve.addr, which mutagen brings to the laptop.

    GET  /health                                  -> {"ok": true, "adapter": ...}
    POST /reply {"messages": [...], "temperature"} -> {"burst": ["...", ...]}
"""
from unsloth import FastModel  # noqa: F401 -- must be imported before transformers

import argparse
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from digital_twin.chat import Twin
from digital_twin.dataset import ROOT

ADDR_FILE = ROOT / "logs" / "serve.addr"


def make_handler(twin, adapter_dir, lock):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, payload):
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True, "adapter": str(adapter_dir)})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/reply":
                return self._send(404, {"error": "not found"})
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            sampling = {k: request[k] for k in ("temperature", "top_p", "min_p") if request.get(k) is not None}
            started = time.time()
            with lock:  # one GPU, one generation at a time
                burst = twin.reply(request["messages"], **sampling)
            print(f"reply: {len(burst)} messages in {time.time() - started:.1f}s", flush=True)
            self._send(200, {"burst": burst})

        def log_message(self, *args):  # no request lines with chat content in the logs
            pass

    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("adapter_dir", type=Path)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--load-in-4bit", action="store_true")
    args = p.parse_args()

    twin = Twin(args.adapter_dir, load_in_4bit=args.load_in_4bit)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(twin, args.adapter_dir, threading.Lock()))
    ADDR_FILE.parent.mkdir(exist_ok=True)
    ADDR_FILE.write_text(f"{socket.gethostname()}:{args.port}\n")
    print(f"serving {args.adapter_dir} on {socket.gethostname()}:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        ADDR_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
