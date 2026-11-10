"""Chat with a trained adapter in the terminal (on the GPU machine).

    python -m digital_twin.chat.terminal outputs/<run>/final
    python -m digital_twin.chat.terminal outputs/<run>/final --load-in-4bit --temperature 0.8

Every line is one message; an empty line hands the turn over to the model.
A line starting with "> text | " replies to the earlier message that starts with text.
"""
import argparse
import readline  # noqa: F401 -- UTF-8-aware line editing for input(); srun --pty ttys lack iutf8
import sys
from datetime import datetime
from pathlib import Path

from digital_twin.model.twin import Twin
from digital_twin.prompt import WAIT, quote, recent


def find(conversation, start):
    """The latest message in the conversation that starts with `start`."""
    for _, turn in reversed(conversation):
        if turn["content"] != WAIT and turn["content"].startswith(start):
            return turn["content"]
    return None


def interactive(twin, **sampling):
    sys.stdin.reconfigure(errors="replace")  # a stray broken byte must not end the chat
    print("One message per line; an empty line hands the turn over. /reset — new chat, /exit — quit.")
    conversation, pending = [], []
    while True:
        line = input("> ").strip()
        if line == "/exit":
            return
        if line == "/reset":
            conversation, pending = [], []
            continue
        if line:
            if line.startswith(">") and " | " in line:
                target, text = line[1:].split(" | ", 1)
                line = quote(text, find(conversation, target.strip()))
            pending.append({"role": "user", "content": line})
            continue
        if not pending:
            continue
        now = datetime.now()
        burst = twin.reply(recent(conversation, now) + pending, now, **sampling)
        conversation += [(now, t) for t in pending] + [(now, {"role": "assistant", "content": t}) for t in burst]
        conversation.append((now, {"role": "assistant", "content": WAIT}))
        pending = []
        for text in burst:
            print(f"{twin.me_name}: {text}")
        print(f"{twin.me_name}: ⏸")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("adapter_dir", type=Path, help="outputs/<run>/final or a checkpoint")
    p.add_argument("--temperature", type=float, help="default: the model's recommended value")
    p.add_argument("--load-in-4bit", action="store_true", help="e.g. a 12B model on a 24 GB GPU")
    args = p.parse_args()
    sampling = {"temperature": args.temperature} if args.temperature is not None else {}
    interactive(Twin(args.adapter_dir, load_in_4bit=args.load_in_4bit), **sampling)


if __name__ == "__main__":
    main()
