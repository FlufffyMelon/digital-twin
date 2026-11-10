"""Telegram bot for the twin. Stdlib only: it can run on a laptop without the GPU stack.

    python -m digital_twin.chat.server outputs/<run>/final   # on the GPU machine: the model
    python3 -m digital_twin.chat.bot                          # anywhere that reaches Telegram

Settings live in .env at the repository root (git-ignored):

    TELEGRAM_TOKEN=123456:ABC...      # from @BotFather
    ALLOWED_USERS=111111111,222222222 # Telegram user ids; /start shows yours
    MODEL_URL=http://127.0.0.1:8765   # the model server (this is the default)
    SSH_HOST=cluster                  # optional: reach the server through `ssh -L` to this host,
                                      # at the node and port the server wrote to logs/serve.addr

Each allowed user has their own conversation. Like in the real chat, the bot
waits for a short silence after her messages, then writes one message at a
time until the model says WAIT. As in the dataset, a Telegram reply shows its
quote (↪ «…») to the model, and a message the model writes with a quote is sent
as a real Telegram reply to that message. Stickers come in and go out as emoji;
other media and forwarded posts are ignored (text and stickers only).
"""
import json
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from digital_twin.prompt import WAIT, quote, recent, unquote

ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env"
ADDR_FILE = ROOT / "logs" / "serve.addr"
DEFAULT_MODEL_URL = "http://127.0.0.1:8765"
LOCAL_PORT = 8765  # local end of the SSH tunnel
DEBOUNCE_SECONDS = 4.0   # her bursts: 65% of pauses inside a burst are under 30 s, most under a few seconds
DEFAULT_TEMPERATURE = 0.7
STICKER = re.compile(r"^\[стикер ?(.*?)\]$")
HELP = (
    "Я цифровой двойник: {me}.\n"
    "Пиши как обычно, можно несколькими сообщениями подряд — я отвечу, когда ты замолчишь на пару секунд.\n\n"
    "/reset — забыть переписку и начать заново\n"
    "/temp 0.7 — температура (меньше — осторожнее, больше — разнообразнее)\n"
    "/help — эта справка"
)


def load_env():
    env = {}
    for line in ENV_FILE.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            env[key.strip()] = value.split(" #")[0].strip()  # allow comments after a value
    return env


class Telegram:
    def __init__(self, token):
        self.base = f"https://api.telegram.org/bot{token}/"

    def call(self, method, http_timeout=20, **params):
        req = urllib.request.Request(self.base + method, data=json.dumps(params).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=http_timeout) as resp:
            data = json.load(resp)
        if not data.get("ok"):
            raise RuntimeError(f"{method}: {data}")
        return data["result"]


class Model:
    """The model server: at MODEL_URL, or on a cluster node through an SSH tunnel to SSH_HOST."""

    def __init__(self, env):
        self.tunnel = None
        self.url = env.get("MODEL_URL", DEFAULT_MODEL_URL)
        if env.get("SSH_HOST"):
            node, port = ADDR_FILE.read_text().strip().split(":")
            self.url = f"http://127.0.0.1:{LOCAL_PORT}"
            self.tunnel = subprocess.Popen(["ssh", "-N", "-o", "ExitOnForwardFailure=yes",
                                            "-L", f"127.0.0.1:{LOCAL_PORT}:{node}:{port}", env["SSH_HOST"]])
        for _ in range(30):
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=5) as resp:
                    health = json.load(resp)
                    self.me_name = health["me_name"]
                    print(f"model server: {health['adapter']} at {self.url}")
                    return
            except (urllib.error.URLError, ConnectionError):
                time.sleep(2)
        raise SystemExit(f"the model server at {self.url} does not answer; is it running?")

    def close(self):
        if self.tunnel:
            self.tunnel.terminate()

    def reply(self, messages, temperature):
        req = urllib.request.Request(self.url + "/reply", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"messages": messages, "temperature": temperature}).encode())
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.load(resp)["burst"]


class Chat:
    """One user's conversation with the twin."""

    def __init__(self):
        self.conversation = []  # (datetime, {"role", "content"}) — content with reply quotes
        self.pending = []       # her messages not answered yet, with reply quotes
        self.texts = {}         # telegram message id -> body, for reply quotes
        self.temperature = DEFAULT_TEMPERATURE
        self.timer = None
        self.busy = False
        self.lock = threading.Lock()

    def reset(self):
        with self.lock:
            self.conversation, self.pending, self.texts = [], [], {}


class Bot:
    def __init__(self):
        env = load_env()
        self.tg = Telegram(env["TELEGRAM_TOKEN"])
        self.allowed = {int(x) for x in env.get("ALLOWED_USERS", "").split(",") if x.strip()}
        self.model = Model(env)
        self.chats = {}

    # ── incoming ───────────────────────────────────────────────────────────
    def handle(self, msg):
        user, chat_id = msg["from"]["id"], msg["chat"]["id"]
        text = msg.get("text", "")
        if user not in self.allowed:
            print(f"ignored a message from user {user} (not in ALLOWED_USERS)")
            if text.startswith("/start"):
                self.send(chat_id, f"Твой Telegram ID: {user}. Попроси владельца бота добавить его в ALLOWED_USERS.")
            return
        chat = self.chats.setdefault(chat_id, Chat())

        if text.startswith("/"):
            command, _, arg = text.partition(" ")
            if command in ("/start", "/help"):
                self.send(chat_id, HELP.format(me=self.model.me_name) + f"\n\nТвой Telegram ID: {user}")
            elif command == "/reset":
                chat.reset()
                self.send(chat_id, "Контекст сброшен, начинаем заново.")
            elif command == "/temp":
                try:
                    chat.temperature = min(max(float(arg.replace(",", ".")), 0.1), 1.5)
                    self.send(chat_id, f"Температура: {chat.temperature}")
                except ValueError:
                    self.send(chat_id, f"Сейчас {chat.temperature}. Пример: /temp 0.8")
            return

        if "forward_origin" in msg or "forward_date" in msg:
            return
        if "sticker" in msg:
            body = f"[стикер {msg['sticker'].get('emoji', '')}]".replace(" ]", "]")
        elif text:
            body = text
        else:
            return  # photos, voice, video notes…: the model only knows text and stickers

        with chat.lock:
            target = msg.get("reply_to_message", {}).get("message_id")
            chat.texts[msg["message_id"]] = body
            chat.pending.append(quote(body, chat.texts.get(target)))
            self.schedule(chat_id, chat)

    def schedule(self, chat_id, chat):
        if chat.timer:
            chat.timer.cancel()
        chat.timer = threading.Timer(DEBOUNCE_SECONDS, self.answer, (chat_id, chat))
        chat.timer.start()

    # ── outgoing ───────────────────────────────────────────────────────────
    def answer(self, chat_id, chat):
        with chat.lock:
            if chat.busy or not chat.pending:
                return
            chat.busy = True
            incoming, chat.pending = chat.pending, []
            now = datetime.now()
            messages = recent(chat.conversation, now) + [{"role": "user", "content": text} for text in incoming]
            chat.conversation += [(now, {"role": "user", "content": text}) for text in incoming]
        try:
            self.tg.call("sendChatAction", chat_id=chat_id, action="typing")
            burst = self.model.reply(messages, chat.temperature)
            for text in burst:
                q, body = unquote(text)
                shown = self.outgoing(body)
                if not shown:
                    continue
                self.tg.call("sendChatAction", chat_id=chat_id, action="typing")
                time.sleep(min(0.6 + 0.04 * len(shown), 4.0))  # roughly how long typing it takes
                sent = self.send(chat_id, shown, reply_to=self.find(chat, q))
                with chat.lock:
                    chat.texts[sent["message_id"]] = body
                    chat.conversation.append((datetime.now(), {"role": "assistant", "content": text}))
        except Exception as e:  # keep the bot alive; say why there is no answer
            print(f"reply failed: {e}")
            self.send(chat_id, "(модель сейчас недоступна, попробуй ещё раз чуть позже)")
            burst = []
        with chat.lock:
            chat.conversation.append((datetime.now(), {"role": "assistant", "content": WAIT}))
            chat.busy = False
            if chat.pending:  # she wrote while I was answering
                self.schedule(chat_id, chat)
        print(f"chat {chat_id}: answered {len(incoming)} messages with {len(burst)}")

    @staticmethod
    def outgoing(text):
        """What Telegram gets: a sticker as its emoji, everything else as the model wrote it."""
        sticker = STICKER.match(text)
        return (sticker.group(1) or "🙂") if sticker else text

    @staticmethod
    def find(chat, q):
        """The most recent message the quote q (possibly cut with …) was taken from."""
        if not q:
            return None
        start = " ".join(q.removesuffix("…").split())
        for message_id in sorted(chat.texts, reverse=True):
            if " ".join(chat.texts[message_id].split()).startswith(start):
                return message_id
        return None

    def send(self, chat_id, text, reply_to=None):
        params = {"chat_id": chat_id, "text": text}
        if reply_to:
            params["reply_parameters"] = {"message_id": reply_to, "allow_sending_without_reply": True}
        return self.tg.call("sendMessage", **params)

    # ── loop ───────────────────────────────────────────────────────────────
    def run(self):
        me = self.tg.call("getMe")
        print(f"bot @{me['username']} is running; allowed users: {sorted(self.allowed) or 'none yet'}")
        offset = None
        while True:
            try:
                updates = self.tg.call("getUpdates", http_timeout=45, timeout=30, offset=offset,
                                       allowed_updates=["message"])
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                print(f"telegram unreachable ({e}), retrying")
                time.sleep(5)
                continue
            for update in updates:
                offset = update["update_id"] + 1
                if "message" in update:
                    self.handle(update["message"])


def main():
    bot = Bot()
    try:
        bot.run()
    except KeyboardInterrupt:
        pass
    finally:
        bot.model.close()


if __name__ == "__main__":
    main()
