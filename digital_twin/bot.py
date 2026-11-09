"""Telegram bot for the twin (runs on the laptop: the cluster cannot reach Telegram).

    sbatch slurm/serve.sh outputs/<run>/final --load-in-4bit   # on mipt: the model server
    python3 -m digital_twin.bot                                 # here: tunnel + bot

Settings live in .env at the repo root (never in git or on the cluster):

    TELEGRAM_TOKEN=123456:ABC...      # from @BotFather
    ALLOWED_USERS=111111111,222222222 # Telegram user ids; /start shows yours

Each allowed user has their own conversation. Like in the real chat, the bot
waits for a short silence after her messages, then writes one message at a
time until the model says WAIT. A reply to an older message carries a quote,
as in the dataset; stickers come in and go out as emoji; other media and
forwarded posts are ignored (the v5 model was trained on text and stickers).
"""
import json
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime

from digital_twin.conversation import prompt, quote
from digital_twin.dataset import ROOT, WAIT

ENV_FILE = ROOT / ".env"
ADDR_FILE = ROOT / "logs" / "serve.addr"
SSH_HOST = "mipt"
LOCAL_PORT = 8765
DEBOUNCE_SECONDS = 4.0   # her bursts: 65% of pauses inside a burst are under 30 s, most under a few seconds
DEFAULT_TEMPERATURE = 0.7
STICKER = re.compile(r"^\[стикер ?(.*?)\]$")
# Media a v4 model may still "send"; the bot cannot, so the placeholder is dropped.
MEDIA = re.compile(r"\[(?:фото|кружок|голосовое|видео|гифка|файл|аудио|пересланное|опрос|геопозиция|контакт|ссылка)\]\s*")
HELP = (
    "Я цифровой двойник Леши.\n"
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
            env[key.strip()] = value.strip()
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
    """The model server on the cluster, reached through an SSH tunnel."""

    def __init__(self):
        node, port = ADDR_FILE.read_text().strip().split(":")
        self.url = f"http://127.0.0.1:{LOCAL_PORT}"
        self.tunnel = subprocess.Popen(
            ["ssh", "-N", "-o", "ExitOnForwardFailure=yes", "-L", f"127.0.0.1:{LOCAL_PORT}:{node}:{port}", SSH_HOST])
        for _ in range(30):
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=5) as resp:
                    print(f"model server: {json.load(resp)['adapter']} on {node}:{port}")
                    return
            except (urllib.error.URLError, ConnectionError):
                time.sleep(2)
        raise SystemExit(f"model server on {node}:{port} does not answer; is the slurm job running?")

    def reply(self, messages, temperature):
        req = urllib.request.Request(self.url + "/reply", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"messages": messages, "temperature": temperature}).encode())
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.load(resp)["burst"]


class Chat:
    """One user's conversation with the twin."""

    def __init__(self):
        self.conversation = []  # (datetime, {"role", "content"}) — content with reply quotes
        self.pending = []       # her messages not answered yet: (body, quoted)
        self.texts = {}         # telegram message id -> (position, body), for reply quotes
        self.count = 0          # messages so far, for the quote distance
        self.temperature = DEFAULT_TEMPERATURE
        self.timer = None
        self.busy = False
        self.lock = threading.Lock()

    def reset(self):
        with self.lock:
            self.conversation, self.pending, self.texts, self.count = [], [], {}, 0


class Bot:
    def __init__(self):
        env = load_env()
        self.tg = Telegram(env["TELEGRAM_TOKEN"])
        self.allowed = {int(x) for x in env.get("ALLOWED_USERS", "").split(",") if x.strip()}
        self.model = Model()
        self.chats = {}

    # ── incoming ───────────────────────────────────────────────────────────
    def handle(self, msg):
        user, chat_id = msg["from"]["id"], msg["chat"]["id"]
        text = msg.get("text", "")
        if user not in self.allowed:
            print(f"ignored a message from user {user} (not in ALLOWED_USERS)")
            if text.startswith("/start"):
                self.send(chat_id, f"Твой Telegram ID: {user}. Попроси Лешу добавить его в бота.")
            return
        chat = self.chats.setdefault(chat_id, Chat())

        if text.startswith("/"):
            command, _, arg = text.partition(" ")
            if command in ("/start", "/help"):
                self.send(chat_id, HELP + f"\n\nТвой Telegram ID: {user}")
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
            position = chat.count + len(chat.pending)
            quoted = body
            target = msg.get("reply_to_message", {}).get("message_id")
            if target in chat.texts:
                target_pos, target_text = chat.texts[target]
                quoted = quote(body, target_text, position - target_pos)
            chat.texts[msg["message_id"]] = (position, body)
            chat.pending.append((body, quoted))
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
            messages = prompt(chat.conversation, [body for body, _ in incoming], now)
            chat.conversation += [(now, {"role": "user", "content": quoted}) for _, quoted in incoming]
            chat.count += len(incoming)
        try:
            self.tg.call("sendChatAction", chat_id=chat_id, action="typing")
            burst = self.model.reply(messages, chat.temperature)
            for text in burst:
                shown = self.outgoing(text)
                if not shown:
                    continue
                self.tg.call("sendChatAction", chat_id=chat_id, action="typing")
                time.sleep(min(0.6 + 0.04 * len(shown), 4.0))  # roughly how long typing it takes
                sent = self.send(chat_id, shown)
                with chat.lock:
                    chat.texts[sent["message_id"]] = (chat.count, text)
                    chat.count += 1
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
        """What Telegram gets: a sticker as its emoji, other media placeholders dropped."""
        sticker = STICKER.match(text)
        if sticker:
            return sticker.group(1) or "🙂"
        return MEDIA.sub("", text).strip()

    def send(self, chat_id, text):
        return self.tg.call("sendMessage", chat_id=chat_id, text=text)

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
        bot.model.tunnel.terminate()


if __name__ == "__main__":
    main()
