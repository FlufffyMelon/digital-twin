"""What the model sees: the system prompt, reply quotes and the recent conversation.

Shared by the dataset builder and the chat front-ends, so a live conversation is
shown to the model exactly like a training example. Stdlib only: the Telegram
bot imports it on a laptop without the GPU stack.

The prompt is in Russian, like the chat it was made for; translate SYSTEM_PROMPT
and WEEKDAYS for a chat in another language.
"""
from datetime import timedelta

WAIT = "<wait>"  # the model's turn that means "nothing more to say, wait for her"

QUOTE_CHARS = 50                      # a reply shows the start of the message it replies to
RECENT_MESSAGES = 12                  # context: at most this many messages...
RECENT_WINDOW = timedelta(minutes=30)  # ...and none older than this

SYSTEM_PROMPT = (
    "Ты — {me}. Ты переписываешься с {partner} в Telegram. Пиши так, как написал бы {me}: "
    "по одному сообщению за раз, а когда сказать больше нечего, ответь " + WAIT + ". "
    "Сейчас {weekday}, {now:%d.%m.%Y %H:%M}."
)
WEEKDAYS = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]


def system_prompt(now, me, partner):
    """me / partner: the names the model knows the two sides by (partner in the instrumental case for Russian)."""
    return SYSTEM_PROMPT.format(me=me, partner=partner, weekday=WEEKDAYS[now.weekday()], now=now)


def quote(text, quoted_text):
    """A Telegram reply as the model sees it: ↪ «start of the replied message» text."""
    if quoted_text is None:
        return text
    q = " ".join(quoted_text.split())
    q = q if len(q) <= QUOTE_CHARS else q[:QUOTE_CHARS].rstrip() + "…"
    return f"↪ «{q}» {text}"


def unquote(text):
    """(quote, text) of a message the model wrote; quote is None when it is not a reply."""
    if text.startswith("↪ «") and "» " in text:
        q, rest = text[3:].split("» ", 1)
        return q, rest
    return None, text


def recent(conversation, now):
    """The last RECENT_MESSAGES messages of the last RECENT_WINDOW, with their WAIT turns.

    conversation: list of (datetime, {"role", "content"}) in chronological order.
    """
    kept, n = [], 0
    for date, turn in reversed(conversation):
        if now - date > RECENT_WINDOW or n == RECENT_MESSAGES:
            break
        kept.insert(0, turn)
        n += turn["content"] != WAIT
    while kept and kept[0]["content"] == WAIT:
        kept.pop(0)
    return kept
