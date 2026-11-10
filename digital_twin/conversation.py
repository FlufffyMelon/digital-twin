"""Live conversation state shared by the chat front-ends (terminal chat, Telegram bot).

Stdlib only, so the Telegram bot can run on a laptop without the GPU stack.
The prompt is built like a v6 training example: the system prompt, the last
half hour of the chat (WAIT after my bursts), then her new messages; Telegram
replies carry their quote (↪ «…») everywhere.
"""
from datetime import datetime

from digital_twin.dataset import WAIT, system_prompt
from digital_twin.telegram import QUOTE_CHARS
from digital_twin.threads import RECENT_MESSAGES, RECENT_WINDOW


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


def prompt(conversation, incoming, now=None):
    """Messages for the model: system prompt, recent context, her new messages last (as in v6, quotes included)."""
    now = now or datetime.now()
    return ([{"role": "system", "content": system_prompt(now)}] + recent(conversation, now)
            + [{"role": "user", "content": text} for text in incoming])


def quote(text, quoted_text):
    """A reply shows what it replies to, as in the v6 dataset: ↪ «start of that message» text."""
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
