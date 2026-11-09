"""Parse Telegram chat exports into a flat, normalized list of messages.

Every message becomes one line of text the model can read: media is replaced
with placeholders, links and personal identifiers are masked, and a reply to
an older message carries a short quote of it.
"""
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ME = "Алексей Семенчук"  # author name of my messages in the export

MEDIA_PLACEHOLDERS = {
    "voice_message": "[голосовое]",
    "video_message": "[кружок]",
    "video_file": "[видео]",
    "animation": "[гифка]",
    "audio_file": "[аудио]",
}
ENTITY_PLACEHOLDERS = {
    "link": "[ссылка]",
    "phone": "[телефон]",
    "email": "[email]",
    "bank_card": "[карта]",
}
# Identifiers Telegram did not mark as entities.
MASKS = [
    (re.compile(r"\b(?:\d[ -]?){15}\d\b"), "[карта]"),
    (re.compile(r"(?<![\w+])\+?\d[\d\s\-()]{9,}\d(?!\w)"), "[телефон]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[email]"),
    (re.compile(r"https?://\S+"), "[ссылка]"),
]

MAX_CHARS = 1500           # a few messages are pasted articles; keep their start only
MAX_FORWARDED_CHARS = 300  # forwarded text is not my style, only context
QUOTE_CHARS = 50
REPLY_NEARBY = 2           # a reply to one of the last N messages needs no quote


@dataclass
class Message:
    id: int
    date: datetime
    from_me: bool
    text: str            # with a quote when it replies to an older message (v2 windows)
    body: str            # without the quote (v3 threads show the replied message itself)
    reply_to: int | None  # id of the replied message, if it is in the export


def _truncate(text, limit):
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _entities_text(msg):
    parts = []
    for entity in msg.get("text_entities", []):
        parts.append(ENTITY_PLACEHOLDERS.get(entity["type"], entity["text"]))
    text = "".join(parts)
    for pattern, placeholder in MASKS:
        text = pattern.sub(placeholder, text)
    return text.strip()


def _render(msg):
    """Text of one message with placeholders, or None if there is nothing to show."""
    parts = []
    media = msg.get("media_type")
    if media == "sticker":
        emoji = msg.get("sticker_emoji", "")
        parts.append(f"[стикер {emoji}]" if emoji else "[стикер]")
    elif media:
        parts.append(MEDIA_PLACEHOLDERS.get(media, "[файл]"))
    elif "photo" in msg:
        parts.append("[фото]")
    elif "file" in msg:
        parts.append("[файл]")
    elif "poll" in msg:
        parts.append("[опрос]")
    elif "location_information" in msg:
        parts.append("[геопозиция]")
    elif "contact_information" in msg:
        parts.append("[контакт]")

    text = _entities_text(msg)
    if "forwarded_from" in msg:
        parts.insert(0, "[пересланное]")
        text = _truncate(text, MAX_FORWARDED_CHARS)
    if text:
        parts.append(_truncate(text, MAX_CHARS))
    return " ".join(parts) if parts else None


def load_export(path):
    """Messages of one export in chronological order."""
    with open(path, encoding="utf-8") as f:
        export = json.load(f)
    raw = [m for m in export["messages"] if m["type"] == "message"]
    my_ids = {m["from_id"] for m in raw if m.get("from") == ME}
    if not my_ids:
        raise ValueError(f"{path}: no messages from {ME!r}")

    rendered = []
    for m in raw:
        text = _render(m)
        if text is not None:
            rendered.append((m, text))

    position = {m["id"]: i for i, (m, _) in enumerate(rendered)}
    messages = []
    for i, (m, body) in enumerate(rendered):
        text = body
        target = position.get(m.get("reply_to_message_id"))
        if target is not None and i - target > REPLY_NEARBY:
            quote = " ".join(rendered[target][1].split())
            text = f"↪ «{_truncate(quote, QUOTE_CHARS)}» {body}"
        messages.append(Message(
            id=m["id"],
            date=datetime.fromisoformat(m["date"]),  # local time of the export, machine-independent
            from_me=m["from_id"] in my_ids,
            text=text,
            body=body,
            reply_to=m["reply_to_message_id"] if target is not None else None,
        ))
    return messages


def load_exports(paths):
    """Merge several exports of the same chat, dropping messages present in more than one."""
    by_id = {}
    for path in paths:
        for message in load_export(Path(path)):
            by_id[message.id] = message
    return sorted(by_id.values(), key=lambda m: (m.date, m.id))
