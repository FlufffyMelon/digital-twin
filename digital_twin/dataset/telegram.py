"""Parse a Telegram Desktop export of a personal chat (result.json) into messages.

The model only learns from plain text and stickers: a message counts as text
when it is written by hand — not forwarded, without a photo, video, voice
message or file, and not just a link. Other messages stay in the list (they
still split the conversation into turns) but are never shown to the model.
Links, phone numbers, emails and card numbers inside the text are masked.
"""
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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
PLACEHOLDERS = {*ENTITY_PLACEHOLDERS.values(), *(p for _, p in MASKS)}
MEDIA_KEYS = ("media_type", "photo", "file", "poll", "location_information", "contact_information")

MAX_CHARS = 1500  # a few messages are pasted articles; keep their start only


@dataclass
class Message:
    id: int
    date: datetime
    from_me: bool
    body: str             # text with masks; a sticker is [стикер 😂]
    reply_to: int | None  # id of the replied message, if it is in the export
    text_only: bool       # text or a sticker written by hand: the only messages the model sees


def _truncate(text, limit):
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _text(msg):
    text = "".join(ENTITY_PLACEHOLDERS.get(e["type"], e["text"]) for e in msg.get("text_entities", []))
    for pattern, placeholder in MASKS:
        text = pattern.sub(placeholder, text)
    return text.strip()


def _parse(msg):
    """(body, text_only) of one message, or None when it has neither text nor media."""
    text = _text(msg)
    if msg.get("media_type") == "sticker":
        emoji = msg.get("sticker_emoji", "")
        return (f"[стикер {emoji}]" if emoji else "[стикер]"), "forwarded_from" not in msg
    if any(k in msg for k in MEDIA_KEYS):
        return _truncate(text, MAX_CHARS), False
    if not text and "forwarded_from" not in msg:
        return None
    bare = text
    for placeholder in PLACEHOLDERS:
        bare = bare.replace(placeholder, "")
    return _truncate(text, MAX_CHARS), "forwarded_from" not in msg and bool(bare.strip())


def load_export(path, me=None):
    """Messages of one export in chronological order.

    me: your name as the export shows it; by default, whoever is not the chat partner
    (a personal chat export is named after the partner and carries their id).
    """
    with open(path, encoding="utf-8") as f:
        export = json.load(f)
    raw = [m for m in export["messages"] if m["type"] == "message"]
    if me is not None:
        my_ids = {m["from_id"] for m in raw if m.get("from") == me}
    elif export.get("type") == "personal_chat":
        my_ids = {m["from_id"] for m in raw} - {f"user{export['id']}"}
    else:
        raise ValueError(f"{path}: not a personal chat, pass your name as it appears in the export (--me)")
    if len(my_ids) != 1:
        raise ValueError(f"{path}: cannot tell your messages apart (senders {my_ids or 'none'}), pass --me")

    parsed = [(m, p) for m in raw if (p := _parse(m)) is not None]
    ids = {m["id"] for m, _ in parsed}
    return [
        Message(
            id=m["id"],
            date=datetime.fromisoformat(m["date"]),  # local time of the export, machine-independent
            from_me=m["from_id"] in my_ids,
            body=body,
            reply_to=m["reply_to_message_id"] if m.get("reply_to_message_id") in ids else None,
            text_only=text_only,
        )
        for m, (body, text_only) in parsed
    ]


def names(path):
    """(your name, partner's name) as the export shows them."""
    with open(path, encoding="utf-8") as f:
        export = json.load(f)
    partner = f"user{export.get('id')}"
    senders = {m["from_id"]: m.get("from") or "" for m in export["messages"] if m["type"] == "message"}
    mine = [name for sender, name in senders.items() if sender != partner]
    return (mine[0] if mine else ""), export.get("name") or ""


def load_exports(paths, me=None):
    """Merge several exports of the same chat, dropping messages present in more than one."""
    by_id = {}
    for path in paths:
        for message in load_export(Path(path), me):
            by_id[message.id] = message
    return sorted(by_id.values(), key=lambda m: (m.date, m.id))
