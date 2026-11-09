"""Build the chat dataset: Telegram messages -> conversation windows (jsonl).

Each Telegram message is its own turn. After the last message of each of my
bursts comes an extra assistant turn with WAIT: the model learns to write one
message at a time and to decide when to stop and wait for a reply.

    python -m digital_twin.dataset                # build data/datasets/chat/v2
    python -m digital_twin.dataset --stats-only   # print statistics, write nothing
"""
import argparse
import json
import random
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from digital_twin.telegram import load_exports

ROOT = Path(__file__).resolve().parent.parent
EXPORTS = [
    ROOT / "data" / "telegram_export_2025-05-03.json",
    ROOT / "data" / "telegram_export_2026-11-09.json",
]
OUT_DIR = ROOT / "data" / "datasets" / "chat" / "v2"

WAIT = "<wait>"
SESSION_GAP = timedelta(hours=2)
VAL_FROM = datetime(2026, 10, 12)  # last four weeks of the newest export
WINDOW_TOKENS = 1500               # estimated, leaves headroom under max_length 2048
CONTEXT_MESSAGES = 12              # messages carried over from the previous window, not trained on
CONTEXT_TOKENS = 400               # ...but no more than this many estimated tokens of them
CHARS_PER_TOKEN = 2.9              # measured on this chat for Gemma 4 / Qwen3.5 tokenizers
TOKENS_PER_TURN = 7                # chat-template header and footer

WEEKDAYS = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]


def system_prompt(now):
    return (
        "Ты — Леша. Ты переписываешься с Соней в Telegram. Пиши так, как написал бы Леша: "
        "по одному сообщению за раз, а когда сказать больше нечего, ответь " + WAIT + ". "
        f"Сейчас {WEEKDAYS[now.weekday()]}, {now:%d.%m.%Y %H:%M}."
    )


@dataclass
class Turn:
    role: str  # "user" (Соня) or "assistant" (я)
    content: str
    date: datetime


@dataclass
class Session:
    turns: list = field(default_factory=list)

    @property
    def start(self):
        return self.turns[0].date


def split_sessions(messages):
    sessions = []
    for m in messages:
        if not sessions or m.date - sessions[-1].turns[-1].date > SESSION_GAP:
            sessions.append(Session())
        sessions[-1].turns.append(Turn("assistant" if m.from_me else "user", m.text, m.date))
    return sessions


def add_wait_turns(session):
    """Insert WAIT after every burst of my messages and after an unanswered burst at the end."""
    turns = []
    for i, turn in enumerate(session.turns):
        turns.append(turn)
        nxt = session.turns[i + 1] if i + 1 < len(session.turns) else None
        if turn.role == "assistant" and (nxt is None or nxt.role == "user"):
            turns.append(Turn("assistant", WAIT, turn.date))
    if turns[-1].role == "user":
        turns.append(Turn("assistant", WAIT, turns[-1].date))
    session.turns = turns
    return session


def estimate_tokens(turn):
    return len(turn.content) / CHARS_PER_TOKEN + TOKENS_PER_TURN


def windows(session):
    """Cut a session into windows; every turn is trained on in exactly one window."""
    turns = session.turns
    start = 0
    while start < len(turns):
        context = []
        for turn in reversed(turns[max(0, start - CONTEXT_MESSAGES):start]):
            if sum(map(estimate_tokens, context)) + estimate_tokens(turn) > CONTEXT_TOKENS:
                break
            context.insert(0, turn)
        budget = WINDOW_TOKENS - sum(estimate_tokens(t) for t in context)
        end = start
        while end < len(turns) and (end == start or budget - estimate_tokens(turns[end]) >= 0):
            budget -= estimate_tokens(turns[end])
            end += 1
        trained = turns[start:end]
        if any(t.role == "assistant" for t in trained):
            yield context, trained
        start = end


def to_example(context, trained):
    first = (context or trained)[0].date
    messages = [{"role": "system", "content": system_prompt(first), "train": False}]
    messages += [{"role": t.role, "content": t.content, "train": False} for t in context]
    messages += [{"role": t.role, "content": t.content, "train": t.role == "assistant"} for t in trained]
    return {"date": first.isoformat(), "messages": messages}


def build(exports):
    sessions = [add_wait_turns(s) for s in split_sessions(load_exports(exports))]
    splits = {"train": [], "val": []}
    for session in sessions:
        split = "val" if session.start >= VAL_FROM else "train"
        splits[split].extend(to_example(c, t) for c, t in windows(session))
    return sessions, splits


def print_stats(sessions, splits):
    turns = [t for s in sessions for t in s.turns]
    mine = [t for t in turns if t.role == "assistant" and t.content != WAIT]
    waits = sum(t.content == WAIT for t in turns)
    print(f"sessions: {len(sessions)}, turns: {len(turns)} "
          f"(user {sum(t.role == 'user' for t in turns)}, my messages {len(mine)}, {WAIT} {waits})")
    print(f"messages per session: median {statistics.median(len(s.turns) for s in sessions)}, "
          f"max {max(len(s.turns) for s in sessions)}")
    for name, examples in splits.items():
        est = [sum(len(m["content"]) / CHARS_PER_TOKEN + TOKENS_PER_TURN for m in e["messages"]) for e in examples]
        trained = sum(m["train"] for e in examples for m in e["messages"])
        print(f"{name}: {len(examples)} windows, {trained} trained turns, "
              f"~{sum(est) / 1e6:.2f}M tokens, window median ~{statistics.median(est):.0f} max ~{max(est):.0f} tokens")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()

    sessions, splits = build(EXPORTS)
    print_stats(sessions, splits)
    if args.stats_only:
        return

    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(3407)
    for name, examples in splits.items():
        if name == "train":
            random.shuffle(examples)
        with open(args.out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
