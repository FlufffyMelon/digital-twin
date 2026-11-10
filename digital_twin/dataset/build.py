"""Build the training set from Telegram exports: one example per exchange.

    python -m digital_twin.dataset.build data/result.json --me-name Саша --partner-name Машей
    python -m digital_twin.dataset.build data/*.json --stats-only

An exchange is everything she wrote since my last message and my whole answer:

    context  the last half hour before her messages (at most RECENT_MESSAGES),
             with WAIT after each of my bursts
    input    her messages since my last one — messages written in a row are
             usually one thought and get one answer
    output   my burst: my messages until her next one or a pause over CONTINUE_GAP,
             then WAIT. The model is trained on these turns only.

A message sent as a Telegram reply carries a quote of what it replies to
(↪ «…»), on both sides: the model sees which of her messages she is replying to,
and learns to answer several topics with several replies or everything at once.
Only text and stickers are kept (see telegram.py).

The last VAL_WEEKS weeks are held out for validation. In the train split every
exchange comes up to three times — with the full context, with only its last
2–4 messages and with none — so the model also learns to start a conversation
from scratch and does not lean on a long history.

Writes train.jsonl, val.jsonl ({"date", "messages": [{"role", "content", "train"}]})
and meta.json with the names used in the system prompt.
"""
import argparse
import json
import random
import re
import statistics
from collections import Counter
from datetime import datetime, time, timedelta
from pathlib import Path

from digital_twin.dataset.telegram import load_exports, names
from digital_twin.prompt import RECENT_MESSAGES, RECENT_WINDOW, WAIT, quote, system_prompt

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "dataset"

CONTINUE_GAP = timedelta(minutes=3)  # 87% of the pauses inside my bursts are shorter
MAX_INPUT_MESSAGES = 8               # her messages since my last one; more than 4 in a row is rare
SHORT_CONTEXT = (2, 4)               # messages kept in the short-context variant
VAL_WEEKS = 4
SEED = 3407


def role(m):
    return "assistant" if m.from_me else "user"


class Chat:
    def __init__(self, messages):
        self.messages = messages
        self.by_id = {m.id: m for m in messages}

    def say(self, m):
        """Text of a message as the model sees it: a Telegram reply shows what it replies to."""
        target = self.by_id.get(m.reply_to)
        return quote(m.body, target.body if target is not None and target.text_only else None)

    def before(self, j):
        """Context: text messages of the last RECENT_WINDOW before position j, WAIT after my bursts."""
        start = self.messages[j].date
        before = [m for m in self.messages[max(0, j - 3 * RECENT_MESSAGES):j]
                  if m.text_only and start - m.date <= RECENT_WINDOW][-RECENT_MESSAGES:]
        turns = []
        for k, m in enumerate(before):
            turns.append({"role": role(m), "content": self.say(m)})
            if m.from_me and (k + 1 == len(before) or not before[k + 1].from_me):
                turns.append({"role": "assistant", "content": WAIT})
        return turns

    def exchanges(self):
        """(context, her messages, my burst) for every burst of mine that follows her messages."""
        for i, m in enumerate(self.messages):
            if not m.from_me or i == 0 or self.messages[i - 1].from_me:
                continue
            j = i - 1
            while j > 0 and not self.messages[j - 1].from_me:
                j -= 1
            hers = [x for x in self.messages[j:i] if x.text_only][-MAX_INPUT_MESSAGES:]
            mine = [m]
            for x in self.messages[i + 1:]:
                if not x.from_me or x.date - mine[-1].date > CONTINUE_GAP:
                    break
                mine.append(x)
            mine = [x for x in mine if x.text_only]
            if hers and mine:
                yield self.before(j), hers, mine


def context_variants(context, rng):
    """The full context, its last 2–4 messages, and none (distinct ones only)."""
    def last(n):
        kept, count = [], 0
        for turn in reversed(context):
            if turn["content"] != WAIT:
                if count == n:
                    break
                count += 1
            kept.insert(0, turn)
        return kept
    variants = []
    for ctx in (context, last(rng.randint(*SHORT_CONTEXT)), []):
        if ctx not in variants:
            variants.append(ctx)
    return variants


def example(chat, context, hers, mine, me, partner):
    messages = [{"role": "system", "content": system_prompt(mine[0].date, me, partner), "train": False}]
    messages += [{**turn, "train": False} for turn in context]
    messages += [{"role": "user", "content": chat.say(m), "train": False} for m in hers]
    messages += [{"role": "assistant", "content": chat.say(m), "train": True} for m in mine]
    messages.append({"role": "assistant", "content": WAIT, "train": True})
    return {"date": mine[0].date.isoformat(), "messages": messages}


def build(exports, me_name, partner_name, me=None, val_weeks=VAL_WEEKS):
    chat = Chat(load_exports(exports, me))
    val_from = datetime.combine(chat.messages[-1].date.date() - timedelta(weeks=val_weeks), time())
    rng = random.Random(SEED)
    splits = {"train": [], "val": []}
    for context, hers, mine in chat.exchanges():
        if mine[0].date >= val_from:
            splits["val"].append(example(chat, context, hers, mine, me_name, partner_name))
        else:
            splits["train"] += [example(chat, ctx, hers, mine, me_name, partner_name)
                                for ctx in context_variants(context, rng)]
    return chat, splits, val_from


def print_stats(chat, splits, val_from):
    first, last = chat.messages[0].date, chat.messages[-1].date
    print(f"messages: {len(chat.messages)} ({sum(m.from_me for m in chat.messages)} mine, "
          f"{sum(m.text_only for m in chat.messages)} text or stickers), {first:%Y-%m-%d} … {last:%Y-%m-%d}; "
          f"validation from {val_from:%Y-%m-%d}")
    for name, examples in splits.items():
        ctx = [sum(not m["train"] for m in e["messages"]) - 1 for e in examples]
        out = Counter(min(sum(m["train"] for m in e["messages"]) - 1, 5) for e in examples)
        chars = sorted(sum(len(m["content"]) for m in e["messages"][1:]) for e in examples)
        print(f"{name}: {len(examples)} examples; context+input messages: median {statistics.median(ctx)}, "
              f"max {max(ctx)}; characters: median {statistics.median(chars)}, p95 {chars[int(.95 * len(chars))]}")
        print(f"  my messages per answer (5 = 5+): {dict(sorted(out.items()))}")


def first_name(name):
    return re.sub(r"[^\w-]", "", name.split()[0]) if name.split() else name


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("exports", nargs="+", type=Path, help="result.json of one or more exports of the same chat")
    p.add_argument("--me-name", help="your name in the system prompt (default: first name from the export)")
    p.add_argument("--partner-name", help="the partner's name as it reads after «переписываешься с» "
                                          "(instrumental case, e.g. Машей; default: first name from the export)")
    p.add_argument("--me", help="your name exactly as the export shows it; only needed for group exports")
    p.add_argument("--val-weeks", type=int, default=VAL_WEEKS, help="weeks at the end held out for validation")
    p.add_argument("--out", type=Path, default=OUT_DIR)
    p.add_argument("--stats-only", action="store_true", help="print statistics, write nothing")
    args = p.parse_args()

    export_me, export_partner = names(args.exports[-1])
    me_name = args.me_name or first_name(args.me or export_me)
    partner_name = args.partner_name or first_name(export_partner)
    print(f"system prompt: {system_prompt(datetime.now(), me_name, partner_name)}")

    chat, splits, val_from = build(args.exports, me_name, partner_name, args.me, args.val_weeks)
    print_stats(chat, splits, val_from)
    if args.stats_only:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    random.Random(SEED).shuffle(splits["train"])
    for name, examples in splits.items():
        with open(args.out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for e in examples:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    meta = {"me_name": me_name, "partner_name": partner_name, "exports": [str(p) for p in args.exports],
            "val_from": val_from.isoformat(), **{name: len(examples) for name, examples in splits.items()}}
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
