"""Build the v3/v4 datasets: reply threads instead of long conversation windows.

Long windows mixed several topics at once and the model lost track of what it
was answering. Here every example is one thread of mine:

    input   what I am answering:
              - the message I replied to with Telegram's reply, or
              - Соня's latest burst, when I started writing without a reply
    context v3: only what explains the input — the reply chain above it (up to
              MAX_CHAIN links) or, if there is none, my message right before it.
              Too little: the model missed the mood and the situation and fell back
              to generic replies (blind test: fooled me in 10% of cases).
            v4: the live conversation before my answer — the last RECENT_MESSAGES
              messages, none older than RECENT_WINDOW (90% of my replies target a
              message within 5 messages and 24 minutes), with WAIT after my bursts;
              the input comes last, so it is clear what is being answered.
            v5: v4 with text only — media ([фото], [стикер], [кружок], …) and
              forwarded posts are dropped from context, input and output. Threads
              and pauses are still found on all messages; an example whose input or
              output has no text left is skipped.
    output  my message and the follow-ups written within CONTINUE_GAP, then WAIT;
            a new reply to Соня starts its own thread

    python -m digital_twin.threads --version v5   # build data/datasets/chat/v5
    python -m digital_twin.threads --version v3 --stats-only
"""
import argparse
import json
import random
import statistics
from collections import Counter
from datetime import timedelta
from pathlib import Path

from digital_twin.dataset import EXPORTS, ROOT, VAL_FROM, WAIT, system_prompt
from digital_twin.telegram import load_exports

DATASETS = ROOT / "data" / "datasets" / "chat"

CONTINUE_GAP = timedelta(minutes=3)   # 87% of pauses inside my bursts are shorter
MAX_LATENCY = timedelta(minutes=15)   # a burst started later than this is not an answer to her last message
INPUT_BURST_GAP = timedelta(minutes=2)
MAX_INPUT_BURST = 4
MAX_CHAIN = 3
PREVIOUS_MINE_WITHIN = timedelta(hours=2)  # same session as in v2
RECENT_MESSAGES = 12
RECENT_WINDOW = timedelta(minutes=30)


def role(m):
    return "assistant" if m.from_me else "user"


class Chat:
    def __init__(self, messages):
        self.messages = messages
        self.by_id = {m.id: m for m in messages}
        self.pos = {m.id: i for i, m in enumerate(messages)}

    def chain(self, m):
        """Messages m replies to, oldest first."""
        links = []
        while m.reply_to in self.by_id and len(links) < MAX_CHAIN:
            m = self.by_id[m.reply_to]
            links.insert(0, m)
        return links

    def context(self, first_input):
        """What explains the input: its reply chain, or else my message right before it."""
        links = self.chain(first_input)
        if links:
            return links
        i = self.pos[first_input.id] - 1
        while i >= 0 and not self.messages[i].from_me:
            i -= 1
        if i >= 0 and first_input.date - self.messages[i].date <= PREVIOUS_MINE_WITHIN:
            return [self.messages[i]]
        return []

    def recent(self, i, inputs, text_only=False):
        """v4/v5 context: the conversation right before position i, without the input messages."""
        skip = {m.id for m in inputs}
        keep = (lambda m: m.text_only) if text_only else (lambda m: True)
        before = [m for m in self.messages[max(0, i - 3 * RECENT_MESSAGES):i]
                  if m.id not in skip and keep(m) and self.messages[i].date - m.date <= RECENT_WINDOW][-RECENT_MESSAGES:]
        if inputs[0].reply_to and not before:  # an old target: at least its reply chain
            before = [m for m in self.chain(inputs[0]) if keep(m)]
        turns = []
        for k, m in enumerate(before):
            turns.append({"role": role(m), "content": m.quoted if text_only else m.text})
            if m.from_me and k + 1 < len(before) and not before[k + 1].from_me:
                turns.append({"role": "assistant", "content": WAIT})
        if turns and turns[-1]["role"] == "assistant" and turns[-1]["content"] != WAIT:
            turns.append({"role": "assistant", "content": WAIT})  # my burst ended before her input
        return turns

    def her_burst_before(self, i):
        """Соня's messages right before position i (pauses within INPUT_BURST_GAP)."""
        burst = []
        j = i - 1
        while j >= 0 and not self.messages[j].from_me and len(burst) < MAX_INPUT_BURST:
            if burst and burst[0].date - self.messages[j].date > INPUT_BURST_GAP:
                break
            burst.insert(0, self.messages[j])
            j -= 1
        return burst

    def output_from(self, i):
        """My message at i and its follow-ups; stops at her message, a new reply to her, or a pause."""
        out = [self.messages[i]]
        for m in self.messages[i + 1:]:
            starts_new_thread = m.reply_to in self.by_id and not self.by_id[m.reply_to].from_me
            if not m.from_me or starts_new_thread or m.date - out[-1].date > CONTINUE_GAP:
                break
            out.append(m)
        return out

    def threads(self, version="v4"):
        """(kind, context turns, input, output) for every thread of mine."""
        for i, m in enumerate(self.messages):
            if not m.from_me:
                continue
            target = self.by_id.get(m.reply_to)
            if target is not None and not target.from_me:
                inputs, kind = [target], "reply"
            elif i > 0 and not self.messages[i - 1].from_me and m.reply_to is None:
                inputs, kind = self.her_burst_before(i), "burst"
                if m.date - inputs[-1].date > MAX_LATENCY:
                    continue
            else:
                continue  # a follow-up inside a thread, or a reply to myself
            output = self.output_from(i)
            if version == "v3":
                context = [{"role": role(c), "content": c.body} for c in self.context(inputs[0])]
            elif version == "v4":
                context = self.recent(i, inputs)
            else:
                context = self.recent(i, inputs, text_only=True)
                inputs = [m for m in inputs if m.text_only]
                output = [m for m in output if m.text_only]
                if not inputs or not output:
                    continue
            yield kind, context, inputs, output


def to_example(kind, context, inputs, output):
    messages = [{"role": "system", "content": system_prompt(output[0].date), "train": False}]
    messages += [{**turn, "train": False} for turn in context]
    messages += [{"role": role(m), "content": m.body, "train": False} for m in inputs]
    messages += [{"role": "assistant", "content": m.body, "train": True} for m in output]
    messages.append({"role": "assistant", "content": WAIT, "train": True})
    return {"date": output[0].date.isoformat(), "kind": kind, "messages": messages}


def build(version="v5", exports=EXPORTS):
    chat = Chat(load_exports(exports))
    splits = {"train": [], "val": []}
    for kind, context, inputs, output in chat.threads(version):
        split = "val" if output[0].date >= VAL_FROM else "train"
        splits[split].append(to_example(kind, context, inputs, output))
    return chat, splits


def print_stats(chat, splits):
    mine = sum(m.from_me for m in chat.messages)
    for name, examples in splits.items():
        trained = sum(m["train"] and m["content"] != WAIT for e in examples for m in e["messages"])
        ctx = [sum(not m["train"] for m in e["messages"]) - 1 for e in examples]
        chars = [sum(len(m["content"]) for m in e["messages"][1:]) for e in examples]
        out = Counter(min(sum(m["train"] for m in e["messages"]) - 1, 5) for e in examples)
        print(f"{name}: {len(examples)} threads {dict(Counter(e['kind'] for e in examples))}, "
              f"my messages covered {trained}/{mine if name == 'train' else '…'}")
        print(f"  context+input messages: median {statistics.median(ctx)}, max {max(ctx)}; "
              f"chars per example: median {statistics.median(chars)}, p95 {sorted(chars)[int(.95 * len(chars))]}")
        print(f"  output messages: {dict(sorted(out.items()))}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", choices=["v3", "v4", "v5"], default="v5")
    p.add_argument("--stats-only", action="store_true")
    args = p.parse_args()
    out = DATASETS / args.version

    chat, splits = build(args.version)
    print_stats(chat, splits)
    if args.stats_only:
        return
    out.mkdir(parents=True, exist_ok=True)
    random.seed(3407)
    random.shuffle(splits["train"])
    for name, examples in splits.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for e in examples:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
