"""Talk to a trained adapter, one message at a time.

The model writes a message, sees it in the context, and either writes the next
one or answers WAIT, after which it stays silent until a new message arrives.

    python -m digital_twin.chat outputs/<run>/final              # interactive
    python -m digital_twin.chat outputs/<run>/final --eval 30    # held-out moments vs. what I actually wrote

Interactive: every line is one message from Соня; an empty line hands the turn over.
"""
from unsloth import FastModel  # must be imported before transformers

import argparse
import json
import random
import readline  # noqa: F401 -- UTF-8-aware line editing for input(); srun --pty ttys lack iutf8
import sys
from datetime import datetime
from pathlib import Path

import torch

from digital_twin.chat_format import encode_prompt, end_token_id, format_of
from digital_twin.dataset import OUT_DIR, WAIT, system_prompt

MAX_MESSAGES = 8  # safety cap on one burst


class Twin:
    def __init__(self, adapter_dir, max_length=2048):
        # Works for final/ and for any intermediate checkpoint-N/.
        with open(Path(adapter_dir) / "adapter_config.json") as f:
            self.fmt = format_of(json.load(f)["base_model_name_or_path"])
        self.model, processor = FastModel.from_pretrained(
            model_name=str(adapter_dir), max_seq_length=max_length, load_in_4bit=False,
        )
        FastModel.for_inference(self.model)
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.end_id = end_token_id(self.tokenizer, self.fmt)

    @torch.no_grad()
    def next_message(self, messages, **sampling):
        ids = encode_prompt(self.tokenizer, self.fmt, messages)
        input_ids = torch.tensor([ids], device=self.model.device)
        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=256,
            do_sample=True,
            eos_token_id=self.end_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **{**self.fmt.sampling, **sampling},
        )
        return self.tokenizer.decode(out[0, len(ids):], skip_special_tokens=True).strip()

    def reply(self, messages, **sampling):
        """My messages until WAIT; `messages` is extended in place."""
        burst = []
        for _ in range(MAX_MESSAGES):
            text = self.next_message(messages, **sampling)
            if text == WAIT or not text:
                break
            burst.append(text)
            messages.append({"role": "assistant", "content": text})
        messages.append({"role": "assistant", "content": WAIT})
        return burst


def interactive(twin):
    sys.stdin.reconfigure(errors="replace")  # a stray broken byte must not end the chat
    messages = [{"role": "system", "content": system_prompt(datetime.now())}]
    print("Пиши сообщения построчно; пустая строка — передать ход. /reset — новый диалог, /exit — выход.")
    pending = []
    while True:
        line = input("Соня: ").strip()
        if line == "/exit":
            return
        if line == "/reset":
            messages, pending = messages[:1], []
            continue
        if line:
            pending.append(line)
            continue
        messages += [{"role": "user", "content": text} for text in pending]
        pending = []
        for text in twin.reply(messages):
            print(f"Леша: {text}")
        print("Леша: ⏸")


def evaluate(twin, n, out_path):
    """Generate at moments right after Соня's bursts in the validation split."""
    with open(OUT_DIR / "val.jsonl", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]
    moments = []
    for e in examples:
        msgs = e["messages"]
        for i in range(1, len(msgs)):
            if msgs[i - 1]["role"] == "user" and msgs[i]["role"] == "assistant":
                real = []
                for m in msgs[i:]:
                    if m["role"] != "assistant" or m["content"] == WAIT:
                        break
                    real.append(m["content"])
                moments.append((msgs[:i], real))
    random.seed(0)
    lines = []
    for context, real in random.sample(moments, min(n, len(moments))):
        generated = twin.reply([dict(m) for m in context])
        lines.append("### Контекст\n")
        lines += [f"- {'Соня' if m['role'] == 'user' else 'Леша'}: {m['content']}"
                  for m in context[-8:] if m["role"] != "system" and m["content"] != WAIT]
        lines.append("\n**Как было:**\n")
        lines += [f"> {t}" for t in real] or ["> ⏸"]
        lines.append("\n**Модель:**\n")
        lines += [f"> {t}" for t in generated] or ["> ⏸"]
        lines.append("\n---\n")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(lines)} lines written to {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("adapter_dir", type=Path)
    p.add_argument("--eval", type=int, metavar="N", help="sample N held-out moments instead of chatting")
    args = p.parse_args()

    twin = Twin(args.adapter_dir)
    if args.eval:
        evaluate(twin, args.eval, args.adapter_dir / "samples.md")
    else:
        interactive(twin)


if __name__ == "__main__":
    main()
