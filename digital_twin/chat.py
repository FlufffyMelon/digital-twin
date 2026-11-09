"""Talk to a trained adapter, one message at a time.

The model writes a message, sees it in the context, and either writes the next
one or answers WAIT, after which it stays silent until a new message arrives.

    python -m digital_twin.chat outputs/<run>/final                       # starts from the last 40 real messages
    python -m digital_twin.chat outputs/<run>/final --history 0 --temperature 0.6

Interactive: every line is one message from Соня; an empty line hands the turn over.
"""
from unsloth import FastModel  # must be imported before transformers

import argparse
import json
import readline  # noqa: F401 -- UTF-8-aware line editing for input(); srun --pty ttys lack iutf8
import sys
from datetime import datetime
from pathlib import Path

import torch

from digital_twin.chat_format import encode_prompt, end_token_id, format_of
from digital_twin.dataset import WAIT, system_prompt

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


def real_history(n):
    """My last n real chat turns (with WAIT turns), so the model knows what is going on."""
    from digital_twin.dataset import EXPORTS, add_wait_turns, split_sessions
    from digital_twin.telegram import load_exports
    turns = [t for s in split_sessions(load_exports(EXPORTS)) for t in add_wait_turns(s).turns]
    return [{"role": t.role, "content": t.content} for t in turns[-n:]] if n else []


def interactive(twin, history=0, **sampling):
    sys.stdin.reconfigure(errors="replace")  # a stray broken byte must not end the chat
    start = [{"role": "system", "content": system_prompt(datetime.now())}] + real_history(history)
    for m in start[-8:]:
        if m["role"] != "system" and m["content"] != WAIT:
            print(f"{'Соня' if m['role'] == 'user' else 'Леша'} (история): {m['content']}")
    print("Пиши сообщения построчно; пустая строка — передать ход. /reset — новый диалог, /exit — выход.")
    messages, pending = list(start), []
    while True:
        line = input("Соня: ").strip()
        if line == "/exit":
            return
        if line == "/reset":
            messages, pending = list(start), []
            continue
        if line:
            pending.append(line)
            continue
        messages += [{"role": "user", "content": text} for text in pending]
        pending = []
        for text in twin.reply(messages, **sampling):
            print(f"Леша: {text}")
        print("Леша: ⏸")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("adapter_dir", type=Path)
    p.add_argument("--history", type=int, default=40, help="real chat messages to start from (0 — empty chat)")
    p.add_argument("--temperature", type=float, help="default: the model's recommended value")
    args = p.parse_args()

    sampling = {"temperature": args.temperature} if args.temperature is not None else {}
    interactive(Twin(args.adapter_dir), args.history, **sampling)


if __name__ == "__main__":
    main()
