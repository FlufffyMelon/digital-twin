"""Generation: the model writes one message at a time until it answers WAIT.

Each message it writes goes back into the context, so it sees its own burst
and decides whether to write the next message or to stop and wait for her.
"""
from unsloth import FastModel  # must be imported before transformers

import json
from datetime import datetime
from pathlib import Path

import torch

from digital_twin.model.chat_format import encode_prompt, end_token_id, format_of
from digital_twin.prompt import WAIT, system_prompt

MAX_MESSAGES = 8  # safety cap on one burst


def load_config(adapter_dir):
    """digital_twin.json written by training: next to the adapter (final/) or in the run folder (checkpoint-N/)."""
    for d in (adapter_dir, adapter_dir.parent):
        if (d / "digital_twin.json").exists():
            return json.loads((d / "digital_twin.json").read_text())
    raise FileNotFoundError(f"no digital_twin.json in {adapter_dir} or {adapter_dir.parent}")


class Twin:
    def __init__(self, adapter_dir, max_length=2048, load_in_4bit=False):
        adapter_dir = Path(adapter_dir)
        config = load_config(adapter_dir)
        self.me_name, self.partner_name = config["me_name"], config["partner_name"]
        with open(adapter_dir / "adapter_config.json") as f:
            self.fmt = format_of(json.load(f)["base_model_name_or_path"])
        self.model, processor = FastModel.from_pretrained(
            model_name=str(adapter_dir), max_seq_length=max_length, load_in_4bit=load_in_4bit,
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

    def reply(self, conversation, now=None, **sampling):
        """My messages until WAIT.

        conversation: the recent turns and her new messages ({"role", "content"}, quotes included),
        without the system prompt, which is added here with the current time.
        """
        messages = [{"role": "system", "content": system_prompt(now or datetime.now(), self.me_name, self.partner_name)}]
        messages += conversation
        burst = []
        for _ in range(MAX_MESSAGES):
            text = self.next_message(messages, **sampling)
            if text == WAIT or not text:
                break
            burst.append(text)
            messages.append({"role": "assistant", "content": text})
        return burst
