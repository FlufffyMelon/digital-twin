"""Render conversations in each model's chat format, with per-turn loss masks.

The models' own chat templates do not fit: Gemma 4 merges consecutive assistant
turns into one, and Qwen3.5 puts an empty think block into the last assistant
turn only. Here every turn is rendered the same way, in training and in
generation, and each piece is tokenized separately so both sides match.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ChatFormat:
    bos: str
    headers: dict   # role -> text that opens a turn
    end: str        # token that closes a turn; generation stops on it
    separator: str  # text between turns
    sampling: dict  # recommended generation settings


FORMATS = {
    "gemma4": ChatFormat(
        bos="<bos>",
        headers={"system": "<|turn>system\n", "user": "<|turn>user\n", "assistant": "<|turn>model\n"},
        end="<turn|>",
        separator="\n",
        sampling={"temperature": 1.0, "top_p": 0.95, "top_k": 64},
    ),
    "qwen3_5": ChatFormat(
        bos="",
        headers={
            "system": "<|im_start|>system\n",
            "user": "<|im_start|>user\n",
            # Non-thinking mode: the same empty think block opens every assistant turn.
            "assistant": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        },
        end="<|im_end|>",
        separator="\n",
        sampling={"temperature": 0.7, "top_p": 0.8, "top_k": 20},
    ),
}


# --model name -> (base model, chat format)
MODELS = {
    "gemma-4-e4b": ("unsloth/gemma-4-E4B-it", "gemma4"),
    "gemma-4-e4b-base": ("unsloth/gemma-4-E4B", "gemma4"),
    "gemma-4-12b": ("unsloth/gemma-4-12b-it", "gemma4"),
    "qwen3.5-4b": ("unsloth/Qwen3.5-4B", "qwen3_5"),
    "qwen3.5-9b": ("unsloth/Qwen3.5-9B", "qwen3_5"),
}


def format_of(base_model):
    """Chat format for a base model id, as written to an adapter's adapter_config.json."""
    for name, fmt in MODELS.values():
        if base_model.removesuffix("-unsloth-bnb-4bit") == name:
            return FORMATS[fmt]
    raise KeyError(f"unknown base model {base_model!r}, add it to MODELS")


def _pieces(fmt, messages):
    """(text, trained) pieces of a rendered conversation."""
    pieces = [(fmt.bos, False)] if fmt.bos else []
    for m in messages:
        pieces.append((fmt.headers[m["role"]], False))
        pieces.append((m["content"] + fmt.end, m.get("train", False)))
        pieces.append((fmt.separator, False))
    return pieces


def encode(tokenizer, fmt, messages, max_length=None):
    """input_ids and labels (-100 outside trained turns) for one conversation."""
    input_ids, labels = [], []
    for text, trained in _pieces(fmt, messages):
        ids = tokenizer(text, add_special_tokens=False).input_ids
        input_ids += ids
        labels += ids if trained else [-100] * len(ids)
    if max_length is not None:
        input_ids, labels = input_ids[:max_length], labels[:max_length]
    return {"input_ids": input_ids, "labels": labels}


def encode_prompt(tokenizer, fmt, messages):
    """Token ids of a conversation followed by an open assistant turn."""
    ids = encode(tokenizer, fmt, messages)["input_ids"]
    return ids + tokenizer(fmt.headers["assistant"], add_special_tokens=False).input_ids


def end_token_id(tokenizer, fmt):
    ids = tokenizer(fmt.end, add_special_tokens=False).input_ids
    assert len(ids) == 1, f"{fmt.end!r} must be a single token, got {ids}"
    return ids[0]
