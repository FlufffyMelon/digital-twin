"""LoRA fine-tuning on the chat dataset (see digital_twin/dataset.py).

    python -m digital_twin.train --model gemma-4-e4b
    python -m digital_twin.train --model qwen3.5-4b --max-steps 20   # smoke test
"""
from unsloth import FastModel  # must be imported before transformers / trl

import argparse
import datetime
import json
from pathlib import Path

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from digital_twin.chat_format import FORMATS, MODELS, encode

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "datasets" / "chat" / "v2"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--run-name", default=None)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--load-in-4bit", action="store_true", help="QLoRA; default is 16-bit LoRA")
    p.add_argument("--r", type=int, default=32)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--warmup-steps", type=int, default=15)  # ~5% of a 2-epoch run
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--save-total-limit", type=int, default=2, help="0 — keep every checkpoint")
    return p.parse_args()


def main():
    args = parse_args()
    model_name, format_name = MODELS[args.model]
    fmt = FORMATS[format_name]
    run_name = args.run_name or f"{args.model}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    output_dir = ROOT / "outputs" / run_name

    model, processor = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_length,
        load_in_4bit=args.load_in_4bit,
        full_finetuning=False,
    )
    tokenizer = getattr(processor, "tokenizer", processor)  # multimodal models return a processor

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # text only
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    data = load_dataset("json", data_files={
        "train": str(args.data_dir / "train.jsonl"),
        "val": str(args.data_dir / "val.jsonl"),
    })
    data = data.map(
        lambda ex: encode(tokenizer, fmt, ex["messages"], args.max_length),
        remove_columns=data["train"].column_names,
    )
    sample = data["train"][0]
    print("=== sample, trained part marked with [[ ]]:")
    print("".join(
        (f"[[{tokenizer.decode([t])}]]" if l != -100 else tokenizer.decode([t]))
        for t, l in zip(sample["input_ids"][:400], sample["labels"][:400])
    ))

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=data["train"],
        eval_dataset=data["val"],
        args=SFTConfig(
            output_dir=str(output_dir),
            max_length=args.max_length,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_steps=args.warmup_steps,
            weight_decay=0.01,
            optim="adamw_8bit",
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.eval_steps,
            save_total_limit=args.save_total_limit or None,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            seed=3407,
            report_to="wandb",
            run_name=run_name,
        ),
    )
    trainer.train()

    final_dir = output_dir / "final"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    with open(final_dir / "digital_twin.json", "w") as f:
        json.dump({"model": args.model, "base_model": model_name, "format": format_name, **vars(args)},
                  f, ensure_ascii=False, indent=2, default=str)
    print(f"saved to {final_dir}, best checkpoint: {trainer.state.best_model_checkpoint}")


if __name__ == "__main__":
    main()
