"""Draft a card "about me" from the chat with a local LLM, to review by hand.

The card goes into the system prompt so the twin stops inventing facts about my
life (work, plans, people, places). The model reads the newest export in chunks,
lists facts about Леша from each chunk, then merges the lists into one card.

    python -m digital_twin.persona          # GPU (A100 or Ada), writes data/persona_draft.md
"""
from unsloth import FastModel  # must be imported before transformers

import argparse
from pathlib import Path

import torch

from digital_twin.dataset import ROOT
from digital_twin.evaluate import JUDGE_MODEL
from digital_twin.telegram import load_export

EXPORT = ROOT / "data" / "telegram_export_2026-11-09.json"
OUT = ROOT / "data" / "persona_draft.md"
CHUNK_CHARS = 30_000  # ~10K tokens

EXTRACT = (
    "Ниже кусок переписки Леши и Сони в Telegram (они пара). Выпиши всё, что из него можно узнать о Леше: "
    "работа и учёба, распорядок, где живёт и бывает, люди вокруг (имена и кто это), увлечения, планы, "
    "привычки, как он обычно называет Соню, его любимые словечки и манера писать. "
    "Только то, что прямо следует из переписки, без догадок. Коротким списком, по факту на строку.\n\n"
    "Переписка:\n{}"
)
MERGE = (
    "Ниже факты о Леше, собранные из разных кусков его переписки с Соней за последние месяцы. "
    "Сведи их в одну карточку «о себе» от третьего лица, не длиннее 25 пунктов: убери повторы, "
    "при противоречиях оставь более позднее, не добавляй ничего от себя. Разделы: Кто он; Работа и учёба; "
    "Люди; Места; Увлечения и привычки; Как пишет.\n\nФакты:\n{}"
)


def chunks(messages):
    current, size = [], 0
    for m in messages:
        line = f"{'Леша' if m.from_me else 'Соня'}: {m.text}"
        if size + len(line) > CHUNK_CHARS and current:
            yield "\n".join(current)
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        yield "\n".join(current)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()

    model, processor = FastModel.from_pretrained(model_name=JUDGE_MODEL, max_seq_length=16384, load_in_4bit=True)
    FastModel.for_inference(model)
    tokenizer = getattr(processor, "tokenizer", processor)

    def ask(prompt, max_new_tokens):
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            out = model.generate(input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False)
        return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    parts = list(chunks(load_export(EXPORT)))
    facts = []
    for i, chunk in enumerate(parts, 1):
        facts.append(ask(EXTRACT.format(chunk), 600))
        print(f"chunk {i}/{len(parts)} done", flush=True)
    card = ask(MERGE.format("\n".join(facts)), 1200)
    args.out.write_text(
        "<!-- Черновик, собран моделью из переписки. Проверь, исправь и сохрани как data/persona.md -->\n\n"
        + card + "\n\n<details><summary>Сырые факты по кускам</summary>\n\n" + "\n\n---\n\n".join(facts)
        + "\n\n</details>\n", encoding="utf-8")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
