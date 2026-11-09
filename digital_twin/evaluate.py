"""How much do the twin's replies look like mine? A Turing test on held-out weeks.

A "moment" is a point in the validation chat right after Соня's messages, with
the context before it and what I actually wrote next. Every system (a model,
a sampling setting, the v1 adapter) generates a reply for the same moments;
then a judge sees the context and two replies, mine and the system's, in random
order, and guesses which one is real.

    fool rate = how often the judge takes the generated reply for mine
                50% — indistinguishable, 0% — always caught

    python -m digital_twin.evaluate prepare                    # fix the moments (local, stdlib)
    python -m digital_twin.evaluate generate outputs/<run>/final --name gemma --temperature 0.7   # GPU
    python -m digital_twin.evaluate generate --v1 outputs/v1_run_20250506_194631/checkpoint-2300 --name v1
    python -m digital_twin.evaluate judge gemma v1             # GPU (A100), local judge model
    python -m digital_twin.evaluate blind gemma                # you are the judge (local, stdlib)
    python -m digital_twin.evaluate report                     # all results side by side

Files live in data/eval/ (personal data: synced to the cluster, never in git).
"""
import argparse
import json
import random
import statistics
from pathlib import Path

from digital_twin.dataset import OUT_DIR, ROOT, WAIT

EVAL_DIR = ROOT / "data" / "eval"
MOMENTS = EVAL_DIR / "moments.jsonl"
N_MOMENTS = 200
JUDGE_MODEL = "unsloth/Qwen3.8-27B-unsloth-bnb-4bit"
JUDGE_CONTEXT = 20  # messages of context shown to the judge
SHUFFLED = "shuffled_real"  # sanity system: my real reply from another moment


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def candidates_path(name):
    return EVAL_DIR / "candidates" / f"{name}.jsonl"


# ── prepare ────────────────────────────────────────────────────────────────

def prepare(n=N_MOMENTS, seed=0):
    moments = []
    for example in read_jsonl(OUT_DIR / "val.jsonl"):
        msgs = example["messages"]
        for i in range(1, len(msgs)):
            if msgs[i - 1]["role"] == "user" and msgs[i]["role"] == "assistant" and msgs[i]["train"]:
                real = []
                for m in msgs[i:]:
                    if m["role"] != "assistant" or m["content"] == WAIT:
                        break
                    real.append(m["content"])
                context = [{"role": m["role"], "content": m["content"]} for m in msgs[:i]]
                moments.append({"context": context, "real": real})
    random.seed(seed)
    moments = random.sample(moments, min(n, len(moments)))
    for i, m in enumerate(moments):
        m["id"] = i
    write_jsonl(MOMENTS, moments)

    # Sanity system for the judge: a real reply of mine, but to a different moment.
    shuffled = [m["real"] for m in moments if m["real"]]
    random.shuffle(shuffled)
    write_jsonl(candidates_path(SHUFFLED),
                [{"id": m["id"], "reply": shuffled[i % len(shuffled)]} for i, m in enumerate(moments)])
    print(f"{len(moments)} moments written to {MOMENTS}")


# ── generate ───────────────────────────────────────────────────────────────

class V1Twin:
    """The May 2025 adapter: one prompt with reply context, several messages joined by newlines."""

    PROMPT = (
        "Ты ассистент, который учавствует в диалоге между Лешей и Соней. Прими во внимание контекст диалога. "
        "Твоя задача ответить на приведенное ниже сообщение максимально похоже на то, как бы это сделал Леша"
        "\n\n### Контекст:\n{}\n\n### Сообщение:\n{}\n\n### Ответ:\n"
    )

    def __init__(self, adapter_dir):
        from unsloth import FastLanguageModel
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_dir), max_seq_length=2048, load_in_4bit=True)
        FastLanguageModel.for_inference(self.model)

    def reply(self, messages, temperature=0.7, **_):
        import torch
        lines = [f"{'Соня' if m['role'] == 'user' else 'Леша'}: {m['content']}"
                 for m in messages if m["role"] != "system" and m["content"] != WAIT]
        prompt = self.PROMPT.format("\n".join(lines[-7:-1]), lines[-1].split(": ", 1)[1])
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=temperature,
                                      top_p=0.9, repetition_penalty=1.1, pad_token_id=self.tokenizer.eos_token_id)
        text = self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return [line.strip() for line in text.split("\n") if line.strip()][:8]


def generate(adapter_dir, name, v1=False, **sampling):
    if v1:
        twin = V1Twin(adapter_dir)
    else:
        from digital_twin.chat import Twin
        twin = Twin(adapter_dir)
    rows = []
    for m in read_jsonl(MOMENTS):
        rows.append({"id": m["id"], "reply": twin.reply([dict(x) for x in m["context"]], **sampling)})
    write_jsonl(candidates_path(name), rows)
    meta = {"adapter": str(adapter_dir), "v1": v1, **sampling}
    (EVAL_DIR / "candidates" / f"{name}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"{len(rows)} replies written to {candidates_path(name)}")


# ── judging ────────────────────────────────────────────────────────────────

def render_context(context, n=JUDGE_CONTEXT):
    lines = [f"{'Соня' if m['role'] == 'user' else 'Леша'}: {m['content']}"
             for m in context if m["role"] != "system" and m["content"] != WAIT]
    return "\n".join(lines[-n:])


def render_reply(reply):
    return "\n".join(f"Леша: {t}" for t in reply) if reply else "(Леша ничего не ответил)"


def judge_prompt(context, a, b):
    return (
        "Ниже — кусок переписки Леши и Сони в Telegram. После него два варианта того, что Леша написал дальше. "
        "Один вариант настоящий, другой сгенерирован нейросетью, которая подражает Леше. "
        "Определи, какой настоящий: смотри на стиль, связность с перепиской и правдоподобие.\n\n"
        f"Переписка:\n{render_context(context)}\n\n"
        f"Вариант A:\n{render_reply(a)}\n\n"
        f"Вариант B:\n{render_reply(b)}\n\n"
        "Какой вариант настоящий? Ответь одной буквой: A или B."
    )


class Judge:
    def __init__(self, model_name=JUDGE_MODEL):
        from unsloth import FastModel
        import torch
        self.torch = torch
        self.model, processor = FastModel.from_pretrained(model_name=model_name, max_seq_length=4096, load_in_4bit=True)
        FastModel.for_inference(self.model)
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.ids = [self.tokenizer(x, add_special_tokens=False).input_ids[0] for x in ("A", "B")]

    def p_a(self, prompt):
        """Probability that the judge answers A (over the two letters)."""
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(self.model.device)
        with self.torch.no_grad():
            logits = self.model(input_ids=ids).logits[0, -1, self.ids].float()
        return self.torch.softmax(logits, dim=0)[0].item()


def judge(names):
    moments = {m["id"]: m for m in read_jsonl(MOMENTS)}
    j = Judge()
    print("judge prompt template check:\n" + j.tokenizer.apply_chat_template(
        [{"role": "user", "content": "…"}], tokenize=False, add_generation_prompt=True, enable_thinking=False))
    for name in names:
        rows = []
        for c in read_jsonl(candidates_path(name)):
            m = moments[c["id"]]
            # Both orders cancel the judge's position bias.
            fooled_as_a = j.p_a(judge_prompt(m["context"], c["reply"], m["real"]))
            fooled_as_b = 1 - j.p_a(judge_prompt(m["context"], m["real"], c["reply"]))
            rows.append({"id": c["id"], "fooled": (fooled_as_a + fooled_as_b) / 2})
        write_jsonl(EVAL_DIR / "judge" / f"{name}.jsonl", rows)
        print(f"{name}: fool rate {statistics.mean(r['fooled'] for r in rows):.1%}")


def blind(name, n=30):
    """You are the judge: pick the real reply. Results are appended, so sessions can be split."""
    moments = {m["id"]: m for m in read_jsonl(MOMENTS)}
    out = EVAL_DIR / "human" / f"{name}.jsonl"
    done = {r["id"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [c for c in read_jsonl(candidates_path(name)) if c["id"] not in done]
    random.shuffle(todo)
    out.parent.mkdir(parents=True, exist_ok=True)
    print("Какой вариант написал настоящий ты? 1 или 2, q — выход.\n")
    with open(out, "a", encoding="utf-8") as f:
        for k, c in enumerate(todo[:n], 1):
            m = moments[c["id"]]
            gen_first = random.random() < 0.5
            options = [c["reply"], m["real"]] if gen_first else [m["real"], c["reply"]]
            print(f"══════ {k}/{min(n, len(todo))} ══════\n{render_context(m['context'], 12)}\n")
            for i, o in enumerate(options, 1):
                print(f"── {i} ──\n{render_reply(o)}\n")
            answer = ""
            while answer not in ("1", "2", "q"):
                answer = input("Настоящий: ").strip()
            if answer == "q":
                break
            fooled = (answer == "1") == gen_first
            f.write(json.dumps({"id": c["id"], "fooled": float(fooled)}) + "\n")
            print("✗ это была модель\n" if fooled else "✓ верно\n")
    report([name])


# ── report ─────────────────────────────────────────────────────────────────

def style_stats(replies):
    msgs = [t for r in replies for t in r]
    return {
        "msgs/reply": statistics.mean(map(len, replies)),
        "silent": sum(not r for r in replies) / len(replies),
        "chars/msg": statistics.median(map(len, msgs)) if msgs else 0,
    }


def report(names=None):
    names = names or sorted(p.stem for p in (EVAL_DIR / "candidates").glob("*.jsonl"))
    real = style_stats([m["real"] for m in read_jsonl(MOMENTS)])
    print(f"{'system':28s} {'judge':>7s} {'human':>11s} {'msgs/reply':>11s} {'silent':>7s} {'chars/msg':>10s}")
    print(f"{'(me, real)':28s} {'':>7s} {'':>11s} {real['msgs/reply']:11.2f} {real['silent']:7.1%} {real['chars/msg']:10.0f}")
    for name in names:
        s = style_stats([c["reply"] for c in read_jsonl(candidates_path(name))])
        cols = []
        for kind in ("judge", "human"):
            p = EVAL_DIR / kind / f"{name}.jsonl"
            rows = read_jsonl(p) if p.exists() else []
            cols.append(f"{statistics.mean(r['fooled'] for r in rows):.1%}" + (f" ({len(rows)})" if kind == "human" else "")
                        if rows else "—")
        print(f"{name:28s} {cols[0]:>7s} {cols[1]:>11s} {s['msgs/reply']:11.2f} {s['silent']:7.1%} {s['chars/msg']:10.0f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    g = sub.add_parser("generate")
    g.add_argument("adapter_dir", type=Path)
    g.add_argument("--name", required=True)
    g.add_argument("--v1", action="store_true")
    g.add_argument("--temperature", type=float)
    g.add_argument("--top-p", type=float)
    g.add_argument("--min-p", type=float)
    j = sub.add_parser("judge")
    j.add_argument("names", nargs="+")
    b = sub.add_parser("blind")
    b.add_argument("name")
    b.add_argument("-n", type=int, default=30)
    r = sub.add_parser("report")
    r.add_argument("names", nargs="*")
    args = p.parse_args()

    if args.cmd == "prepare":
        prepare()
    elif args.cmd == "generate":
        sampling = {k: v for k, v in (("temperature", args.temperature), ("top_p", args.top_p),
                                      ("min_p", args.min_p)) if v is not None}
        generate(args.adapter_dir, args.name, v1=args.v1, **sampling)
    elif args.cmd == "judge":
        judge(args.names)
    elif args.cmd == "blind":
        blind(args.name, args.n)
    else:
        report(args.names)


if __name__ == "__main__":
    main()
