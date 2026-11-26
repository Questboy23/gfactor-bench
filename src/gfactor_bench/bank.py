"""Calibrated item bank: Open LLM Leaderboard v1 questions, fetched one row at a time from the
Hugging Face datasets-server API (no bulk download), turned into prompts with shuffled choices."""
from __future__ import annotations

import json
import re
import string
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

CACHE = Path.home() / ".cache" / "gfactor-bench" / "rows"
FACTORS = ["math", "stem", "humanities_social", "commonsense", "truthfulness"]

MATH = {"abstract_algebra", "college_mathematics", "elementary_mathematics", "high_school_mathematics",
        "high_school_statistics", "formal_logic", "econometrics"}
STEM = {"anatomy", "astronomy", "college_biology", "college_chemistry", "college_computer_science",
        "college_medicine", "college_physics", "computer_security", "conceptual_physics",
        "electrical_engineering", "high_school_biology", "high_school_chemistry",
        "high_school_computer_science", "high_school_physics", "machine_learning", "medical_genetics",
        "clinical_knowledge", "nutrition", "virology", "professional_medicine", "human_aging"}

# leaderboard key -> (hf dataset, config, split, row format)
FIXED = {
    "harness_arc_challenge_25": ("allenai/ai2_arc", "ARC-Challenge", "test", "arc"),
    "harness_hellaswag_10": ("Rowan/hellaswag", "default", "validation", "hellaswag"),
    "harness_truthfulqa_mc_0": ("truthfulqa/truthful_qa", "multiple_choice", "validation", "truthfulqa"),
    "harness_winogrande_5": ("allenai/winogrande", "winogrande_xl", "validation", "winogrande"),
    "harness_gsm8k_5": ("openai/gsm8k", "main", "test", "gsm8k"),
}


def source(key: str):
    if key in FIXED:
        return FIXED[key]
    subject = key.removeprefix("harness_hendrycksTest_").removesuffix("_5")
    return ("cais/mmlu", subject, "test", "mmlu")


def factor_of(key: str) -> str:
    if key == "harness_gsm8k_5":
        return "math"
    if key == "harness_arc_challenge_25":
        return "stem"
    if key in ("harness_hellaswag_10", "harness_winogrande_5"):
        return "commonsense"
    if key == "harness_truthfulqa_mc_0":
        return "truthfulness"
    subject = source(key)[1]
    return "math" if subject in MATH else "stem" if subject in STEM else "humanities_social"


def guess_floor(key: str) -> float:
    fmt = source(key)[3]
    return {"winogrande": 0.5, "gsm8k": 0.0, "truthfulqa": 0.0}.get(fmt, 0.25)


def fetch_row(key: str, idx: int) -> dict:
    ds, cfg, split, _ = source(key)
    f = CACHE / ds.replace("/", "__") / cfg / f"{split}_{idx}.json"
    if f.exists():
        return json.loads(f.read_text())
    url = "https://datasets-server.huggingface.co/rows"
    for attempt in range(4):
        r = requests.get(url, params={"dataset": ds, "config": cfg, "split": split, "offset": idx, "length": 1}, timeout=30)
        if r.status_code == 200:
            row = r.json()["rows"][0]["row"]
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(row))
            return row
        import time
        time.sleep(2 ** attempt)
    r.raise_for_status()


def fetch_rows(pairs, workers=16):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda kv: fetch_row(*kv), pairs))


MC_INSTR = "Answer the multiple-choice question. Think briefly if needed, then end with a final line 'Answer: X' where X is the letter."


def _mc(question, choices, correct, rng):
    order = rng.permutation(len(choices))  # shuffle choice order each run against memorized letters
    letters = string.ascii_uppercase
    body = "\n".join(f"{letters[i]}. {choices[o]}" for i, o in enumerate(order))
    gold = letters[int(list(order).index(correct))]
    return f"{MC_INSTR}\n\n{question}\n\n{body}", gold, "mc"


def make_prompt(key: str, row: dict, rng):
    """Returns (prompt, gold, kind) with kind in {'mc', 'number'}."""
    fmt = source(key)[3]
    if fmt == "mmlu":
        return _mc(row["question"], row["choices"], int(row["answer"]), rng)
    if fmt == "arc":
        labels = row["choices"]["label"]
        return _mc(row["question"], row["choices"]["text"], labels.index(row["answerKey"]), rng)
    if fmt == "hellaswag":
        q = f"Which ending best continues the text?\n\n{row['ctx']}"
        return _mc(q, row["endings"], int(row["label"]), rng)
    if fmt == "winogrande":
        q = f"Which option correctly fills the blank (_)?\n\n{row['sentence']}"
        return _mc(q, [row["option1"], row["option2"]], int(row["answer"]) - 1, rng)
    if fmt == "truthfulqa":
        t = row["mc1_targets"]
        return _mc(row["question"], t["choices"], t["labels"].index(1), rng)
    if fmt == "gsm8k":
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        p = f"Solve the problem. End with a final line 'Answer: N' where N is the number only.\n\n{row['question']}"
        return p, gold, "number"
    raise ValueError(fmt)


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)


def score(text: str, gold: str, kind: str) -> int:
    t = _strip_think(text)
    if kind == "mc":
        m = re.findall(r"[Aa]nswer\s*[:：]?\s*\**\(?([A-J])\b", t)
        if not m:
            m = re.findall(r"\b([A-J])\b", t[-40:])
        return int(bool(m) and m[-1].upper() == gold)
    if kind == "number":
        m = re.findall(r"[Aa]nswer\s*[:：]?\s*\$?\s*(-?[\d,]*\.?\d+)", t) or re.findall(r"-?[\d,]*\.?\d+", t)
        if not m:
            return 0
        try:
            return int(abs(float(m[-1].replace(",", "")) - float(gold)) < 1e-6)
        except ValueError:
            return 0
    if kind == "exact":
        m = re.findall(r"[Aa]nswer\s*[:：]\s*(.+)", t)
        ans = (m[-1] if m else t.strip().splitlines()[-1] if t.strip() else "").strip().strip("*`'\". ")
        return int(ans.lower() == gold.lower())
    raise ValueError(kind)
