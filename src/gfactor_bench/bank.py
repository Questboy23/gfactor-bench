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



def guess_floor(key: str) -> float:
    fmt = source(key)[3]
    return {"winogrande": 0.5, "gsm8k": 0.0, "truthfulqa": 0.0}.get(fmt, 0.25)


PAGE = 100


def _hf_headers():
    import os
    tok = os.environ.get("HF_TOKEN")
    f = Path.home() / ".cache" / "huggingface" / "token"
    if not tok and f.exists():
        tok = f.read_text().strip()
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _fetch_page(ds, cfg, split, page):
    """One request returns 100 rows; all are cached, so later draws from the same page are free."""
    d = CACHE / ds.replace("/", "__") / cfg
    marker = d / f"{split}_page{page}.done"
    if marker.exists():
        return
    import time
    for attempt in range(8):
        r = requests.get("https://datasets-server.huggingface.co/rows", headers=_hf_headers(), timeout=60,
                         params={"dataset": ds, "config": cfg, "split": split, "offset": page * PAGE, "length": PAGE})
        if r.status_code == 200:
            d.mkdir(parents=True, exist_ok=True)
            for item in r.json()["rows"]:
                (d / f"{split}_{item['row_idx']}.json").write_text(json.dumps(item["row"]))
            marker.touch()
            return
        wait = float(r.headers.get("Retry-After", 0) or 0) or min(60, 2 ** attempt)
        time.sleep(wait)
    r.raise_for_status()


def fetch_row(key: str, idx: int) -> dict:
    ds, cfg, split, _ = source(key)
    f = CACHE / ds.replace("/", "__") / cfg / f"{split}_{idx}.json"
    if not f.exists():
        _fetch_page(ds, cfg, split, idx // PAGE)
    return json.loads(f.read_text())


def fetch_rows(pairs, workers=4):
    pages = sorted({(*source(k)[:3], i // PAGE) for k, i in pairs})
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda p: _fetch_page(*p), pages))
    return [fetch_row(k, i) for k, i in pairs]


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


def _last_boxed(t):
    i = t.rfind("\\boxed{")
    if i < 0:
        return None
    i += 7
    depth, j = 1, i
    while j < len(t) and depth:
        depth += {"{": 1, "}": -1}.get(t[j], 0)
        j += 1
    return t[i:j - 1] if depth == 0 else None


def _same_math(a, b):
    def n(x):
        x = re.sub(r"\\(left|right|!|,|;|displaystyle)", "", str(x)).replace("dfrac", "frac").replace("tfrac", "frac")
        x = re.sub(r"\\text\{([^}]*)\}", r"\1", x)
        return re.sub(r"[\s$]", "", x).rstrip(".")
    a, b = n(a), n(b)
    if a == b:
        return True
    try:
        return abs(float(a.replace(",", "")) - float(b.replace(",", ""))) < 1e-6
    except ValueError:
        return False


def score(text: str, gold: str, kind: str) -> float:
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
    if kind == "boxed":  # MathArena final answers
        got = _last_boxed(t)
        return int(got is not None and _same_math(got, gold))
    if kind == "connections":  # LiveBench: share of the 4-word groups found, a soft score in [0, 1]
        m = re.findall(r"<solution>(.*?)</solution>", t, flags=re.S)
        words = [w.strip(" *`'\".\n").lower() for w in (m[-1] if m else t.strip().splitlines()[-1] if t.strip() else "").split(",")]
        want = [w.strip().lower() for w in gold.split(",")]
        groups = [set(want[i:i + 4]) for i in range(0, len(want), 4)]
        got = [set(words[i:i + 4]) for i in range(0, len(words), 4)]
        return sum(g in got for g in groups) / len(groups)
    if kind == "contains":  # LiveBench typos: the corrected text must appear verbatim
        squash = lambda x: " ".join(x.split()).lower()  # noqa: E731
        return int(squash(gold) in squash(t))
    if kind == "exact":
        m = re.findall(r"[Aa]nswer\s*[:：]\s*(.+)", t)
        ans = (m[-1] if m else t.strip().splitlines()[-1] if t.strip() else "").strip().strip("*`'\". ")
        return int(ans.lower() == gold.lower())
    raise ValueError(kind)
