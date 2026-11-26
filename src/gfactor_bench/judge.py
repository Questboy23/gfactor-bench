"""Judge panel for open-ended answers (poetry, advice, ideas, explanations).

- Pairwise: each judge compares the tested model's answer with a fixed reference answer, in both orders
  (cancels position bias), as in RLHF preference data and AlpacaEval.
- Several free models from different families run in parallel; the panel reports mean and SD.
- Novelty: embedding distance from a pool of standard answers, credited only when quality wins.
- Audits measure each judge's error rate on planted pairs with a known right answer, how those errors
  correlate between judges, and the score shift when only the conclusion flips between consensus and contrarian.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .models import chat, embed

DATA = Path(__file__).parent / "data"
DEFAULT_JUDGES = [
    "groq/openai/gpt-oss-120b",
    "groq/qwen/qwen3.8-27b",
    "gemini/gemini-3.5-flash-lite",
    "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
    "ollama/qwen2.5-coder:7b",  # not gemma3: gemma3 wrote the reference answers, so it would judge its own work
]

JUDGE_SYS = (
    "You are a blind evaluator. Judge only the quality of the work: correctness, depth of reasoning, "
    "usefulness, craft, and originality that serves the task. Do not reward agreement with popular opinion "
    "or penalize an unusual conclusion if its reasoning is sound. Ignore length unless it changes quality. "
    "Do not guess who wrote either response."
)
PAIR = """Task given to both writers:
{task}

=== Response 1 ===
{a}

=== Response 2 ===
{b}

Which response is better overall? Reply with one line: 'Winner: 1' or 'Winner: 2'."""


def _clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    # remove self-identification so judges cannot favor a brand
    return re.sub(r"(?i)\b(as an ai|i am|i'm) (an? )?(ai|language model|chatgpt|claude|gemini|llama|qwen)[^.]*\.", "", text).strip()


def _winner(text: str):
    m = re.findall(r"[Ww]inner\s*[:：]?\s*\**\s*([12])", _clean(text))
    return int(m[-1]) if m else None


def compare(judge, task, x, y):
    """Returns P(x better than y) from this judge over both orders: 1, 0.5 or 0 (None if unusable)."""
    r1 = _winner(chat(judge, PAIR.format(task=task, a=_clean(x), b=_clean(y)), max_tokens=1500, system=JUDGE_SYS))
    r2 = _winner(chat(judge, PAIR.format(task=task, a=_clean(y), b=_clean(x)), max_tokens=1500, system=JUDGE_SYS))
    votes = [v for v in ((r1 == 1) if r1 else None, (r2 == 2) if r2 else None) if v is not None]
    return float(np.mean(votes)) if votes else None


def load_reference():
    f = DATA / "reference_answers.json"
    return json.loads(f.read_text()) if f.exists() else {}


def build_reference(models, prompts):
    """Pool of standard answers per prompt. The first model's answer is the fixed opponent.
    Each model runs in its own lane (local models one request at a time, APIs three) so a slow
    local model never blocks the fast APIs."""
    import threading

    results, lock = [], threading.Lock()

    def lane(m):
        def one(p):
            out = chat(m, p["prompt"], max_tokens=700, temperature=0.7, retries=3)
            with lock:
                results.append((m, p["id"], _clean(out)))
                print(f"{m}: {sum(r[0] == m for r in results)}/{len(prompts)}", flush=True)
        with ThreadPoolExecutor(1 if m.startswith("ollama") else 3) as ex:
            list(ex.map(one, prompts))

    with ThreadPoolExecutor(len(models)) as ex:
        list(ex.map(lane, models))
    ref = {}
    for m, pid, out in results:
        if out and not out.startswith("[error"):
            ref.setdefault(pid, {"answers": {}})["answers"][m] = out
    ref = {pid: v for pid, v in ref.items() if models[0] in v["answers"]}  # opponent answer is required
    for v in ref.values():
        v["answers"] = {m: v["answers"][m] for m in models if m in v["answers"]}
        v["embeddings"] = embed(list(v["answers"].values()))
        v["opponent"] = models[0]
    (DATA / "reference_answers.json").write_text(json.dumps(ref))
    return ref


def novelty(answer, pool_emb):
    e = np.array(embed([answer])[0])
    P = np.array(pool_emb)
    sims = P @ e / (np.linalg.norm(P, axis=1) * np.linalg.norm(e) + 1e-9)
    return float(1 - sims.mean())


def judge_answers(items, judges, weights=None, rng=None, novelty_weight=0.25, quality_gate=0.5):
    """items: list of (prompt_dict, answer). Returns per-item and panel summaries with 90% bootstrap intervals."""
    ref = load_reference()
    weights = weights or {}
    jobs = [(i, j) for i in range(len(items)) for j in judges]

    def run(job):
        i, j = job
        p, ans = items[i]
        opp = ref[p["id"]]["answers"][ref[p["id"]]["opponent"]]
        return i, j, compare(j, p["prompt"], ans, opp)

    with ThreadPoolExecutor(min(16, len(jobs))) as ex:
        res = list(ex.map(run, jobs))
    W = np.full((len(items), len(judges)), np.nan)
    for i, j, v in res:
        if v is not None:
            W[i, judges.index(j)] = v
    w = np.array([weights.get(j, 1.0) for j in judges])
    per_item = []
    for i, (p, ans) in enumerate(items):
        row = W[i]
        ok = ~np.isnan(row)
        q = float(np.average(row[ok], weights=w[ok])) if ok.any() else np.nan
        nov = novelty(_clean(ans), ref[p["id"]]["embeddings"])
        bonus = novelty_weight * nov if q >= quality_gate else 0.0  # unusual and bad earns nothing
        per_item.append({"id": p["id"], "cat": p["cat"], "win_rate": q, "judge_sd": float(np.nanstd(row)) if ok.sum() > 1 else None,
                         "novelty": nov, "score": q + bonus if not np.isnan(q) else np.nan})
    scores = np.array([x["score"] for x in per_item])
    bonus = np.array([x["score"] - x["win_rate"] for x in per_item])
    boot = _bootstrap_rows_cols(W, w, bonus, rng)
    judge_means = {j: (float(np.nanmean(W[:, k])) if (~np.isnan(W[:, k])).any() else None) for k, j in enumerate(judges)}
    return {
        "items": per_item,
        "win_rate": float(np.nanmean([x["win_rate"] for x in per_item])),
        "score": float(np.nanmean(scores)),
        "ci90": boot,
        "judge_means": judge_means,
        "judge_sd": float(np.nanstd([v for v in judge_means.values() if v is not None])),
        "novelty": float(np.mean([x["novelty"] for x in per_item])),
    }


def _bootstrap_rows_cols(W, w, bonus, rng, n=2000):
    """90% interval for the panel score, resampling both prompts and judges."""
    rng = rng or np.random.default_rng(0)
    I, J = W.shape
    out = []
    for _ in range(n):
        ri, cj = rng.integers(0, I, I), rng.integers(0, J, J)
        sub, ww = W[np.ix_(ri, cj)], w[cj]
        m = ~np.isnan(sub)
        if not m.any():
            continue
        rows = [np.average(sub[k][m[k]], weights=ww[m[k]]) for k in range(I) if m[k].any()]
        out.append(np.mean(rows) + np.nanmean(bonus[ri]))
    return [float(np.percentile(out, 5)), float(np.percentile(out, 95))] if out else [None, None]


# ---------- judge audits ----------

def damage(text, rng):
    """Planted flaw with a known right answer: the original should beat the damaged copy."""
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    kind = rng.choice(["drop", "shuffle", "number"]) if len(sents) > 3 else "number"
    if kind == "drop":
        k = int(rng.integers(1, len(sents) - 1))
        sents = sents[:k] + sents[k + max(1, len(sents) // 3):]
        return " ".join(sents)
    if kind == "shuffle":
        return " ".join(rng.permutation(sents))
    nums = re.findall(r"\d+", text)
    if nums:
        n = nums[int(rng.integers(0, len(nums)))]
        return text.replace(n, str(int(n) * 3 + 7), 1)
    return " ".join(sents[: max(1, len(sents) // 2)])


def audit(judges, n=12, seed=0):
    rng = np.random.default_rng(seed)
    ref = load_reference()
    prompts = {p["id"]: p for p in json.loads((DATA / "open_prompts.json").read_text())}
    ids = [str(x) for x in rng.choice(list(ref), min(n, len(ref)), replace=False)]
    pairs = []
    for pid in ids:
        good = ref[pid]["answers"][ref[pid]["opponent"]]
        pairs.append((prompts[pid]["prompt"], good, damage(good, rng)))
    flips = json.loads((DATA / "consensus_pairs.json").read_text())

    def run_planted(args):
        j, (task, good, bad) = args
        v = compare(j, task, good, bad)
        return j, None if v is None else 1 - v  # error = preferring the damaged copy

    def run_flip(args):
        j, f = args
        return j, compare(j, f["task"], f["contrarian"], f["consensus"])  # 0.5 = no bias

    with ThreadPoolExecutor(16) as ex:
        planted = list(ex.map(run_planted, [(j, p) for p in pairs for j in judges]))
        flipped = list(ex.map(run_flip, [(j, f) for f in flips for j in judges]))
    E = np.full((len(pairs), len(judges)), np.nan)
    for k, (j, e) in enumerate(planted):
        if e is not None:
            E[k // len(judges), judges.index(j)] = e
    out = {}
    for k, j in enumerate(judges):
        fl = [v for jj, v in flipped if jj == j and v is not None]
        err = float(np.nanmean(E[:, k])) if (~np.isnan(E[:, k])).any() else None
        bias = float(np.mean(fl) - 0.5) if fl else None  # negative = favors consensus
        weight = max(0.0, 1 - 2 * (err or 0)) * max(0.0, 1 - 2 * abs(bias or 0))
        out[j] = {"planted_error_rate": err, "consensus_bias": bias, "weight": weight}
    with np.errstate(invalid="ignore"):
        C = np.ma.corrcoef(np.ma.masked_invalid(E).T).filled(np.nan)
    result = {"judges": out, "error_correlation": C.tolist(), "judge_order": judges, "n_planted": len(pairs), "n_flips": len(flips)}
    (DATA / "judge_audit.json").write_text(json.dumps(result, indent=1))
    return result


def load_weights():
    f = DATA / "judge_audit.json"
    if not f.exists():
        return {}
    return {j: v["weight"] for j, v in json.loads(f.read_text())["judges"].items()}
