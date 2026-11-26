"""Adaptive test of one model: random calibrated items until every 90% interval is narrow enough,
then fresh odd tasks and judged open-ended tasks."""
from __future__ import annotations

import json
import secrets
import time
import warnings
from pathlib import Path

import numpy as np

from . import generators
from .bank import fetch_rows, make_prompt, score
from .discover import short
from .irt import Z90, estimate_theta, loading_matrix, select_items
from .models import chat_many

DATA = Path(__file__).parent / "data"


BANKS = {"open": "calibration.npz", "frontier": "calibration_frontier.npz"}


def load_calibration(bank="open"):
    z = np.load(DATA / BANKS[bank], allow_pickle=False)
    return {k: z[k] for k in z.files}


BASE, PER_SD = 1000.0, 200.0  # open-ended game-style score: no ceiling or floor


def scales(cal):
    """Map latent units to score points: 1000 = reference-model average, 200 points per reference SD."""
    tg, ts = cal["ref_theta_g"], cal["ref_theta_s"]
    return {"g_mean": float(tg.mean()), "g_sd": float(tg.std()), "s_mean": ts.mean(0), "s_sd": ts.std(0)}


def to_points(th, cov, sc):
    g = BASE + PER_SD * (th[0] - sc["g_mean"]) / sc["g_sd"]
    g_hw = Z90 * PER_SD * np.sqrt(cov[0, 0]) / sc["g_sd"]
    s = PER_SD * (th[1:] - sc["s_mean"]) / sc["s_sd"]  # deviation from what g predicts, in points
    s_hw = Z90 * PER_SD * np.sqrt(np.diag(cov)[1:]) / sc["s_sd"]
    return g, g_hw, s, s_hw


def wilson90(k, n):
    if n == 0:
        return [None, None]
    p, z = k / n, Z90
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def run(model, seed=None, target_g=80.0, target_factor=130.0, max_items=300, batch=15, start_per_factor=4,
        pool_frac=0.4, n_odd=16, n_open=8, judges=None, workers=4, log=print, skip_open=False, bank="open"):
    seed = seed if seed is not None else secrets.randbits(32)
    rng = np.random.default_rng(seed)
    cal = load_calibration(bank)
    served = json.loads((DATA / "frontier_items.json").read_text()) if bank == "frontier" else None
    sc = scales(cal)
    FACTORS = [str(x) for x in cal["factor_names"]]
    F = len(FACTORS)
    L = loading_matrix(cal["a_g"], cal["a_s"], cal["fidx"], F)
    b, c = cal["b"].astype(float), cal["c"].astype(float)
    can_ask = np.ones(len(b), bool) if served is None else np.array([f"{k}|{i}" in served for k, i in zip(cal["keys"], cal["idxs"])])
    pool = np.flatnonzero((rng.random(len(b)) < pool_frac) & can_ask)  # a different random slice of the bank each run
    asked, ys, log_items, tried = [], [], [], set()
    t0 = time.time()

    def ask(idx):
        pairs = [(str(cal["keys"][j]), str(cal["idxs"][j])) for j in idx]
        if served is None:
            rows = fetch_rows([(k, int(i)) for k, i in pairs])
            prompts = [make_prompt(k, r, rng) for (k, _), r in zip(pairs, rows)]
        else:
            prompts = [tuple(served[f"{k}|{i}"]) for k, i in pairs]
        outs = chat_many(model, [p for p, _, _ in prompts], workers=workers, max_tokens=4096)
        for j, (k, i), (p, gold, kind), o in zip(idx, pairs, prompts, outs):
            err = o.startswith("[error")
            y = score(o, gold, kind)
            if not err:  # an API failure says nothing about ability; leave it out of the estimate
                asked.append(j)
                ys.append(y)
            tried.add(j)
            log_items.append({"key": k, "row": i, "factor": FACTORS[cal["fidx"][j]], "gold": gold, "correct": y,
                              "error": err, "response_tail": o[-200:]})

    start = [int(j) for f in range(F) for j in rng.permutation(pool[cal["fidx"][pool] == f])[:start_per_factor]]
    ask(start)
    while True:
        if not asked:
            raise RuntimeError(f"every request to {model} failed; first error: {log_items[0]['response_tail']}")
        A = np.array(asked, dtype=int)
        th, cov = estimate_theta(L[A], b[A], c[A], np.array(ys, float))
        g, g_hw, s, s_hw = to_points(th, cov, sc)
        ratios = np.concatenate([[g_hw / target_g], s_hw / target_factor])
        left = np.setdiff1d(pool, np.array(sorted(tried)))
        ratios[1:][np.bincount(cal["fidx"][left], minlength=F) == 0] = 0  # no askable questions left for that factor
        log(f"items {len(asked)}  g {g:.1f} ±{g_hw:.1f}  widest {['g', *FACTORS][int(ratios.argmax())]} ±{max(g_hw, *s_hw):.1f}  {time.time() - t0:.0f}s")
        if ratios.max() <= 1 or len(tried) >= max_items or len(left) == 0:
            break
        dim = int(ratios.argmax())
        cand = left
        if dim > 0:
            cand = cand[cal["fidx"][cand] == dim - 1]
        ask(select_items(L, b, c, th, cov, cand, dim, min(batch, max_items - len(tried)), rng))

    # fresh odd tasks
    odd = generators.sample(rng, n_odd)
    outs = chat_many(model, [p for _, p, _, _ in odd], workers=workers, max_tokens=4096)
    odd_res = [{"task": n, "gold": g_, "correct": score(o, g_, k), "response_tail": o[-200:]}
               for (n, _, g_, k), o in zip(odd, outs) if not o.startswith("[error")]
    k_ok = sum(r["correct"] for r in odd_res)

    open_res = None
    if not skip_open and n_open:
        from .judge import DEFAULT_JUDGES, judge_answers, load_reference, load_weights
        ref = load_reference()
        prompts = [p for p in json.loads((DATA / "open_prompts.json").read_text()) if p["id"] in ref]
        pick = [prompts[i] for i in rng.choice(len(prompts), min(n_open, len(prompts)), replace=False)]
        answers = chat_many(model, [p["prompt"] for p in pick], workers=workers, max_tokens=1500, temperature=0.7)
        open_res = judge_answers(list(zip(pick, answers)), judges or DEFAULT_JUDGES, load_weights(), rng)
        log(f"open-ended win rate {open_res['win_rate']:.2f}, judge SD {open_res['judge_sd']:.2f}")

    return summarize(model, seed, bank, cal, sc, th, cov, asked, ys, log_items, odd_res, k_ok, open_res, time.time() - t0)


def summarize(model, seed, bank, cal, sc, th, cov, asked, ys, log_items, odd_res, k_ok, open_res, secs):
    FACTORS = [str(x) for x in cal["factor_names"]]
    g, g_hw, s, s_hw = to_points(th, cov, sc)
    # spread: SD of the skill deviations after removing what measurement noise alone would add
    se2 = (s_hw / Z90) ** 2
    spread = float(np.sqrt(max(0.0, s.var() - se2.mean())))

    # residuals per benchmark or subject: observed accuracy minus model-predicted accuracy
    L = loading_matrix(cal["a_g"], cal["a_s"], cal["fidx"], len(FACTORS))
    A = np.array(asked)
    p = cal["c"][A] + (1 - cal["c"][A]) / (1 + np.exp(-(L[A] @ th - cal["b"][A])))
    keys = cal["keys"][A]
    resid = []
    for k in sorted(set(keys)):
        m = keys == k
        if m.sum() >= 3:
            resid.append({"subject": short(k),
                          "n": int(m.sum()), "observed": float(np.mean(np.array(ys)[m])), "predicted": float(p[m].mean()),
                          "gap": float(np.mean(np.array(ys)[m]) - p[m].mean())})
    resid.sort(key=lambda r: -r["gap"])

    # skill correlations across the reference models, per factor (accuracy averaged over its subjects);
    # pairwise over the models tested on both factors
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fac_acc = np.stack([np.nanmean(cal["ref_subscores"][:, cal["subject_factor"] == f], 1) for f in range(len(FACTORS))], 1)
        corr = np.ma.corrcoef(np.ma.masked_invalid(fac_acc).T).filled(np.nan)

    return {
        "model": model, "seed": seed, "bank": bank, "n_ref": int(len(cal["ref_models"])), "seconds": round(secs), "n_items": len(asked),
        "api_errors": int(sum(x["error"] for x in log_items)),
        "g": {"score": float(g), "ci90": [float(g - g_hw), float(g + g_hw)]},
        "factors": {f: {"deviation": float(s[i]), "ci90": [float(s[i] - s_hw[i]), float(s[i] + s_hw[i])],
                        "index": float(g + s[i]), "items": int(sum(x["factor"] == f and not x["error"] for x in log_items)),
                        "accuracy": float(np.mean([x["correct"] for x in log_items if x["factor"] == f and not x["error"]] or [np.nan]))}
                    for i, f in enumerate(FACTORS)},
        "spread": {"score": spread, "raw_sd": float(s.std())},
        "odd_tasks": {"accuracy": k_ok / max(1, len(odd_res)), "ci90": wilson90(k_ok, len(odd_res)), "n": len(odd_res),
                      "by_task": _by_task(odd_res), "calibrated": False},
        "open_ended": open_res,
        "residuals": resid,
        "factor_correlations": {"factors": FACTORS, "matrix": corr.tolist()},
        "items": log_items, "odd_items": odd_res,
    }


def _by_task(res):
    out = {}
    for r in res:
        out.setdefault(r["task"], []).append(r["correct"])
    return {k: float(np.mean(v)) for k, v in out.items()}
