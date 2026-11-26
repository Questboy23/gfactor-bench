"""Fit the bifactor model on the Open LLM Leaderboard correctness matrix (395 models x 28,659 questions)
released by tinyBenchmarks (Polo et al. 2024, MIT) and save item parameters to data/calibration.npz."""
from __future__ import annotations

import importlib
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import requests

from .bank import guess_floor
from .discover import discover
from .irt import fit_bifactor

LB_URL = "https://github.com/felipemaiapolo/tinyBenchmarks/raw/main/tutorials/data/lb.pickle"
DATA = Path(__file__).parent / "data"


class _NumpyOnly(pickle.Unpickler):
    """The file is a pickle; allow only numpy array reconstruction so it cannot run code."""

    def find_class(self, module, name):
        if module.startswith("numpy") and name in ("_frombuffer", "dtype", "_reconstruct", "ndarray", "scalar"):
            return getattr(importlib.import_module(module), name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def load_leaderboard(path: Path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {LB_URL} (90 MB, once)")
        path.write_bytes(requests.get(LB_URL, timeout=600).content)
    return _NumpyOnly(path.open("rb")).load()


def build_matrix(lb):
    keys, idxs, cols = [], [], []
    for k, v in lb["data"].items():
        x = v["correctness"]
        if k == "harness_truthfulqa_mc_0":
            x = (x > 0.5).astype(float)  # mc2 is a probability; binarize
        cols.append(x)
        keys += [k] * x.shape[0]
        idxs += list(range(x.shape[0]))
    Y = np.concatenate(cols, 0).T  # (models, items)
    return Y, np.array(keys), np.array(idxs)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv  # [] from the CLI means the default cache path
    if argv[:1] == ["frontier"]:
        return frontier()
    cache = Path.home() / ".cache" / "gfactor-bench" / "lb.pickle"
    lb = load_leaderboard(Path(argv[0]) if argv else cache)
    Y, keys, idxs = build_matrix(lb)
    d = discover(Y, keys)
    sub_f = dict(zip(d["subjects"], d["subject_factor"]))
    fidx = np.array([sub_f[k] for k in keys])
    c = np.array([guess_floor(k) for k in keys])
    print(f"matrix {Y.shape}; {d['n_factors']} factors found: {d['factor_names']}")
    print("eigenvalues", np.round(d["eigen"], 2), "null 95%", np.round(d["null95"], 2))
    fit = fit_bifactor(Y, fidx, c, d["n_factors"], iters=800)
    DATA.mkdir(exist_ok=True)
    np.savez_compressed(
        DATA / "calibration.npz",
        keys=keys, idxs=idxs, fidx=fidx, c=c.astype(np.float32),
        a_g=fit["a_g"].astype(np.float32), a_s=fit["a_s"].astype(np.float32), b=fit["b"].astype(np.float32),
        ref_theta_g=fit["theta_g"], ref_theta_s=fit["theta_s"], ref_models=np.array(lb["models"]),
        ref_subscores=subscores(Y, keys),
        subscore_names=np.array(sorted(set(keys))),
        factor_names=np.array(d["factor_names"]), subject_factor=d["subject_factor"],
    )
    print("saved", DATA / "calibration.npz")


def subscores(Y, keys):
    """Accuracy per benchmark or subject for each reference model (for the correlation heatmap); NaN if untested."""
    names = sorted(set(keys))
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.stack([np.nanmean(Y[:, keys == n], 1) for n in names], 1).astype(np.float32)


SERVE = {"livebench_connections": "connections", "livebench_typos": "contains"}  # LiveBench tasks with a local scorer


def frontier(min_models=15, min_items=20):
    """Frontier panel: MathArena + LiveBench per-question scores, models joined by normalized name.
    Saves data/calibration_frontier.npz and data/frontier_items.json (prompts for the questions a test can ask)."""
    import json

    from .sources import livebench, matharena, norm_model
    rows, items = [], {}
    for r, it in (livebench(), matharena()):
        rows += r
        items.update(it)
    cell, shown = {}, {}
    for subj, q, m, v in rows:
        k = norm_model(m)
        shown.setdefault(k, m)
        cell.setdefault((k, f"{subj}|{q}"), []).append(v)
    cols = sorted({c for _, c in cell})
    n_by_col = {}
    for _, c in cell:
        n_by_col[c] = n_by_col.get(c, 0) + 1
    cols = [c for c in cols if n_by_col[c] >= min_models]  # too few answers to place a question
    ci = {c: j for j, c in enumerate(cols)}
    models = sorted({m for m, _ in cell})
    mi = {m: i for i, m in enumerate(models)}
    Y = np.full((len(models), len(cols)), np.nan)
    for (m, c), v in cell.items():
        if c in ci:
            Y[mi[m], ci[c]] = np.mean(v)
    keep = (~np.isnan(Y)).sum(1) >= min_items
    Y, models = Y[keep], [m for m, k in zip(models, keep) if k]
    keys = np.array([c.split("|")[0] for c in cols])
    idxs = np.array([c.split("|", 1)[1] for c in cols])
    src = np.array([k.split("_")[0] for k in keys])
    both = sum(((~np.isnan(Y[:, src == "matharena"])).any(1) & (~np.isnan(Y[:, src == "livebench"])).any(1)))
    print(f"frontier matrix {Y.shape}, {np.isnan(Y).mean():.0%} missing, {both} models link MathArena and LiveBench")
    # 12 shared models cannot measure skill correlations across leaderboards, so factors are found within each
    sub_f, names, subject_factor, all_subjects = {}, [], [], []
    for s_ in ("livebench", "matharena"):
        m = src == s_
        rows_ = (~np.isnan(Y[:, m])).any(1)
        d = discover(Y[np.ix_(rows_, m)], keys[m])
        print(f"{s_}: {d['n_factors']} factors {d['factor_names']}; eigenvalues {np.round(d['eigen'][:6], 2)} null {np.round(d['null95'][:6], 2)}")
        for subj, f in zip(d["subjects"], d["subject_factor"]):
            sub_f[subj] = len(names) + f
        names += d["factor_names"]
    d = {"subjects": sorted(sub_f), "factor_names": names, "n_factors": len(names)}
    d["subject_factor"] = np.array([sub_f[x] for x in d["subjects"]])
    fidx = np.array([sub_f[k] for k in keys])
    fit = fit_bifactor(Y, fidx, np.zeros(len(keys)), d["n_factors"], iters=800)
    serve = {}
    for k, i in zip(keys, idxs):
        prob, gold = items.get(k, {}).get(i, (None, None))
        # broken-problem and Lean sets are graded by other means, so only plain final-answer sets are asked
        plain = k.startswith("matharena_") and not k.startswith(("matharena_brokenarxiv", "matharena_arxivlean"))
        kind = SERVE.get(k, "boxed" if plain else None)
        if prob and gold and kind:
            if kind == "boxed":
                prob += "\n\nPut the final answer in \\boxed{}."
            serve[f"{k}|{i}"] = [prob, str(gold), kind]
    np.savez_compressed(
        DATA / "calibration_frontier.npz",
        keys=keys, idxs=idxs, fidx=fidx, c=np.zeros(len(keys), np.float32),
        a_g=fit["a_g"].astype(np.float32), a_s=fit["a_s"].astype(np.float32), b=fit["b"].astype(np.float32),
        ref_theta_g=fit["theta_g"], ref_theta_s=fit["theta_s"], ref_models=np.array([shown[m] for m in models]),
        ref_subscores=subscores(Y, keys), subscore_names=np.array(sorted(set(keys))),
        factor_names=np.array(d["factor_names"]), subject_factor=d["subject_factor"],
    )
    (DATA / "frontier_items.json").write_text(json.dumps(serve))
    print(f"saved calibration_frontier.npz; {len(serve)} of {len(keys)} questions can be asked at test time")


if __name__ == "__main__":
    main()
