"""Fit the bifactor model on the Open LLM Leaderboard correctness matrix (395 models x 28,659 questions)
released by tinyBenchmarks (Polo et al. 2024, MIT) and save item parameters to data/calibration.npz."""
from __future__ import annotations

import importlib
import pickle
import sys
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
    argv = argv or sys.argv[1:]
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
    """Accuracy per benchmark or MMLU subject for each reference model (for the correlation heatmap)."""
    names = sorted(set(keys))
    return np.stack([Y[:, keys == n].mean(1) for n in names], 1).astype(np.float32)


if __name__ == "__main__":
    main()
