"""Find skill factors from the data instead of naming them by hand.

1. Accuracy per benchmark or MMLU subject for every reference model, split into two random halves of questions.
2. Remove g (first principal component) from both halves.
3. Correlate residuals across halves: sampling noise in one half does not correlate with the other half,
   so what remains is real shared skill.
4. Parallel analysis: keep as many factors as there are eigenvalues above the 95th percentile of a null
   where model order is shuffled in one half.
5. Place subjects by direction in that many eigenvectors and cluster them into that many + 1 groups; groups under 3 subjects join their nearest group.
Each factor is named after its 3 most central subjects.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage


def _logit(a):
    return np.log((a + 0.01) / (1.01 - a))


def short(name: str) -> str:
    for pre in ("matharena_", "livebench_"):
        if name.startswith(pre):
            return name.removeprefix(pre)
    return name.removeprefix("harness_hendrycksTest_").removeprefix("harness_").rsplit("_", 1)[0]


def discover(Y, keys, seed=0, max_factors=8, min_size=3, n_null=100, min_overlap=20):
    rng = np.random.default_rng(seed)
    names = sorted(set(keys))
    halves = [[], []]
    for n in names:
        cols = np.flatnonzero(keys == n)
        rng.shuffle(cols)
        h = len(cols) // 2
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            halves[0].append(_logit(np.nanmean(Y[:, cols[:h]], 1)))
            halves[1].append(_logit(np.nanmean(Y[:, cols[h:]], 1)))
    H = [np.stack(h, 1) for h in halves]
    H = [(h - np.nanmean(h, 0)) / (np.nanstd(h, 0) + 1e-9) for h in H]
    Z = [np.nan_to_num(h) for h in H]  # untested = subject mean, only for finding g
    g = np.linalg.svd(Z[0] + Z[1], full_matrices=False)[0][:, 0]
    R = [h - np.outer(g, g @ z) for h, z in zip(H, Z)]  # NaN stays NaN
    S = len(names)

    def cross(a, b):
        # pairwise over models tested on both subjects (models differ across leaderboards)
        C = np.ma.corrcoef(np.ma.masked_invalid(np.concatenate([a, b], 1)).T)[:S, S:].filled(0.0)
        shared = (~np.isnan(a)).astype(float).T @ (~np.isnan(b))
        C[shared < min_overlap] = 0.0  # too few models tested on both subjects to tell
        return (C + C.T) / 2

    C = cross(R[0], R[1])
    ev = np.linalg.eigvalsh(C)[::-1]
    def shuffled(r):
        # null: shuffle each subject's scores among the models tested on it, so missing cells stay where they are
        r = r.copy()
        for j in range(r.shape[1]):
            ok = np.flatnonzero(~np.isnan(r[:, j]))
            r[ok, j] = r[rng.permutation(ok), j]
        return r

    null = np.percentile([np.linalg.eigvalsh(cross(R[0], shuffled(R[1])))[::-1] for _ in range(n_null)], 95, axis=0)
    below = np.flatnonzero(ev <= null)
    k = int(min(max_factors, below[0] if len(below) else len(ev)))  # every eigenvalue above the null: keep all
    k = max(1, min(k, S // min_size - 1))  # k + 1 groups of at least min_size subjects must fit
    rel = np.clip(np.diag(C), 0.05, None)
    Cd = np.clip(C / np.sqrt(np.outer(rel, rel)), -1, 1)  # correct for each subject's reliability
    np.fill_diagonal(Cd, 1)
    # Removing g costs one dimension (k skill groups leave k - 1 independent leftovers), so subjects are placed
    # by their direction in the top k eigenvectors and grouped into k + 1 clusters. Grouping in this embedding
    # stays stable when many subject pairs share no models (raw clustering chained them into one group).
    w, V = np.linalg.eigh(Cd)
    E = V[:, ::-1][:, :k] * np.sqrt(np.clip(w[::-1][:k], 0, None))
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    lab = fcluster(linkage(E, "average"), k + 1, "maxclust") - 1  # unit rows: euclidean ranks like cosine
    # merge tiny groups into the group they correlate with most
    while True:
        sizes = np.bincount(lab)
        small = [f for f in range(len(sizes)) if 0 < sizes[f] < min_size]
        if not small or (sizes > 0).sum() == 1:
            break
        f = small[0]
        members = lab == f
        others = [o for o in range(len(sizes)) if o != f and sizes[o] > 0]
        best = max(others, key=lambda o: Cd[np.ix_(members, lab == o)].mean())
        lab[members] = best
    uniq = sorted(set(lab))
    lab = np.array([uniq.index(x) for x in lab])
    factor_names = []
    for f in range(len(uniq)):
        m = np.flatnonzero(lab == f)
        central = m[np.argsort(-Cd[np.ix_(m, m)].mean(1))][:3]
        factor_names.append("+".join(short(names[i]) for i in central))
    return {"subjects": names, "subject_factor": lab, "factor_names": factor_names,
            "eigen": ev[:10], "null95": null[:10], "n_factors": len(uniq)}
