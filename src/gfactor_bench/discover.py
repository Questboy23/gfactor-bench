"""Find skill factors from the data instead of naming them by hand.

1. Accuracy per benchmark or MMLU subject for every reference model, split into two random halves of questions.
2. Remove g (first principal component) from both halves.
3. Correlate residuals across halves: sampling noise in one half does not correlate with the other half,
   so what remains is real shared skill.
4. Parallel analysis: keep as many factors as there are eigenvalues above the 95th percentile of a null
   where model order is shuffled in one half.
5. Average-linkage clustering of subjects into that many groups; groups under 3 subjects join their nearest group.
Each factor is named after its 3 most central subjects.
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage


def _logit(a):
    return np.log((a + 0.01) / (1.01 - a))


def short(name: str) -> str:
    return name.removeprefix("harness_hendrycksTest_").removeprefix("harness_").rsplit("_", 1)[0]


def discover(Y, keys, seed=0, max_factors=8, min_size=3, n_null=100):
    rng = np.random.default_rng(seed)
    names = sorted(set(keys))
    halves = [[], []]
    for n in names:
        cols = np.flatnonzero(keys == n)
        rng.shuffle(cols)
        h = len(cols) // 2
        halves[0].append(_logit(Y[:, cols[:h]].mean(1)))
        halves[1].append(_logit(Y[:, cols[h:]].mean(1)))
    H = [np.stack(h, 1) for h in halves]
    H = [(h - h.mean(0)) / (h.std(0) + 1e-9) for h in H]
    g = np.linalg.svd(H[0] + H[1], full_matrices=False)[0][:, 0]
    R = [h - np.outer(g, g @ h) for h in H]
    S = len(names)

    def cross(a, b):
        C = np.corrcoef(a.T, b.T)[:S, S:]
        return (C + C.T) / 2

    C = cross(R[0], R[1])
    ev = np.linalg.eigvalsh(C)[::-1]
    null = np.percentile([np.linalg.eigvalsh(cross(R[0], R[1][rng.permutation(len(g))]))[::-1] for _ in range(n_null)], 95, axis=0)
    k = int(min(max_factors, np.argmax(ev <= null)))
    k = max(k, 1)
    rel = np.clip(np.diag(C), 0.05, None)
    Cd = np.clip(C / np.sqrt(np.outer(rel, rel)), -1, 1)  # correct for each subject's reliability
    np.fill_diagonal(Cd, 1)
    D = 1 - Cd
    lab = fcluster(linkage(D[np.triu_indices(S, 1)], "average"), k, "maxclust") - 1
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
