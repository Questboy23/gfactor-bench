import numpy as np

from gfactor_bench.discover import discover


def test_discover_finds_planted_groups():
    # 12 subjects in 3 planted skill groups on top of a strong g; 300 models, 60 questions per subject
    rng = np.random.default_rng(0)
    N, S, Q = 300, 12, 60
    group = np.repeat([0, 1, 2], 4)
    g, sk = rng.normal(size=N), rng.normal(size=(N, 3))
    keys, cols = [], []
    for s in range(S):
        z = 1.5 * g + 0.9 * sk[:, group[s]]
        b = rng.normal(0, 1, Q)
        cols.append((rng.random((N, Q)) < 1 / (1 + np.exp(-(z[:, None] - b)))).astype(float))
        keys += [f"harness_hendrycksTest_subj{s}_5"] * Q
    d = discover(np.concatenate(cols, 1), np.array(keys), n_null=30)
    lab = dict(zip(d["subjects"], d["subject_factor"]))
    found = np.array([lab[f"harness_hendrycksTest_subj{s}_5"] for s in range(S)])
    assert d["n_factors"] == 3
    for k in range(3):  # every planted group lands in a single found factor
        assert len(set(found[group == k])) == 1
