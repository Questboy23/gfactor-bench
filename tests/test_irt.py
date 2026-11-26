import numpy as np

from gfactor_bench.irt import Z90, estimate_theta, fit_bifactor, loading_matrix, select_items


def _sim(N=300, J=600, F=3, seed=1):
    rng = np.random.default_rng(seed)
    fidx = rng.integers(0, F, J)
    ag, as_ = rng.uniform(0.6, 1.8, J), rng.uniform(0.3, 1.0, J)
    b = rng.normal(0, 1, J)
    c = np.where(rng.random(J) < 0.5, 0.25, 0.0)
    tg, ts = rng.normal(0, 1, N), rng.normal(0, 1, (N, F))
    z = tg[:, None] * ag + ts[:, fidx] * as_ - b
    p = c + (1 - c) / (1 + np.exp(-z))
    Y = (rng.random((N, J)) < p).astype(float)
    return Y, fidx, c, F, tg, ts, ag, as_, b


def test_fit_recovers_g():
    Y, fidx, c, F, tg, ts, *_ = _sim()
    fit = fit_bifactor(Y, fidx, c, F, iters=600, log=lambda *_: None)
    assert np.corrcoef(fit["theta_g"], tg)[0, 1] > 0.9
    for f in range(F):
        assert np.corrcoef(fit["theta_s"][:, f], ts[:, f])[0, 1] > 0.5
        assert abs(np.corrcoef(fit["theta_s"][:, f], fit["theta_g"])[0, 1]) < 0.05  # bifactor: orthogonal to g


def test_intervals_cover_about_90_percent():
    # true item parameters, many simulated models: the 90% interval for g should cover ~90%
    _, fidx, c, F, _, _, ag, as_, b = _sim()
    L = loading_matrix(ag, as_, fidx, F)
    rng = np.random.default_rng(7)
    hits = 0
    n = 300
    for _ in range(n):
        th = rng.normal(0, 1, 1 + F)
        idx = rng.choice(len(b), 120, replace=False)
        p = c[idx] + (1 - c[idx]) / (1 + np.exp(-(L[idx] @ th - b[idx])))
        y = (rng.random(len(idx)) < p).astype(float)
        est, cov = estimate_theta(L[idx], b[idx], c[idx], y)
        hits += abs(est[0] - th[0]) <= Z90 * np.sqrt(cov[0, 0])
    assert 0.84 <= hits / n <= 0.96


def test_selection_is_random_but_informative():
    _, fidx, c, F, _, _, ag, as_, b = _sim()
    L = loading_matrix(ag, as_, fidx, F)
    th, cov = np.zeros(1 + F), np.eye(1 + F)
    picks = [tuple(select_items(L, b, c, th, cov, range(len(b)), 1, 5, np.random.default_rng(s))) for s in range(5)]
    assert len(set(picks)) > 1  # different seeds give different items
    # dim 1 = factor 0; later picks may use g-loaded items that help through the g correlation
    assert all(fidx[p[0]] == 0 for p in picks)
    assert np.mean([fidx[j] == 0 for p in picks for j in p]) > 0.6


def test_fit_with_missing_data_and_soft_labels():
    # two leaderboards with different question sets, linked only by 40 models that took both;
    # scores are averages of 4 attempts (soft labels), as in MathArena
    Y, fidx, c, F, tg, ts, ag, as_, b = _sim(seed=3)
    rng = np.random.default_rng(3)
    pr = c + (1 - c) / (1 + np.exp(-(tg[:, None] * ag + ts[:, fidx] * as_ - b)))
    soft = (rng.random((4, *Y.shape)) < pr).mean(0)
    board = rng.integers(0, 2, Y.shape[1])
    who = rng.integers(0, 2, Y.shape[0])
    both = rng.random(Y.shape[0]) < 0.15
    mask = (board[None] == who[:, None]) | both[:, None]
    fit = fit_bifactor(np.where(mask, soft, np.nan), fidx, c, F, iters=600, log=lambda *_: None)
    assert np.corrcoef(fit["theta_g"], tg)[0, 1] > 0.85
