"""Bifactor 3PL item response model: one general factor (g) plus one specific factor per item.

P(correct) = c + (1 - c) * sigmoid(a_g * theta_g + a_s * theta_s[factor(item)] - b)
c is fixed per item (1 / number of choices), so only a_g, a_s, b are fitted.
"""
from __future__ import annotations

import numpy as np

Z90 = 1.6448536269514722


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _softplus(x):
    return np.logaddexp(0.0, x)


def fit_bifactor(Y, fidx, c, n_factors, iters=1500, lr=0.05, seed=0, log=print):
    """Two stages. Stage 1 fits g alone, each factor weighted equally so a large benchmark cannot
    define g. Stage 2 holds person g fixed and fits the specific factors on what g leaves unexplained."""
    J = Y.shape[1]
    w = (1.0 / np.bincount(fidx, minlength=n_factors)[fidx]) * (J / n_factors)
    s1 = _fit(Y, fidx, c, n_factors, iters // 2, lr, seed, log, item_w=w, use_s=False)
    return _fit(Y, fidx, c, n_factors, iters // 2, lr, seed, log, init=s1, fix_g=True)


def _fit(Y, fidx, c, n_factors, iters, lr, seed, log, item_w=None, use_s=True, init=None, fix_g=False):
    """Joint MAP fit of item and person parameters by Adam.

    Y: (models, items) scores in [0, 1] (fractions are soft labels); NaN = not tested, left out of the likelihood. fidx: (items,) specific factor index. c: (items,) guessing floor.
    Priors: theta ~ N(0, 1); raw loadings ~ N(0.5, 1) before softplus; b ~ N(0, 2^2).
    """
    rng = np.random.default_rng(seed)
    N, J = Y.shape
    M = (~np.isnan(Y)).astype(np.float32)
    Y = np.nan_to_num(Y).astype(np.float32)
    c = c.astype(np.float32)
    onehot = np.zeros((J, n_factors), np.float32)
    onehot[np.arange(J), fidx] = 1
    p = {
        "tg": rng.normal(0, 0.1, N).astype(np.float32),
        "ts": rng.normal(0, 0.1, (N, n_factors)).astype(np.float32),
        "ag": np.full(J, 0.5, np.float32),
        "as": np.full(J, 0.0, np.float32),
        "b": np.zeros(J, np.float32),
    }
    # start g at standardized total score so the sign of g is "more correct = higher"
    tot = (Y * M).sum(1) / np.maximum(M.sum(1), 1)
    p["tg"] = ((tot - tot.mean()) / (tot.std() + 1e-9)).astype(np.float32)
    if init is not None:
        p["tg"] = init["theta_g"].astype(np.float32)
        # start each specific factor at its observed leftover: factor accuracy (logit) minus what g predicts
        for f in range(n_factors):
            n = M[:, fidx == f].sum(1)
            a = (Y * M)[:, fidx == f].sum(1) / np.maximum(n, 1)
            lg = np.log((a + 0.01) / (1.01 - a))
            X = np.stack([np.ones(N), p["tg"], p["tg"] ** 2], 1)
            ok = n > 0
            r = np.zeros(N)
            r[ok] = lg[ok] - X[ok] @ np.linalg.lstsq(X[ok], lg[ok], rcond=None)[0]
            r[ok] = (r[ok] - r[ok].mean()) / (r[ok].std() + 1e-9)
            p["ts"][:, f] = r.astype(np.float32)  # models never tested on this factor start at 0
        p["ag"] = np.log(np.expm1(np.maximum(init["a_g"], 1e-4))).astype(np.float32)
        p["b"] = init["b"].astype(np.float32)
    iw = np.ones(J, np.float32) if item_w is None else item_w.astype(np.float32)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v2 = {k: np.zeros_like(v) for k, v in p.items()}
    for t in range(1, iters + 1):
        ag, as_ = _softplus(p["ag"]), _softplus(p["as"])
        TS = p["ts"][:, fidx]  # (N, J)
        z = p["tg"][:, None] * ag[None] + (TS * as_[None] if use_s else 0) - p["b"][None]
        s = _sig(z)
        pr = np.clip(c + (1 - c) * s, 1e-6, 1 - 1e-6)
        G = (Y / pr - (1 - Y) / (1 - pr)) * (1 - c) * s * (1 - s) * iw[None] * M  # dlogL/dz
        grads = {
            "tg": G @ ag - p["tg"],
            "ts": (G * as_[None]) @ onehot - p["ts"],
            "ag": (G * p["tg"][:, None]).sum(0) * _sig(p["ag"]) - (p["ag"] - 0.5),
            "as": (G * TS).sum(0) * _sig(p["as"]) - (p["as"] - 0.5),
            "b": -G.sum(0) - p["b"] / 4.0,
        }
        if fix_g:  # stage 2: person scores stay at g and the observed leftovers; only question parameters move
            grads["tg"] = np.zeros_like(grads["tg"])
            grads["ts"] = np.zeros_like(grads["ts"])
        if not use_s:
            grads["ts"] = np.zeros_like(grads["ts"])
            grads["as"] = np.zeros_like(grads["as"])
        for k in p:  # Adam ascent
            g = grads[k]
            m[k] = 0.9 * m[k] + 0.1 * g
            v2[k] = 0.999 * v2[k] + 0.001 * g * g
            p[k] += lr * (m[k] / (1 - 0.9**t)) / (np.sqrt(v2[k] / (1 - 0.999**t)) + 1e-8)
        # bifactor constraints: g has mean 0, SD 1; each specific factor is uncorrelated with g, mean 0, SD 1.
        # Without this, specific factors absorb general variance and g loses meaning.
        tg = p["tg"]
        tg = (tg - tg.mean()) / (tg.std() + 1e-9)
        ts = p["ts"] - p["ts"].mean(0)
        ts = ts - np.outer(tg, tg @ ts / len(tg))
        p["tg"], p["ts"] = tg, ts / (ts.std(0) + 1e-9)
        if t % 100 == 0 or t == 1:
            ll = (M * (Y * np.log(pr) + (1 - Y) * np.log(1 - pr))).sum() / M.sum()
            log(f"iter {t} mean loglik {ll:.4f}")
    return {
        "a_g": _softplus(p["ag"]), "a_s": _softplus(p["as"]), "b": p["b"], "c": c,
        "fidx": fidx, "theta_g": p["tg"], "theta_s": p["ts"] if use_s else np.zeros_like(p["ts"]),
    }


def loading_matrix(a_g, a_s, fidx, n_factors):
    """(items, 1 + n_factors) loadings: column 0 is g, column 1+f is specific factor f."""
    L = np.zeros((len(a_g), 1 + n_factors))
    L[:, 0] = a_g
    L[np.arange(len(a_g)), 1 + fidx] = a_s
    return L


def estimate_theta(L, b, c, y, iters=200):
    """MAP ability with N(0, I) prior and Laplace posterior covariance.

    L: (n, D) loadings of answered items, y: (n,) 0/1. Returns (theta, cov).
    """
    D = L.shape[1]
    th = np.zeros(D)
    for _ in range(iters):
        s = _sig(L @ th - b)
        pr = np.clip(c + (1 - c) * s, 1e-9, 1 - 1e-9)
        w = (1 - c) * s * (1 - s)
        gz = (y / pr - (1 - y) / (1 - pr)) * w
        grad = L.T @ gz - th
        info = (w**2) / (pr * (1 - pr))  # expected information per item
        H = (L * info[:, None]).T @ L + np.eye(D)
        step = np.linalg.solve(H, grad)
        step *= min(1.0, 0.5 / (np.abs(step).max() + 1e-12))  # damped Newton: 3PL surfaces make full steps oscillate
        th += step
        if np.abs(step).max() < 1e-6:
            break
    s = _sig(L @ th - b)
    pr = np.clip(c + (1 - c) * s, 1e-9, 1 - 1e-9)
    info = ((1 - c) * s * (1 - s)) ** 2 / (pr * (1 - pr))
    cov = np.linalg.inv((L * info[:, None]).T @ L + np.eye(D))
    return th, cov


def item_info(L, b, c, th):
    s = _sig(L @ th - b)
    pr = np.clip(c + (1 - c) * s, 1e-9, 1 - 1e-9)
    return ((1 - c) * s * (1 - s)) ** 2 / (pr * (1 - pr))


def select_items(L, b, c, th, cov, candidates, dim, k, rng, top=8):
    """Pick k items that most shrink the posterior variance of dimension `dim`.

    Uses the exact rank-one covariance update, then picks at random among the `top` best
    (randomesque exposure control) so no single item is always chosen.
    """
    chosen = []
    cand = np.array(candidates)
    cov = cov.copy()
    for _ in range(k):
        if len(cand) == 0:
            break
        Lc = L[cand]
        w = item_info(Lc, b[cand], c[cand], th)
        Sv = Lc @ cov  # (n, D)
        gain = w * Sv[:, dim] ** 2 / (1 + w * np.einsum("nd,nd->n", Sv, Lc))
        best = np.argsort(-gain)[:top]
        pick = rng.choice(best)
        j = cand[pick]
        chosen.append(int(j))
        sv, wj = Sv[pick], w[pick]
        cov = cov - np.outer(sv, sv) * wj / (1 + wj * sv @ L[j])
        cand = np.delete(cand, pick)
    return chosen
