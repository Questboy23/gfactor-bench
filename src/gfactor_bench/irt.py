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
    """Joint MAP fit of item and person parameters by Adam.

    Y: (models, items) 0/1 matrix. fidx: (items,) specific factor index. c: (items,) guessing floor.
    Priors: theta ~ N(0, 1); raw loadings ~ N(0.5, 1) before softplus; b ~ N(0, 2^2).
    """
    rng = np.random.default_rng(seed)
    N, J = Y.shape
    Y = Y.astype(np.float32)
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
    tot = Y.mean(1)
    p["tg"] = ((tot - tot.mean()) / (tot.std() + 1e-9)).astype(np.float32)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v2 = {k: np.zeros_like(v) for k, v in p.items()}
    for t in range(1, iters + 1):
        ag, as_ = _softplus(p["ag"]), _softplus(p["as"])
        TS = p["ts"][:, fidx]  # (N, J)
        z = p["tg"][:, None] * ag[None] + TS * as_[None] - p["b"][None]
        s = _sig(z)
        pr = np.clip(c + (1 - c) * s, 1e-6, 1 - 1e-6)
        G = (Y / pr - (1 - Y) / (1 - pr)) * (1 - c) * s * (1 - s)  # dlogL/dz
        grads = {
            "tg": G @ ag - p["tg"],
            "ts": (G * as_[None]) @ onehot - p["ts"],
            "ag": (G * p["tg"][:, None]).sum(0) * _sig(p["ag"]) - (p["ag"] - 0.5),
            "as": (G * TS).sum(0) * _sig(p["as"]) - (p["as"] - 0.5),
            "b": -G.sum(0) - p["b"] / 4.0,
        }
        for k in p:  # Adam ascent
            g = grads[k]
            m[k] = 0.9 * m[k] + 0.1 * g
            v2[k] = 0.999 * v2[k] + 0.001 * g * g
            p[k] += lr * (m[k] / (1 - 0.9**t)) / (np.sqrt(v2[k] / (1 - 0.999**t)) + 1e-8)
        if t % 100 == 0 or t == 1:
            ll = (Y * np.log(pr) + (1 - Y) * np.log(1 - pr)).sum() / Y.size
            log(f"iter {t} mean loglik {ll:.4f}")
    return {
        "a_g": _softplus(p["ag"]), "a_s": _softplus(p["as"]), "b": p["b"], "c": c,
        "fidx": fidx, "theta_g": p["tg"], "theta_s": p["ts"],
    }


def loading_matrix(a_g, a_s, fidx, n_factors):
    """(items, 1 + n_factors) loadings: column 0 is g, column 1+f is specific factor f."""
    L = np.zeros((len(a_g), 1 + n_factors))
    L[:, 0] = a_g
    L[np.arange(len(a_g)), 1 + fidx] = a_s
    return L


def estimate_theta(L, b, c, y, iters=50):
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
