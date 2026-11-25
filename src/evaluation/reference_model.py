"""Reference model g for the deference target (Pillar 2).

g keeps the assessor's statement-free log-probabilities as a fixed offset and adds a
small number of statement terms:

    log g(y) = log q0(y) + phi(y; y_src, rel, H(q0)) . beta - log Z

g0 is the same model with beta = 0, i.e. g0 = q0. The reference update is therefore

    r*(y) = log g(y) - log q0(y) = phi(y) . beta - log sum_y' q0(y') exp(phi(y') . beta)

beta is fitted on (statement, held-out rater rating) pairs by Firth-penalised maximum
likelihood (Jeffreys-prior penalty 0.5 * log det I(beta)), which keeps the fit finite on
small samples such as PC-GITA.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

FEATURE_NAMES = (
    "match",  # 1[y == y_src]
    "match_x_rel",  # rel * 1[y == y_src]
    "match_x_entropy",  # H(q0) * 1[y == y_src]   (normalised entropy in [0, 1])
    "neg_dist",  # -|y - y_src| / (K - 1)
    "neg_dist_x_rel",  # rel * -|y - y_src| / (K - 1)
)
N_FEATURES = len(FEATURE_NAMES)


def normalized_entropy(log_q: np.ndarray) -> np.ndarray:
    q = np.exp(log_q)
    k = log_q.shape[-1]
    return -(q * log_q).sum(-1) / np.log(k)


def features(log_q0: np.ndarray, y_src: np.ndarray, rel: np.ndarray) -> np.ndarray:
    """Feature tensor phi with shape (n_items, K, N_FEATURES)."""
    _, k = log_q0.shape
    ys = np.arange(k)[None, :]
    match = (ys == y_src[:, None]).astype(float)
    neg_dist = -np.abs(ys - y_src[:, None]) / (k - 1)
    rel = rel[:, None]
    ent = normalized_entropy(log_q0)[:, None]
    return np.stack([match, rel * match, ent * match, neg_dist, rel * neg_dist], axis=-1)


def _log_g(log_q0: np.ndarray, phi: np.ndarray, beta: np.ndarray) -> np.ndarray:
    eta = log_q0 + phi @ beta
    return eta - logsumexp(eta, axis=1, keepdims=True)


def fisher_information(log_q0: np.ndarray, phi: np.ndarray, beta: np.ndarray) -> np.ndarray:
    p = np.exp(_log_g(log_q0, phi, beta))
    m = np.einsum("nk,nkp->np", p, phi)
    return np.einsum("nk,nkp,nkq->pq", p, phi, phi) - m.T @ m


def fit_reference(
    log_q0: np.ndarray,
    phi: np.ndarray,
    y_target: np.ndarray,
    *,
    firth: bool = True,
    ridge: float = 1e-8,
) -> np.ndarray:
    """Fit beta by (Firth-penalised) maximum likelihood."""
    n = len(y_target)
    rows = np.arange(n)
    phi_obs = phi[rows, y_target]  # (n, P)
    eye = ridge * np.eye(phi.shape[-1])

    def objective(beta):
        log_g = _log_g(log_q0, phi, beta)
        p = np.exp(log_g)
        ll = log_g[rows, y_target].sum()
        grad = phi_obs.sum(0) - np.einsum("nk,nkp->p", p, phi)
        if not firth:
            return -ll, -grad
        m = np.einsum("nk,nkp->np", p, phi)
        info = np.einsum("nk,nkp,nkq->pq", p, phi, phi) - m.T @ m + eye
        sign, logdet = np.linalg.slogdet(info)
        if sign <= 0:
            return np.inf, np.zeros_like(beta)
        # d/d beta_j of 0.5 log det I = 0.5 tr(I^-1 dI/d beta_j)
        inv = np.linalg.inv(info)
        dp = p[:, :, None] * (phi - m[:, None, :])  # dp_k/d beta_j  (n, K, P)
        dm = np.einsum("nkj,nkp->npj", dp, phi)  # dm_p/d beta_j
        dI = np.einsum("nkj,nkp,nkq->pqj", dp, phi, phi) - (
            np.einsum("npj,nq->pqj", dm, m) + np.einsum("np,nqj->pqj", m, dm)
        )
        grad_pen = 0.5 * np.einsum("pq,qpj->j", inv, dI)
        return -(ll + 0.5 * logdet), -(grad + grad_pen)

    res = minimize(objective, np.zeros(phi.shape[-1]), jac=True, method="L-BFGS-B")
    return res.x


def reference_update(log_q0: np.ndarray, phi: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """r*(y) = log g(y) - log g0(y), with g0 = q0."""
    return _log_g(log_q0, phi, beta) - log_q0


def cluster_robust_cov(
    log_q0: np.ndarray,
    phi: np.ndarray,
    y_target: np.ndarray,
    beta: np.ndarray,
    clusters: np.ndarray,
) -> np.ndarray:
    """Case-clustered sandwich covariance  I^-1 (sum_c S_c S_c^T) I^-1 * C/(C-1)."""
    p = np.exp(_log_g(log_q0, phi, beta))
    m = np.einsum("nk,nkp->np", p, phi)
    scores = phi[np.arange(len(y_target)), y_target] - m
    _, idx = np.unique(clusters, return_inverse=True)
    n_c = idx.max() + 1
    S = np.zeros((n_c, phi.shape[-1]))
    np.add.at(S, idx, scores)
    info = fisher_information(log_q0, phi, beta) + 1e-8 * np.eye(phi.shape[-1])
    inv = np.linalg.inv(info)
    return inv @ (S.T @ S) @ inv * n_c / max(n_c - 1, 1)


def update_sampler(log_q0: np.ndarray, phi: np.ndarray, beta: np.ndarray, cov: np.ndarray):
    """Return f(rng, b) -> (b, n, K) reference updates with beta* ~ N(beta, cov).

    Used inside the bootstrap so that intervals for kappa include the uncertainty of the
    fitted reference model, without refitting it thousands of times.
    """

    def draw(rng: np.random.Generator, b: int) -> np.ndarray:
        betas = rng.multivariate_normal(beta, cov, size=b, method="eigh")  # (b, P)
        eta = np.einsum("nkp,bp->bnk", phi, betas)
        log_g = log_q0[None] + eta
        log_g = log_g - logsumexp(log_g, axis=2, keepdims=True)
        return log_g - log_q0[None]

    return draw
