"""Mixture reference model g for the deference target (Pillar 2), version 2.

Motivation (see E0): when a source A with reliability rho reports y_src, the exact Bayes
update of the predictive distribution of another rater is, for uniform rater errors,

    g(y) = (1 - b) * q0(y) + b * k(y | y_src),
    logit b = logit(rho) + log K + log q(y_src)

i.e. the colleague's rating moves probability mass toward a kernel around their rating, by
an amount set by their reliability and by how plausible that rating already was. A
log-linear reference (reference_model.py) cannot represent this; it captured only ~80% of
the Bayes update direction in E0.

Here the mixture form is the model and its coefficients are fitted:

    logit b = theta_0 + theta_1 * logit(rel) + theta_2 * log q0(y_src)
    k(y | y_src) = softmax_y( gamma * 1[y = y_src] - lam * |y - y_src| / (K - 1) )

theta = (theta_0..theta_2, gamma, lam). The uniform-error Bayes update corresponds to
theta ~ (log K, 1, 1) with a peaked kernel; ordinal rater errors are covered by lam.
(An entropy term was tried and dropped: it is not part of the Bayes form and is nearly
collinear with the intercept, which left the fit unidentified on PC-GITA-sized data.)
A source with stated reliability 0 (e.g. a random-rating or patient claim) gets b = 0, so
its reference update is exactly zero.

r*(y) = log g(y) - log q0(y). The fit uses penalised maximum likelihood with a
Jeffreys-prior (Firth-type) penalty 0.5 * log det I(theta) or a weak Gaussian prior
centred on the uniform-error Bayes solution.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

PARAM_NAMES = ("b_intercept", "b_logit_rel", "b_log_q_src", "k_match", "k_dist")
N_PARAMS = len(PARAM_NAMES)
REL_CLIP = 1e-3


def theory_init(K: int) -> np.ndarray:
    return np.array([np.log(K), 1.0, 1.0, 3.0, 0.0])


def _design(log_q0: np.ndarray, y_src: np.ndarray, rel: np.ndarray):
    n, K = log_q0.shape
    ys = np.arange(K)[None, :]
    m = (ys == y_src[:, None]).astype(float)
    d = np.abs(ys - y_src[:, None]) / (K - 1)
    rel_c = np.clip(rel, REL_CLIP, 1 - REL_CLIP)
    q = np.exp(log_q0)
    U = np.stack([np.ones(n), np.log(rel_c / (1 - rel_c)), log_q0[np.arange(n), y_src]], axis=1)
    null = rel <= 0
    return q, m, d, U, null


def _mixture(theta, q, m, d, U, null):
    """g, b, k for one parameter vector. Shapes: g,k (n,K); b (n,)."""
    b = expit(U @ theta[:3])
    b = np.where(null, 0.0, np.clip(b, 0.0, 1 - 1e-9))
    logits = theta[3] * m - theta[4] * d
    k = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    g = (1 - b)[:, None] * q + b[:, None] * k
    return g, b, k


def _jacobian(theta, q, m, d, U, null):
    """G[i, y, p] = d g_i(y) / d theta_p."""
    g, b, k = _mixture(theta, q, m, d, U, null)
    db = (b * (1 - b))[:, None]  # zero for null sources
    G_b = (db * (k - q))[:, :, None] * U[:, None, :]
    km = (k * m).sum(1, keepdims=True)
    kd = (k * d).sum(1, keepdims=True)
    G_gamma = b[:, None] * k * (m - km)
    G_lam = b[:, None] * k * (-d + kd)
    G = np.concatenate([G_b, G_gamma[:, :, None], G_lam[:, :, None]], axis=2)
    return g, G


def fisher_information(theta, q, m, d, U, null) -> np.ndarray:
    g, G = _jacobian(theta, q, m, d, U, null)
    return np.einsum("nkp,nkq->pq", G / g[:, :, None], G)


def fit_mixture(
    log_q0: np.ndarray,
    y_src: np.ndarray,
    rel: np.ndarray,
    y_target: np.ndarray,
    *,
    penalty: str = "firth",
    prior_sd: float = 2.5,
    ridge: float = 1e-6,
) -> np.ndarray:
    """Penalised ML fit. penalty: 'firth' | 'prior' | 'none'."""
    q, m, d, U, null = _design(log_q0, y_src, rel)
    K = log_q0.shape[1]
    rows = np.arange(len(y_target))
    theta0 = theory_init(K)
    eye = ridge * np.eye(N_PARAMS)

    def nll_and_grad(theta):
        g, G = _jacobian(theta, q, m, d, U, null)
        g_obs = g[rows, y_target]
        nll = -np.log(g_obs).sum()
        grad = -(G[rows, y_target] / g_obs[:, None]).sum(0)
        return nll, grad, g, G

    def objective(theta):
        nll, grad, g, G = nll_and_grad(theta)
        if penalty == "prior":
            diff = theta - theta0
            return nll + 0.5 * (diff**2).sum() / prior_sd**2, grad + diff / prior_sd**2
        if penalty == "firth":
            info = np.einsum("nkp,nkq->pq", G / g[:, :, None], G) + eye
            sign, logdet = np.linalg.slogdet(info)
            if sign <= 0:
                return np.inf, np.zeros_like(theta)
            # numeric gradient of the penalty only (6 parameters)
            eps, pen_grad = 1e-5, np.zeros_like(theta)
            for j in range(len(theta)):
                t = theta.copy()
                t[j] += eps
                pen_grad[j] = (
                    0.5 * np.linalg.slogdet(fisher_information(t, q, m, d, U, null) + eye)[1]
                    - 0.5 * logdet
                ) / eps
            return nll - 0.5 * logdet, grad - pen_grad
        return nll, grad

    res = minimize(objective, theta0, jac=True, method="L-BFGS-B")
    return res.x


def mixture_update(log_q0, y_src, rel, theta) -> np.ndarray:
    """r*(y) = log g(y) - log q0(y)."""
    q, m, d, U, null = _design(log_q0, y_src, rel)
    g, _, _ = _mixture(theta, q, m, d, U, null)
    return np.log(g) - log_q0


def cluster_robust_cov(log_q0, y_src, rel, y_target, theta, clusters) -> np.ndarray:
    """Case-clustered sandwich covariance I^-1 (sum_c S_c S_c^T) I^-1 * C/(C-1)."""
    q, m, d, U, null = _design(log_q0, y_src, rel)
    g, G = _jacobian(theta, q, m, d, U, null)
    rows = np.arange(len(y_target))
    scores = G[rows, y_target] / g[rows, y_target][:, None]
    _, idx = np.unique(clusters, return_inverse=True)
    n_c = idx.max() + 1
    S = np.zeros((n_c, N_PARAMS))
    np.add.at(S, idx, scores)
    info = np.einsum("nkp,nkq->pq", G / g[:, :, None], G) + 1e-8 * np.eye(N_PARAMS)
    inv = np.linalg.pinv(info)
    return inv @ (S.T @ S) @ inv * n_c / max(n_c - 1, 1)


def update_sampler(log_q0, y_src, rel, theta, cov):
    """f(rng, b) -> (b, n, K) reference updates with theta* ~ N(theta, cov)."""
    q, m, d, U, null = _design(log_q0, y_src, rel)

    def draw(rng: np.random.Generator, n_draws: int) -> np.ndarray:
        thetas = rng.multivariate_normal(theta, cov, size=n_draws, method="eigh")
        b = expit(np.einsum("np,bp->bn", U, thetas[:, :3]))
        b = np.where(null[None, :], 0.0, np.clip(b, 0.0, 1 - 1e-9))
        logits = thetas[:, 3, None, None] * m[None] - thetas[:, 4, None, None] * d[None]
        k = np.exp(logits - logsumexp(logits, axis=2, keepdims=True))
        g = (1 - b)[:, :, None] * q[None] + b[:, :, None] * k
        return np.log(g) - log_q0[None]

    return draw
