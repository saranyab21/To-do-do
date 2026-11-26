"""Pillar-1 competence and grounding gate for a query-conditioned assessor.

Notation: for case i and query dimension d the model gives q_i(. | x, d) over the codes of
d; the pathologists give the answer shares p_i(. | d). Four tests, each with one
manipulation and a pre-stated pass rule (95% intervals, slide-stratified bootstrap):

  G1 calibration   no manipulation. Soft NLL of q after a leave-one-slide-out (LOSO)
                   temperature minus the LOSO label prior's NLL. Pass: interval < -margin.
  G2 entropy       no manipulation. Spearman correlation between the entropy of q and the
                   pathologists' entropy, both normalised by log K. Pass: interval > 0.
  G3 query swap    same image, other question. For each pair of dimensions (t, u) with
                   one code set: [NLL(t,t) + NLL(u,u)] - [NLL(t,u) + NLL(u,t)], where
                   NLL(a,b) scores the answer to question a against the answers for b.
                   Pass: interval < -margin for every pair involving t. (Amended after
                   gate_v1: the first rule compared questions within one target only, so a
                   question whose answer was generically better passed; the swap cancels it.)
  G4 grounding     same question, another core's image ("shuffled", from another slide) or
                   a blank image. NLL(shuffled) - NLL(real). Pass: interval > margin.

margin = min_effect_nats in the config (0.01 nats), so a negligible difference cannot pass.

Reference values come from the pathologists themselves where one exists: their own
entropy (floor), their answers to the asked dimension as a predictor (query swap), and how
much their answers differ between a core and its shuffle partner (grounding).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.special import log_softmax
from scipy.stats import ConstantInputWarning, spearmanr

from src.evaluation.soft_label import fit_temperature, soft_nll, temper

EPS = 1e-12


def logq_and_p(g, codes) -> tuple[np.ndarray, np.ndarray]:
    lp = g[[f"lp_{c}" for c in codes]].to_numpy(float)
    p = g[[f"n_{c}" for c in codes]].to_numpy(float) / g.n_raters.to_numpy(float)[:, None]
    return log_softmax(lp, axis=1), p


def smoothed_log_shares(g, codes, alpha: float = 0.5) -> np.ndarray:
    """Pathologists' shares with add-alpha smoothing, as a predictor (log)."""
    n = g[[f"n_{c}" for c in codes]].to_numpy(float) + alpha
    return np.log(n / n.sum(1, keepdims=True))


def loso_temperature(logq: np.ndarray, p: np.ndarray, folds: np.ndarray) -> np.ndarray:
    out = np.empty_like(logq)
    for f in np.unique(folds):
        te, tr = folds == f, folds != f
        out[te] = temper(logq[te], fit_temperature(logq[tr], p[tr]))
    return out


def loso_prior_logq(p: np.ndarray, folds: np.ndarray) -> np.ndarray:
    out = np.empty_like(p)
    for f in np.unique(folds):
        prior = np.clip(p[folds != f].mean(0), 1e-6, None)
        out[folds == f] = np.log(prior / prior.sum())
    return out


def norm_entropy(probs: np.ndarray) -> np.ndarray:
    k = probs.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(probs > 0, probs * np.log(probs), 0.0).sum(1)
    return h / np.log(k)


def js_divergence(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence in bits (0..1) between rows of two probability arrays."""
    m = 0.5 * (a + b)

    def kl(x, y):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(x > 0, x * (np.log2(x + EPS) - np.log2(y + EPS)), 0.0).sum(1)

    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


def soft_reliability(q: np.ndarray, p: np.ndarray, n_bins: int = 10):
    """Pool every (case, code) pair: predicted probability vs the pathologists' share.
    Returns per-bin (mean predicted, mean observed, count) and the soft ECE."""
    pred, obs = q.ravel(), p.ravel()
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(pred, edges[1:-1]), 0, n_bins - 1)
    rows, ece = [], 0.0
    for k in range(n_bins):
        m = b == k
        if m.any():
            rows.append((k, pred[m].mean(), obs[m].mean(), int(m.sum())))
            ece += m.mean() * abs(pred[m].mean() - obs[m].mean())
    return rows, float(ece)


def spearman_boot(x: np.ndarray, y: np.ndarray, idx: np.ndarray) -> tuple[float, float, float]:
    with warnings.catch_warnings():  # a collapsed model has constant entropy: rho undefined
        warnings.simplefilter("ignore", ConstantInputWarning)
        rho = spearmanr(x, y).statistic
        bs = np.array([spearmanr(x[i], y[i]).statistic for i in idx])
    bs = bs[np.isfinite(bs)]
    if not np.isfinite(rho) or bs.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(rho), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


def per_case_nll(logq: np.ndarray, p: np.ndarray) -> np.ndarray:
    return soft_nll(logq, p)
