"""Scoring a predicted code distribution against the rater distribution (soft labels).

Per case i with rater shares p_i over the K codes and model log-probabilities log q_i:

    soft NLL_i = -sum_y p_i(y) log q_i(y)      (cross-entropy; the Pillar-1 fitting loss)
    KL_i       = soft NLL_i - H(p_i)           (0 iff q_i reproduces the raters' spread)

Baselines: uniform over codes, and the leave-one-slide-out (LOSO) marginal of the rater
shares. "loso_T" rescales the model's log-probabilities by one temperature fitted on the
other slides, which separates ranking quality from over/under-confidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax


def soft_nll(logq: np.ndarray, p: np.ndarray) -> np.ndarray:
    return -(p * logq).sum(1)


def entropy(p: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(p > 0, p * np.log(p), 0.0).sum(1)


def temper(logq: np.ndarray, T: float) -> np.ndarray:
    return log_softmax(logq / T, axis=1)


def fit_temperature(logq: np.ndarray, p: np.ndarray) -> float:
    """Temperature minimising mean soft NLL (searched on log T in [-4, 4])."""
    res = minimize_scalar(lambda s: soft_nll(temper(logq, np.exp(s)), p).mean(),
                          bounds=(-4.0, 4.0), method="bounded")
    return float(np.exp(res.x))


def top_match(logq: np.ndarray, p: np.ndarray) -> np.ndarray:
    """1 if the model's top code is a (possibly tied) most common rater answer."""
    pred = logq.argmax(1)
    return (p[np.arange(len(p)), pred] >= p.max(1) - 1e-12).astype(float)


def evaluate(df: pd.DataFrame, codes) -> pd.DataFrame:
    """df: one row per (core, size, dimension) with lp_<c> (full-vocabulary log-probs),
    n_<c> (rater counts), n_raters, fold. Returns one metrics row per (size, dimension)."""
    codes = [str(c) for c in codes]
    rows = []
    for (size, dim), g in df[df.n_raters > 0].groupby(["size", "dimension"]):
        lp = g[[f"lp_{c}" for c in codes]].to_numpy(float)
        logq = log_softmax(lp, axis=1)  # renormalise over the answer codes
        p = g[[f"n_{c}" for c in codes]].to_numpy(float) / g.n_raters.to_numpy(float)[:, None]
        h = entropy(p)
        folds = g.fold.to_numpy()

        nll_T, nll_prior, temps = np.empty(len(g)), np.empty(len(g)), []
        for f in np.unique(folds):
            te, tr = folds == f, folds != f
            T = fit_temperature(logq[tr], p[tr])
            temps.append(T)
            nll_T[te] = soft_nll(temper(logq[te], T), p[te])
            prior = np.clip(p[tr].mean(0), 1e-6, None)
            nll_prior[te] = soft_nll(np.log(prior / prior.sum())[None, :], p[te])

        nll = soft_nll(logq, p)
        pred = logq.argmax(1)
        rows.append({
            "size": size, "dimension": dim, "n": len(g),
            "mass_mean": float(np.exp(lp).sum(1).mean()),
            "mass_min": float(np.exp(lp).sum(1).min()),
            "top_match": float(top_match(logq, p).mean()),
            "nll": float(nll.mean()), "kl": float((nll - h).mean()),
            "nll_loso_T": float(nll_T.mean()), "kl_loso_T": float((nll_T - h).mean()),
            "T_median": float(np.median(temps)),
            "nll_uniform": float(np.log(len(codes))),
            "nll_loso_prior": float(nll_prior.mean()),
            "rater_entropy": float(h.mean()),
            **{f"pred_share_{c}": float((pred == k).mean()) for k, c in enumerate(codes)},
        })
    return pd.DataFrame(rows)
