"""Scoring a predicted code distribution against the rater distribution (soft labels).

Per case i with rater shares p_i over the K codes and model log-probabilities log q_i:

    soft NLL_i = -sum_y p_i(y) log q_i(y)      (cross-entropy; the Pillar-1 fitting loss)
    KL_i       = soft NLL_i - H(p_i)           (0 iff q_i reproduces the raters' spread)

Baselines: uniform over codes, and the leave-one-slide-out (LOSO) marginal of the rater
shares. Two LOSO recalibrations of the model, fitted on the other slides:
  * loso_T     one temperature: separates ranking quality from over/under-confidence
  * loso_TB    temperature plus one bias per code: also removes a constant answer
               preference (e.g. always "3"). It can always fall back to the prior (T large),
               so nll_loso_TB clearly below nll_loso_prior means the image carries signal.
Signal diagnostics that ignore calibration entirely:
  * auroc_cancer   P(cancer) = 1 - q(0) against the raters' majority (cancer vs benign)
  * spearman_grade expected code rank under q against the raters' mean code rank
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import log_softmax
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


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


def apply_tb(logq: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """theta = [log T, b_1, ..., b_{K-1}] (b_0 = 0)."""
    return log_softmax(logq / np.exp(theta[0]) + np.r_[0.0, theta[1:]], axis=1)


def fit_temperature_bias(logq: np.ndarray, p: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Temperature plus per-code bias minimising mean soft NLL (tiny ridge for stability)."""
    k = logq.shape[1]
    res = minimize(lambda th: soft_nll(apply_tb(logq, th), p).mean() + ridge * th @ th,
                   np.zeros(k), method="L-BFGS-B", bounds=[(-4.0, 6.0)] + [(-20.0, 20.0)] * (k - 1))
    return res.x


def top_match(logq: np.ndarray, p: np.ndarray) -> np.ndarray:
    """1 if the model's top code is a (possibly tied) most common rater answer."""
    pred = logq.argmax(1)
    return (p[np.arange(len(p)), pred] >= p.max(1) - 1e-12).astype(float)


def auroc_cancer(logq: np.ndarray, p: np.ndarray) -> float:
    """Code 0 = benign. Label: majority of raters say cancer. NaN if one class is absent."""
    y = (p[:, 0] < 0.5).astype(int)
    if y.min() == y.max():
        return np.nan
    return float(roc_auc_score(y, 1 - np.exp(logq[:, 0])))


def spearman_grade(logq: np.ndarray, p: np.ndarray) -> float:
    ranks = np.arange(logq.shape[1])
    rho = spearmanr(np.exp(logq) @ ranks, p @ ranks).statistic
    return float(rho)


def per_case(g: pd.DataFrame, codes) -> tuple[pd.DataFrame, list, np.ndarray, np.ndarray]:
    """Per-case losses for one (size, dimension) group: raw soft NLL, after LOSO
    temperature (nll_T) and temperature+bias (nll_TB), the LOSO prior's NLL, the rater
    entropy and top_match. Also returns the fitted temperatures, log q and p."""
    codes = [str(c) for c in codes]
    lp = g[[f"lp_{c}" for c in codes]].to_numpy(float)
    logq = log_softmax(lp, axis=1)  # renormalise over the answer codes
    p = g[[f"n_{c}" for c in codes]].to_numpy(float) / g.n_raters.to_numpy(float)[:, None]
    folds = g.fold.to_numpy()

    nll_T, nll_TB, nll_prior = (np.full(len(g), np.nan) for _ in range(3))
    temps = []
    # LOSO needs at least two slides (a --limit smoke test may cover only one)
    for f in np.unique(folds) if len(np.unique(folds)) > 1 else []:
        te, tr = folds == f, folds != f
        T = fit_temperature(logq[tr], p[tr])
        temps.append(T)
        nll_T[te] = soft_nll(temper(logq[te], T), p[te])
        nll_TB[te] = soft_nll(apply_tb(logq[te], fit_temperature_bias(logq[tr], p[tr])), p[te])
        prior = np.clip(p[tr].mean(0), 1e-6, None)
        nll_prior[te] = soft_nll(np.log(prior / prior.sum())[None, :], p[te])

    cases = pd.DataFrame({
        "core": g.core.to_numpy(), "fold": folds, "nll": soft_nll(logq, p), "nll_T": nll_T,
        "nll_TB": nll_TB, "nll_prior": nll_prior, "entropy": entropy(p),
        "top_match": top_match(logq, p), "mass": np.exp(lp).sum(1)})
    return cases, temps, logq, p


def evaluate(df: pd.DataFrame, codes) -> pd.DataFrame:
    """df: one row per (core, size, dimension) with lp_<c> (full-vocabulary log-probs),
    n_<c> (rater counts), n_raters, fold. Returns one metrics row per (size, dimension)."""
    codes = [str(c) for c in codes]
    rows = []
    for (size, dim), g in df[df.n_raters > 0].groupby(["size", "dimension"]):
        c, temps, logq, p = per_case(g, codes)
        pred = logq.argmax(1)
        rows.append({
            "size": size, "dimension": dim, "n": len(g),
            "mass_mean": float(c.mass.mean()), "mass_min": float(c.mass.min()),
            "top_match": float(c.top_match.mean()),
            "nll": float(c.nll.mean()), "kl": float((c.nll - c.entropy).mean()),
            "nll_loso_T": float(c.nll_T.mean()), "kl_loso_T": float((c.nll_T - c.entropy).mean()),
            "T_median": float(np.median(temps)) if temps else np.nan,
            "nll_loso_TB": float(c.nll_TB.mean()),
            "nll_uniform": float(np.log(len(codes))),
            "nll_loso_prior": float(c.nll_prior.mean()),
            "rater_entropy": float(c.entropy.mean()),
            "auroc_cancer": auroc_cancer(logq, p),
            "spearman_grade": spearman_grade(logq, p),
            **{f"pred_share_{c_}": float((pred == k).mean()) for k, c_ in enumerate(codes)},
        })
    return pd.DataFrame(rows)
