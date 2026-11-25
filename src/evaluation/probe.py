"""Soft-label linear probe with nested leave-one-slide-out (LOSO) evaluation.

Model: standardise -> PCA (fitted on the training slides only) -> multinomial logistic
regression trained on the rater distribution p_i (soft labels). Soft labels enter as
sample weights: case i appears once per code y with weight p_i(y), which makes the
weighted log-loss exactly the soft cross-entropy -sum_y p_i(y) log q(y | x_i).

The L2 strength C is chosen by an inner LOSO over the training slides only, so the
held-out slide never influences any choice. Output: out-of-fold log-probabilities for
every case, which src.evaluation.soft_label.evaluate scores like the zero-shot model.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LOG_FLOOR = np.log(1e-4)  # for a code never rated in the training slides


def fit_soft_probe(X: np.ndarray, P: np.ndarray, C: float, n_components: int | None):
    """Scaler and PCA are fitted on the cases (each once), the classifier on the
    soft-label expansion."""
    n, k = P.shape
    steps = [StandardScaler()]
    if n_components and n_components < min(X.shape):
        steps.append(PCA(n_components=n_components, random_state=0))
    reduce = make_pipeline(*steps).fit(X)
    Z = reduce.transform(X)
    Zr, y, w = np.repeat(Z, k, axis=0), np.tile(np.arange(k), n), P.ravel()
    keep = w > 0
    clf = LogisticRegression(C=C, max_iter=5000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(Zr[keep], y[keep], sample_weight=w[keep])
    return reduce, clf


def predict_logq(probe, X: np.ndarray, k: int) -> np.ndarray:
    reduce, clf = probe
    lp = np.full((len(X), k), LOG_FLOOR)
    lp[:, clf.classes_] = clf.predict_log_proba(reduce.transform(X))
    return lp - np.logaddexp.reduce(lp, axis=1, keepdims=True)


def _cv_nll(X, P, folds, C, n_components) -> float:
    losses = []
    for f in np.unique(folds):
        te, tr = folds == f, folds != f
        logq = predict_logq(fit_soft_probe(X[tr], P[tr], C, n_components), X[te], P.shape[1])
        losses.append(-(P[te] * logq).sum(1))
    return float(np.concatenate(losses).mean())


def nested_loso(X: np.ndarray, P: np.ndarray, folds: np.ndarray, C_grid, n_components):
    """Out-of-fold log q for every case, and the C chosen for each outer fold."""
    logq = np.empty(P.shape)
    chosen = {}
    for f in np.unique(folds):
        te, tr = folds == f, folds != f
        inner = [_cv_nll(X[tr], P[tr], folds[tr], C, n_components) for C in C_grid]
        C = C_grid[int(np.argmin(inner))]
        chosen[int(f)] = C
        logq[te] = predict_logq(fit_soft_probe(X[tr], P[tr], C, n_components), X[te], P.shape[1])
    return logq, chosen
