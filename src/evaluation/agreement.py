"""Inter-rater agreement for ordinal ratings with missing values."""

from __future__ import annotations

import itertools

import numpy as np


def krippendorff_alpha(ratings: np.ndarray, metric: str = "interval") -> float:
    """Krippendorff's alpha. ratings: (n_units, n_raters) with NaN for missing.

    metric: "nominal" or "interval" (on the given numeric values; pass ordinal ranks
    0..K-1 for an ordinal-as-interval analysis).
    """
    values = np.unique(ratings[~np.isnan(ratings)])
    k = len(values)
    pos = {v: i for i, v in enumerate(values)}
    o = np.zeros((k, k))  # coincidence matrix
    for row in ratings:
        r = row[~np.isnan(row)]
        m = len(r)
        if m < 2:
            continue
        for a, b in itertools.permutations(range(m), 2):
            o[pos[r[a]], pos[r[b]]] += 1.0 / (m - 1)
    n_c = o.sum(1)
    n = n_c.sum()
    if metric == "nominal":
        d = 1.0 - np.eye(k)
    elif metric == "interval":
        d = (values[:, None] - values[None, :]) ** 2
    else:
        raise ValueError(metric)
    d_o = (o * d).sum() / n
    d_e = (np.outer(n_c, n_c) * d).sum() / (n * (n - 1))
    return float(1.0 - d_o / d_e) if d_e > 0 else np.nan


def pairwise_weighted_kappa(ratings: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Quadratic-weighted Cohen's kappa for every rater pair (NaN if < 10 shared units)."""
    from sklearn.metrics import cohen_kappa_score

    r = ratings.shape[1]
    out = np.full((r, r), np.nan)
    for i, j in itertools.combinations(range(r), 2):
        both = ~np.isnan(ratings[:, i]) & ~np.isnan(ratings[:, j])
        if both.sum() >= 10:
            k = cohen_kappa_score(ratings[both, i], ratings[both, j], labels=levels,
                                  weights="quadratic")
            out[i, j] = out[j, i] = k
    return out


def leave_one_out_agreement(ratings: np.ndarray) -> np.ndarray:
    """Per rater: share of units where the rating equals the modal rating of the others."""
    r = ratings.shape[1]
    out = np.full(r, np.nan)
    for i in range(r):
        hits, n = 0, 0
        for row in ratings:
            if np.isnan(row[i]):
                continue
            others = np.delete(row, i)
            others = others[~np.isnan(others)]
            if len(others) < 2:
                continue
            vals, cnt = np.unique(others, return_counts=True)
            if (cnt == cnt.max()).sum() > 1:  # tie among the others: skip
                continue
            hits += row[i] == vals[np.argmax(cnt)]
            n += 1
        out[i] = hits / n if n else np.nan
    return out
