"""Case bootstrap stratified by slide (or speaker): each group keeps its size."""

from __future__ import annotations

import numpy as np


def stratified_indices(groups: np.ndarray, n_boot: int, rng) -> np.ndarray:
    """(n_boot, n) row indices; cases are resampled with replacement within each group.
    Intervals are therefore conditional on the observed groups."""
    groups = np.asarray(groups)
    members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    return np.stack([np.concatenate([rng.choice(m, len(m)) for m in members])
                     for _ in range(n_boot)])


def ci(values: np.ndarray, idx: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    """Mean of per-case values and its percentile interval over bootstrap rows idx."""
    values = np.asarray(values, float)
    bs = np.nanmean(values[idx], axis=1)  # NaN = case not scored (e.g. no hard negative)
    a = (1 - level) / 2
    return float(np.nanmean(values)), float(np.quantile(bs, a)), float(np.quantile(bs, 1 - a))
