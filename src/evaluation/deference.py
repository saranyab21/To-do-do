"""Direction-gated deference estimator (Pillar 2 of the proposal).

Given, for every item i (a case x dimension x statement),

* ``delta[i]``  the assessor's actual update  Delta(y) = log q(y|x,q_d,s) - log q(y|x,q_d)
* ``r_a[i]``, ``r_b[i]``  the reference update r*(y) = log g(y) - log g0(y) predicted by
  reference models fitted on two disjoint halves (a, b) of the cases,

we compute

    D        = sum_i <r_a_i, r_b_i>
    kappa    = sum_i <Delta_i, rbar_i> / D          (rbar = (r_a + r_b) / 2)
    rho_size = sqrt(sum_i ||Delta_i||^2 / D)
    cos      = kappa / rho_size

Using two independently fitted references in D removes the upward bias that estimation
noise would add to ||r||^2.

Design choice (to be preregistered): every update vector is **centred over categories**
before taking inner products. An update is only defined up to its normalising constant
(log q is shifted by -log Z), so the uncentred inner product mixes direction with that
constant. With centring, an assessor whose logits move by kappa * r has a centred Delta of
exactly kappa * centred(r).

Inference uses a case-clustered bootstrap. Each decision rule is a one-sided bootstrap
test; the family of tests is Holm-adjusted and the rules are applied in the order of the
proposal: Misdirected, then (if direction passes) Over-, Under-deference, Calibrated,
otherwise Inconclusive.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

EQUIVALENCE_MARGIN = np.log(1.25)

LABELS = ("misdirected", "over", "under", "calibrated", "inconclusive")


def center(v: np.ndarray) -> np.ndarray:
    """Centre update vectors over the last (category) axis."""
    return v - v.mean(axis=-1, keepdims=True)


@dataclass
class ClusterSums:
    """Per-cluster sums of the three quantities kappa, rho and cos are built from."""

    num: np.ndarray  # sum <Delta, rbar>
    den: np.ndarray  # sum <r_a, r_b>
    sq: np.ndarray  # sum ||Delta||^2


def cluster_sums(
    delta: np.ndarray, r_a: np.ndarray, r_b: np.ndarray, clusters: np.ndarray
) -> ClusterSums:
    delta, r_a, r_b = center(delta), center(r_a), center(r_b)
    r_bar = 0.5 * (r_a + r_b)
    num_i = np.einsum("ik,ik->i", delta, r_bar)
    den_i = np.einsum("ik,ik->i", r_a, r_b)
    sq_i = np.einsum("ik,ik->i", delta, delta)
    _, idx = np.unique(clusters, return_inverse=True)
    n_clusters = idx.max() + 1
    return ClusterSums(
        num=np.bincount(idx, num_i, n_clusters),
        den=np.bincount(idx, den_i, n_clusters),
        sq=np.bincount(idx, sq_i, n_clusters),
    )


def _ratios(num, den, sq):
    """kappa, rho_size, cos from (possibly bootstrapped) totals. NaN where D <= 0."""
    num, den, sq = (np.asarray(a, dtype=float) for a in (num, den, sq))
    ok = den > 0
    safe_den = np.where(ok, den, 1.0)
    kappa = np.where(ok, num / safe_den, np.nan)
    rho = np.where(ok, np.sqrt(sq / safe_den), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.where(ok & (sq > 0), num / np.sqrt(sq * safe_den), np.nan)
    return kappa, rho, cos


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values."""
    names = sorted(pvalues, key=pvalues.get)
    m = len(names)
    adjusted, running = {}, 0.0
    for rank, name in enumerate(names):
        running = max(running, min(1.0, (m - rank) * pvalues[name]))
        adjusted[name] = running
    return adjusted


def _boot_p(null_side: np.ndarray) -> float:
    """One-sided bootstrap p-value: share of resamples on the null side (NaN counts as null)."""
    null_side = np.asarray(null_side)
    return (np.count_nonzero(null_side) + 1) / (null_side.size + 1)


@dataclass
class DeferenceResult:
    kappa: float
    rho_size: float
    cos: float
    D: float
    ci: dict[str, tuple[float, float]]
    p_raw: dict[str, float]
    p_holm: dict[str, float]
    label: str
    n_clusters: int
    boot_valid_frac: float
    extra: dict = field(default_factory=dict)


def deference_test(
    delta: np.ndarray,
    r_a: np.ndarray,
    r_b: np.ndarray,
    clusters: np.ndarray,
    *,
    tau_cos: float = 0.5,
    alpha: float = 0.05,
    n_boot: int = 2000,
    ci_level: float = 0.95,
    rng: np.random.Generator | None = None,
    ref_samplers: tuple[Callable, Callable] | None = None,
    chunk: int = 100,
    interval: str = "percentile",
) -> DeferenceResult:
    """Point estimates, clustered-bootstrap intervals and the preregistered decision.

    If ``ref_samplers`` = (draw_a, draw_b) is given (see reference_model.update_sampler),
    every bootstrap draw also replaces r_a and r_b by perturbed reference updates, so the
    intervals include the estimation uncertainty of the two reference models.
    """
    rng = np.random.default_rng() if rng is None else rng
    s = cluster_sums(delta, r_a, r_b, clusters)
    n_c = s.num.size

    kappa, rho, cos = (float(v) for v in _ratios(s.num.sum(), s.den.sum(), s.sq.sum()))

    weights = rng.multinomial(n_c, np.full(n_c, 1.0 / n_c), size=n_boot).astype(float)
    if ref_samplers is None:
        num_b, den_b = weights @ s.num, weights @ s.den
    else:
        _, idx = np.unique(clusters, return_inverse=True)
        onehot = np.zeros((len(idx), n_c))
        onehot[np.arange(len(idx)), idx] = 1.0
        d_c = center(delta)
        num_b, den_b = np.empty(n_boot), np.empty(n_boot)
        for start in range(0, n_boot, chunk):
            b = min(chunk, n_boot - start)
            ra = center(ref_samplers[0](rng, b))
            rb = center(ref_samplers[1](rng, b))
            num_i = np.einsum("ik,bik->bi", d_c, 0.5 * (ra + rb))
            den_i = np.einsum("bik,bik->bi", ra, rb)
            w = weights[start : start + b]
            num_b[start : start + b] = ((num_i @ onehot) * w).sum(1)
            den_b[start : start + b] = ((den_i @ onehot) * w).sum(1)
    k_b, r_b_, c_b = _ratios(num_b, den_b, weights @ s.sq)
    valid = np.isfinite(k_b)
    boot_mean_kappa = float(np.nanmean(k_b)) if valid.any() else np.nan
    if interval not in ("percentile", "basic"):
        raise ValueError(interval)
    orig = (k_b, r_b_, c_b)
    # reflected ("basic") bootstrap: corrects a shift of the bootstrap distribution relative
    # to the estimate, e.g. from the ratio and from perturbing a non-linear reference model
    refl = (2 * kappa - k_b, 2 * rho - r_b_, 2 * cos - c_b)
    lo, hi = (1 - ci_level) / 2 * 100, (1 + ci_level) / 2 * 100

    def pct(a):
        a = a[np.isfinite(a)]
        return (float(np.percentile(a, lo)), float(np.percentile(a, hi))) if a.size else (np.nan, np.nan)

    def decide(kb, rb, cb):
        ci = {"kappa": pct(kb), "rho_size": pct(rb), "cos": pct(cb)}
        with np.errstate(invalid="ignore", divide="ignore"):
            log_k = np.where(kb > 0, np.log(np.where(kb > 0, kb, 1.0)), -np.inf)
        log_k = np.where(valid, log_k, np.nan)
        p_raw = {
            # H1: cos > tau  (direction passes)
            "direction": _boot_p(~(cb > tau_cos)),
            # H1: cos < tau  (misdirected)
            "misdirected": _boot_p(~(cb < tau_cos)),
            # H1: kappa > 1
            "over": _boot_p(~(kb > 1)),
            # H1: rho_size < 1
            "under": _boot_p(~(rb < 1)),
            # TOST, H1: |log kappa| < log 1.25
            "calibrated": max(
                _boot_p(~(log_k > -EQUIVALENCE_MARGIN)), _boot_p(~(log_k < EQUIVALENCE_MARGIN))
            ),
        }
        p_holm = holm(p_raw)
        rej = {k: p <= alpha for k, p in p_holm.items()}
        if not s.den.sum() > 0:
            label = "inconclusive"
        elif rej["misdirected"]:
            label = "misdirected"
        elif rej["direction"] and rej["over"]:
            label = "over"
        elif rej["direction"] and rej["under"]:
            label = "under"
        elif rej["direction"] and rej["calibrated"]:
            label = "calibrated"
        else:
            label = "inconclusive"
        return ci, p_raw, p_holm, label

    main_draws, alt_draws = (refl, orig) if interval == "basic" else (orig, refl)
    ci, p_raw, p_holm, label = decide(*main_draws)
    alt_ci, _, _, alt_label = decide(*alt_draws)
    alt_name = "percentile" if interval == "basic" else "basic"

    return DeferenceResult(
        kappa=kappa,
        rho_size=rho,
        cos=cos,
        D=float(s.den.sum()),
        ci=ci,
        p_raw=p_raw,
        p_holm=p_holm,
        label=label,
        n_clusters=n_c,
        boot_valid_frac=float(valid.mean()),
        extra={"boot_mean_kappa": boot_mean_kappa, "interval": interval,
               "alt_interval": alt_name, "alt_ci": alt_ci, "alt_label": alt_label},
    )
