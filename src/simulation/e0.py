"""E0: simulation check of the deference estimator before any real data is touched.

Generative world
----------------
* Each case i has a latent severity u_i ~ N(0, 1). For each dimension d its true category
  is z = clip(round(c + a * u_i + b * e_id)), so items within a case are correlated (this
  is why the bootstrap resamples cases, not items).
* Each case is rated by a subset of a fixed rater pool. Rater j reports y ~ E_j[z, :]:
    - "uniform" errors: correct with probability rho_j, otherwise uniform over K;
    - "ordinal" errors: P(y | z) proportional to exp(-lambda_j |y - z|), lambda_j set so
      that the average probability of the correct category is rho_j.
* The assessor sees a noisy view z_hat = z + N(0, sigma_x) and reports the calibrated
  statement-free read  q0(y) = sum_z p(z | z_hat) * mean_j E_j[z, y].
* A statement relays the rating y_src of a source rater A, with a stated reliability
  ("true": rho_A; "coarse": 0.9 for experts, 0.5 otherwise). The target is the rating y_J
  of another rater J of the same case.

What is planted
---------------
kappa is defined relative to the reference model class g (see reference_model.py). Its
population-limit reference r_pop is obtained by fitting g on a very large sample from the
same world. A simulated assessor then updates with

    Delta = kappa * centre(r_pop) + sigma_noise * centre(xi),   xi ~ N(0, I)

so the planted value is known exactly. The "bayes" scenario instead plants the exact
Bayes-optimal update r_bayes; there the estimate measures how well g captures the true
update (reference adequacy), not the estimator's coverage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from src.evaluation import reference_mixture as rm
from src.evaluation.deference import center, deference_test
from src.evaluation.reference_model import (
    cluster_robust_cov,
    features,
    fit_reference,
    reference_update,
    update_sampler,
)


@dataclass(frozen=True)
class Design:
    name: str
    K: int
    n_cases: int
    n_dims: int
    rater_reliabilities: tuple[float, ...]
    raters_per_case: tuple[int, int]
    pairs_per_item: int
    error_model: str = "uniform"  # "uniform" | "ordinal"
    stated_reliability: str = "true"  # "true" | "coarse"
    sigma_x: float = 0.6
    expert_threshold: float = 0.75


# --------------------------------------------------------------------------------------
# rater error matrices
# --------------------------------------------------------------------------------------


def _ordinal_matrix(K: int, lam: float) -> np.ndarray:
    d = np.abs(np.arange(K)[:, None] - np.arange(K)[None, :])
    m = np.exp(-lam * d)
    return m / m.sum(1, keepdims=True)


def error_matrix(K: int, rho: float, model: str) -> np.ndarray:
    """E[z, y] = P(rater reports y | true category z)."""
    if model == "uniform":
        return rho * np.eye(K) + (1 - rho) / K
    if model == "ordinal":
        lo, hi = 0.0, 50.0
        for _ in range(60):  # bisection on the mean diagonal
            mid = 0.5 * (lo + hi)
            if np.diag(_ordinal_matrix(K, mid)).mean() < rho:
                lo = mid
            else:
                hi = mid
        return _ordinal_matrix(K, 0.5 * (lo + hi))
    raise ValueError(model)


# --------------------------------------------------------------------------------------
# world
# --------------------------------------------------------------------------------------


class World:
    def __init__(self, design: Design, rng: np.random.Generator):
        self.d = design
        K = design.K
        self.E = np.stack(
            [error_matrix(K, r, design.error_model) for r in design.rater_reliabilities]
        )
        # marginal prior over z under the latent process (Monte Carlo, used by the assessor)
        z = self._latent(rng.standard_normal(200_000), rng.standard_normal(200_000))
        self.prior = np.bincount(z, minlength=K) / z.size

    def _latent(self, u, e):
        K = self.d.K
        c = (K - 1) * 0.4
        return np.clip(np.rint(c + 1.1 * u + 0.6 * e), 0, K - 1).astype(int)

    def stated(self, rho: float) -> float:
        if self.d.stated_reliability == "true":
            return rho
        return 0.9 if rho >= self.d.expert_threshold else 0.5

    def sample(self, rng: np.random.Generator, n_cases: int | None = None) -> dict:
        d, K = self.d, self.d.K
        n_cases = d.n_cases if n_cases is None else n_cases
        n_pool = len(d.rater_reliabilities)
        ks = np.arange(K)

        rows = {k: [] for k in ("case", "log_q0", "y_src", "rel", "y_tgt", "r_bayes")}
        for i in range(n_cases):
            u = rng.standard_normal()
            n_r = rng.integers(d.raters_per_case[0], d.raters_per_case[1] + 1)
            raters = rng.choice(n_pool, size=n_r, replace=False)
            for _ in range(d.n_dims):
                z = self._latent(u, rng.standard_normal())
                ratings = np.array([rng.choice(K, p=self.E[j, z]) for j in raters])

                z_hat = z + d.sigma_x * rng.standard_normal()
                log_post = np.log(self.prior + 1e-300) - 0.5 * ((z_hat - ks) / d.sigma_x) ** 2
                post = np.exp(log_post - logsumexp(log_post))  # p(z | z_hat)
                E_mean = self.E[raters].mean(0)  # mean_j E_j[z, y]
                q0 = post @ E_mean
                log_q0 = np.log(q0)

                pairs = [(a, b) for a in range(n_r) for b in range(n_r) if a != b]
                pick = rng.choice(len(pairs), size=min(d.pairs_per_item, len(pairs)), replace=False)
                for p_idx in pick:
                    a, b = pairs[p_idx]
                    A = raters[a]
                    others = np.delete(raters, a)
                    y_src = ratings[a]
                    # exact Bayes update about a held-out rater of this case
                    joint = post * self.E[A][:, y_src]  # p(z | z_hat, y_src) unnormalised
                    pred = joint @ self.E[others].mean(0)
                    log_pred = np.log(pred / pred.sum())
                    rows["case"].append(i)
                    rows["log_q0"].append(log_q0)
                    rows["y_src"].append(y_src)
                    rows["rel"].append(self.stated(d.rater_reliabilities[A]))
                    rows["y_tgt"].append(ratings[b])
                    rows["r_bayes"].append(log_pred - log_q0)
        out = {k: np.asarray(v) for k, v in rows.items()}
        out["phi"] = features(out["log_q0"], out["y_src"], out["rel"])
        return out


# --------------------------------------------------------------------------------------
# one simulation replicate
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    name: str
    design: Design
    kappa: float  # planted kappa (sign -1 = reversed direction)
    plant: str = "reference"  # "reference" | "bayes"
    noise_frac: float = 0.5


class LinearReference:
    """v1: log-linear statement terms on top of the log q0 offset (reference_model.py)."""

    name = "linear"

    def fit(self, s, mask, *, population=False):
        return fit_reference(s["log_q0"][mask], s["phi"][mask], s["y_tgt"][mask],
                             firth=not population)

    def update(self, s, p):
        return reference_update(s["log_q0"], s["phi"], p)

    def cov(self, s, mask, p):
        return cluster_robust_cov(s["log_q0"][mask], s["phi"][mask], s["y_tgt"][mask], p,
                                  s["case"][mask])

    def sampler(self, s, p, cov):
        return update_sampler(s["log_q0"], s["phi"], p, cov)


class MixtureReference:
    """v2: Bayes-shaped mixture toward the source's rating (reference_mixture.py)."""

    name = "mixture"

    def __init__(self, penalty: str = "firth"):
        self.penalty = penalty

    def fit(self, s, mask, *, population=False):
        return rm.fit_mixture(s["log_q0"][mask], s["y_src"][mask], s["rel"][mask],
                              s["y_tgt"][mask], penalty="none" if population else self.penalty)

    def update(self, s, p):
        return rm.mixture_update(s["log_q0"], s["y_src"], s["rel"], p)

    def cov(self, s, mask, p):
        return rm.cluster_robust_cov(s["log_q0"][mask], s["y_src"][mask], s["rel"][mask],
                                     s["y_tgt"][mask], p, s["case"][mask])

    def sampler(self, s, p, cov):
        return rm.update_sampler(s["log_q0"], s["y_src"], s["rel"], p, cov)


def make_reference(kind: str = "linear", penalty: str = "firth"):
    return MixtureReference(penalty) if kind == "mixture" else LinearReference()


def population_beta(world: World, rng: np.random.Generator, n_cases: int, ref=None) -> np.ndarray:
    ref = ref or LinearReference()
    big = world.sample(rng, n_cases=n_cases)
    return ref.fit(big, np.ones(len(big["case"]), bool), population=True)


def run_replicate(
    scen: Scenario,
    world: World,
    beta_pop: np.ndarray,
    noise_scale: float,
    rng: np.random.Generator,
    *,
    n_boot: int,
    tau_cos: float,
    alpha: float,
    ref=None,
    interval: str = "percentile",
) -> dict:
    ref = ref or LinearReference()
    s = world.sample(rng)
    cases = np.unique(s["case"])
    half_a = rng.choice(cases, size=cases.size // 2, replace=False)
    in_a = np.isin(s["case"], half_a)

    beta_a = ref.fit(s, in_a)
    beta_b = ref.fit(s, ~in_a)
    r_a = ref.update(s, beta_a)
    r_b = ref.update(s, beta_b)

    r_pop = ref.update(s, beta_pop)
    target = r_pop if scen.plant == "reference" else s["r_bayes"]
    xi = rng.standard_normal(target.shape)
    delta = scen.kappa * center(target) + noise_scale * center(xi)

    # bootstrap that holds the fitted references fixed (shown for comparison only)
    res_fixed = deference_test(
        delta, r_a, r_b, s["case"], tau_cos=tau_cos, alpha=alpha, n_boot=n_boot, rng=rng
    )
    # bootstrap that also propagates reference-model uncertainty (the method used)
    cov_a = ref.cov(s, in_a, beta_a)
    cov_b = ref.cov(s, ~in_a, beta_b)
    samplers = (ref.sampler(s, beta_a, cov_a), ref.sampler(s, beta_b, cov_b))
    res = deference_test(
        delta, r_a, r_b, s["case"], tau_cos=tau_cos, alpha=alpha, n_boot=n_boot, rng=rng,
        ref_samplers=samplers, interval=interval,
    )
    # adequacy of g: how well the population reference matches the true Bayes update
    rp, rb = center(r_pop), center(s["r_bayes"])
    adequacy_cos = float((rp * rb).sum() / np.sqrt((rp**2).sum() * (rb**2).sum()))

    return {
        "scenario": scen.name,
        "design": scen.design.name,
        "reference": ref.name,
        "interval": interval,
        "boot_mean_kappa": res.extra["boot_mean_kappa"],
        "kappa_lo_alt": res.extra["alt_ci"]["kappa"][0],
        "kappa_hi_alt": res.extra["alt_ci"]["kappa"][1],
        "covered_alt": bool(res.extra["alt_ci"]["kappa"][0] <= scen.kappa
                            <= res.extra["alt_ci"]["kappa"][1]),
        "label_alt": res.extra["alt_label"],
        "planted_kappa": scen.kappa,
        "kappa_hat": res.kappa,
        "kappa_lo": res.ci["kappa"][0],
        "kappa_hi": res.ci["kappa"][1],
        "rho_size": res.rho_size,
        "cos": res.cos,
        "cos_lo": res.ci["cos"][0],
        "D": res.D,
        "label": res.label,
        "covered": bool(res.ci["kappa"][0] <= scen.kappa <= res.ci["kappa"][1]),
        "kappa_lo_fixed": res_fixed.ci["kappa"][0],
        "kappa_hi_fixed": res_fixed.ci["kappa"][1],
        "covered_fixed": bool(
            res_fixed.ci["kappa"][0] <= scen.kappa <= res_fixed.ci["kappa"][1]
        ),
        "label_fixed": res_fixed.label,
        "boot_valid_frac": res.boot_valid_frac,
        "n_items": len(s["case"]),
        "adequacy_cos_r_pop_vs_bayes": adequacy_cos,
    }


# expected decision per planted kappa (used to score decision errors)
def decision_outcome(planted: float, label: str) -> str:
    """'correct', 'inconclusive' or 'error' for a planted kappa and a decision label."""
    if label == "inconclusive":
        return "inconclusive"
    if planted < 0:
        correct = {"misdirected"}
    elif planted > 1.25:
        correct = {"over"}
    elif planted < 0.8:
        correct = {"under"}
    else:
        correct = {"calibrated"}
    return "correct" if label in correct else "error"
