import numpy as np
import pytest
from scipy.optimize import approx_fprime

from src.evaluation.deference import center, cluster_sums, deference_test, holm
from src.evaluation.reference_model import (
    features,
    fisher_information,
    fit_reference,
    reference_update,
)


def _random_updates(rng, n=400, K=5):
    r = rng.normal(size=(n, K))
    clusters = np.repeat(np.arange(n // 10), 10)
    return r, clusters


def test_exact_recovery_without_noise():
    rng = np.random.default_rng(0)
    r, cl = _random_updates(rng)
    for kappa in (0.5, 1.0, 2.0):
        res = deference_test(kappa * r, r, r, cl, n_boot=200, rng=rng)
        assert res.kappa == pytest.approx(kappa)
        assert res.rho_size == pytest.approx(kappa)
        assert res.cos == pytest.approx(1.0)


def test_invariant_to_normalising_constant():
    rng = np.random.default_rng(1)
    r, cl = _random_updates(rng)
    shift = rng.normal(size=(len(r), 1))  # per-item constant, as from log Z
    a = cluster_sums(2 * r, r, r, cl)
    b = cluster_sums(2 * r + shift, r + shift, r - shift, cl)
    np.testing.assert_allclose(a.num, b.num)
    np.testing.assert_allclose(a.den, b.den)


def test_reversed_direction_is_misdirected():
    rng = np.random.default_rng(2)
    r, cl = _random_updates(rng)
    res = deference_test(-r, r, r, cl, n_boot=500, rng=rng)
    assert res.cos == pytest.approx(-1.0)
    assert res.label == "misdirected"


def test_labels_on_clean_large_signal():
    rng = np.random.default_rng(3)
    r, cl = _random_updates(rng, n=2000)
    noise = 0.3 * center(rng.normal(size=r.shape))
    assert deference_test(2 * r + noise, r, r, cl, n_boot=500, rng=rng).label == "over"
    assert deference_test(0.4 * r + noise, r, r, cl, n_boot=500, rng=rng).label == "under"
    assert deference_test(r + 0.1 * noise, r, r, cl, n_boot=500, rng=rng).label == "calibrated"


def test_holm_matches_manual():
    p = {"a": 0.01, "b": 0.04, "c": 0.03}
    adj = holm(p)
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)  # monotone: max(0.06, 0.04)


def _sample_from_g(rng, beta, n=20000, K=5):
    log_q0 = np.log(rng.dirichlet(np.ones(K), size=n))
    y_src = rng.integers(K, size=n)
    rel = rng.choice([0.5, 0.6, 0.85, 0.9], size=n)
    phi = features(log_q0, y_src, rel)
    eta = log_q0 + phi @ beta
    p = np.exp(eta - eta.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    y = np.array([rng.choice(K, p=pi) for pi in p])
    return log_q0, phi, y


def test_reference_fit_recovers_update():
    # distance and distance x reliability are nearly collinear, so individual coefficients
    # are weakly identified; the reference *update* is what kappa uses and must be recovered.
    rng = np.random.default_rng(4)
    beta = np.array([0.3, 1.5, -0.4, 0.8, 0.6])
    log_q0, phi, y = _sample_from_g(rng, beta)
    est = fit_reference(log_q0, phi, y, firth=True)
    r_true = center(reference_update(log_q0, phi, beta))
    r_est = center(reference_update(log_q0, phi, est))
    cos = (r_true * r_est).sum() / np.sqrt((r_true**2).sum() * (r_est**2).sum())
    scale = (r_true * r_est).sum() / (r_true**2).sum()
    assert cos > 0.98
    assert scale == pytest.approx(1.0, abs=0.1)


def test_firth_gradient_matches_finite_differences():
    rng = np.random.default_rng(5)
    log_q0, phi, y = _sample_from_g(rng, np.array([0.2, 1.0, 0.0, 0.5, 0.3]), n=300)
    rows = np.arange(len(y))

    def penalised_ll(b):
        lg = log_q0 + phi @ b
        lg = lg - np.log(np.exp(lg).sum(1, keepdims=True))
        return lg[rows, y].sum() + 0.5 * np.linalg.slogdet(fisher_information(log_q0, phi, b))[1]

    # the fitted optimum must be a stationary point of the penalised likelihood
    est = fit_reference(log_q0, phi, y, firth=True)
    grad = approx_fprime(est, penalised_ll, 1e-6)
    assert np.max(np.abs(grad)) < 1e-2


def test_reference_update_is_zero_at_beta_zero():
    rng = np.random.default_rng(6)
    log_q0 = np.log(rng.dirichlet(np.ones(4), size=10))
    phi = features(log_q0, rng.integers(4, size=10), np.full(10, 0.7))
    np.testing.assert_allclose(reference_update(log_q0, phi, np.zeros(5)), 0, atol=1e-12)
