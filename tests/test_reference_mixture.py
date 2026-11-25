import numpy as np
from scipy.special import logsumexp

from src.evaluation import reference_mixture as rm
from src.evaluation.deference import center


def _items(rng, n=3000, K=5):
    log_q0 = np.log(rng.dirichlet(np.ones(K) * 2, size=n))
    y_src = rng.integers(K, size=n)
    rel = rng.choice([0.55, 0.6, 0.85, 0.9], size=n)
    return log_q0, y_src, rel


def test_jacobian_matches_finite_differences():
    rng = np.random.default_rng(0)
    log_q0, y_src, rel = _items(rng, n=50)
    parts = rm._design(log_q0, y_src, rel)
    theta = np.array([0.5, 0.8, 0.7, 2.0, 0.5])
    _, G = rm._jacobian(theta, *parts)
    eps = 1e-6
    for p in range(rm.N_PARAMS):
        t = theta.copy()
        t[p] += eps
        g1, _, _ = rm._mixture(t, *parts)
        g0, _, _ = rm._mixture(theta, *parts)
        np.testing.assert_allclose((g1 - g0) / eps, G[:, :, p], atol=1e-5)


def test_null_source_has_zero_update():
    rng = np.random.default_rng(1)
    log_q0, y_src, _ = _items(rng, n=20)
    r = rm.mixture_update(log_q0, y_src, np.zeros(20), rm.theory_init(5))
    np.testing.assert_allclose(r, 0.0, atol=1e-12)


def test_recovers_exact_uniform_error_bayes_update():
    # world: latent z ~ q, source correct w.p. rho else uniform; target rater noiseless
    rng = np.random.default_rng(2)
    K, n = 5, 6000
    log_q0, y_src, rel = _items(rng, n=n, K=K)
    q = np.exp(log_q0)
    z = np.array([rng.choice(K, p=p) for p in q])
    correct = rng.random(n) < rel
    y_src = np.where(correct, z, rng.integers(K, size=n))
    lik = rel[:, None] * (np.arange(K)[None] == y_src[:, None]) + (1 - rel[:, None]) / K
    log_post = log_q0 + np.log(lik)
    log_post -= logsumexp(log_post, axis=1, keepdims=True)
    r_bayes = log_post - log_q0

    theta = rm.fit_mixture(log_q0, y_src, rel, z, penalty="firth")
    r_hat = rm.mixture_update(log_q0, y_src, rel, theta)
    a, b = center(r_bayes), center(r_hat)
    cos = (a * b).sum() / np.sqrt((a**2).sum() * (b**2).sum())
    assert cos > 0.99
    assert abs((a * b).sum() / (a**2).sum() - 1) < 0.1
