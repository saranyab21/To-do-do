import numpy as np
import pytest
from scipy.special import softmax

from src.evaluation.probe import fit_soft_probe, nested_loso, predict_logq
from src.models.features import colour_features


def _data(n=210, d=40, k=4, signal=True, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    W = rng.normal(size=(d, k)) * (0.6 if signal else 0.0)
    P_true = softmax(X @ W + np.array([0.0, 1.5, 0.5, -1.0]), axis=1)
    counts = np.array([rng.multinomial(4, p) for p in P_true])
    return X, counts / 4.0, np.arange(n) % 7


def _nll(logq, P):
    return float(-(P * logq).sum(1).mean())


def test_probe_finds_signal():
    X, P, folds = _data(signal=True)
    logq, chosen = nested_loso(X, P, folds, [0.01, 0.1, 1.0], n_components=None)
    prior = np.log(P.mean(0))
    assert _nll(logq, P) < _nll(np.tile(prior, (len(P), 1)), P) - 0.1
    assert set(chosen) == set(range(7))


def test_probe_without_signal_stays_near_prior():
    X, P, folds = _data(signal=False, seed=1)
    logq, _ = nested_loso(X, P, folds, [0.001, 0.01, 0.1], n_components=16)
    prior = np.log(P.mean(0))
    assert _nll(logq, P) < _nll(np.tile(prior, (len(P), 1)), P) + 0.05


def test_missing_code_gets_floor_and_rows_normalise():
    X, P, _ = _data(seed=2)
    P[:, 0] = 0.0
    P[P.sum(1) == 0, 1] = 1.0
    P /= P.sum(1, keepdims=True)
    logq = predict_logq(fit_soft_probe(X, P, 1.0, None), X[:5], 4)
    assert np.allclose(np.exp(logq).sum(1), 1.0)
    assert np.all(np.exp(logq[:, 0]) < 1e-3)


def test_colour_features():
    rng = np.random.default_rng(3)
    img = np.full((60, 60, 3), 245, np.uint8)  # glass
    img[10:50, 10:50] = rng.integers(0, 255, size=(40, 40, 3))  # "tissue"
    f = colour_features(img)
    assert f.shape == (12 + 3 * 8 + 1,) and np.all(np.isfinite(f))
    assert f[-1] == pytest.approx(0.44, abs=0.05)  # stained fraction ~ 1600/3600
