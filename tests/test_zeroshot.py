import numpy as np
import pandas as pd
import pytest
from scipy.special import log_softmax

from src.evaluation.soft_label import (
    entropy,
    evaluate,
    fit_temperature,
    soft_nll,
    temper,
    top_match,
)
from src.models.zeroshot import build_messages, code_token_ids


class FakeTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def encode(self, s, add_special_tokens=False):
        return self.vocab[s]


def test_code_token_ids():
    tok = FakeTokenizer({"0": [15], "3": [18], "4": [19], "5": [20], "10": [16, 15]})
    assert code_token_ids(tok, [0, 3, 4, 5]) == {"0": 15, "3": 18, "4": 19, "5": 20}
    with pytest.raises(ValueError):
        code_token_ids(tok, [10])
    with pytest.raises(ValueError):
        code_token_ids(FakeTokenizer({"0": [1], "3": [1]}), [0, 3])


def test_build_messages():
    prompts = {"system": "S", "context": "C.", "questions": {"primary": "Q1?", "secondary": "Q2?"}}
    m = build_messages(prompts, "secondary", "img.jpg")
    assert m[0]["role"] == "system"
    assert m[1]["content"][0] == {"type": "image", "image": "img.jpg"}
    assert m[1]["content"][1]["text"] == "C. Q2?"


def test_soft_nll_and_kl():
    p = np.array([[0.5, 0.5, 0, 0]])
    assert soft_nll(np.log(p.clip(1e-300)), p)[0] == pytest.approx(entropy(p)[0])  # KL = 0
    assert soft_nll(np.log(np.full((1, 4), 0.25)), p)[0] == pytest.approx(np.log(4))


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    true = log_softmax(rng.normal(size=(2000, 4)), axis=1)
    p = np.exp(true)  # soft labels equal the true distribution
    T = fit_temperature(true * 3.0, p)  # model 3x overconfident
    assert T == pytest.approx(3.0, rel=0.02)
    assert np.allclose(temper(true * 3.0, T), true, atol=0.05)


def test_top_match_counts_ties():
    p = np.array([[0.5, 0.5, 0, 0], [0, 0, 1, 0]])
    logq = np.log(np.array([[0.1, 0.7, 0.1, 0.1], [0.7, 0.1, 0.1, 0.1]]))
    assert top_match(logq, p).tolist() == [1.0, 0.0]


def test_evaluate_perfect_model_beats_baselines():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(140):
        counts = rng.multinomial(4, rng.dirichlet([0.4, 1.2, 0.5, 0.2]))  # skewed marginal
        lp = np.log((counts + 0.01) / (counts + 0.01).sum()) + np.log(0.9)  # 90% mass on codes
        rows.append({"core": f"c{i}", "size": 448, "dimension": "primary", "fold": i % 7,
                     "n_raters": 4, **{f"n_{c}": n for c, n in zip((0, 3, 4, 5), counts)},
                     **{f"lp_{c}": v for c, v in zip((0, 3, 4, 5), lp)}})
    m = evaluate(pd.DataFrame(rows), [0, 3, 4, 5]).iloc[0]
    assert m.n == 140 and m.mass_mean == pytest.approx(0.9)
    assert m.top_match == 1.0 and m.kl < 0.05
    assert m.nll < m.nll_loso_prior < m.nll_uniform

    one_slide = evaluate(pd.DataFrame(rows).assign(fold=0), [0, 3, 4, 5]).iloc[0]
    assert np.isnan(one_slide.nll_loso_T) and np.isnan(one_slide.nll_loso_prior)
    assert one_slide.nll == pytest.approx(m.nll)
