import numpy as np
import pandas as pd
import pytest

from src.data.gleason2019 import dimension_distributions, grade_group, rating_dimensions
from src.evaluation import gate as G
from src.evaluation.bootstrap import ci, stratified_indices


@pytest.mark.parametrize(("p", "s", "gg"), [(0, 0, 0), (3, 3, 1), (3, 4, 2), (4, 3, 3), (4, 4, 4),
                                            (3, 5, 4), (5, 3, 4), (4, 5, 5), (5, 4, 5), (5, 5, 5)])
def test_grade_group(p, s, gg):
    assert grade_group(p, s) == gg


def _labels():
    return pd.DataFrame({
        "core": ["c1"] * 3 + ["c2"] * 2, "slide": ["s1"] * 3 + ["s2"] * 2, "rater": [1, 2, 3, 1, 2],
        "primary": [3, 4, 0, 4, np.nan], "secondary": [4, 4, 0, 3, np.nan],
        "frac_3": [0.6, 0.0, 0.0, 0.3, 0.0], "frac_4": [0.39, 0.95, 0.0, 0.6, 0.0],
        "frac_5": [0.01, 0.005, 0.0, 0.1, 0.0]})


def test_rating_dimensions_and_distributions():
    long = rating_dimensions(_labels(), min_pattern_frac=0.01)
    w = long.pivot_table(index=["core", "rater"], columns="dimension", values="value")
    assert w.loc[("c1", 1)].to_dict() == {"cancer": 1, "grade_group": 2, "highest": 5,
                                          "primary": 3, "secondary": 4}
    assert w.loc[("c1", 2), "highest"] == 4          # 0.5% pattern 5 is below the 1% floor
    assert w.loc[("c1", 3)].sum() == 0               # benign
    assert ("c2", 2) not in w.index                  # unrated map is dropped
    d = dimension_distributions(long).set_index(["core", "dimension"])
    assert d.loc[("c1", "cancer"), ["n_0", "n_1", "n_raters"]].tolist() == [1, 2, 3]
    assert d.loc[("c2", "grade_group"), "n_3"] == 1


def test_js_divergence_bounds():
    a = np.array([[1.0, 0.0], [0.5, 0.5]])
    b = np.array([[0.0, 1.0], [0.5, 0.5]])
    assert G.js_divergence(a, b) == pytest.approx([1.0, 0.0])


def test_soft_reliability_perfect():
    rng = np.random.default_rng(0)
    q = rng.dirichlet(np.ones(4), size=500)
    _, ece = G.soft_reliability(q, q)
    assert ece == pytest.approx(0.0, abs=1e-12)


def test_norm_entropy():
    assert G.norm_entropy(np.array([[0.25] * 4, [1, 0, 0, 0]])) == pytest.approx([1.0, 0.0])


def test_stratified_bootstrap_keeps_group_sizes():
    groups = np.array(["a"] * 5 + ["b"] * 3)
    idx = stratified_indices(groups, 50, np.random.default_rng(0))
    assert idx.shape == (50, 8)
    assert np.all(idx[:, :5] < 5) and np.all(idx[:, 5:] >= 5)
    m, lo, hi = ci(np.arange(8.0), idx)
    assert lo <= m <= hi


def test_shuffle_partners_cross_slide():
    from scripts.run_gleason_gate import shuffle_partners
    cores = [f"slide00{s}_core00{i}" for s in (1, 2, 3) for i in range(4)]
    part = shuffle_partners(cores, seed=1)
    assert all(c.split("_")[0] != p.split("_")[0] for c, p in part.items())
