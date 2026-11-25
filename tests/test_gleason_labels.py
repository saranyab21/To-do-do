import numpy as np
import pandas as pd
import pytest

from src.data import gleason2019 as g
from src.evaluation.agreement import krippendorff_alpha, leave_one_out_agreement


def _row(**px):
    base = {f"px_{v}": 0 for v in range(7)}
    base.update({f"px_{k}": v for k, v in px.items()})
    return base


@pytest.mark.parametrize(
    "px, expected",
    [
        ({"1": 900, "3": 100}, (3, 3, "graded")),           # benign area never competes
        ({"3": 300, "4": 600, "1": 100}, (4, 3, "graded")),
        ({"4": 990, "5": 5, "1": 5}, (4, 4, "graded")),      # 0.5% G5 sliver below the floor
        ({"1": 1000}, (0, 0, "benign")),
        ({"6": 500, "1": 500}, (0, 0, "benign")),            # 6 never changes a pattern label
        ({"6": 500, "3": 500}, (3, 3, "graded")),
        ({"6": 1000}, (0, 0, "only_label6")),
        ({}, (None, None, "no_tissue")),
    ],
)
def test_core_label_rules(px, expected):
    p, s, status = g.core_label(_row(**{k: v for k, v in px.items()}))
    assert status == expected[2]
    if expected[0] is None:
        assert np.isnan(p) and np.isnan(s)
    else:
        assert (p, s) == expected[:2]


def test_whole_core_6_missing_option():
    p, _, status = g.core_label(_row(**{"6": 1000}), whole_core_6="missing")
    assert np.isnan(p) and status == "only_label6"


def test_equal_areas_prefer_higher_grade():
    assert g.core_label(_row(**{"3": 500, "4": 500}))[:2] == (4, 3)


def test_rater_distributions_counts_and_entropy():
    labels = pd.DataFrame({
        "core": ["slide001_core001"] * 3, "slide": ["slide001"] * 3,
        "rater": [1, 2, 3], "primary": [3, 3, 4], "secondary": [3, 3, 3],
    })
    d = g.rater_distributions(labels).set_index("dimension")
    assert d.loc["primary", "n_3"] == 2 and d.loc["primary", "n_4"] == 1
    assert d.loc["secondary", "entropy"] == 0
    assert d.loc["primary", "mode"] == 3


def test_krippendorff_alpha_extremes():
    perfect = np.array([[1, 1, 1], [2, 2, 2], [3, 3, np.nan]], float)
    assert krippendorff_alpha(perfect, "nominal") == pytest.approx(1.0)
    # textbook example (Krippendorff 2011, nominal, 4 raters x 12 units) gives alpha ~0.743
    data = np.array([
        [1, 1, np.nan, 1], [2, 2, 3, 2], [3, 3, 3, 3], [3, 3, 3, 3], [2, 2, 2, 2],
        [1, 2, 3, 4], [4, 4, 4, 4], [1, 1, 2, 1], [2, 2, 2, 2], [np.nan, 5, 5, 5],
        [np.nan, np.nan, 1, 1], [np.nan, np.nan, 3, np.nan]], float)
    assert krippendorff_alpha(data, "nominal") == pytest.approx(0.743, abs=0.001)


def test_leave_one_out_agreement():
    r = np.array([[1, 1, 1, 2], [2, 2, 2, 2], [3, 3, 1, 3]], float)
    loo = leave_one_out_agreement(r)
    assert loo[0] == 1.0 and loo[3] == pytest.approx(2 / 3)


def test_slide_folds():
    f = g.slide_folds(pd.Series(["slide002_core001", "slide001_core004", "slide001_core002"]))
    assert f.set_index("core").fold.to_dict() == {
        "slide001_core002": 0, "slide001_core004": 0, "slide002_core001": 1}
