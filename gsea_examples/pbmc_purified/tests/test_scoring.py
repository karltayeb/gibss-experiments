import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scoring import select_hits, significance_score


def test_select_hits_requires_exactly_one_mode():
    score = np.arange(10.0)
    with pytest.raises(ValueError):
        select_hits(score)
    with pytest.raises(ValueError):
        select_hits(score, threshold=1.0, topk=3)


def test_select_hits_threshold():
    score = np.array([0.0, 1.0, 2.0, 3.0])
    assert select_hits(score, threshold=2.0).tolist() == [False, False, True, True]


def test_select_hits_topk_picks_largest():
    score = np.array([5.0, 1.0, 4.0, 2.0, 3.0])
    mask = select_hits(score, topk=2)
    assert mask.tolist() == [True, False, True, False, False]


def test_select_hits_quantile():
    score = np.arange(100.0)
    mask = select_hits(score, quantile=0.95)
    # top 5% -> scores >= 95th percentile
    assert mask.sum() == pytest.approx(5, abs=1)


def test_significance_score_clamps_zero_lfsr():
    lfsr = np.array([0.0, 0.05, 0.5])
    pm = np.array([0.4, 0.2, 0.01])
    sd = np.array([0.001, 0.001, 0.001])
    score = significance_score(lfsr, pm, sd, lfsr_floor=1e-20)
    assert np.all(np.isfinite(score))
    # smaller LFSR -> larger score
    assert score[0] > score[1] > score[2]
