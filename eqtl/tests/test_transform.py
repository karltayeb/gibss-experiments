import numpy as np
from scipy.stats import norm

from eqtl.transform import TRANSFORMS, inverse_normal, library_normalize, log_normalized


def test_inverse_normal_is_rank_based_and_standard():
    x = np.random.default_rng(0).gamma(2.0, size=1000)
    z = inverse_normal(x)
    assert np.all(np.diff(z[np.argsort(x)]) >= 0)
    assert abs(z.mean()) < 1e-6 and abs(z.std() - 1) < 0.05
    assert np.allclose(z[np.argmax(x)], norm.ppf((1000 - 0.5) / 1000))


def test_inverse_normal_ties_get_the_same_value():
    z = inverse_normal(np.array([0, 0, 1, 2, 2, 2]))
    assert z[0] == z[1] and z[3] == z[4] == z[5]


def test_library_normalize_and_log():
    y = np.array([10, 20, 40]); ll = np.log(np.array([1.0, 2.0, 4.0]))
    assert np.allclose(library_normalize(y, ll), 10.0)
    assert np.allclose(log_normalized(y, ll), np.log(11.0))
    assert set(TRANSFORMS) == {"int", "log"}
