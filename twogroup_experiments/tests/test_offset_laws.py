"""Offset axis: laws, their reductions, the three fits, and hash/reproducibility invariants."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.loader import (
    library_methods,
    load_library,
    resolve_simulation,
    run_method,
    simulation_coordinate,
)
from core import simulate
from simulations.offset.laws import (
    GaussianMixtureOffsetLaw,
    StudentTOffsetLaw,
    gaussian_offset,
    multimodal_offset,
    t_offset,
)


# --- Law reductions -------------------------------------------------------------------

def test_gaussian_mixture_reductions_match_closed_form():
    n, = (4,)
    w = np.array([[0.3, 0.7]] * n)
    m = np.array([[-1.0, 2.0]] * n)
    v = np.array([[0.5, 1.5]] * n)
    law = GaussianMixtureOffsetLaw(weights=w, means=m, vars=v)
    exp_mean = 0.3 * -1.0 + 0.7 * 2.0
    exp_var = (0.3 * (0.5 + 1.0) + 0.7 * (1.5 + 4.0)) - exp_mean**2
    assert np.allclose(law.mean, exp_mean)
    assert np.allclose(law.variance, exp_var)
    weights, means, vars_ = law.mixture()
    assert weights.shape == means.shape == vars_.shape == (n, 2)


def test_student_t_reductions_and_mixture_shape():
    n = 5
    law = StudentTOffsetLaw(means=np.zeros(n), scales=np.full(n, 2.0), df=5.0, n_components=8)
    assert np.allclose(law.mean, 0.0)
    assert np.allclose(law.variance, 4.0 * (5.0 / 3.0))  # scale^2 * df/(df-2)
    w, m, v = law.mixture()
    assert w.shape == m.shape == v.shape == (n, 8)
    assert np.allclose(w.sum(1), 1.0)
    # scale-mixture is variance-consistent with the t in expectation over components
    assert np.allclose((w * v).sum(1), law.variance, rtol=1e-6)


def test_student_t_requires_finite_variance():
    with pytest.raises(ValueError):
        StudentTOffsetLaw(means=np.zeros(3), scales=np.ones(3), df=2.0)


def test_samplers_return_law_and_draw_of_right_shape():
    rng = np.random.default_rng(0)
    psi, law = gaussian_offset(rng, 50, scale=1.0)
    assert psi.shape == (50,) and law.mean.shape == (50,)
    psi, law = multimodal_offset(rng, 50, weights=[0.5, 0.5], locs=[-2.0, 2.0], scales=[0.5, 0.5])
    assert psi.shape == (50,) and law.mixture()[0].shape == (50, 2)
    psi, law = t_offset(rng, 50, scale=1.0, df=5.0)
    assert psi.shape == (50,) and isinstance(law, StudentTOffsetLaw)


# --- Coordinate / hash invariants -----------------------------------------------------

def test_no_offset_coordinate_omits_offset_key():
    lib = load_library()
    base = simulation_coordinate(lib, "gaussian_p8", "ser_b2", "loc_2.0", "gaussian")
    explicit_none = simulation_coordinate(lib, "gaussian_p8", "ser_b2", "loc_2.0", "gaussian", "none")
    assert "offset" not in base
    assert base == explicit_none  # existing sim hashes are unchanged


def test_offset_coordinate_carries_entry_and_changes_hash():
    from experiments.loader import sim_hash
    lib = load_library()
    base = simulation_coordinate(lib, "gaussian_p8", "ser_b2", "binary", "noiseless")
    withoff = simulation_coordinate(lib, "gaussian_p8", "ser_b2", "binary", "noiseless", "gaussian_s1.0")
    assert "offset" in withoff and sim_hash(base) != sim_hash(withoff)


# --- DGP: reproducibility and law storage ---------------------------------------------

def test_no_offset_simulation_has_no_law_and_is_reproducible():
    lib = load_library()
    spec = resolve_simulation(lib, "gaussian_p8", "ser_b2", "loc_2.0", "gaussian")
    a, b = simulate(spec, 0), simulate(spec, 0)
    assert a.offset_law is None
    assert np.array_equal(a.z, b.z) and np.array_equal(a.thetahat, b.thetahat)


def test_offset_simulation_stores_law_not_draw_and_is_reproducible():
    lib = load_library()
    spec = resolve_simulation(lib, "gaussian_p8", "ser_b2", "binary", "noiseless", "gaussian_s1.0")
    a, b = simulate(spec, 0), simulate(spec, 0)
    assert a.offset_law is not None
    assert not hasattr(a, "offset")  # realized psi is discarded
    assert np.array_equal(a.z, b.z)
    assert np.allclose(a.offset_law.variance, 1.0)


# --- Fits: the three quadratures --------------------------------------------------------

def _run_all(lib, offset, replicate=0):
    spec = resolve_simulation(
        lib, "gaussian_rho0.00_n500_p100", "ser_b1_int0", "binary", "noiseless", offset
    )
    sim = simulate(spec, replicate)
    methods = library_methods(lib)
    return {
        m: run_method(methods[m], sim)["single_effects"][0]["ser_log_bf"]
        for m in ("logistic_point", "logistic_gaussq", "logistic_mixq")
    }


def test_zero_variance_offset_collapses_all_three():
    bf = _run_all(load_library(), "gaussian_s0.0")
    assert np.allclose(bf["logistic_point"], bf["logistic_gaussq"])
    assert np.allclose(bf["logistic_gaussq"], bf["logistic_mixq"])


def test_gaussian_offset_gaussq_equals_mixq():
    # A K=1 Gaussian law: MixtureGH reduces to GH, so the two integrated fits coincide,
    # and both should differ from the point estimate at appreciable variance.
    bf = _run_all(load_library(), "gaussian_s2.0")
    assert np.allclose(bf["logistic_gaussq"], bf["logistic_mixq"], rtol=1e-4)
    assert not np.isclose(bf["logistic_point"], bf["logistic_gaussq"], rtol=1e-3)
