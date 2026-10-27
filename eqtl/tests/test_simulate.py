import numpy as np
import pytest

from eqtl.simulate import ExpressionSpec, load_sim, rep_seed, save_sim, simulate_rep


def test_reproducible_and_roundtrip(small_population, tmp_path):
    spec = ExpressionSpec(n=100, mu0=50.0, phi=0.1, k_causal=2, beta=0.3)
    a = simulate_rep(small_population, spec, seed=1)
    b = simulate_rep(small_population, spec, seed=1)
    assert np.array_equal(a.y, b.y) and np.array_equal(a.causal_idx, b.causal_idx)
    save_sim(a, tmp_path / "s.npz")
    c = load_sim(str(tmp_path / "s.npz"))
    assert np.array_equal(a.X, c.X) and np.array_equal(a.y, c.y) and c.r == a.r


def test_mean_count_tracks_mu0_and_overdispersion(small_population):
    spec = ExpressionSpec(n=300, mu0=100.0, phi=0.5, k_causal=1, beta=0.0, libsize_sd=0.0)
    s = simulate_rep(small_population, spec, seed=2)
    assert abs(s.y.mean() - 100) < 15
    # NB variance mu + phi mu^2 = 100 + 5000
    assert 3000 < s.y.var() < 8000


def test_poisson_limit(small_population):
    spec = ExpressionSpec(n=300, mu0=100.0, phi=0.0, beta=0.0, libsize_sd=0.0)
    s = simulate_rep(small_population, spec, seed=2)
    assert not np.isfinite(s.r)
    assert 60 < s.y.var() < 150


def test_causal_effect_direction(small_population):
    spec = ExpressionSpec(n=300, mu0=100.0, phi=0.05, k_causal=1, beta=1.0, libsize_sd=0.0)
    s = simulate_rep(small_population, spec, seed=4)
    g = s.X[:, s.causal_idx[0]]
    slope = np.polyfit(g, np.log(s.y + 1), 1)[0]
    assert np.sign(slope) == np.sign(s.beta[0]) and abs(slope) > 0.5


def test_causals_are_in_band_and_not_in_tight_ld(small_population):
    spec = ExpressionSpec(n=200, mu0=100.0, k_causal=3, beta=0.2, causal_max_abs_r=0.5)
    s = simulate_rep(small_population, spec, seed=7)
    assert s.causal_idx.shape == (3,)
    assert np.all(s.maf[s.causal_idx] >= 0.05)
    C = np.corrcoef(s.X[:, s.causal_idx].T.astype(float))
    assert np.all(np.abs(C[np.triu_indices(3, 1)]) <= 0.5)
    r2 = s.r2_to_causal()
    assert np.allclose(r2[s.causal_idx], 1.0)


def test_rep_seed_distinct():
    assert rep_seed("abc", 0) != rep_seed("abc", 1) != rep_seed("abd", 1)


def test_bad_k_raises(small_population):
    with pytest.raises(ValueError):
        simulate_rep(small_population, ExpressionSpec(n=50, k_causal=10_000), seed=0)


def test_null_gene(small_population):
    s = simulate_rep(small_population, ExpressionSpec(n=100, mu0=20.0, k_causal=0, beta=0.3), seed=3)
    assert s.causal_idx.shape == (0,) and s.beta.shape == (0,)
    assert not s.causal_mask.any() and np.all(s.r2_to_causal() == 0)
    assert abs(s.y.mean() - 20) < 8


def test_alpha_model_scales_effects_by_maf(small_population):
    spec = ExpressionSpec(n=200, mu0=100.0, k_causal=3, beta=0.2, causal_max_abs_r=0.5)
    s = simulate_rep(small_population, spec, seed=11)
    v = 2 * s.maf[s.causal_idx] * (1 - s.maf[s.causal_idx])
    # alpha = -1: standardized effect is the spec's beta, variance explained beta^2 for every causal
    assert np.allclose(np.abs(s.beta) * np.sqrt(v), 0.2)
    chi2 = spec.expected_chi2(s.maf[s.causal_idx])
    assert np.allclose(chi2, 200 * 0.04 / (1 / 100 + spec.phi))
    s0 = simulate_rep(small_population, ExpressionSpec(n=200, mu0=100.0, k_causal=3, beta=0.2, alpha=0.0), seed=11)
    assert np.allclose(np.abs(s0.beta), 0.2)
    assert s0.causal_idx.tolist() == s.causal_idx.tolist()  # alpha does not touch the draw


def test_batch_round_trip_matches_per_rep_files(small_population, tmp_path):
    from eqtl.simulate import load_batch, save_batch
    spec = ExpressionSpec(n=60, mu0=20.0, phi=0.3, libsize_sd=0.3, k_causal=1, beta=0.2, maf_min=0.05)
    sims = {r: simulate_rep(small_population, spec, rep_seed("abc", r)) for r in (3, 4, 5)}
    save_batch(sims, tmp_path / "b.npz")
    back = load_batch(str(tmp_path / "b.npz"))
    assert list(back) == [3, 4, 5]
    for r in sims:
        a, b = sims[r], back[r]
        assert np.array_equal(a.X, b.X) and np.array_equal(a.y, b.y) and a.r == b.r
        assert np.array_equal(a.causal_idx, b.causal_idx) and np.allclose(a.beta, b.beta)
