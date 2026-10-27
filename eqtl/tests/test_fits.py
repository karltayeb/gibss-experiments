import numpy as np
import polars as pl
import pytest

from eqtl.fits import ARMS, fit_arm, rep_tables


@pytest.mark.parametrize("arm", ARMS)
def test_each_arm_runs_and_tables_are_well_formed(strong_sim, arm):
    s = strong_sim
    f = fit_arm(arm, s, L=2, max_iter=20)
    pip, cs, fit = rep_tables(f, s, rep=0, L=2)
    assert pip.height == s.p and set(pip["arm"]) == {arm}
    assert pip["causal"].sum() == 1
    assert fit.height == 1 and fit["L"][0] == "2"
    assert {"ser_log_bf", "cs_size", "purity", "covered", "cs_r2_max"} <= set(cs.columns)
    if arm.startswith("nb_"):
        assert fit["r_used"][0] is not None and fit["r_used"][0] > 0
    if arm in ("nb_est", "nb_score_rss", "nb_cavi"):
        assert fit["null_at_cap"][0] is False
    # the strong effect is found by every arm: top PIP mass sits in tight LD with the causal
    tight = s.r2_to_causal() > 0.8
    assert pip.filter(pl.Series(tight))["pip"].sum() > 0.9, arm


def test_nb_true_uses_the_simulated_r(strong_sim):
    f = fit_arm("nb_true", strong_sim, L=1, max_iter=5)
    assert f.r_used == strong_sim.r
    assert np.allclose(np.asarray(f.state.family_state.dispersion), strong_sim.r)


def test_nb_est_dispersion_near_truth(strong_sim):
    f = fit_arm("nb_est", strong_sim, L=1, max_iter=5)
    # intercept-only null absorbs the causal's variance: r_hat <= r roughly, within 2x here
    assert 0.4 < f.r_used / strong_sim.r < 1.5


def test_auto_L_runs_for_every_arm(strong_sim):
    for arm in ARMS:
        f = fit_arm(arm, strong_sim, L="auto", max_iter=20)
        pip, cs, fit = rep_tables(f, strong_sim, rep=0, L="auto")
        assert fit["L"][0] == "auto" and 1 <= fit["n_effects"][0] <= 5


def test_poisson_arms_on_poisson_counts(small_population):
    """phi = 0 draws Poisson counts; the Poisson arms fit them with a finite exact ELBO and
    no dispersion. The null-model dispersion plug-in only runs to the cap on a NULL gene: with
    a causal effect in the counts the intercept + offset null reads the eQTL variance as
    overdispersion (a strong effect gives r_hat ~ 3 on Poisson data)."""
    from eqtl.simulate import ExpressionSpec, simulate_rep
    spec = ExpressionSpec(n=120, mu0=50.0, phi=0.0, libsize_sd=0.3, k_causal=1, beta=0.8, maf_min=0.05)
    sim = simulate_rep(small_population, spec, seed=5)
    assert not np.isfinite(sim.r)
    for arm in ("poisson", "poisson_cavi"):
        f = fit_arm(arm, sim, L=2)
        _, _, fit_df = rep_tables(f, sim, rep=0, L=2)
        assert f.r_used is None
        assert fit_df["elbo"][0] is not None and np.isfinite(fit_df["elbo"][0])
    f = fit_arm("nb_est", sim, L=2)
    assert np.isfinite(f.r_used) and not f.null.at_cap
    null_gene = simulate_rep(small_population, ExpressionSpec(n=120, mu0=50.0, phi=0.0, libsize_sd=0.3, k_causal=0, beta=0.0, maf_min=0.05), seed=6)
    assert fit_arm("nb_est", null_gene, L=1).null.at_cap
    with pytest.raises(ValueError):
        fit_arm("nb_true", sim, L=1)
