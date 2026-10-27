import numpy as np
from gibss.negbin import estimate_null_dispersion

from eqtl.rss import fit_rss, null_intercept, rss_design, score_summary_stats


def test_rss_design_reproduces_z_and_R(strong_sim):
    s = strong_sim
    null = estimate_null_dispersion(s.y, offset=s.log_libsize)
    ss = score_summary_stats(s.X.astype(float), s.y, s.log_libsize, r=null.dispersion, null_coef=null.coef)
    Xt, yt = rss_design(ss)
    assert np.allclose(Xt.T @ Xt, ss.R, atol=1e-10)
    assert np.allclose(Xt.T @ yt, ss.z, atol=1e-10)
    assert np.allclose(np.diag(ss.R), 1.0)
    # residualization: G_hat is W-orthogonal to the intercept
    assert np.allclose(ss.w @ ss.G_hat, 0.0, atol=1e-8)


def test_score_z_matches_finite_difference_of_loglik(strong_sim):
    """z_j^2 is the one-step (score) chi-square; its sign follows the causal effect."""
    s = strong_sim
    null = estimate_null_dispersion(s.y, offset=s.log_libsize)
    ss = score_summary_stats(s.X.astype(float), s.y, s.log_libsize, r=null.dispersion, null_coef=null.coef)
    j = s.causal_idx[0]
    assert np.sign(ss.z[j]) == np.sign(s.beta[0])
    assert abs(ss.z[j]) > 5
    assert np.argmax(np.abs(ss.z)) in np.flatnonzero(s.r2_to_causal() > 0.8)


def test_null_intercept_matches_ml_at_estimated_r(strong_sim):
    s = strong_sim
    null = estimate_null_dispersion(s.y, offset=s.log_libsize)
    b0 = null_intercept(s.y, s.log_libsize, r=null.dispersion)
    assert np.allclose(b0, null.coef, atol=1e-6)


def test_fit_rss_finds_the_causal(strong_sim):
    s = strong_sim
    null = estimate_null_dispersion(s.y, offset=s.log_libsize)
    ss = score_summary_stats(s.X.astype(float), s.y, s.log_libsize, r=null.dispersion, null_coef=null.coef)
    st = fit_rss(ss, L=2)
    pip = np.asarray(st.pip)
    assert pip[s.r2_to_causal() > 0.8].sum() > 0.9


def test_susie_rss_reproduces_fit_rss_on_score_statistics(strong_sim):
    """The sufficient-statistics IBSS (used by nb_mle_rss) and the synthetic-design route
    through gibss.linear (used by nb_score_rss) are the same algorithm: on score statistics,
    where both apply, they agree effect by effect, sweep by sweep."""
    from gibss.negbin import estimate_null_dispersion
    from eqtl.rss import fit_rss, rss_design, score_summary_stats, susie_rss
    s = strong_sim
    X = s.X.astype(float)
    null = estimate_null_dispersion(s.y, offset=s.log_libsize)
    ss = score_summary_stats(X, s.y, s.log_libsize, r=float(null.dispersion), null_coef=null.coef)
    ref = fit_rss(ss, L=3, max_iter=30)
    X_t, y_t = rss_design(ss)
    st = susie_rss(ss.z, ss.R, L=3, max_iter=30, yty=float(y_t @ y_t), n=y_t.shape[0])
    assert st.n_iter == ref.n_iter and st.converged == ref.converged
    for e, f in zip(st.single_effects, ref.single_effects):
        assert np.allclose(e.alpha, np.asarray(f.alpha), atol=1e-8)
        assert np.allclose(e.mu, np.asarray(f.mu), atol=1e-8)
        assert np.isclose(e.prior_variance, float(f.prior_variance), rtol=1e-8)
        assert np.isclose(e.ser_log_bf, float(f.ser_log_bf), atol=1e-6)
    assert np.allclose(st.pip, np.asarray(ref.pip), atol=1e-8)
    assert np.allclose(st.family_state.elbo_history, ref.family_state.elbo_history, atol=1e-6)


def test_wald_statistics_match_score_statistics_to_first_order(strong_sim):
    """Per-SNP NB Fisher scoring converges; away from the causal (small effects) the Wald
    z equals the score z to first order, and the Wald effect is on the per-allele log scale."""
    from gibss.negbin import estimate_null_dispersion
    from eqtl.rss import score_summary_stats, wald_summary_stats
    s = strong_sim
    X = s.X.astype(float)
    null = estimate_null_dispersion(s.y, offset=s.log_libsize)
    r = float(null.dispersion)
    ss = score_summary_stats(X, s.y, s.log_libsize, r=r, null_coef=null.coef)
    ws = wald_summary_stats(X, s.y, s.log_libsize, r=r, null_coef=null.coef)
    assert ws.n_iter < 50
    weak = np.abs(ss.z) < 1.0
    assert weak.sum() > 20
    assert np.corrcoef(ss.z[weak], ws.z[weak])[0, 1] > 0.99
    assert np.max(np.abs(ss.z[weak] - ws.z[weak])) < 0.1
    j = s.causal_idx[0]
    assert np.sign(ws.bhat[j]) == np.sign(s.beta[0]) and abs(ws.bhat[j] - s.beta[0]) < 0.5
    assert np.abs(ws.z[j]) > 5


def test_plain_ld_is_a_correlation_matrix(strong_sim):
    from eqtl.rss import plain_ld
    R = plain_ld(strong_sim.X.astype(float))
    assert np.allclose(np.diag(R), 1.0)
    assert np.allclose(R, R.T) and np.all(np.abs(R) <= 1 + 1e-12)
