"""The four arms, all through the gibss front doors, plus their per-rep result tables.

    linear_int  rank inverse-normal of library-normalized counts -> gibss.linear.fit_linear_susie
    linear_log  log(library-normalized count + 1)                -> gibss.linear.fit_linear_susie
    nb_true     counts, offset = log library size, dispersion = the simulation's r
    nb_est      counts, offset = log library size, dispersion = negbin.estimate_null_dispersion
                (intercept + offset null model, the jaxQTL / edgeR plug-in)
    nb_score_rss NB score-test z per SNP at the estimated-dispersion null + GLM-weighted
                in-sample LD, then SuSiE-RSS (eqtl.rss): the jaxQTL calling + fine-mapping path
    nb_score_rss_true the same summary-statistic path at the simulation's r
    nb_mle_rss  one NB regression per SNP (intercept + SNP + offset, Fisher scoring at the
                estimated-dispersion null's r), Wald z = bhat / se, PLAIN in-sample genotype
                LD, then SuSiE-RSS on the sufficient statistics (eqtl.rss.susie_rss): the
                DESeq2-style effect-and-standard-error path a summary-statistics user takes
    nb_cavi     counts, estimated dispersion, CAVI in Q2 via the spectral kernel
                (method="cf_cavi_spectral"; null init, EB prior variance, 50-sweep cap). The
                reference gIBSS is measured against. The 022 protocol's
                freeze_prior_variance=0.01 (stop updating an effect once its prior variance
                drops below 0.01) is NOT used: on 003-poisson at mu0 = 1000 it froze
                early-sweep components in place, CAVI "converged" in 5 sweeps 187 nats below
                gIBSS with false single-SNP sets; without it cf_cavi reproduces the gIBSS fit
                to the cent.
    nb_cavi_true the same at the simulation's r
    nb_polya_gamma   counts, estimated dispersion, NB by its local Polya-Gamma augmentation
                (method="polya_gamma": the PG factor's parameter is the JJ tilt, so this is
                localjj on the NB base; the Gaussian base problem). gIBSS, closed form.
    nb_poisson_gamma counts, estimated dispersion, NB by its local Poisson-Gamma augmentation
                (method="poisson_gamma": Poisson base problem with a Gamma exposure tuned per
                row and feature, other effects through their exact mixture log-MGF). gIBSS,
                closed form. Known caveat: posterior variances deflate by about 1/(1 - s),
                s = mu/(mu + r); means and credible sets stay close.

    poisson     counts, offset = log library size, family="poisson" (no dispersion): gIBSS
    poisson_cavi the same likelihood, exact CAVI-Q2 via the characteristic-function builder
                (method="cf_cavi"; the spectral kernel is logistic-only, and the Poisson fold
                is closed form so cf is cheap here), same settings as nb_cavi. The phi = 0 companion: with Poisson counts, log-linear vs Poisson SuSiE isolates
                the link and the count likelihood from the overdispersion question.

The three NB routes (logistic cumulant = nb_est / nb_cavi, Polya-Gamma, Poisson-Gamma) are
named by the base regression problem each reduces to (binomial, Gaussian, Poisson).

Every arm centers the design, estimates the intercept, and estimates the prior variance by
EB (gibss defaults). The linear arms estimate the residual variance. `fit_arm` returns the
state and timing; `rep_tables` turns one fitted rep into the three long tables the
pipeline concatenates (per-feature PIPs, per-component credible sets, fit-level header).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import polars as pl
from gibss import glm
from gibss.elbo import compute_elbo
from gibss.linear import fit_linear_susie
from gibss.methods import fit_glm_susie
from gibss.negbin import NullDispersion, estimate_null_dispersion

from .cs import component_table
from .rss import fit_rss, null_intercept, plain_ld, score_summary_stats, susie_rss, wald_summary_stats
from .simulate import Sim
from .transform import TRANSFORMS

__all__ = ["ARMS", "ArmFit", "fit_arm", "is_rss", "rep_tables"]

ARMS = (
    "linear_int", "linear_log", "nb_true", "nb_est", "nb_score_rss", "nb_score_rss_true", "nb_mle_rss", "nb_cavi", "nb_cavi_true",
    "nb_polya_gamma", "nb_poisson_gamma", "poisson", "poisson_cavi",
)
def is_rss(arm: str) -> bool:
    """Summary-statistics arms: no individual-level likelihood, so no exact ELBO."""
    return arm.endswith("_rss") or arm.endswith("_rss_true")


AUGMENTATION = {"nb_polya_gamma": "polya_gamma", "nb_poisson_gamma": "poisson_gamma"}
CAVI_MAX_ITER = 50  # cap the expensive CAVI arm only (gIBSS keeps the engine default)


@dataclass(frozen=True)
class ArmFit:
    arm: str
    state: object
    seconds: float
    r_used: float | None  # NB size the fit ran at (None for the linear and Poisson arms)
    null: NullDispersion | None  # the null-model fit behind nb_est


def _design(sim: Sim) -> np.ndarray:
    return sim.X.astype(np.float64)


def parse_L(L) -> tuple:
    """"auto" -> ("auto", max_L); an int or digit string -> (int, None)."""
    if isinstance(L, str) and L == "auto":
        return "auto", MAX_L_AUTO
    return int(L), None


MAX_L_AUTO = 5  # cap for L="auto" (greedy forward selection; gibss adds effects while ser_log_bf >= tol_L)


def fit_arm(arm: str, sim: Sim, *, L, max_iter: int = 100) -> ArmFit:
    """`L` is an int or "auto" (gibss greedy forward selection up to MAX_L_AUTO)."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {ARMS}")
    X = _design(sim)
    L, max_L = parse_L(L)
    Lkw = {"L": L, "max_L": max_L} if max_L is not None else {"L": L}
    null = None
    r_used = None
    t0 = time.perf_counter()
    if arm.startswith("linear_"):
        z = TRANSFORMS[arm.removeprefix("linear_")](sim.y, sim.log_libsize)
        state = fit_linear_susie(X, z, center=True, max_iter=max_iter, **Lkw)
    elif arm.startswith("poisson"):
        kw = dict(method="cf_cavi", max_iter=min(max_iter, CAVI_MAX_ITER)) if arm == "poisson_cavi" else dict(max_iter=max_iter)
        state = fit_glm_susie(
            X, sim.y.astype(np.float64), family="poisson", offset=sim.log_libsize, center=True, **kw, **Lkw,
        )
    elif arm == "nb_mle_rss":
        null = estimate_null_dispersion(sim.y, offset=sim.log_libsize)
        r_used = float(null.dispersion)
        ws = wald_summary_stats(X, sim.y, sim.log_libsize, r=r_used, null_coef=null.coef)
        state = susie_rss(ws.z, plain_ld(X), max_iter=max_iter, **Lkw)
    elif arm.startswith("nb_score_rss"):
        if arm == "nb_score_rss_true":
            if not np.isfinite(sim.r):
                raise ValueError("nb_score_rss_true needs a finite simulated dispersion (phi > 0)")
            r_used = float(sim.r)
            coef = null_intercept(sim.y, sim.log_libsize, r=r_used)
        else:
            null = estimate_null_dispersion(sim.y, offset=sim.log_libsize)
            r_used = float(null.dispersion)
            coef = null.coef
        ss = score_summary_stats(X, sim.y, sim.log_libsize, r=r_used, null_coef=coef)
        state = fit_rss(ss, max_iter=max_iter, **Lkw)
    else:
        if arm.endswith("_true"):
            if not np.isfinite(sim.r):
                raise ValueError(f"{arm} needs a finite simulated dispersion (phi > 0)")
            r_used = float(sim.r)
        else:
            null = estimate_null_dispersion(sim.y, offset=sim.log_libsize)
            r_used = float(null.dispersion)
        if arm.startswith("nb_cavi"):
            kw = dict(method="cf_cavi_spectral", max_iter=min(max_iter, CAVI_MAX_ITER))
        elif arm in AUGMENTATION:
            kw = dict(method=AUGMENTATION[arm], max_iter=max_iter)
        else:
            kw = dict(max_iter=max_iter)
        state = fit_glm_susie(
            X, sim.y.astype(np.float64), dispersion=r_used, offset=sim.log_libsize,
            center=True, **kw, **Lkw,
        )
    return ArmFit(arm=arm, state=state, seconds=time.perf_counter() - t0, r_used=r_used, null=null)


def rep_tables(fit: ArmFit, sim: Sim, *, rep: int, L, coverage: float = 0.95):
    L = str(L)
    """(pip, cs, fit) polars frames for one fitted replicate.

    pip: one row per feature. `pip`, `posterior_mean` (sum_l alpha_l mu_l, on the arm's own
         scale), `causal`, `beta_true` (0 off the causals), `r2_causal` (max r^2 with any
         causal), `maf`, `position`.
    cs:  one row per component from `eqtl.cs.component_table` (ser_log_bf, cs_size,
         purity, beta, ...) plus `covered` (a causal is in the set), `n_causal_in_cs`,
         `cs_r2_max` (max r^2 between a set member and a causal) and the member list.
    fit: one row: wall seconds, sweeps, convergence, the dispersion used vs the truth, the
         null-model fit behind nb_est, the fitted prior variances and, for the count arms,
         `elbo`: the EXACT NB ELBO of the fitted state (gibss.elbo.compute_elbo, Q1 or Q2 by
         the state's shape), so the three NB routes (logistic cumulant, Polya-Gamma,
         Poisson-Gamma) and gIBSS vs CAVI are scored on one scale. Null for linear / RSS arms.
    """
    st = fit.state
    X = _design(sim)
    pip = np.asarray(st.pip)
    pm = np.asarray(st.posterior_mean)
    causal = sim.causal_mask
    beta_true = np.zeros(sim.p)
    beta_true[sim.causal_idx] = sim.beta
    r2 = sim.r2_to_causal()
    pip_df = pl.DataFrame({
        "rep": rep, "arm": fit.arm, "L": L,
        "feature": np.arange(sim.p, dtype=np.int64),
        "site_id": sim.site_id, "position": sim.position, "maf": sim.maf,
        "pip": pip, "posterior_mean": pm, "causal": causal, "beta_true": beta_true,
        "r2_causal": r2,
    })

    tab = component_table(st, X, coverage=coverage)
    members = [list(map(int, cs)) for cs in tab["cs"].to_list()]
    covered = [bool(causal[m].any()) for m in members]
    n_in = [int(causal[m].sum()) for m in members]
    cs_r2 = [float(r2[m].max()) if m else float("nan") for m in members]
    cs_df = tab.with_columns(
        pl.lit(rep).alias("rep"), pl.lit(fit.arm).alias("arm"), pl.lit(L).alias("L"),
        pl.Series("covered", covered), pl.Series("n_causal_in_cs", n_in),
        pl.Series("cs_r2_max", cs_r2),
    ).select(["rep", "arm", "L", *[c for c in tab.columns]] + ["covered", "n_causal_in_cs", "cs_r2_max"])

    pv = [float(e.prior_variance) for e in st.single_effects]
    elbo = None
    if fit.arm.startswith("poisson"):
        elbo = float(compute_elbo(glm.prep_data(X, sim.y.astype(np.float64), center=True), st))
    elif fit.r_used is not None and not is_rss(fit.arm):
        data = glm.prep_data(X, sim.y.astype(np.float64), center=True, dispersion=fit.r_used)
        elbo = float(compute_elbo(data, st))
    fs = st.family_state
    intercept = float(getattr(fs, "intercept_value", getattr(fs, "intercept", float("nan"))))
    residual_variance = float(fs.residual_variance) if hasattr(fs, "residual_variance") else None
    fit_df = pl.DataFrame({
        "rep": rep, "arm": fit.arm, "L": L, "n_effects": len(st.single_effects), "n": sim.n, "p": sim.p,
        "seconds": fit.seconds, "n_iter": int(st.n_iter), "converged": bool(st.converged),
        "r_true": float(sim.r), "r_used": fit.r_used,
        "null_at_cap": None if fit.null is None else bool(fit.null.at_cap),
        "null_n_iter": None if fit.null is None else int(fit.null.n_iter),
        "null_log_likelihood": None if fit.null is None else float(fit.null.log_likelihood),
        "intercept": intercept,
        "residual_variance": residual_variance,
        "prior_variance": [pv],
        "n_active": int(sum(float(e.ser_log_bf) >= 1.0 for e in st.single_effects)),
        "elbo": elbo,
    })
    return pip_df, cs_df, fit_df
