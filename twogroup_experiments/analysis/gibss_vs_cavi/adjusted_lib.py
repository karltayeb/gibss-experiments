"""Adjusted gIBSS vs gIBSS vs exact offset integration, as the offset variance V varies.

`Smoothed(base, Adjusted())` (gibss-mono 96cc7d7, branch adjusted-gibss) replaces the
offset-smoothed loglik `y*eta - E_o A(eta + o)` by the family's own GLM at a per-row adjusted
response y + delta and offset shift nu, tuned to match the Taylor-smoothed value, score and
curvature at the current predictor. Plain gIBSS is delta = nu = 0 (it ignores V). The question
here: how does the adjusted SER behave against the exact offset integration, and against plain
gIBSS, as V grows?

Two ways V enters, and we run both:

  EXOGENOUS (known V). A per-row Gaussian offset o_i ~ N(0, V_i) with KNOWN variance is
    added to the predictor when simulating, and handed to the fit as the state's
    `random_intercept_var` with the per-sweep random-intercept update REMOVED from the
    schedule (so m_i stays 0 and v_i stays V_i). Every `Smoothed` arm then folds V_i into
    the offset variance `ov` the kernel sees; the plain base ignores it. The exact reference
    is `Smoothed(base, GH(21))`: for a Gaussian offset of known variance the GH fold IS the
    exact offset-integrated SER (this is what CAVI computes when the offset law is that
    Gaussian). Four arms, all free-form Q1 in the quad kernel, same intercept treatment:
        gibss     = base                       (plug-in; ignores V)
        taylor2   = Smoothed(base, Taylor())   (A + 1/2 V A'' at every eta; no anchor)
        adjusted  = Smoothed(base, Adjusted()) (Taylor2's value/score/curvature at the
                                                anchor, carried by the family's own cumulant)
        exact     = Smoothed(base, GH(21))
    V is swept directly, so we see each approximation's error as a function of V alone.

  ENDOGENOUS (V from the other effects). The ordinary multi-effect fit: V_i is the
    leave-one-out message variance sum_{l' != l} Var[x_i b_{l'}] plus the intercept factor's
    variance. We sweep sample size and design correlation (which set how unresolved the
    other components stay), measure the realized V per row at convergence, and compare
        gibss     = method="logistic" / "poisson"   (plug-in)
        taylor2   = the same, offset_integration="taylor2"
        adjusted  = method="adjusted"
        cavi      = method="cf_cavi_q1"             (exact free-form CAVI in Q1)
    through the front door `fit_glm_susie`, fixed sweep count so the arms are compared at
    the same number of coordinate updates.

Both parts run logistic and Poisson. Poisson is special: E_o e^{eta+o} = e^{eta + V/2}, so a
HOMOSKEDASTIC V is a pure intercept shift the free intercept absorbs (gibss == exact on alpha,
E[b] and log BF); only a HETEROSKEDASTIC V_i can separate the arms, which is why the exogenous
part has a subgroup design (V_i = V on half the rows, 0 elsewhere) next to the homoskedastic one.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import jax.numpy as jnp

from gibss import glm
from gibss.engine import fit_ibss
from gibss.methods import fit_glm_susie
from gibss.response import GH, Adjusted, Bernoulli, Poisson, Smoothed, Taylor

import gvc

# ------------------------------------------------------------------ palette / labels
# gIBSS orange and CAVI/exact blue follow gvc; adjusted takes the third Okabe-Ito hue used
# across the twogroup method palette (bluish green); Taylor2 the reddish purple.
COLOR = {"gibss": gvc.C_GIBSS, "taylor2": "#CC79A7", "adjusted": "#009E73",
         "exact": gvc.C_CAVI, "cavi": gvc.C_CAVI}
LABEL = {"gibss": "gIBSS", "taylor2": "Taylor2", "adjusted": "adjusted", "exact": "exact (GH)",
         "cavi": "CAVI (exact Q1)"}
FAMILY_LABEL = {"logistic": "logistic", "poisson": "Poisson"}
BASES = {"logistic": Bernoulli, "poisson": Poisson}

# adjusted_nr: the same Adjusted fit, but the b=0 null is NOT re-anchored. `glm._fit_effect`
# re-scores the null of every row-parameterized scheme with the row parameter tuned at the
# leave-one-out predictor (right for TaylorFixed / JJFixed, whose surrogate is tight at its
# own point). Adjusted's surrogate carries a per-row constant C(z) that pins its VALUE to the
# Taylor value at the anchor z, so scoring the null at a different anchor than the
# alternative leaks sum_i [C_i(z_alt) - C_i(z_null)] into the log BF (Poisson: C_i(z) =
# -y_i V/2 + e^z (e^{V/2} - 1 - V/2), with the eta-shape exact). This arm keeps the kernel's
# own baseline (same aux for null and alternative), the fix the notebook argues for.
# taylor2: `Smoothed(base, Taylor())`, the closed-form second-order smoothing Atilde = A +
# 1/2 V A'' applied at EVERY eta (no anchor, no row parameter). Its Newton weight is
# A'' + 1/2 V A'''', which for logistic is p(1-p)[1 + V/2 (1 - 6p(1-p))]: zero at p = 1/2
# when V = 4 and negative beyond, so the fit can lose convexity and return NaN. The
# shipped Adjusted takes its value/score/curvature targets from this same function at
# the anchor, so the two agree there and differ only in the shape away from it.
EXO_ARMS = ("gibss", "taylor2", "adjusted", "adjusted_nr", "exact")
ENDO_ARMS = ("gibss", "taylor2", "adjusted", "cavi")
ENDO_METHOD = {"gibss": {"logistic": "logistic", "poisson": "poisson"},
               "taylor2": {"logistic": "logistic", "poisson": "poisson"},
               "adjusted": {"logistic": "adjusted", "poisson": "adjusted"},
               "cavi": {"logistic": "cf_cavi_q1", "poisson": "cf_cavi_q1"}}
ENDO_EXTRA = {"taylor2": dict(offset_integration="taylor2")}  # front-door kwargs per arm

_QUAD_ORDER = 15
_GH_EXACT = 21
_GAUSS_RHO = 0.9

# Fixed sweep count, no early stop: the row-parameterized scheme converges on the full SKL
# while the comparators stop on alpha alone, so an early stop would compare different sweeps
# (the branch's own test does the same). before_sweep=() drops the random-intercept update,
# which is what keeps the injected V fixed (see the module docstring).
SCHED_EXO = replace(glm.default_schedule(), before_sweep=(), check_convergence=None)
SCHED_ENDO = replace(glm.default_schedule(), check_convergence=None)


# ------------------------------------------------------------------ simulation
def design(n: int, p: int, rho: float, seed: int) -> np.ndarray:
    """Standardized Gaussian AR(1) design, corr(x_j, x_k) = rho^|j-k|, drawn from `seed`."""
    rng = np.random.default_rng(seed)
    Z = np.empty((n, p))
    Z[:, 0] = rng.standard_normal(n)
    c = np.sqrt(1.0 - rho**2)
    for j in range(1, p):
        Z[:, j] = rho * Z[:, j - 1] + c * rng.standard_normal(n)
    return (Z - Z.mean(0)) / Z.std(0)


def offset_variance(structure: str, V: float, n: int) -> np.ndarray:
    """Per-row known offset variance V_i. 'homo': V on every row. 'hetero': V on the first
    half of the rows (the subgroup H), 0 elsewhere."""
    if structure == "homo":
        return np.full(n, float(V))
    if structure == "hetero":
        v = np.zeros(n)
        v[: n // 2] = float(V)
        return v
    raise ValueError(structure)


def draw_response(family: str, eta: np.ndarray, rng) -> np.ndarray:
    if family == "logistic":
        return (rng.random(eta.shape[0]) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    if family == "poisson":
        return rng.poisson(np.exp(eta)).astype(float)
    raise ValueError(family)


def simulate_exo(family: str, structure: str, V: float, b0: float, beta: float, causal,
                 n: int, p: int, seed: int):
    """One exogenous-V dataset: eta = b0 + X[:, causal] beta + o, o_i ~ N(0, V_i). The design
    is drawn per seed too (full generative sampling). `beta` may be a scalar (shared by all
    causal columns) or a sequence. Returns X, y, V_i."""
    rng = np.random.default_rng(seed)
    X = design(n, p, _GAUSS_RHO, seed=10_000 + seed)
    causal = list(causal)
    b = np.zeros(p)
    b[causal] = beta
    v = offset_variance(structure, V, n)
    eta = b0 + X @ b + rng.standard_normal(n) * np.sqrt(v)
    return X, draw_response(family, eta, rng), v


def simulate_endo(family: str, n: int, rho: float, b0: float, beta: float, causal, p: int,
                  seed: int):
    """One endogenous dataset: eta = b0 + X[:, causal] beta, no exogenous offset."""
    rng = np.random.default_rng(seed)
    X = design(n, p, rho, seed=20_000 + seed)
    b = np.zeros(p)
    b[list(causal)] = beta
    return X, draw_response(family, b0 + X @ b, rng)


# ------------------------------------------------------------------ fits
def exo_response(family: str, arm: str):
    base = BASES[family]()
    if arm == "gibss":
        return base
    if arm == "taylor2":
        return Smoothed(base, Taylor())
    if arm in ("adjusted", "adjusted_nr"):
        return Smoothed(base, Adjusted())
    if arm == "exact":
        return Smoothed(base, GH(order=_GH_EXACT))
    raise ValueError(arm)


def fit_exo(X, y, v, family: str, arm: str, L: int = 1, pv: float = 1.0, n_sweeps: int = 20):
    """Fit one exogenous arm with the known per-row offset variance `v` injected through the
    state's random-intercept factor (mean 0, variance v, never updated). Centered design,
    estimated shared (point) intercept, fixed prior variance, free-form Q1 quad kernel."""
    data = glm.prep_data(X, y, center=True)
    n = X.shape[0]
    v = np.asarray(v, dtype=float)
    state = glm.initialize_state(
        data, L=L, response=exo_response(family, arm), prior_variance=pv,
        family_state_kwargs=dict(
            kernel="quad", intercept="shared", estimate_intercept=True,
            estimate_prior_variance=False, quadrature_order=_QUAD_ORDER,
            random_intercept=True, random_intercept_mean=jnp.zeros(n),
            random_intercept_var=jnp.asarray(v),
            random_intercept_prior_variance=jnp.asarray(np.maximum(v, 1e-12)),
        ),
    )
    if arm != "adjusted_nr":
        return fit_ibss(data, state, SCHED_EXO, max_iter=n_sweeps)
    # keep the kernel's own null baseline for Adjusted (see EXO_ARMS); the predicate is
    # looked up by name inside glm._fit_effect at call time, so patching the module works.
    orig = glm._row_param_retuned
    glm._row_param_retuned = lambda fs: (
        orig(fs) and not isinstance(getattr(fs.response, "smoother", None), Adjusted)
    )
    try:
        return fit_ibss(data, state, SCHED_EXO, max_iter=n_sweeps)
    finally:
        glm._row_param_retuned = orig


def fit_endo(X, y, family: str, arm: str, L: int = 5, pv: float = 1.0, n_sweeps: int = 30):
    """Fit one endogenous arm through the front door at a fixed sweep count."""
    return fit_glm_susie(
        np.asarray(X), np.asarray(y), L=L, method=ENDO_METHOD[arm][family], family=family,
        prior_variance=pv, estimate_prior_variance=False, center=True,
        max_iter=n_sweeps, schedule=SCHED_ENDO, **ENDO_EXTRA.get(arm, {}),
    )


# ------------------------------------------------------------------ summaries
def pip_of(state) -> np.ndarray:
    return 1.0 - np.prod(1.0 - np.stack([np.asarray(e.alpha) for e in state.single_effects]), 0)


def summarize(state) -> dict:
    """Compact per-fit record: per-component alpha / mu / sd / log BF / feature log BF, the
    PIP, total posterior mean per feature, the intercept, and the realized offset variance
    distribution (message variance at convergence, the V the endogenous arms see).
    `failed` flags a fit that returned a non-finite alpha (Taylor2 losing convexity)."""
    eff = state.single_effects
    alpha = np.stack([np.asarray(e.alpha) for e in eff])
    mu = np.stack([np.asarray(e.mu) for e in eff])
    var = np.stack([np.asarray(e.var) for e in eff])
    flbf = np.stack([np.asarray(e.feature_log_bf) for e in eff])
    mv = np.asarray(state.total_message.var)
    return dict(
        alpha=alpha, mu=mu, sd=np.sqrt(np.maximum(var, 0.0)), feature_log_bf=flbf,
        ser_log_bf=np.array([float(e.ser_log_bf) for e in eff]),
        pip=pip_of(state), eb=np.sum(alpha * mu, 0),
        intercept=float(state.family_state.intercept_value),
        msg_var_mean=float(mv.mean()), msg_var_p95=float(np.quantile(mv, 0.95)),
        msg_var_max=float(mv.max()), failed=bool(not np.all(np.isfinite(alpha))),
    )


def gap(rec: dict, ref: dict) -> dict:
    """Discrepancies of one arm's record against the reference record (same L).
    alpha_gap   : max_j |alpha_j - alpha_j^ref| over components (L=1: the SER alpha)
    pip_gap     : max_j |PIP_j - PIP_j^ref|
    logbf_gap   : SER log BF - reference (component 0 for L=1; summed over components else)
    eb_gap      : E[b] at the reference's top feature minus the reference's
    sd_ratio    : posterior sd at the reference's top feature / reference's
    icpt_gap    : intercept - reference intercept
    """
    j = int(np.argmax(ref["pip"]))
    l = int(np.argmax(ref["alpha"][:, j]))
    return dict(
        alpha_gap=float(np.max(np.abs(rec["alpha"] - ref["alpha"]))),
        pip_gap=float(np.max(np.abs(rec["pip"] - ref["pip"]))),
        logbf_gap=float(rec["ser_log_bf"].sum() - ref["ser_log_bf"].sum()),
        eb_gap=float(rec["eb"][j] - ref["eb"][j]),
        sd_ratio=float(rec["sd"][l, j] / max(ref["sd"][l, j], 1e-12)),
        icpt_gap=float(rec["intercept"] - ref["intercept"]),
        top_feature=j,
    )
