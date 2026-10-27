"""eQTL calling in the NB model, then fine-mapping on the summary statistics (jaxQTL).

Null model: y_i ~ NB(mu_i, r), log mu_i = b0 + offset_i (+ covariates), r the null-model
plug-in. jaxQTL's score test for SNP j uses the null-model GLM weights

    W = diag(w_i),   w_i = mu_i / (1 + mu_i / r)        (Fisher information in eta_i)
    s_i = (y_i - mu_i) / (1 + mu_i / r)                 (score in eta_i)
    G_hat = G - X (X' W X)^{-1} X' W G                   (W-residualized genotypes)
    U_j = g_hat_j' s,   D = diag(G_hat' W G_hat),   z_j = U_j / sqrt(D_j)
    R_hat = D^{-1/2} G_hat' W G_hat D^{-1/2}             (in-sample weighted LD)

SuSiE-RSS (Zou et al. 2022, lambda = 0): z ~ N(R_hat b, R_hat), b the standardized effects,
residual variance fixed at 1, prior variance by EB. Here it runs through
`gibss.linear.fit_linear_susie` on the synthetic design

    X_tilde = W^{1/2} G_hat D^{-1/2},   y_tilde = W^{-1/2} s,

which has X_tilde' X_tilde = R_hat and X_tilde' y_tilde = z exactly, so the Gaussian SuSiE
on (X_tilde, y_tilde) with sigma^2 = 1, no intercept and no centering is SuSiE-RSS on
(z, R_hat) to the arithmetic. `score_summary_stats` is the calling step; `fit_rss` the
fine-mapping step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gibss.linear import fit_linear_susie

__all__ = ["SummaryStats", "null_intercept", "score_summary_stats", "rss_design", "fit_rss",
           "WaldStats", "wald_summary_stats", "plain_ld", "RSSState", "susie_rss"]


@dataclass(frozen=True)
class SummaryStats:
    z: np.ndarray  # (p,) score-test z
    D: np.ndarray  # (p,) score variance = 1 / se^2
    G_hat: np.ndarray  # (n, p) W-residualized genotypes
    w: np.ndarray  # (n,) null GLM weights
    s: np.ndarray  # (n,) null score in eta
    r: float
    null_coef: np.ndarray

    @property
    def se(self) -> np.ndarray:
        return 1.0 / np.sqrt(self.D)

    @property
    def bhat(self) -> np.ndarray:
        """Score-test effect estimate U_j / D_j (one Newton step from the null)."""
        return self.z * self.se

    @property
    def R(self) -> np.ndarray:
        """The weighted in-sample LD matrix R_hat (p, p)."""
        Gw = self.G_hat * np.sqrt(self.w)[:, None]
        R = Gw.T @ Gw
        d = 1.0 / np.sqrt(self.D)
        return d[:, None] * R * d[None, :]


def null_intercept(y: np.ndarray, offset: np.ndarray, *, r: float, tol: float = 1e-10, max_iter: int = 100) -> np.ndarray:
    """Null-model coefficient (intercept only) by Newton at a FIXED NB size r. The
    estimated-dispersion arms take the intercept from `estimate_null_dispersion`; the
    true-dispersion arm refits it here so the score test is anchored at the right null."""
    y = np.asarray(y, float)
    offset = np.asarray(offset, float)
    b0 = np.log(np.mean(y) / np.mean(np.exp(offset)))
    for _ in range(max_iter):
        mu = np.exp(b0 + offset)
        denom = 1.0 + mu / r
        grad = np.sum((y - mu) / denom)
        info = np.sum(mu / denom)
        step = grad / info
        b0 += step
        if abs(step) < tol:
            break
    return np.array([b0])


def score_summary_stats(
    G: np.ndarray, y: np.ndarray, offset: np.ndarray, *, r: float, null_coef: np.ndarray,
    covariates: np.ndarray | None = None,
) -> SummaryStats:
    """Per-SNP NB score statistics at the null model (coefficients `null_coef` over
    [1 | covariates], as `negbin.estimate_null_dispersion` returns them)."""
    G = np.asarray(G, dtype=float)
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    X = np.ones((n, 1)) if covariates is None else np.column_stack([np.ones(n), np.asarray(covariates, float)])
    eta = X @ np.asarray(null_coef, float) + np.asarray(offset, float)
    mu = np.exp(eta)
    denom = 1.0 + mu / r
    w = mu / denom
    s = (y - mu) / denom
    XtW = X.T * w[None, :]
    G_hat = G - X @ np.linalg.solve(XtW @ X, XtW @ G)
    D = np.einsum("ij,i,ij->j", G_hat, w, G_hat)
    U = G_hat.T @ s
    return SummaryStats(z=U / np.sqrt(D), D=D, G_hat=G_hat, w=w, s=s, r=float(r), null_coef=np.asarray(null_coef, float))


def rss_design(ss: SummaryStats) -> tuple[np.ndarray, np.ndarray]:
    """(X_tilde, y_tilde) with X_tilde'X_tilde = R_hat and X_tilde'y_tilde = z."""
    sw = np.sqrt(ss.w)
    X_tilde = (ss.G_hat * sw[:, None]) / np.sqrt(ss.D)[None, :]
    y_tilde = ss.s / sw
    return X_tilde, y_tilde


def fit_rss(ss: SummaryStats, *, L, max_L=None, max_iter: int = 100):
    """SuSiE-RSS on (z, R_hat): Gaussian SuSiE on the sufficient-statistic design, residual
    variance fixed at 1, no intercept, no centering (G_hat is already W-centered)."""
    X_tilde, y_tilde = rss_design(ss)
    Lkw = {"L": L, "max_L": max_L} if max_L is not None else {"L": L}
    return fit_linear_susie(
        X_tilde, y_tilde, center=False, residual_variance=1.0,
        estimate_residual_variance=False, estimate_intercept=False, max_iter=max_iter, **Lkw,
    )


# ----------------------------------------------------------------------------------------
# Wald summary statistics (per-SNP NB regression) and SuSiE-RSS on sufficient statistics.
#
# The score path above has an exact synthetic design because z = D^{-1/2} G_hat' s lies in
# the row space of G_hat. Wald z from p separate NB fits does not, and with p > n the
# in-sample LD is singular, so no (X_tilde, y_tilde) reproduces (z, R) exactly. `susie_rss`
# therefore runs the Gaussian IBSS on the sufficient statistics (R, z) directly, the way
# susieR's `susie_rss` does (lambda = 0, sigma^2 = 1, no intercept), with gibss's update
# order and EB rule so that on score statistics it reproduces `fit_rss` to the cent
# (tests/test_rss.py).
# ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WaldStats:
    z: np.ndarray  # (p,) bhat / se
    bhat: np.ndarray  # (p,) per-SNP NB log fold change per allele
    se: np.ndarray  # (p,) expected-information standard error
    r: float
    n_iter: int  # Fisher-scoring iterations until every SNP converged (or the cap)


def wald_summary_stats(
    G: np.ndarray, y: np.ndarray, offset: np.ndarray, *, r: float, null_coef: np.ndarray,
    tol: float = 1e-8, max_iter: int = 50,
) -> WaldStats:
    """One NB regression per SNP (intercept + SNP, library offset) at the FIXED size r by
    Fisher scoring, vectorized over SNPs; the DESeq2 / edgeR-style per-SNP effect and
    standard error. Starts every SNP at the null intercept with a zero effect."""
    G = np.asarray(G, dtype=float)
    y = np.asarray(y, dtype=float)[:, None]
    off = np.asarray(offset, dtype=float)[:, None]
    p = G.shape[1]
    b0 = np.full(p, float(np.asarray(null_coef, float)[0]))
    b1 = np.zeros(p)
    n_iter = max_iter
    for it in range(1, max_iter + 1):
        mu = np.exp(b0[None, :] + G * b1[None, :] + off)
        denom = 1.0 + mu / r
        w = mu / denom
        s = (y - mu) / denom
        u0 = s.sum(0)
        u1 = (G * s).sum(0)
        i00 = w.sum(0)
        i01 = (G * w).sum(0)
        i11 = (G * G * w).sum(0)
        det = i00 * i11 - i01 * i01
        d0 = (i11 * u0 - i01 * u1) / det
        d1 = (i00 * u1 - i01 * u0) / det
        b0 += d0
        b1 += d1
        if max(np.abs(d0).max(), np.abs(d1).max()) < tol:
            n_iter = it
            break
    mu = np.exp(b0[None, :] + G * b1[None, :] + off)
    w = mu / (1.0 + mu / r)
    i00 = w.sum(0)
    i01 = (G * w).sum(0)
    i11 = (G * G * w).sum(0)
    se = np.sqrt(i00 / (i00 * i11 - i01 * i01))
    if not (np.all(np.isfinite(b1)) and np.all(np.isfinite(se))):
        raise FloatingPointError("per-SNP NB Fisher scoring diverged")
    return WaldStats(z=b1 / se, bhat=b1, se=se, r=float(r), n_iter=n_iter)


def plain_ld(G: np.ndarray) -> np.ndarray:
    """In-sample Pearson correlation of the genotype columns: the LD a summary-statistics
    user computes from the genotypes, with no GLM weights."""
    G = np.asarray(G, dtype=float)
    Gc = G - G.mean(0)
    Gc /= np.sqrt(np.sum(Gc * Gc, 0))
    return Gc.T @ Gc


@dataclass(frozen=True)
class RSSEffect:
    alpha: np.ndarray
    mu: np.ndarray
    var: np.ndarray
    prior_variance: float
    ser_log_bf: float
    kl: float


@dataclass
class RSSFamilyState:
    intercept: float = 0.0
    residual_variance: float = 1.0
    elbo_history: list = None


@dataclass
class RSSState:
    """The slice of gibss's state that `eqtl.cs.component_table` and `eqtl.fits.rep_tables`
    read: `single_effects` with alpha / mu / var / prior_variance / ser_log_bf, `pip`,
    `posterior_mean`, `n_iter`, `converged`, `family_state`."""
    single_effects: list
    family_state: RSSFamilyState
    n_iter: int = 0
    converged: bool = False

    @property
    def pip(self) -> np.ndarray:
        a = np.stack([e.alpha for e in self.single_effects]) if self.single_effects else np.zeros((0, 0))
        return 1.0 - np.prod(1.0 - a, axis=0)

    @property
    def posterior_mean(self) -> np.ndarray:
        return sum(e.alpha * e.mu for e in self.single_effects)


def _ser_ss(r: np.ndarray, d: np.ndarray, V: float) -> RSSEffect:
    """Gaussian SER on sufficient statistics: x_j'r = r_j, x_j'x_j = d_j, sigma^2 = 1."""
    p = r.shape[0]
    bhat = r / d
    s2 = 1.0 / d
    log_bf = 0.5 * np.log(s2 / (V + s2)) + 0.5 * bhat * bhat * (1.0 / s2 - 1.0 / (V + s2))
    var = 1.0 / (1.0 / s2 + 1.0 / V)
    mu = var * bhat / s2
    m = log_bf.max()
    alpha = np.exp(log_bf - m)
    alpha /= alpha.sum()
    ser_log_bf = float(m + np.log(np.sum(np.exp(log_bf - m))) - np.log(p))
    return RSSEffect(alpha=alpha, mu=mu, var=var, prior_variance=V, ser_log_bf=ser_log_bf, kl=_kl(alpha, mu, var, V))


def _kl(alpha, mu, var, V) -> float:
    p = alpha.size
    var = np.maximum(var, 1e-8)
    kl_cat = np.sum(alpha * (np.log(alpha + 1e-30) - np.log(1.0 / p)))
    kl_gauss = 0.5 * (np.log(V / var) + (var + mu * mu) / V - 1.0)
    return float(kl_cat + np.sum(alpha * kl_gauss))


def _eb_prior_variance(e: RSSEffect) -> RSSEffect:
    """gibss's coordinate update: sigma^2 <- E_q[b^2], floored at 1e-8."""
    V = max(float(np.sum(e.alpha * (e.mu * e.mu + e.var))), 1e-8)
    return RSSEffect(alpha=e.alpha, mu=e.mu, var=e.var, prior_variance=V, ser_log_bf=e.ser_log_bf, kl=_kl(e.alpha, e.mu, e.var, V))


def _elbo_ss(z, R, d, effects, *, yty: float, n: int) -> float:
    bbar = sum(e.alpha * e.mu for e in effects) if effects else np.zeros_like(z)
    rss = yty - 2.0 * z @ bbar + bbar @ R @ bbar
    msg_var = 0.0
    for e in effects:
        am = e.alpha * e.mu
        msg_var += float(np.sum(e.alpha * (e.mu * e.mu + e.var) * d) - am @ R @ am)
    expected_ll = -0.5 * n * np.log(2.0 * np.pi) - 0.5 * (rss + msg_var)
    return float(expected_ll - sum(e.kl for e in effects))


def susie_rss(
    z: np.ndarray, R: np.ndarray, *, L, max_L=None, prior_variance: float = 1.0,
    max_iter: int = 100, tol: float = 1e-4, tol_L: float = 1.0, yty: float | None = None, n: int | None = None,
) -> RSSState:
    """SuSiE-RSS on (z, R): z ~ N(R b, R), lambda = 0, sigma^2 = 1, no intercept, EB prior
    variance per effect, gibss's sweep order (residualize, SER, prior-variance update) and
    ELBO stopping rule. `yty` and `n` only shift the ELBO by a constant (needed to compare
    with a gibss fit on a real design; default p and p). `L="auto"` adds effects one at a
    time while the new effect's log BF reaches `tol_L`, up to `max_L`."""
    z = np.asarray(z, dtype=float)
    R = np.asarray(R, dtype=float)
    p = z.shape[0]
    d = np.diag(R).copy()
    yty = float(p) if yty is None else float(yty)
    n = p if n is None else int(n)
    greedy = L == "auto"
    if greedy:
        max_L = min(20, p) if max_L is None else int(max_L)

    def run(effects: list, iters: int) -> tuple[list, int, bool, list]:
        history = [-np.inf]  # gibss seeds the trace with -inf
        converged = False
        it = 0
        for it in range(1, iters + 1):
            bbar = sum(e.alpha * e.mu for e in effects)
            for l, e in enumerate(effects):
                bbar = bbar - e.alpha * e.mu
                r_l = z - R @ bbar
                new = _eb_prior_variance(_ser_ss(r_l, d, e.prior_variance))
                effects[l] = new
                bbar = bbar + new.alpha * new.mu
            history.append(_elbo_ss(z, R, d, effects, yty=yty, n=n))
            if len(history) >= 2 and abs(history[-1] - history[-2]) < tol:
                converged = True
                break
        return effects, it, converged, history

    empty = RSSEffect(alpha=np.full(p, 1.0 / p), mu=np.zeros(p), var=np.zeros(p), prior_variance=float(prior_variance), ser_log_bf=0.0, kl=np.inf)
    if not greedy:
        effects, it, conv, hist = run([empty] * int(L), max_iter)
    else:
        effects, it, conv, hist = [], 0, True, []
        while len(effects) < max_L:
            trial, it, conv, hist = run(list(effects) + [empty], max_iter)
            if trial[-1].ser_log_bf < tol_L:
                break
            effects = trial
        if not effects:
            effects, it, conv, hist = run([empty], max_iter)
    return RSSState(single_effects=effects, family_state=RSSFamilyState(elbo_history=hist), n_iter=it, converged=conv)
