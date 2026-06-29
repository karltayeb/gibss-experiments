# Approximate single-effect logistic regression: a comparison of variational, Laplace, and quadrature Bayes factors

**Technical report — draft (numbers pending build of `000_binary05`)**
**Date:** 2026-06-28

> All quantities in this report are computed by `scripts/report_binary05.py` from
> the `000_binary05` supercollection. No figures are used; results are reported as
> tables. Sections marked **[PENDING]** are filled once the fits complete.

## 1. Setup

We study the single-effect regression (SER), the atom of logistic SuSiE. Given a
design `X ∈ R^{n×p}`, an intercept `b0`, and one causal column `j*` with effect
`b`, the response is `y_i ~ Bernoulli(σ(b0 + b·x_{ij*}))`. The SER places a
spike-uniform prior over which single column carries a `N(0, σ0²)` effect and
returns, per feature, a posterior inclusion probability (PIP) and, collectively,
a log Bayes factor (logBF) for "some effect" vs the null, plus a credible set
(CS).

The marginal likelihood of each feature requires integrating the logistic
likelihood over the `N(0, σ0²)` effect — analytically intractable. The methods
here differ **only** in how that one-dimensional integral (and the nuisance
intercept) is approximated. We therefore have a clean setting in which to isolate
the inferential cost of each approximation.

### 1.1 Design and grid

`n = 1000`, `p = 500`. The design is a **binary ±1 Markov chain** with marginal
frequency `q = 0.5` (`binary_markov_X`, `freq=0.5`): each column is `2·1(U>0)−1`
of a stationary Gaussian AR(1) latent `U`, with the latent correlation chosen so
adjacent columns have the target tetrachoric correlation `ρ`. The causal column
sits in this correlated block, so `ρ` controls how hard localisation is.

The grid (single batch, 50 replicates per cell):

| axis | values |
|---|---|
| intercept `b0` | −2 (≈12% cases) |
| correlation `ρ` | 0, 0.5, 0.9 |
| target univariate logBF | 0 (null), 4, 8, 16, 32 |

Effect sizes `b` are frozen so the *exact* expected univariate logBF hits the
target (see `notes/2026-06-27-univariate-logbf-z-grid.md`); `ρ` does not change
the causal column's own marginal evidence, only localisation. `target=0` is the
null (no causal), used for false-positive control.

### 1.2 The twenty methods

Every method approximates the same SER. They factor as
**{taylor, jj} × {local, global} × {centered, not} + two exact references**,
each fit with a **fixed** prior (`σ0²=1`) and with **empirical-Bayes** (EB,
`σ0²` estimated) — twenty in all.

- **taylor (Laplace).** A second-order expansion of the per-feature
  log-likelihood. *Global* (`irls`): working weights `w_i = σ(η_i)(1−σ(η_i))`
  are computed once at the current linear predictor `η` and **shared** across all
  `p` univariate regressions. *Local* (`quadrature` at `m=1` node): a single
  Gauss–Hermite node — the Laplace approximation — re-evaluated **per feature**.
- **jj (Jaakkola–Jordan).** A variational quadratic *lower bound* on the logistic
  likelihood with a per-observation variational parameter `ξ`. *Local* (`localjj`):
  `ξ` is optimised per `(feature, observation)`, including the candidate column's
  contribution. *Global* (`globaljj`): a single `ξ` from the aggregate `η`,
  **shared** across features.
- **centered vs not.** "Centered" applies weighted column centering, making the
  intercept and the effect orthogonal under the working metric; "not" leaves the
  intercept shared/unprofiled. The two exact references instantiate this axis:
  `quadrature` folds a single shared intercept (non-centered), `profile` profiles
  the intercept per feature (centered).
- **exact references.** `quadrature` and `profile` at the default order
  (`m=15` Gauss–Hermite nodes) are numerically exact marginals — the
  non-centered and centered ground truth respectively. (`taylor·local` is exactly
  these two at `m=1`.)
- **EB vs fixed.** EB estimates `σ0²` by maximising the SER evidence; fixed holds
  `σ0²=1`.

The naming map (config → contrast cell):

| name | family | locality | centering | exact at m=15? |
|---|---|---|---|---|
| `taylor_global` | Laplace | global | no | — |
| `taylor_global_c` | Laplace | global | yes | — |
| `taylor_local` | Laplace | local | no | → `quadrature` |
| `taylor_local_c` | Laplace | local | yes | → `profile` |
| `jj_local` | JJ | local | no | — |
| `jj_local_c` | JJ | local | yes | — |
| `jj_global` | JJ | global | no | — |
| `jj_global_c` | JJ | global | yes | — |
| `quadrature` | exact | local | no | yes |
| `profile` | exact | local | yes | yes |

## 2. Quantities

For each `(method, prior)` we report, computed by the cited hook:

- **Approximation error** of the SER logBF against the matched exact marginal
  (`quadrature` for non-centered, `profile` for centered), at the *same prior* and
  on the *same simulated datasets* (paired): bias, RMSE, max |error|.
- **PIP calibration:** class-stratified Brier scores `B_causal` (target 1, lower
  = sharper on causals) and `B_null` (target 0, lower = nulls correctly near 0).
- **Detection:** PIP-level power at FDP ≤ 0.1; and the logBF detection AUC
  (signal vs null simulations).
- **Credible sets:** empirical coverage of the nominal 95% CS; mean CS size; and
  the calibrated `β` — the nominal level whose *empirical* coverage is 0.95 (β
  near 0.95 = honest; β≪0.95 = over-covering).

## 3. Approximation accuracy (logBF vs exact)

**[PENDING build]** — Table of bias / RMSE / max|err| per method, grouped by the
global/local and centering contrasts. Expected reading: local approximations
(per-feature reweighting) track the exact marginal more tightly than global
(shared-weight) ones; JJ, a one-sided bound, is biased in a fixed direction.

## 4. Calibration

**[PENDING build]** — PIP Brier and CS coverage, by contrast.

## 5. Detection and resolution

**[PENDING build]** — power/FDP, logBF AUC, CS size, calibrated β.

## 6. Discussion

**[PENDING build]** — practical recommendation.
