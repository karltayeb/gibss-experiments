# Approximate single-effect logistic regression: a comparison of variational, Laplace, and quadrature Bayes factors

**Technical report**
**Date:** 2026-06-28

> All quantities in this report are computed by `scripts/report_binary05.py` from
> the `000_binary05` supercollection. No figures are used; results are reported as
> tables.

## Abstract

We compare twenty approximations to the single-effect logistic Bayes factor — the
cross of {Laplace, Jaakkola–Jordan} × {local, global} × {centered, uncentered},
each fit with a fixed and an empirical-Bayes slab variance — against exact
Gauss–Hermite quadrature, on a correlated binary design (`n=1000`, `p=500`, 50
replicates per cell). Three findings emerge. First, the per-feature single-node
Laplace is *numerically the exact marginal* (RMSE < 0.01 nats); the entire
approximation error resides in the *global* (shared-weight) and *variational*
families and grows with the effect. Second, at a fixed prior this error is
inferentially inert — inclusion probabilities and credible sets are rank
statistics and tolerate even a six-nat Bayes-factor error — the lone exception
being global JJ, whose non-monotone error shifts coverage. Third, empirical Bayes
is the amplifier: estimating the slab variance from a method's own biased evidence
converts approximation error into lost power and inflated sets, sharply for the JJ
bound (global JJ detection AUC 0.91 → 0.77) and negligibly for Laplace/quadrature.
We recommend the local Laplace under a fixed prior, and caution against empirical
Bayes layered on a global variational bound.

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

## 3. Approximation accuracy of the Bayes factor

Table 1 reports the SER logBF error against the *matched exact marginal*
(`quadrature` for non-centered cells, `profile` for centered), paired on the same
600 signal datasets, fixed prior. `quadrature` and `profile` are the references
(error ≡ 0) and are omitted.

**Table 1. logBF error vs exact (fixed prior, signal cells, nats).**

| cell | locality | RMSE | bias | max\|err\| |
|---|---|---:|---:|---:|
| `taylor_local` | local | **0.009** | +0.004 | 0.19 |
| `taylor_local_c` | local | **0.000** | −0.000 | 0.00 |
| `jj_local_c` | local | 0.357 | −0.352 | 0.56 |
| `jj_local` | local | 0.785 | +0.087 | 5.79 |
| `jj_global_c` | global | 0.817 | −0.592 | 2.33 |
| `jj_global` | global | 1.054 | −0.301 | 5.61 |
| `taylor_global` | global | 1.861 | +0.927 | 11.5 |
| `taylor_global_c` | global | **3.569** | −2.062 | 17.2 |

Three facts. (i) The single-node **local Laplace is numerically the exact
marginal** (RMSE < 0.01 nats) — on this design full 15-node quadrature buys
nothing over one node. The entire approximation gap lives in the *global* and
*variational* families. (ii) **Global Laplace bias grows with signal**; (iii) the
**JJ bound is one-sided** (understates weak signal). Table 2 makes the
signal-dependence explicit.

**Table 2. Mean logBF error vs exact, by target univariate logBF (fixed prior).**

| cell | L4 | L8 | L16 | L32 |
|---|---:|---:|---:|---:|
| `taylor_local` | 0.00 | 0.00 | 0.01 | 0.01 |
| `taylor_local_c` | 0.00 | 0.00 | −0.00 | −0.00 |
| `jj_local` | −0.26 | −0.24 | −0.11 | +0.96 |
| `jj_local_c` | −0.31 | −0.32 | −0.35 | −0.42 |
| `jj_global` | −0.97 | −0.86 | −0.27 | +0.89 |
| `jj_global_c` | −1.03 | −0.93 | −0.41 | −0.00 |
| `taylor_global` | +0.03 | +0.11 | +0.58 | **+2.99** |
| `taylor_global_c` | −0.09 | −0.35 | −1.56 | **−6.25** |

The global Laplace shares one working-weight vector across all `p` candidates;
that vector, set at the aggregate linear predictor, misfits an individually
strong effect, so its error climbs to +3 nats (non-centered) and the centered
variant to −6 nats at logBF 32. The JJ bound understates by ~1 nat at weak
signal and relaxes toward the truth as the effect strengthens (the bound tightens
where the likelihood is sharp). Local JJ is more accurate than global
JJ (RMSE 0.79 vs 1.05 uncentered, 0.36 vs 0.82 centered), and centering tightens
both (0.36 vs 0.79 local, 0.82 vs 1.05 global).

The ordering is robust to correlation. RMSE at ρ = 0 / 0.5 / 0.9 is
0.004 / 0.014 / 0.007 for `taylor_local`, rising monotonically through
`jj_local` (0.94 / 0.94 / 0.30), `jj_global` (1.12 / 1.15 / 0.87),
`taylor_global` (2.16 / 2.16 / 1.04), to `taylor_global_c`
(3.79 / 3.86 / 3.00) — local Laplace is exact and the global/variational gap is
present at every ρ (errors shrink at ρ=0.9, where the shared block weakens each
column's individual evidence).

## 4. Calibration, and why absolute BF accuracy is the wrong yardstick

A negative result of practical importance: at the fixed prior the large logBF
errors of §3 **do not propagate** to the inference. Table 3 (fixed prior, pooled
over the signal grid; `B_null ≈ 0.0001` for every method — none is over-confident
on null features — so it is omitted).

**Table 3. Downstream inference at fixed prior.**

| cell | B_causal↓ | logBF AUC↑ | cov@95 | size@95 | cal β |
|---|---:|---:|---:|---:|---:|
| `quadrature` (exact) | 0.341 | 0.915 | 0.993 | 134 | 0.75 |
| `profile` (exact) | 0.338 | 0.915 | 0.992 | 132 | 0.76 |
| `taylor_local` | 0.341 | 0.915 | 0.993 | 134 | 0.75 |
| `taylor_local_c` | 0.338 | 0.915 | 0.992 | 132 | 0.76 |
| `taylor_global` | 0.340 | 0.915 | 0.990 | 133 | 0.79 |
| `taylor_global_c` | 0.341 | 0.915 | 0.995 | 137 | 0.76 |
| `jj_local` | 0.340 | 0.916 | 0.992 | 132 | 0.71 |
| `jj_local_c` | 0.339 | 0.916 | 0.993 | 134 | 0.73 |
| `jj_global` | **0.373** | 0.913 | **0.937** | 133 | **0.99** |
| `jj_global_c` | 0.372 | 0.913 | 0.938 | 133 | 0.99 |

The PIPs and CSs depend on the *ranking* of feature evidence, not its absolute
scale, and the Laplace/global errors are near-monotone in the true evidence — so
they preserve the ranking and leave calibration, detection AUC, and coverage
essentially at the exact values. `taylor_global_c`, whose logBF is wrong by 6
nats at strong signal (Table 2), nonetheless has AUC 0.915 and coverage 0.995,
indistinguishable from exact.

This is verifiable directly. Comparing each method to its matched exact, paired
per dataset (fixed prior): the probability that the causal feature receives the
*same rank* in the posterior, and the Spearman correlation of the causal PIP, are

| cell | P(same causal rank) | Spearman(causal PIP) |
|---|---:|---:|
| `taylor_local` | 0.983 | 1.000 |
| `taylor_global` | 0.950 | 1.000 |
| `taylor_global_c` | 0.935 | 0.999 |
| `jj_local` | 0.965 | 1.000 |
| `jj_global` | **0.868** | **0.973** |

Even the 6-nat `taylor_global_c` error leaves the ranking essentially intact
(Spearman 0.999); the SER reads off the ranking, so its inference is unmoved.
The **global-JJ pair are the only methods that perturb the ranking** (`jj_global`
0.868/0.973, `jj_global_c` 0.855/0.974) — precisely the methods whose coverage and
PIP calibration depart from exact. Rank preservation, not Bayes-factor fidelity,
is the operative property.

The one method that *does* leak is **global JJ**: its error is non-monotone
enough to shift PIP sharpness (`B_causal` 0.37 vs 0.34) and, notably, coverage —
it is the only method near nominal (0.94) while all others **over-cover** (≈0.99;
the level delivering honest 95% coverage is β≈0.71–0.79). On a near-collinear
p=500 block the spike-and-slab CS is conservative; global JJ's looseness happens
to offset that, but §5 shows the offset is not free.

## 5. Empirical Bayes is the amplifier; detection and resolution

The fixed prior hides the approximations; empirical Bayes exposes them.
Estimating σ0² from a method's *own* approximate evidence feeds that method's
bias back into the prior, and the harm scales with the bias.

**Table 4. EB amplification: logBF detection AUC and CS size, fixed → EB.**

| cell | AUC fixed | AUC eb | Δ | size fixed | size eb |
|---|---:|---:|---:|---:|---:|
| `quadrature` (exact) | 0.915 | 0.912 | −0.003 | 134 | 156 |
| `taylor_global` | 0.915 | 0.907 | −0.008 | 133 | 156 |
| `taylor_local` | 0.915 | 0.901 | −0.014 | 134 | 162 |
| `jj_local` | 0.916 | 0.835 | −0.081 | 132 | 160 |
| `jj_global_c` | 0.913 | 0.774 | −0.139 | 133 | 192 |
| `jj_global` | 0.913 | 0.773 | **−0.140** | 133 | **191** |

EB is benign for the exact and Laplace families (AUC loss ≤ 0.014) but corrosive
for the variational bound: global JJ's detection AUC falls from 0.913 to 0.773
and its mean CS grows 44% (133→191). The same ordering holds for PIP sharpness
(`B_causal` under EB: quadrature 0.371, `jj_local` 0.390, `jj_global` 0.470).

The mechanism is *not* a mis-set σ0² — estimated σ0² agrees across methods at
strong signal (≈0.6 at logBF 32). It is a loss of separation at the detection
margin. Empirical Bayes removes the fixed prior's Occam penalty: averaged over the
grid the null logBF rises from negative to ≈0, and so does the weak-signal logBF
(Table 5). Detection then hinges on the residual null-vs-weak-signal gap, which
the JJ bound — understating weak signal, most for its global variant (Table 2) —
nearly closes.

**Table 5. Mean SER logBF on null and signal datasets, fixed → EB.**

| cell | null (fix) | null (eb) | L4 (fix) | L4 (eb) | L32 (eb) |
|---|---:|---:|---:|---:|---:|
| `quadrature` | −1.34 | 0.04 | −0.46 | **0.57** | 26.2 |
| `jj_local` | −1.62 | 0.02 | −0.73 | 0.46 | 27.1 |
| `jj_global` | −2.20 | −0.02 | −1.43 | **0.22** | 27.0 |
| `taylor_global` | −1.34 | 0.06 | −0.43 | 0.60 | 29.1 |

Under EB, global JJ's weak (L4) signal sits at 0.22 against a null of −0.02 — a gap
of 0.24 — versus 0.57 against 0.04 (gap 0.53) for quadrature. The strong-signal
logBF is unaffected; the damage is entirely at the margin, where the loose bound
and the lost Occam penalty compound. The exact/Laplace evidence keeps a clear
weak-signal margin and is barely touched.

Resolution is dominated by the design, not the method: exact CS size@95 is
140 / 159 / 104 features at ρ = 0 / 0.5 / 0.9 — large throughout, as expected on a
correlated p=500 block — and EB inflates every method's size by ~20–45%.

## 6. Conclusions for practice

1. **Use the local approximation.** The per-feature single-node Laplace
   (`taylor_local`) reproduces the exact marginal to < 0.01 nats at every signal
   level and a fraction of the cost; quadrature with more nodes is unnecessary on
   this design. The *global* shared-weight Laplace is biased, and the bias grows
   with signal (and is worst when centered).
2. **Judge methods by calibration/coverage/power, not logBF fidelity.** At a
   fixed prior even a 6-nat logBF error is inferentially invisible, because the
   SER is rank-based. The exception to watch is a *non-monotone* error (global
   JJ), which does move coverage.
3. **Empirical Bayes is where approximation quality matters.** EB is safe with
   exact or Laplace evidence but degrades sharply with the JJ bound — worst for
   global JJ (AUC 0.91→0.77, CS +44%). If σ0² must be estimated, pair EB with a
   local Laplace/quadrature evidence, never a global variational one.
4. **Centering** is immaterial for the (already exact) local methods, slightly
   helps the local variational fit, and is *harmful* for the global Laplace.

Net recommendation: **`taylor_local` (≡ local quadrature) with a fixed or
well-chosen prior** is the robust, cheap default; **global JJ under EB** is the
configuration to avoid.

---
*Reproduce: `uv run python scripts/report_binary05.py` on branch
`analysis/binary05-methods` (supercollection `000_binary05`, 15 sims × 20 methods
× 50 reps).*

## Appendix A. Full per-method table (all 20 cells)

Pooled over the signal grid (ρ ∈ {0,0.5,0.9}, logBF ∈ {4,8,16,32}); coverage/size
at nominal β=0.95; `cal β` is the nominal level whose empirical coverage is 0.95.

| method | prior | B_causal | B_null | pow@FDP.1 | logBF AUC | cov@95 | size@95 | cal β |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| taylor_local | fixed | 0.341 | .0001 | 0.618 | 0.915 | 0.993 | 134.0 | 0.75 |
| taylor_local | eb | 0.389 | .0001 | 0.608 | 0.901 | 0.995 | 162.4 | 0.75 |
| taylor_local_c | fixed | 0.338 | .0001 | 0.603 | 0.915 | 0.992 | 132.2 | 0.76 |
| taylor_local_c | eb | 0.386 | .0001 | 0.607 | 0.905 | 0.995 | 160.7 | 0.76 |
| taylor_global | fixed | 0.340 | .0001 | 0.607 | 0.915 | 0.990 | 132.6 | 0.79 |
| taylor_global | eb | 0.371 | .0001 | 0.607 | 0.907 | 0.992 | 155.9 | 0.76 |
| taylor_global_c | fixed | 0.341 | .0001 | 0.615 | 0.915 | 0.995 | 137.1 | 0.76 |
| taylor_global_c | eb | 0.372 | .0001 | 0.607 | 0.908 | 0.995 | 159.6 | 0.75 |
| jj_local | fixed | 0.340 | .0001 | 0.608 | 0.916 | 0.992 | 132.3 | 0.71 |
| jj_local | eb | 0.390 | .0001 | 0.600 | 0.835 | 0.995 | 160.0 | 0.76 |
| jj_local_c | fixed | 0.339 | .0001 | 0.603 | 0.916 | 0.993 | 133.5 | 0.73 |
| jj_local_c | eb | 0.396 | .0001 | 0.593 | 0.822 | 0.995 | 162.4 | 0.73 |
| jj_global | fixed | 0.373 | .0001 | 0.593 | 0.913 | 0.937 | 133.0 | 0.99 |
| jj_global | eb | 0.470 | .0001 | 0.522 | 0.773 | 0.958 | 191.2 | 0.99 |
| jj_global_c | fixed | 0.372 | .0001 | 0.595 | 0.913 | 0.938 | 133.0 | 0.99 |
| jj_global_c | eb | 0.469 | .0001 | 0.522 | 0.774 | 0.962 | 192.0 | 0.99 |
| quadrature | fixed | 0.341 | .0001 | 0.618 | 0.915 | 0.993 | 134.0 | 0.75 |
| quadrature | eb | 0.371 | .0001 | 0.607 | 0.912 | 0.995 | 156.4 | 0.76 |
| profile | fixed | 0.338 | .0001 | 0.603 | 0.915 | 0.992 | 132.2 | 0.76 |
| profile | eb | 0.367 | .0001 | 0.608 | 0.912 | 0.995 | 154.2 | 0.76 |
