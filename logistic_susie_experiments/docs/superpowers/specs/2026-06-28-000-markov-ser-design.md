# 000_markov — SER calibration batch (design spec)

**Date:** 2026-06-28
**Status:** design (for review)
**Companion:** `notes/2026-06-27-univariate-logbf-z-grid.md`,
`notebooks/effect_size_calibration.py`

## Goal

A clean, cluster-scale **L=1 (SER)** simulation batch over synthetic Markov
designs, with effects sized to a grid of **expected univariate logBF**, to study
how the logistic-SuSiE SER variants trade off **detection, localization, and CS
calibration** as a function of **signal strength × correlation × outcome
imbalance**. Replaces the ad-hoc effects of `002_global` (which sat in the
saturated regime, AUC≈1, so methods didn't separate).

This is the first of a family: `000_markov` (synthetic, controllable rho); gene
sets and L>1 SuSiE come later, reusing the same machinery.

## Scope

In scope: synthetic Markov designs (gaussian, uniform, binary), exact-LRT effect
sizing + frozen table, the 000_markov experiment config, removal of the dead
`000_global_local` experiment.

Out of scope (later): gene-set designs (hallmark/c4/c2), L>1 SuSiE, the dedicated
imbalanced study. (b0=−4 *is* included here — it is now fully computable.)

## Grid

| axis | values |
|---|---|
| n | 1000 |
| p | 500 |
| design | gaussian, uniform, binary q=0.5, binary q=0.1 |
| rho | 0, 0.5, 0.8, 0.9, 0.95, 0.99 |
| target logBF | 4, 8, 16, 32, 64 |
| b0 (intercept) | 0, −2, −4 |
| + null (b=0) | one per (design, rho, b0) |

Cells: 4 designs × 6 rho × 3 b0 × (5 logBF + 1 null) = **432 sim configs** ×
methods × replicates. Cluster-scale.

## Effect sizing — exact LRT (`Λ = n·MI`), frozen offline

**Why:** size by the expected univariate log Bayes factor. The asymptotic logBF ≈
the expected log-likelihood ratio `Λ(b) = n·MI(x;y) = n·[H(p̄) − E_x H(p_x)]`,
`p_x = σ(b0 + b·x)` over the causal feature's **marginal**. `Λ` is **monotone**
in b (saturates only at the deviance `n·H(p̄_∞)`, hundreds of nats), so every rung
is reachable by a 1-D root find — **no Wald ceiling** (that is a Wald-method
artifact, not a sizing limit; see the note). Sizing is **rho-independent** (uses
the marginal), so one b per (design, b0, logBF) serves the whole rho sweep.

**Component: `simulations/effect/logbf_sizing.py`**
- `expected_lrt(marginal, b0, b, n) -> float` — `Λ` via 1-D quadrature (binary =
  2 atoms; gaussian/uniform = dense grid).
- `solve_b(marginal, b0, target_logbf, n) -> float` — monotone `brentq` on
  `Λ(b)=target`.
- `MARGINALS = {"gaussian", "uniform", "binary q=0.5", "binary q=0.1"}`.
- A `main()` that prints/emits the frozen table (the offline generator).
- `numpy` + `scipy.optimize.brentq` only. No glmbf.

**Frozen table (n=1000), baked into the config:**

| design | b0 | L4 | L8 | L16 | L32 | L64 |
|---|---:|---:|---:|---:|---:|---:|
| gaussian | 0 | 0.180 | 0.256 | 0.366 | 0.531 | 0.789 |
| uniform | 0 | 0.311 | 0.441 | 0.628 | 0.902 | 1.316 |
| binary q=0.5 | 0 | 0.179 | 0.254 | 0.361 | 0.514 | 0.740 |
| binary q=0.1 | 0 | 0.299 | 0.425 | 0.607 | 0.874 | 1.286 |
| gaussian | −2 | 0.275 | 0.387 | 0.545 | 0.766 | 1.091 |
| uniform | −2 | 0.477 | 0.675 | 0.953 | 1.347 | 1.910 |
| binary q=0.5 | −2 | 0.277 | 0.392 | 0.556 | 0.793 | 1.137 |
| binary q=0.1 | −2 | 0.440 | 0.613 | 0.851 | 1.180 | 1.649 |
| gaussian | −4 | 0.621 | 0.830 | 1.086 | 1.405 | 1.829 |
| uniform | −4 | 1.124 | 1.543 | 2.077 | 2.735 | 3.537 |
| binary q=0.5 | −4 | 0.674 | 0.951 | 1.337 | 1.853 | 2.502 |
| binary q=0.1 | −4 | 0.974 | 1.313 | 1.752 | 2.316 | 3.039 |

(Validated: mean realized LRT ≈ target, e.g. 4→4.5 … 64→64.3, all designs.)

## How the table reaches the simulations (no new sampler, no lookup bug)

The b depends on **both** design and b0, so the existing cartesian
`intercept × causal_effect` enrichment family would mis-pair them. Instead we
**bake one concrete enrichment per (design, b0, logBF)** using the existing
`uniform_single_effect` sampler with `causal_effect = frozen b`, named by the
target logBF for readability:

```
ser__{design}__b0={b0}__lbf={logbf}   ->  uniform_single_effect(causal_effect=b), intercept=b0
ser__{design}__b0={b0}_null           ->  uniform_single_effect(causal_effect=0), intercept=b0
```

These are **generated** by `logbf_sizing.main()` into a yaml fragment (designs +
enrichments) that is committed into `library.yaml` (or an included file) — frozen,
human-readable, no sim-time computation. The causal-feature marginal is set by the
**design** (binary_markov `freq`, gaussian/uniform unit/centered), so the baked b
is correct for that design. Generation is reproducible; a test guards drift.

Rationale for baking vs. a lookup sampler: simplest path, reuses
`uniform_single_effect`, keeps `core.simulate` untouched, and the values are
inspectable in the config. The `logbf_sizing` module remains the single source of
truth + regenerator.

## Designs

`library.yaml` gains 24 design entries (4 types × 6 rho) at n=1000, p=500:
- `gaussian_markov_X {n:1000, p:500, rho:ρ}`
- `uniform_markov_X {n:1000, p:500, rho:ρ}`
- `binary_markov_X {n:1000, p:500, rho:ρ, freq:0.5}`
- `binary_markov_X {n:1000, p:500, rho:ρ, freq:0.1}`

(These samplers already exist and are tested: `tests/test_markov_designs.py`.)

## Experiment `experiments/000_markov.yaml`

One supercollection. Collections pair each (design type, rho) design with the
matching (b0, logBF) signal enrichments + the (b0) nulls.

**Methods (6) — all `estimate_prior_variance=false` (fixed prior), `center=false`
except `null_score`:**

| label | function | config |
|---|---|---|
| `linear` | `run_linear_susie_method` | L=1, fixed prior, residual variance 0.25 |
| `null_score` | `run_irls_method` | `ser_cadence=block, n_outer=1, center=true`, fixed prior — the one-step null-orthogonalized reference (alias for irls-1-centered) |
| `irls` | `run_irls_method` | `ser_cadence=block, n_outer=50, center=false`, fixed prior (Laplace) |
| `localjj` | `run_logistic_method` | `impl=localjj`, fixed prior |
| `globaljj` | `run_logistic_method` | `impl=globaljj`, fixed prior |
| `quadrature` | `run_logistic_method` | `impl=logistic_quadrature`, fixed prior (exact reference) |

- Fixed-prior variance value: **`prior_variance = 1.0`** (sensible for the
  effect scale here, b ≲ 3.5; flag if a different σ0² is wanted — it shifts the
  Occam term uniformly across methods).
- `null_score` is `center=true` by request (the calibrated one-step); the other
  five are `center=false`. In `viz_dims` it resolves to family=`irls`,
  step=`one_step` — add a `null_score` family alias so it plots distinctly.

**Replicates:** `replicates_per_batch = 50`, `n_batches = 2` → 100 reps/cell.

**Plots** (reuse the faceting system; per design, per b0; facet **rho × logBF**,
color = method):
- `pip_calibration`
- `power_fdp` (PIP-level power vs FDP)
- `cs_power_fdp` (logBF detection precision-recall) + `cs_roc` (logBF ROC)
- `cs_calibration` (coverage calibration curve)
- `cs_coverage_size` (CS size vs empirical coverage)

`viz_dims` already maps `binary_markov_X → "binary"`, `uniform`, `gaussian`. The
`logbf` facet dimension comes from the enrichment (surface `target_logbf` from
the enrichment name/arg in `sim_dims`). Detailed plot specs in the plan.

## Removal

Delete `experiments/000_global_local.yaml` (dead; superseded). Confirm no other
file references it before removing.

## Testing

- `tests/test_logbf_sizing.py`:
  - `expected_lrt` monotone in b; equals 0 at b=0.
  - `solve_b` round-trips: `Λ(solve_b(target)) ≈ target` for each (design, b0,
    target) at n=1000.
  - **Frozen-table guard**: the committed b values reproduce their target logBF
    via `expected_lrt` (so the table can't silently rot if the solver changes).
  - (optional, slow) a small simulation check that mean realized LRT ≈ target.
- `binary_markov_X` correlation/marginal already covered by
  `tests/test_markov_designs.py`.

## Resolved decisions

1. Methods: the 6 above, all fixed prior (`prior_variance=1.0`), `center=false`
   except `null_score` (centered one-step).
2. Replicates: 50 × 2 batches = 100/cell.
3. Plots: pip_calibration, power_fdp, cs_power_fdp + cs_roc, cs_calibration,
   cs_coverage_size.

## Remaining detail (decide in plan, non-blocking)

- `prior_variance=1.0` for the fixed prior (flag if another σ0² preferred).
- `null_score` family alias in `viz_dims` so it plots distinctly from `irls`.
- Config layout: bake the generated 000_markov designs/enrichments into
  `library.yaml` vs. a separate included file (lean either way; default: append
  to `library.yaml` with a clear comment banner).
