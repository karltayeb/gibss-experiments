# Deterministic membership for well-specified Cox (009)

Date: 2026-06-21

## Problem

Experiment 009 (`009_cox_well_specified.yaml`) aims to feed Cox proportional-hazards
(PH) data to the `cox_reversed` method. The current generative model is *not*
well-specified for Cox.

Generative path (`core.py:113-124`):

```python
b[causal_indices] = causal_effects
logits = intercept + X @ b
z = rng.binomial(1, sigmoid(logits))      # stochastic membership
theta = Exp(1)  if z == 0 else Exp(lambda)
```

`X` is binary gene-set membership (`simulations/design/genesets.py`); the effect
sampler picks one random set column (`simulations/effect/effects.py:13`).

With `ser_b2` (`causal_effect=2`, `intercept=-2`) the causal column gives:

- in-set gene `x=1`  -> `logits = 0`  -> `P(z=1) = 0.5`
- out-set gene `x=0` -> `logits = -2` -> `P(z=1) = sigmoid(-2) = 0.12`

So time given `x` is a **mixture** of `Exp(1)` and `Exp(lambda)`. A
mixture-of-exponentials hazard is not proportional, so Cox sees misspecified data
even on out-of-set genes.

## Goal

A well-specified Cox benchmark: hazard_i = h0 * exp(beta * x_i) **exactly**. For
binary `x` this requires deterministic membership `z = x_causal`:

- out-set genes -> `Exp(1)` (hazard 1)
- in-set genes  -> `Exp(lambda)` (hazard lambda)
- hazard ratio `lambda = exp(beta)`, exact PH.

## Design

Add a `membership` mode to the generative model.

- `stochastic` (default, current behavior): `z = Bernoulli(sigmoid(logits))`.
- `deterministic`: `z = 1[logits > 0]`.

With `effect=2, intercept=-1` on binary `X`: in-set `logits=+1 -> z=1`, out-set
`logits=-1 -> z=0`. Hence `z = x_causal` exactly. The error layer still applies on
top of `theta` unchanged, so deterministic membership composes with any
`error_sampler` for later robustness sweeps.

### Constraint

Clean `z = x_causal` requires `intercept < 0` and `intercept + effect > 0`.
`(intercept=-1, effect=2)` satisfies both. Document this on the `_det` enrichments.

### Components (4 touch points + test)

1. `core.py` (`SimulationSpec`, ~line 22) — add field:
   ```python
   membership: str = "stochastic"
   ```
2. `core.py` (`simulate`, ~line 116) — branch:
   ```python
   if simulation_spec.membership == "deterministic":
       z = (logits > 0).astype(int)
   else:
       z = rng.binomial(1, _sigmoid(logits)).astype(int)
   ```
3. `experiments/loader.py` (`resolve_simulation_from_coord`) — plumb the flag:
   ```python
   membership=enrich.get("membership", "stochastic"),
   ```
   Default keeps every existing enrichment byte-identical.
4. `experiments/library.yaml` — new enrichments:
   - `ser_b2_det`: `causal_effect: 2.0`, `intercept: -1.0`, `membership: deterministic`
   - `null_b0_det`: `causal_effect: 0.0`, `intercept: -1.0`, `membership: deterministic`
     (no causal column -> all `logits = -1 < 0` -> all `z = 0`, a valid null)
5. `experiments/009_cox_well_specified.yaml` — swap
   `enrichment: [ser_b2, null_b0]` -> `[ser_b2_det, null_b0_det]` in all four
   supercollections (hallmark, c2, c4, c5).
6. Test (`tests/test_core_run_methods.py` or sibling) — a deterministic-membership
   sim asserts `z == X[:, causal]` for the in-set column and `z == 0` elsewhere;
   confirms exact PH construction.

## Cache / invalidation

Two distinct mechanisms:

- **Hash:** unchanged. `sim_hash` hashes the coordinate dict including the
  enrichment entry (`loader.py`). Existing `ser_b2` / `null_b0` entries are
  untouched; the new `_det` entries hash distinct. No hash-level invalidation.
- **Code-dep tracking:** editing `core.py` *does* re-trigger. `simulation_code_files`
  lists `core.py` as a plain (non-`ancient()`) `input:` to the `simulate` and `fit`
  rules (`twogroup_experiments.snk:56,78`). Bumping `core.py` mtime/content marks all
  twogroup sims out of date.

Re-run produces **identical** results (stochastic path byte-unchanged, seeds
deterministic via `replicate_seed`), so there is no scientific invalidation — only
wasted compute.

**Mitigation (chosen):** after the `core.py` edit, run `snakemake --touch` on the
existing stochastic sim/fit outputs to mark them up to date (their logic is verified
unchanged). Only the new `_det` batches build fresh.

## Out of scope

- Continuous / multi-column membership thresholds (flag generalizes but unused here).
- Changing the `stochastic` default or any non-009 experiment.
