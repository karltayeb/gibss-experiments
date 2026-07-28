# Cox well-specified arrival-time simulations

**Date:** 2026-06-18
**Status:** Design — pending user review

## Goal

Add simulations that are *well-specified* generative examples of the cox model,
expressed inside the existing design → enrichment → f1 → noise framework, and
that let us study the cox-light vs cox-heavy asymmetry the project notes are
about.

The cox proportional-hazards model depends only on the **ranks** of the
observations. A clean well-specified instance is a two-group model where each
group is exponentially distributed and the enrichment indicator `z` selects the
group:

- background arrivals `~ Exp(1)`, enriched arrivals `~ Exp(λ)`.

`λ` is the f1 rate, the swept signal parameter (analogous to `loc`/`scale` in
the gaussian sims). `λ = 1` recovers the background (f0 = f1), so it is the null
sanity point.

### Why exponential only (no inverse-exponential)

Earlier drafts added an inverse-exponential ("reversed") family for cox-heavy.
It is redundant in the well-specified setting:

- inv-exp data + forward cox = misspecified (wrong ranking direction);
- inv-exp data + reversed cox ranks by `1/T`, and `1/InvExp(λ) = Exp(λ)`, so it
  is **identical** to exp data + forward cox.

So the well-specified inv-exp problem is just the exp problem relabeled — it
adds nothing. The cox-light / cox-heavy distinction instead lives in **λ**, read
through a single forward cox fit (`time_sign = +1`, rank ascending):

- **λ > 1** → enriched arrive early → cox-light regime;
- **λ < 1** → enriched arrive late → cox-heavy regime.

Both are well-specified. Sweeping λ across both sides of 1 exposes the
partial-likelihood asymmetry (power differs for matched signal on either side)
without any inverse-exponential draws or `time_sign = -1` reversal.

### Other decisions settled during brainstorming

- **Raw arrival times, no transform.** We considered a PIT-to-gaussian map
  (`Φ⁻¹(F_ref(T))`, which yields a `Beta(1,λ)` percentile) so the statistic
  would share the z-score scale of the gaussian sims. **Rejected** in favor of
  raw arrival times: simpler, no transform plumbing, and *more* correct —
  arrival times are strictly positive, so `score = |thetahat/se| = thetahat`
  (the `abs()` in `fit_cox_method` is a no-op) and the ranking is the exact
  arrival order. Cost: no shared threshold scale with the gaussian sims, so
  threshold methods are excluded (below).
- **No `time_sign = -1`.** Rank order is never reversed by multiplying the
  statistic by `-1`. A single forward cox fit is used.
- **Enrichment unchanged.** `z ~ Bernoulli(sigmoid(intercept + X·b))` still
  selects the mixture component. The covariate effect enters through `z`, not a
  per-observation rate.

## Components

### 1. Distribution (in `core.py`, alongside `Normal`/`PointMass` usage)

One new frozen dataclass following the existing distribution shape
(`sample(rng, size)`; dehydratable via the `is_dataclass` path the spec hasher
already supports). Only `sample` is required — these sims are fit by the cox
method, which never calls `log_likelihood_nm`.

```python
@dataclass(frozen=True)
class Exponential:
    rate: float = 1.0
    def sample(self, rng, size):
        return rng.exponential(scale=1.0 / self.rate, size=size)
```

Note: `np.random.Generator.exponential` is parameterized by `scale = 1/rate`.

### 2. Noiseless error sampler (in `core.py`, beside `t_error_sampler`)

```python
def noiseless_error_sampler(rng, se):
    del rng
    return np.zeros(len(se), dtype=float)
```

Required because `simulate()` defaults to `rng.normal(scale=se)` when
`error_sampler is None`; we need `thetahat = theta` exactly.

`simulate()` itself is **not modified** — f1/f0 produce the arrival times and
the noiseless sampler zeroes the additive noise.

### 3. New simulation specs (in `config.py`)

A constructor mirroring `_make_t_error_simulation`:

```python
def _make_cox_arrival_simulation(*, design_name, design_sampler, rate):
    return SimulationSpec(
        name=_simulation_name(
            design=design_name,
            enrichment=SER_ENRICH,
            signal=f"exp_rate_{format_float(rate)}",
        ),
        design_sampler=design_sampler,
        effect_sampler=partial(uniform_single_effect, causal_effect=2.0),
        intercept=-2.0,
        f0=Exponential(1.0),
        f1=Exponential(rate),
        base_seed=BASE_SEED,
        error_sampler=noiseless_error_sampler,
    )
```

Registered like the other families (`SIMULATION_BY_NAME.update`,
`REGISTRY.register_simulations`, `REGISTRY.register_batches`).

**Sweep:**
- Designs: same set as `T_ERROR_SIGNAL_VALUES` keys (hallmark, c4, the gaussian
  markov variants, uniform markov).
- `RATE_GRID = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)` — log-symmetric around 1
  so `λ` and `1/λ` pair up, covering the cox-heavy (`λ<1`) and cox-light (`λ>1`)
  regimes. `λ = 1` is the null (f0 = f1) sanity point.

### 4. Fit method

A single forward cox fit — `time_sign = +1`, `threshold = 0` (all positive
arrival times are events; equivalent to `None` here but explicit). Add one
method spec:

```python
def _cox_method_spec(*, L: int) -> MethodSpec:
    return MethodSpec(
        name=f"cox_L{L}",
        fit_function=fit_cox_method,
        summarize_function=summarize_cox_method,
        kwargs={"threshold": 0.0, "time_sign": 1.0, "L": int(L)},
    )
```

Naming note: the existing `cox_light_*` means `time_sign = -1` + thresholding;
this new `cox_L*` is distinct (no `-1`, threshold 0). λ>1 read through `cox_L*` =
cox-light regime; λ<1 = cox-heavy regime.

**Threshold methods excluded** for this family: `cox_light_threshold` and
`logistic_threshold` are dropped — they are not calibrated on the arrival-time
scale.

The methods evaluated on these sims are therefore a small set, e.g. `cox_L1`
(and `cox_L5`), plus whichever non-threshold methods are useful for comparison
(`logistic_oracle`, `twogroup_*`). Exact method membership for the batches is
set when wiring the batch specs.

## Open items for review

None outstanding. Flag anything in this revision before planning.

## Testing

- Unit: `Exponential.sample` shape, positivity, approximate rate (mean ≈
  `1/rate`). `noiseless_error_sampler` returns zeros.
- Unit: each new `SimulationSpec` dehydrates/rehydrates round-trip (hash stable)
  like the existing specs.
- Integration: `simulate()` on a `cox_arrival` spec yields `thetahat == theta`
  (noiseless), enriched-group mean ≈ `1/λ` vs background ≈ `1`, and a finite
  `fit_cox_method` (`cox_L*`) run end-to-end.
