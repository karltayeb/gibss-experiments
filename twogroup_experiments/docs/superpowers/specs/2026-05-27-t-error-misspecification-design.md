# T-Error Misspecification Experiment

**Date:** 2026-05-27

## Goal

Show that Cox regression is more robust than the twogroup model when the observation error distribution is non-normal. The twogroup model assumes `bhat | b ~ N(b, 1)`; this experiment replaces that with a standardized t-distribution, holding variance fixed at 1 so only tail heaviness varies.

## Error Distribution

Standardized t-error: `t_ν(0, √((ν-2)/ν))` — unit variance, excess kurtosis `6/(ν-4)` for `ν > 4`, infinite for `ν ≤ 4`.

Degrees of freedom: **ν ∈ {3, 5, 10, 30}** plus normal baseline (reuses existing simulations).

| ν | Excess kurtosis |
|---|---|
| 3 | ∞ |
| 5 | 6.0 |
| 10 | 1.0 |
| 30 | 0.22 |
| normal | 0 |

## Signal Values

One fixed signal per design, chosen for moderate power in the Gaussian case (based on c4 f1_boxplot results; revisable after gaussian/uniform results are computed).

| design | loc signal | scale signal |
|---|---|---|
| hallmark | 2.00 | 2.00 |
| c4 | 2.00 | 2.00 |
| gaussian_markov_rho_0.90_n_features_100 | 2.00 | 2.00 |
| uniform_markov_rho_0.90_n_features_100 | 2.00 | 2.00 |

## Code Changes

### `core.py` — `SimulationSpec` + `simulate()`

Add optional `error_sampler: Any = None` field to `SimulationSpec`. `None` preserves existing normal behavior.

```python
@dataclass(frozen=True)
class SimulationSpec:
    name: str
    design_sampler: Any
    effect_sampler: Any
    intercept: float
    f0: Any
    f1: Any
    base_seed: int
    error_sampler: Any = None  # None = N(0, se)
```

Update `simulate()` line 144:

```python
if simulation_spec.error_sampler is None:
    noise = rng.normal(scale=se, size=X.shape[0])
else:
    noise = simulation_spec.error_sampler(rng, se)
thetahat = theta + noise
```

Update `dehydrate_spec()` to skip None-default fields (preserves existing hashes):

```python
def dehydrate_spec(spec: SimulationSpec) -> dict[str, Any]:
    result = {}
    for field in fields(spec):
        value = getattr(spec, field.name)
        if value is None and field.default is None:
            continue
        result[field.name] = dehydrate_node(value)
    return result
```

Update `rehydrate_spec()` to handle missing optional fields:

```python
def rehydrate_spec(node: dict[str, Any]) -> SimulationSpec:
    canonical_node = canonicalize_node(node)
    return SimulationSpec(
        ...,  # existing fields unchanged
        error_sampler=rehydrate_node(canonical_node["error_sampler"])
            if "error_sampler" in canonical_node else None,
    )
```

### `config.py` — error samplers + builders

Add importable error sampler functions:

```python
def normal_error_sampler(rng, se):
    return rng.normal(scale=se, size=len(se))

def t_error_sampler(rng, se, df):
    scale = se * np.sqrt((df - 2.0) / df)
    return rng.standard_t(df, size=len(se)) * scale
```

Extend `_simulation_name` with optional `error` param:

```python
def _simulation_name(*, design, enrichment, signal, error=None):
    base = f"design={design}__enrichment={enrichment}__signal={signal}"
    return base if error is None else f"{base}__error={error}"
```

Add builder and registration:

```python
T_ERROR_DFS = (3, 5, 10, 30)

T_ERROR_SIGNAL_VALUES = {
    "hallmark":     {"loc": 2.0, "scale": 2.0},
    "c4":           {"loc": 2.0, "scale": 2.0},
    _markov_design_name(family="gaussian", rho=0.90, n_features=100): {"loc": 2.0, "scale": 2.0},
    _markov_design_name(family="uniform",  rho=0.90, n_features=100): {"loc": 2.0, "scale": 2.0},
}

def _make_t_error_simulation(*, design_name, design_sampler, signal_kind, signal_value, error_df):
    f1 = fixed_normal(loc=signal_value, scale=LOC_SCALE_FIXED) if signal_kind == "loc" \
         else fixed_normal(loc=0.0, scale=signal_value)
    return SimulationSpec(
        name=_simulation_name(
            design=design_name,
            enrichment=SER_ENRICH,
            signal=_signal_name(signal_kind, signal_value),
            error=f"t_df_{error_df}",
        ),
        design_sampler=design_sampler,
        effect_sampler=partial(uniform_single_effect, causal_effect=2.0),
        intercept=-2.0,
        f0=F0,
        f1=f1,
        base_seed=BASE_SEED,
        error_sampler=partial(t_error_sampler, df=error_df),
    )

T_ERROR_SIMULATION_SPECS = tuple(
    _make_t_error_simulation(
        design_name=design_name,
        design_sampler=DESIGN_KWARGS[design_name]["design_sampler"],
        signal_kind=signal_kind,
        signal_value=T_ERROR_SIGNAL_VALUES[design_name][signal_kind],
        error_df=df,
    )
    for design_name in T_ERROR_SIGNAL_VALUES
    for signal_kind in ("loc", "scale")
    for df in T_ERROR_DFS
)

REGISTRY.register_simulations(T_ERROR_SIMULATION_SPECS)
REGISTRY.register_batches(tuple(
    batch
    for spec in T_ERROR_SIMULATION_SPECS
    for batch in batch_specs_for_simulation(spec, replicates_per_batch=REPLICATES_PER_BATCH, n_batches=N_BATCHES)
))
```

### `plot_config.yaml` — collections + supercollections

**Collection naming:**
- t-error: `design=X__enrichment=ser_enrich__signal=loc_2.00__error=t_df_5`
- Normal baseline: existing `design=X__enrichment=ser_enrich__signal=loc_2.00` (no new data)

**Aggregate collection naming:** `t-error-agg-{loc|scale}-{df|normal}` with `simulations:` list of all 4 designs.

**10 supercollections** (prefix `0010-`):

Per-design (8): `0010-{hallmark|c4|gaussian|uniform}-t-error-{loc|scale}`
- 5 collections: aliases `df=3`, `df=5`, `df=10`, `df=30`, `normal`

Aggregate (2): `0010-all-designs-t-error-{loc|scale}`
- 5 collections: same df aliases, each bundles all 4 designs via `simulations:` list

**Plots spec** (new YAML anchor `&t_error_plots`):
```yaml
plots: &t_error_plots
  - settings:
      - minimal
    plot_type_groups:
      - standard
      - cs
```

All 10 supercollections reference `&t_error_plots`.

## Future Extension

Heteroskedastic errors: add a new `error_sampler` implementation. No further changes to `SimulationSpec` or `simulate()` required.
