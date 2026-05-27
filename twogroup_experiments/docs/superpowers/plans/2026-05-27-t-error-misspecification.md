# T-Error Misspecification Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add t-distributed observation error to the simulation framework and register 10 new supercollections comparing twogroup vs Cox robustness under error misspecification.

**Architecture:** (1) Extend `SimulationSpec` with an optional `error_sampler` field, preserving existing hashes via `dehydrate_spec` skip-None logic. (2) Add `t_error_sampler` to `core.py` and register 32 new simulation specs in `config.py`. (3) Add collection definitions and 10 supercollections to `plot_config.yaml`.

**Tech Stack:** Python dataclasses, numpy, PyYAML, Snakemake, pytest (via `uv run pytest`)

---

### Task 1: Extend `SimulationSpec` with `error_sampler`, preserve backward-compat hashes

**Files:**
- Modify: `core.py` — `SimulationSpec`, `dehydrate_spec`, `rehydrate_spec`
- Modify: `tests/test_twogroup_experiments.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_twogroup_experiments.py`:

```python
def test_simulation_spec_hash_unchanged_after_adding_error_sampler():
    """Adding error_sampler=None must not change existing hashes."""
    from core import SimulationSpec, simulation_hash, uniform_single_effect, identity_design_sampler
    from gibss.distributions import Normal, PointMass
    from functools import partial

    spec_without = SimulationSpec(
        name="tiny_simulation",
        design_sampler=identity_design_sampler,
        effect_sampler=partial(uniform_single_effect, causal_effect=2.0),
        intercept=-1.0,
        f0=PointMass(0.0),
        f1=Normal(loc=1.0, scale=0.1, estimate_loc=False, estimate_scale=False),
        base_seed=123,
    )
    # Record hash before the field exists (captured from current passing tests)
    hash_before = simulation_hash(spec_without)

    spec_with_none = SimulationSpec(
        name="tiny_simulation",
        design_sampler=identity_design_sampler,
        effect_sampler=partial(uniform_single_effect, causal_effect=2.0),
        intercept=-1.0,
        f0=PointMass(0.0),
        f1=Normal(loc=1.0, scale=0.1, estimate_loc=False, estimate_scale=False),
        base_seed=123,
        error_sampler=None,
    )
    assert simulation_hash(spec_with_none) == hash_before


def test_rehydrate_spec_handles_missing_error_sampler():
    """Old serialized specs (no error_sampler key) must rehydrate without error."""
    from core import rehydrate_spec, dehydrate_spec, SimulationSpec, uniform_single_effect, identity_design_sampler
    from gibss.distributions import Normal, PointMass
    from functools import partial

    spec = SimulationSpec(
        name="tiny_simulation",
        design_sampler=identity_design_sampler,
        effect_sampler=partial(uniform_single_effect, causal_effect=2.0),
        intercept=-1.0,
        f0=PointMass(0.0),
        f1=Normal(loc=1.0, scale=0.1, estimate_loc=False, estimate_scale=False),
        base_seed=123,
        error_sampler=None,
    )
    dehydrated = dehydrate_spec(spec)
    assert "error_sampler" not in dehydrated  # omitted when None

    rehydrated = rehydrate_spec(dehydrated)
    assert rehydrated.error_sampler is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_twogroup_experiments.py::test_simulation_spec_hash_unchanged_after_adding_error_sampler tests/test_twogroup_experiments.py::test_rehydrate_spec_handles_missing_error_sampler -v
```

Expected: both FAIL (field doesn't exist yet).

- [ ] **Step 3: Add `error_sampler` field to `SimulationSpec`**

In `core.py`, change `SimulationSpec` from:
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
```
to:
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
    error_sampler: Any = None
```

- [ ] **Step 4: Update `dehydrate_spec` to skip None-default fields**

In `core.py`, replace:
```python
def dehydrate_spec(spec: SimulationSpec) -> dict[str, Any]:
    return {
        field.name: dehydrate_node(getattr(spec, field.name)) for field in fields(spec)
    }
```
with:
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

- [ ] **Step 5: Update `rehydrate_spec` to handle missing optional fields**

In `core.py`, replace:
```python
def rehydrate_spec(node: dict[str, Any]) -> SimulationSpec:
    canonical_node = canonicalize_node(node)
    return SimulationSpec(
        name=str(canonical_node["name"]),
        design_sampler=rehydrate_node(canonical_node["design_sampler"]),
        effect_sampler=rehydrate_node(canonical_node["effect_sampler"]),
        intercept=float(canonical_node["intercept"]),
        f0=rehydrate_node(canonical_node["f0"]),
        f1=rehydrate_node(canonical_node["f1"]),
        base_seed=int(canonical_node["base_seed"]),
    )
```
with:
```python
def rehydrate_spec(node: dict[str, Any]) -> SimulationSpec:
    canonical_node = canonicalize_node(node)
    return SimulationSpec(
        name=str(canonical_node["name"]),
        design_sampler=rehydrate_node(canonical_node["design_sampler"]),
        effect_sampler=rehydrate_node(canonical_node["effect_sampler"]),
        intercept=float(canonical_node["intercept"]),
        f0=rehydrate_node(canonical_node["f0"]),
        f1=rehydrate_node(canonical_node["f1"]),
        base_seed=int(canonical_node["base_seed"]),
        error_sampler=rehydrate_node(canonical_node["error_sampler"])
            if "error_sampler" in canonical_node else None,
    )
```

- [ ] **Step 6: Run tests**

```bash
uv run pytest tests/test_twogroup_experiments.py::test_simulation_spec_hash_unchanged_after_adding_error_sampler tests/test_twogroup_experiments.py::test_rehydrate_spec_handles_missing_error_sampler -v
```

Expected: both PASS.

- [ ] **Step 7: Run full test suite**

```bash
uv run pytest tests/ -q
```

Expected: 37 passed.

- [ ] **Step 8: Commit**

```bash
git add core.py tests/test_twogroup_experiments.py
git commit -m "feat: add optional error_sampler to SimulationSpec, preserve backward-compat hashes"
```

---

### Task 2: Add `t_error_sampler` to `core.py` and wire into `simulate()`

**Files:**
- Modify: `core.py` — add sampler functions, update `simulate()`
- Modify: `tests/test_twogroup_experiments.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_twogroup_experiments.py`:

```python
def test_t_error_sampler_has_unit_variance():
    """Standardized t-error sampler should produce unit variance regardless of df."""
    from core import t_error_sampler
    import numpy as np

    rng = np.random.default_rng(42)
    se = np.ones(10_000)
    for df in (3, 5, 10, 30):
        samples = t_error_sampler(rng, se, df=df)
        assert abs(np.var(samples) - 1.0) < 0.05, f"df={df}: var={np.var(samples):.3f}"


def test_simulate_uses_error_sampler_when_set():
    """simulate() with a t error_sampler should produce heavier tails than normal."""
    from core import SimulationSpec, simulate, t_error_sampler, uniform_single_effect, identity_design_sampler
    from gibss.distributions import Normal, PointMass
    from functools import partial
    import numpy as np

    base_spec = SimulationSpec(
        name="tiny_simulation",
        design_sampler=identity_design_sampler,
        effect_sampler=partial(uniform_single_effect, causal_effect=2.0),
        intercept=-1.0,
        f0=PointMass(0.0),
        f1=Normal(loc=1.0, scale=0.1, estimate_loc=False, estimate_scale=False),
        base_seed=123,
    )
    t_spec = SimulationSpec(
        name="tiny_simulation",
        design_sampler=identity_design_sampler,
        effect_sampler=partial(uniform_single_effect, causal_effect=2.0),
        intercept=-1.0,
        f0=PointMass(0.0),
        f1=Normal(loc=1.0, scale=0.1, estimate_loc=False, estimate_scale=False),
        base_seed=123,
        error_sampler=partial(t_error_sampler, df=3),
    )
    normal_sim = simulate(base_spec, replicate=0)
    t_sim = simulate(t_spec, replicate=0)
    # Both should have same theta (same seed, same spec except error)
    np.testing.assert_array_equal(normal_sim.theta, t_sim.theta)
    # thetahat should differ
    assert not np.allclose(normal_sim.thetahat, t_sim.thetahat)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_twogroup_experiments.py::test_t_error_sampler_has_unit_variance tests/test_twogroup_experiments.py::test_simulate_uses_error_sampler_when_set -v
```

Expected: both FAIL (`t_error_sampler` not defined).

- [ ] **Step 3: Add `t_error_sampler` to `core.py`**

Add after the `uniform_single_effect` function (around line 619):

```python
def t_error_sampler(
    rng: np.random.Generator,
    se: np.ndarray,
    *,
    df: float,
) -> np.ndarray:
    """Standardized t-distributed error: unit variance regardless of df."""
    scale = se * np.sqrt((df - 2.0) / df)
    return rng.standard_t(df, size=len(se)) * scale
```

- [ ] **Step 4: Update `simulate()` to use `error_sampler` when set**

In `core.py`, replace line 144:
```python
    thetahat = theta + rng.normal(scale=se, size=X.shape[0])
```
with:
```python
    if simulation_spec.error_sampler is None:
        noise = rng.normal(scale=se, size=X.shape[0])
    else:
        noise = simulation_spec.error_sampler(rng, se)
    thetahat = theta + noise
```

- [ ] **Step 5: Export `t_error_sampler` so it's importable for serialization**

`t_error_sampler` is already a module-level function in `core.py`, so it's importable. Verify the `_callable_path` check will pass by confirming it has no `<locals>` or `<lambda>` in its qualname (it won't, since it's top-level).

- [ ] **Step 6: Run tests**

```bash
uv run pytest tests/test_twogroup_experiments.py::test_t_error_sampler_has_unit_variance tests/test_twogroup_experiments.py::test_simulate_uses_error_sampler_when_set -v
```

Expected: both PASS.

- [ ] **Step 7: Run full test suite**

```bash
uv run pytest tests/ -q
```

Expected: 39 passed.

- [ ] **Step 8: Commit**

```bash
git add core.py tests/test_twogroup_experiments.py
git commit -m "feat: add t_error_sampler and wire error_sampler into simulate()"
```

---

### Task 3: Register `T_ERROR_SIMULATION_SPECS` in `config.py`

**Files:**
- Modify: `config.py`
- Modify: `tests/test_twogroup_experiments.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_twogroup_experiments.py`:

```python
def test_t_error_simulation_specs_count():
    """4 designs × 2 signal kinds × 4 df values = 32 specs."""
    from config import T_ERROR_SIMULATION_SPECS
    assert len(T_ERROR_SIMULATION_SPECS) == 32


def test_t_error_simulation_spec_names_include_error_field():
    from config import T_ERROR_SIMULATION_SPECS
    for spec in T_ERROR_SIMULATION_SPECS:
        assert "__error=t_df_" in spec.name, f"Missing error field: {spec.name}"


def test_t_error_simulation_specs_have_registered_batches():
    from config import REGISTRY, T_ERROR_SIMULATION_SPECS
    batch_sim_names = {b.simulation_spec.name for b in REGISTRY.batches}
    for spec in T_ERROR_SIMULATION_SPECS:
        assert spec.name in batch_sim_names, f"No batch for {spec.name}"


def test_t_error_simulation_spec_hash_differs_from_normal_baseline():
    """t-error spec hash must differ from the equivalent normal-error spec."""
    from config import T_ERROR_SIMULATION_SPECS, SIMULATION_BY_NAME
    for spec in T_ERROR_SIMULATION_SPECS:
        # Derive the equivalent normal baseline name (strip __error=... suffix)
        normal_name = spec.name.split("__error=")[0]
        if normal_name in SIMULATION_BY_NAME:
            from core import simulation_hash
            assert simulation_hash(spec) != simulation_hash(SIMULATION_BY_NAME[normal_name])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_twogroup_experiments.py::test_t_error_simulation_specs_count tests/test_twogroup_experiments.py::test_t_error_simulation_spec_names_include_error_field tests/test_twogroup_experiments.py::test_t_error_simulation_specs_have_registered_batches tests/test_twogroup_experiments.py::test_t_error_simulation_spec_hash_differs_from_normal_baseline -v
```

Expected: all FAIL.

- [ ] **Step 3: Extend `_simulation_name` to accept optional `error` param**

In `config.py`, replace:
```python
def _simulation_name(*, design: str, enrichment: str, signal: str) -> str:
    return f"design={design}__enrichment={enrichment}__signal={signal}"
```
with:
```python
def _simulation_name(*, design: str, enrichment: str, signal: str, error: str | None = None) -> str:
    base = f"design={design}__enrichment={enrichment}__signal={signal}"
    return base if error is None else f"{base}__error={error}"
```

- [ ] **Step 4: Add `T_ERROR_DFS`, `T_ERROR_SIGNAL_VALUES`, `_make_t_error_simulation`, and `T_ERROR_SIMULATION_SPECS` to `config.py`**

Add after `NULL_ENRICH_SIMULATION_SPECS` registration block (after line ~486):

```python
T_ERROR_DFS = (3, 5, 10, 30)

T_ERROR_SIGNAL_VALUES: dict[str, dict[str, float]] = {
    "hallmark": {"loc": 2.0, "scale": 2.0},
    "c4": {"loc": 2.0, "scale": 2.0},
    _markov_design_name(
        family="gaussian", rho=SIGNAL_RHO, n_features=SIGNAL_N_FEATURES
    ): {"loc": 2.0, "scale": 2.0},
    _markov_design_name(
        family="uniform", rho=SIGNAL_RHO, n_features=SIGNAL_N_FEATURES
    ): {"loc": 2.0, "scale": 2.0},
}


def _make_t_error_simulation(
    *,
    design_name: str,
    design_sampler,
    signal_kind: str,
    signal_value: float,
    error_df: int,
) -> SimulationSpec:
    if signal_kind == "loc":
        f1 = fixed_normal(loc=signal_value, scale=LOC_SCALE_FIXED)
    elif signal_kind == "scale":
        f1 = fixed_normal(loc=0.0, scale=signal_value)
    else:
        raise ValueError(f"Unknown signal kind: {signal_kind!r}")
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

SIMULATION_BY_NAME.update({spec.name: spec for spec in T_ERROR_SIMULATION_SPECS})
REGISTRY.register_simulations(T_ERROR_SIMULATION_SPECS)
REGISTRY.register_batches(tuple(
    batch
    for spec in T_ERROR_SIMULATION_SPECS
    for batch in batch_specs_for_simulation(
        spec,
        replicates_per_batch=REPLICATES_PER_BATCH,
        n_batches=N_BATCHES,
    )
))
```

- [ ] **Step 5: Add `t_error_sampler` import to `config.py`**

At the top of `config.py`, add `t_error_sampler` to the import from `core`:

```python
from core import (
    ...
    t_error_sampler,
    ...
)
```

Find the existing `from core import (` block and add `t_error_sampler` to it.

- [ ] **Step 6: Export `T_ERROR_SIMULATION_SPECS` in `__all__`**

In `config.py`, add `"T_ERROR_SIMULATION_SPECS"` to the `__all__` list.

- [ ] **Step 7: Run tests**

```bash
uv run pytest tests/test_twogroup_experiments.py::test_t_error_simulation_specs_count tests/test_twogroup_experiments.py::test_t_error_simulation_spec_names_include_error_field tests/test_twogroup_experiments.py::test_t_error_simulation_specs_have_registered_batches tests/test_twogroup_experiments.py::test_t_error_simulation_spec_hash_differs_from_normal_baseline -v
```

Expected: all PASS.

- [ ] **Step 8: Run full test suite**

```bash
uv run pytest tests/ -q
```

Expected: 43 passed.

- [ ] **Step 9: Commit**

```bash
git add core.py config.py tests/test_twogroup_experiments.py
git commit -m "feat: register T_ERROR_SIMULATION_SPECS for t-distributed error misspecification"
```

---

### Task 4: Add per-design collection definitions to `plot_config.yaml`

Each of the 8 per-design supercollections needs collection entries. The normal baseline reuses existing simulation names (no new collection definition needed — the name IS the simulation name, and the snk file resolves it automatically). Only t-error collections need explicit entries if they require non-default method collections; otherwise they are also auto-resolved.

**Files:**
- Modify: `notebooks/plot_config.yaml`

- [ ] **Step 1: Add the 32 t-error collection definitions to the `collections:` block**

The t-error collections use the `default` method collection (same as standard signal collections). Since all 32 t-error collection names are new (not in the `collections:` block), the snk file will auto-resolve them with `method_collections: ["default"]`. No explicit `collections:` entries are needed unless you want a non-default method collection.

Verify this by checking `twogroup_experiments.snk:93`: `_sims = _spec.get("simulations", [_name])` and `_spec.get("method_collections", ["default"])`. Since `_spec = _COLLECTION_DEFS.get(_name, {})` returns `{}` for unknown names, default method collection is used automatically.

No YAML change needed for Task 4. Skip to Task 5.

---

### Task 5: Add aggregate collection definitions and per-design supercollections to `plot_config.yaml`

The aggregate supercollections need explicit `collections:` entries because they bundle multiple simulations. The per-design supercollections reference single-simulation collections (auto-resolved).

**Files:**
- Modify: `notebooks/plot_config.yaml`

- [ ] **Step 1: Add aggregate collection definitions to the `collections:` block**

Add to the `collections:` section of `notebooks/plot_config.yaml`. Use the exact simulation names from Task 3.

The gaussian design name is `gaussian_markov_rho_0.90_n_features_100`.
The uniform design name is `uniform_markov_rho_0.90_n_features_100`.

```yaml
  # --- t-error aggregate collections (loc) ---
  t-error-agg-loc-df-3:
    simulations:
      - design=hallmark__enrichment=ser_enrich__signal=loc_2.00__error=t_df_3
      - design=c4__enrichment=ser_enrich__signal=loc_2.00__error=t_df_3
      - design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_2.00__error=t_df_3
      - design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_2.00__error=t_df_3
  t-error-agg-loc-df-5:
    simulations:
      - design=hallmark__enrichment=ser_enrich__signal=loc_2.00__error=t_df_5
      - design=c4__enrichment=ser_enrich__signal=loc_2.00__error=t_df_5
      - design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_2.00__error=t_df_5
      - design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_2.00__error=t_df_5
  t-error-agg-loc-df-10:
    simulations:
      - design=hallmark__enrichment=ser_enrich__signal=loc_2.00__error=t_df_10
      - design=c4__enrichment=ser_enrich__signal=loc_2.00__error=t_df_10
      - design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_2.00__error=t_df_10
      - design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_2.00__error=t_df_10
  t-error-agg-loc-df-30:
    simulations:
      - design=hallmark__enrichment=ser_enrich__signal=loc_2.00__error=t_df_30
      - design=c4__enrichment=ser_enrich__signal=loc_2.00__error=t_df_30
      - design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_2.00__error=t_df_30
      - design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_2.00__error=t_df_30
  t-error-agg-loc-normal:
    simulations:
      - design=hallmark__enrichment=ser_enrich__signal=loc_2.00
      - design=c4__enrichment=ser_enrich__signal=loc_2.00
      - design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_2.00
      - design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_2.00
  # --- t-error aggregate collections (scale) ---
  t-error-agg-scale-df-3:
    simulations:
      - design=hallmark__enrichment=ser_enrich__signal=scale_2.00__error=t_df_3
      - design=c4__enrichment=ser_enrich__signal=scale_2.00__error=t_df_3
      - design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_2.00__error=t_df_3
      - design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_2.00__error=t_df_3
  t-error-agg-scale-df-5:
    simulations:
      - design=hallmark__enrichment=ser_enrich__signal=scale_2.00__error=t_df_5
      - design=c4__enrichment=ser_enrich__signal=scale_2.00__error=t_df_5
      - design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_2.00__error=t_df_5
      - design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_2.00__error=t_df_5
  t-error-agg-scale-df-10:
    simulations:
      - design=hallmark__enrichment=ser_enrich__signal=scale_2.00__error=t_df_10
      - design=c4__enrichment=ser_enrich__signal=scale_2.00__error=t_df_10
      - design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_2.00__error=t_df_10
      - design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_2.00__error=t_df_10
  t-error-agg-scale-df-30:
    simulations:
      - design=hallmark__enrichment=ser_enrich__signal=scale_2.00__error=t_df_30
      - design=c4__enrichment=ser_enrich__signal=scale_2.00__error=t_df_30
      - design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_2.00__error=t_df_30
      - design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_2.00__error=t_df_30
  t-error-agg-scale-normal:
    simulations:
      - design=hallmark__enrichment=ser_enrich__signal=scale_2.00
      - design=c4__enrichment=ser_enrich__signal=scale_2.00
      - design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_2.00
      - design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_2.00
```

- [ ] **Step 2: Add 8 per-design supercollections and 2 aggregate supercollections to the `supercollections:` block**

Add at the end of the `supercollections:` section:

```yaml
  0010-hallmark-t-error-loc:
    collections:
      - name: design=hallmark__enrichment=ser_enrich__signal=loc_2.00
        alias: normal
      - name: design=hallmark__enrichment=ser_enrich__signal=loc_2.00__error=t_df_30
        alias: df=30
      - name: design=hallmark__enrichment=ser_enrich__signal=loc_2.00__error=t_df_10
        alias: df=10
      - name: design=hallmark__enrichment=ser_enrich__signal=loc_2.00__error=t_df_5
        alias: df=5
      - name: design=hallmark__enrichment=ser_enrich__signal=loc_2.00__error=t_df_3
        alias: df=3
    default_settings:
      thresholds:
        - 2.0
      min_log_bf: 2.0
      max_cs_size: 10000
      max_fdp: 0.5
    plots: &t_error_plots
      - settings:
          - minimal
        plot_type_groups:
          - standard
          - cs
  0010-hallmark-t-error-scale:
    collections:
      - name: design=hallmark__enrichment=ser_enrich__signal=scale_2.00
        alias: normal
      - name: design=hallmark__enrichment=ser_enrich__signal=scale_2.00__error=t_df_30
        alias: df=30
      - name: design=hallmark__enrichment=ser_enrich__signal=scale_2.00__error=t_df_10
        alias: df=10
      - name: design=hallmark__enrichment=ser_enrich__signal=scale_2.00__error=t_df_5
        alias: df=5
      - name: design=hallmark__enrichment=ser_enrich__signal=scale_2.00__error=t_df_3
        alias: df=3
    default_settings:
      thresholds:
        - 2.0
      min_log_bf: 2.0
      max_cs_size: 10000
      max_fdp: 0.5
    plots: *t_error_plots
  0010-c4-t-error-loc:
    collections:
      - name: design=c4__enrichment=ser_enrich__signal=loc_2.00
        alias: normal
      - name: design=c4__enrichment=ser_enrich__signal=loc_2.00__error=t_df_30
        alias: df=30
      - name: design=c4__enrichment=ser_enrich__signal=loc_2.00__error=t_df_10
        alias: df=10
      - name: design=c4__enrichment=ser_enrich__signal=loc_2.00__error=t_df_5
        alias: df=5
      - name: design=c4__enrichment=ser_enrich__signal=loc_2.00__error=t_df_3
        alias: df=3
    default_settings:
      thresholds:
        - 2.0
      min_log_bf: 2.0
      max_cs_size: 10000
      max_fdp: 0.5
    plots: *t_error_plots
  0010-c4-t-error-scale:
    collections:
      - name: design=c4__enrichment=ser_enrich__signal=scale_2.00
        alias: normal
      - name: design=c4__enrichment=ser_enrich__signal=scale_2.00__error=t_df_30
        alias: df=30
      - name: design=c4__enrichment=ser_enrich__signal=scale_2.00__error=t_df_10
        alias: df=10
      - name: design=c4__enrichment=ser_enrich__signal=scale_2.00__error=t_df_5
        alias: df=5
      - name: design=c4__enrichment=ser_enrich__signal=scale_2.00__error=t_df_3
        alias: df=3
    default_settings:
      thresholds:
        - 2.0
      min_log_bf: 2.0
      max_cs_size: 10000
      max_fdp: 0.5
    plots: *t_error_plots
  0010-gaussian-t-error-loc:
    collections:
      - name: design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_2.00
        alias: normal
      - name: design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_2.00__error=t_df_30
        alias: df=30
      - name: design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_2.00__error=t_df_10
        alias: df=10
      - name: design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_2.00__error=t_df_5
        alias: df=5
      - name: design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_2.00__error=t_df_3
        alias: df=3
    default_settings:
      thresholds:
        - 2.0
      min_log_bf: 2.0
      max_cs_size: 10000
      max_fdp: 0.5
    plots: *t_error_plots
  0010-gaussian-t-error-scale:
    collections:
      - name: design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_2.00
        alias: normal
      - name: design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_2.00__error=t_df_30
        alias: df=30
      - name: design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_2.00__error=t_df_10
        alias: df=10
      - name: design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_2.00__error=t_df_5
        alias: df=5
      - name: design=gaussian_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_2.00__error=t_df_3
        alias: df=3
    default_settings:
      thresholds:
        - 2.0
      min_log_bf: 2.0
      max_cs_size: 10000
      max_fdp: 0.5
    plots: *t_error_plots
  0010-uniform-t-error-loc:
    collections:
      - name: design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_2.00
        alias: normal
      - name: design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_2.00__error=t_df_30
        alias: df=30
      - name: design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_2.00__error=t_df_10
        alias: df=10
      - name: design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_2.00__error=t_df_5
        alias: df=5
      - name: design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=loc_2.00__error=t_df_3
        alias: df=3
    default_settings:
      thresholds:
        - 2.0
      min_log_bf: 2.0
      max_cs_size: 10000
      max_fdp: 0.5
    plots: *t_error_plots
  0010-uniform-t-error-scale:
    collections:
      - name: design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_2.00
        alias: normal
      - name: design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_2.00__error=t_df_30
        alias: df=30
      - name: design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_2.00__error=t_df_10
        alias: df=10
      - name: design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_2.00__error=t_df_5
        alias: df=5
      - name: design=uniform_markov_rho_0.90_n_features_100__enrichment=ser_enrich__signal=scale_2.00__error=t_df_3
        alias: df=3
    default_settings:
      thresholds:
        - 2.0
      min_log_bf: 2.0
      max_cs_size: 10000
      max_fdp: 0.5
    plots: *t_error_plots
  0010-all-designs-t-error-loc:
    collections:
      - name: t-error-agg-loc-normal
        alias: normal
      - name: t-error-agg-loc-df-30
        alias: df=30
      - name: t-error-agg-loc-df-10
        alias: df=10
      - name: t-error-agg-loc-df-5
        alias: df=5
      - name: t-error-agg-loc-df-3
        alias: df=3
    default_settings:
      thresholds:
        - 2.0
      min_log_bf: 2.0
      max_cs_size: 10000
      max_fdp: 0.5
    plots: *t_error_plots
  0010-all-designs-t-error-scale:
    collections:
      - name: t-error-agg-scale-normal
        alias: normal
      - name: t-error-agg-scale-df-30
        alias: df=30
      - name: t-error-agg-scale-df-10
        alias: df=10
      - name: t-error-agg-scale-df-5
        alias: df=5
      - name: t-error-agg-scale-df-3
        alias: df=3
    default_settings:
      thresholds:
        - 2.0
      min_log_bf: 2.0
      max_cs_size: 10000
      max_fdp: 0.5
    plots: *t_error_plots
```

- [ ] **Step 3: Verify YAML parses without error**

```bash
uv run python -c "import yaml; yaml.safe_load(open('notebooks/plot_config.yaml'))"
```

Expected: no output (clean parse).

- [ ] **Step 4: Verify Snakemake sees the new supercollections**

```bash
uv run python -c "
import yaml
cfg = yaml.safe_load(open('notebooks/plot_config.yaml'))
scs = [k for k in cfg['supercollections'] if k.startswith('0010-')]
print(f'{len(scs)} new supercollections:', *scs, sep='\n  ')
"
```

Expected: 10 supercollections listed.

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest tests/ -q
```

Expected: 43 passed (no regressions; YAML tasks have no new Python tests).

- [ ] **Step 6: Commit**

```bash
git add notebooks/plot_config.yaml
git commit -m "feat: add t-error misspecification supercollections to plot_config.yaml"
```

---

### Task 6: Regenerate manifest and verify Snakemake dry-run

The manifest encodes all registered simulations/batches. After adding new specs, it must be regenerated so Snakemake knows about the new batches.

**Files:**
- Modify: `results/manifest.json` (regenerated, not hand-edited)

- [ ] **Step 1: Check how manifest is regenerated**

```bash
grep -n "manifest" Snakefile twogroup_experiments.snk | grep "rule\|output\|shell" | head -10
```

Find the rule that writes `results/manifest.json` and run it.

- [ ] **Step 2: Regenerate manifest**

```bash
snakemake results/manifest.json --cores 1
```

Or if there's a dedicated script:
```bash
uv run python -c "
from config import manifest_dict
import json
from pathlib import Path
Path('results/manifest.json').write_text(json.dumps(manifest_dict(), indent=2))
print('done')
"
```

- [ ] **Step 3: Snakemake dry-run for one new supercollection**

```bash
snakemake --snakefile Snakefile --config supercollection=0010-c4-t-error-loc -n 2>&1 | tail -20
```

Expected: dry-run lists fit jobs for t-error batches and plot jobs for `minimal` setting.

- [ ] **Step 4: Commit manifest**

```bash
git add results/manifest.json
git commit -m "chore: regenerate manifest with t-error simulation specs"
```
