# YAML-driven Experiment Config Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the registry-driven config (`config.py`/`config_builders.py`/`config_registry.py` + `plot_configs/`) with a declarative YAML config under `experiments/`, loaded by a thin `experiments/loader.py`, driving simulations, methods, atomic reductions, and per-supercollection analyses through two generic snakemake rules.

**Architecture:** A `library.yaml` defines named `function`+args entries (designs, enrichments, signals, errors, methods, reductions, analyses, analysis_groups). Numbered `experiments/NNN_*.yaml` files define supercollections (collections via `template`/`over` expansion + methods + outputs). The loader builds `core.SimulationSpec`/`MethodSpec`/`BatchSpec` objects, emits the same hash-keyed manifest the snakefile consumes, and resolves the collection/analysis structures. Reductions are fit-level (atomic), content-addressed under `by_batch/`; collections are logical, concatenated in-memory at analysis time.

**Tech Stack:** Python 3, polars, numpy, PyYAML, snakemake, matplotlib, gibss. Tests with pytest.

## Global Constraints

- Run all Python via `uv run` (e.g. `uv run pytest`, `uv run python`) — never bare `python`/`pip`.
- Clean break on hashes; no coexistence — done on branch `yaml-experiment-config`. `config.py`, `config_builders.py`, `config_registry.py`, `plot_configs/` are deleted by end of plan.
- All `function` names in YAML resolve to importable top-level callables via `core._callable_path` (`module:qualname`, no lambdas/locals).
- Library entry keys (design/enrichment/signal/error/method/reduction/analysis names) MUST NOT contain `__` (the over-key join separator). Validate at load.
- Generated method name = `{base}__{key}={value}` per `over` key in declared order; float values formatted `f"{x:.2f}"`, ints bare, `None`→`none`, bool→`true`/`false`, str verbatim (use `config_builders.format_float` relocated to loader).
- `defaults.base_seed = 20260501`, `replicates_per_batch = 50`, `n_batches = 1`.

---

## File Structure

- `experiments/loader.py` — **new.** Pure-Python config loader (no snakemake import). Library parsing, distribution/simulation/method resolution, collection/supercollection expansion, manifest emission, reduction/analysis resolution, `load_sc_bundle`, `method_metadata`.
- `experiments/library.yaml` — **new.** Shared library.
- `experiments/003_loc_snr.yaml`, `experiments/000_t_errors.yaml` — **new.** Proof-migration supercollections.
- `core.py` — **modify.** `MethodSpec` collapse + 4 `run_*` entrypoints; drop `summarize_method_spec`; delete dead `MethodSpec` constants.
- `utils.py` — **modify.** `fit_batch_method` uses `run_method_spec` only; `manifest_dict` repoint; relocate `batch_specs_for_simulation`.
- `plot_ready.py` — **modify.** Reduction builders refactored to atomic signatures.
- `generate_plots.py` — **modify.** Renderers become analysis functions; `_foreground_methods` → name membership; entry points use loader.
- `viz_utils.py` — **modify.** `method_metadata_from_method_spec_json` reworked for new method-spec shape (or replaced by loader-built metadata).
- `twogroup_experiments.snk` — **modify.** Two generic rules + `sample_metadata` output + loader-backed targets.
- `manifest_cache.py` — **modify.** Watch `experiments/`, import loader.
- `tests/test_loader.py` — **new.** Loader unit tests.
- Delete: `config.py`, `config_builders.py`, `config_registry.py`, `plot_configs/`.

---

## Phase A — core.py MethodSpec collapse

### Task A1: Branch + `run_*` entrypoints in core.py

**Files:**
- Modify: `core.py` (after `summarize_linear_method`, ~line 455)
- Test: `tests/test_core_run_methods.py` (create)

**Interfaces:**
- Produces: `run_cox_method(simulation, **kwargs) -> dict`, `run_logistic_method(...)`, `run_twogroup_method(...)`, `run_linear_method(...)`. Each returns the summary-row dict produced by the matching `summarize_*`.

- [ ] **Step 1: Create the branch**

```bash
cd /Users/ktayeb/research/gibss-experiments/twogroup_experiments
git checkout -b yaml-experiment-config
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_core_run_methods.py
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import core


def _tiny_simulation():
    from core import SimulationSpec, simulate
    from functools import partial
    from gibss.distributions import Normal, PointMass
    spec = SimulationSpec(
        name="tiny",
        design_sampler=partial(core.gaussian_markov_X, n=30, p=8, rho=0.5),
        effect_sampler=partial(core.uniform_single_effect, causal_effect=2.0),
        intercept=-1.0,
        f0=PointMass(0.0),
        f1=Normal(loc=2.0, scale=0.1, estimate_loc=False, estimate_scale=False),
        base_seed=1,
    )
    return simulate(spec, 0)


def test_run_cox_method_returns_summary_row():
    sim = _tiny_simulation()
    row = core.run_cox_method(sim, threshold=None, time_sign=1.0, L=1)
    assert row["method"] is None or "single_effects" in row  # row is the summarize dict
    assert "single_effects" in row and "fit_summary" in row
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_core_run_methods.py -v`
Expected: FAIL with `AttributeError: module 'core' has no attribute 'run_cox_method'`

- [ ] **Step 4: Add the four entrypoints in core.py**

Insert after `summarize_linear_method` (around line 455):

```python
def run_cox_method(simulation: TwoGroupSimulation, **kwargs) -> dict[str, Any]:
    return summarize_cox_method(fit_cox_method(simulation, **kwargs), simulation, **kwargs)


def run_logistic_method(simulation: TwoGroupSimulation, **kwargs) -> dict[str, Any]:
    return summarize_logistic_method(fit_logistic_method(simulation, **kwargs), simulation, **kwargs)


def run_twogroup_method(simulation: TwoGroupSimulation, **kwargs) -> dict[str, Any]:
    return summarize_twogroup_method(fit_twogroup_method(simulation, **kwargs), simulation, **kwargs)


def run_linear_method(simulation: TwoGroupSimulation, **kwargs) -> dict[str, Any]:
    return summarize_linear_method(fit_linear_method(simulation, **kwargs), simulation, **kwargs)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_core_run_methods.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add core.py tests/test_core_run_methods.py
git commit -m "feat(core): add run_* method entrypoints"
```

### Task A2: Collapse MethodSpec to single `function`

**Files:**
- Modify: `core.py:33-38` (`MethodSpec`), `core.py:561-575` (`run_method_spec`/`summarize_method_spec`), `core.py:457-558` (delete dead constants)
- Modify: `utils.py:13-24` (imports), `utils.py:109-135` (`fit_batch_method`)
- Test: `tests/test_core_run_methods.py` (extend)

**Interfaces:**
- Consumes: `run_cox_method` etc. from A1.
- Produces: `MethodSpec(name: str, function: Any, kwargs: dict)`. `run_method_spec(method_spec, simulation) -> dict` returns `{"method": method_spec.name, **method_spec.function(simulation, **method_spec.kwargs)}`. `summarize_method_spec` removed.

- [ ] **Step 1: Write the failing test (append to tests/test_core_run_methods.py)**

```python
def test_method_spec_single_function_and_run_method_spec():
    sim = _tiny_simulation()
    spec = core.MethodSpec(name="cox_heavy__L=1", function=core.run_cox_method,
                           kwargs={"threshold": None, "time_sign": 1.0, "L": 1})
    row = core.run_method_spec(spec, sim)
    assert row["method"] == "cox_heavy__L=1"
    assert "single_effects" in row
    assert not hasattr(spec, "fit_function")
    assert not hasattr(spec, "summarize_function")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_core_run_methods.py::test_method_spec_single_function_and_run_method_spec -v`
Expected: FAIL (`MethodSpec.__init__() got an unexpected keyword argument 'function'`)

- [ ] **Step 3: Rewrite MethodSpec (core.py:33-38)**

```python
@dataclass(frozen=True)
class MethodSpec:
    name: str
    function: Any
    kwargs: dict[str, Any]
```

- [ ] **Step 4: Rewrite run_method_spec, delete summarize_method_spec (core.py:561-575)**

Replace both functions with:

```python
def run_method_spec(method_spec: MethodSpec, simulation: TwoGroupSimulation) -> dict[str, Any]:
    return {"method": method_spec.name, **method_spec.function(simulation, **method_spec.kwargs)}
```

- [ ] **Step 5: Delete dead module-level MethodSpec constants**

Delete `core.py:457-558` (the block from `TWOGROUP_DEFAULT_F1INIT = ...` through `COX_LIGHT_THRESHOLD_2_0 = MethodSpec(...)` — these constants and the `_TG_ITER_KWARGS` near them are unused once the library replaces them). Keep `TWOGROUP_DEFAULT_F1INIT`/`TWOGROUP_SCALE_FAM_F1INIT`/`TWOGROUP_LOC_FAM_F1INIT` only if referenced elsewhere; verify with `grep -rn "TWOGROUP_DEFAULT_F1INIT\|COX_HEAVY\|LOGISTIC_ORACLE\|TWOGROUP_ORACLE\b" --include=*.py .` and delete those with zero hits outside `core.py`.

- [ ] **Step 6: Update utils.py imports and fit_batch_method**

`utils.py:13-24` — remove `summarize_method_spec` from the import list.
`utils.py:127-134` — replace the loop body:

```python
        simulation = simulate(simulation_spec, replicate)
        rows.append({"replicate": replicate, **run_method_spec(method_spec, simulation)})
```

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/test_core_run_methods.py -v`
Expected: PASS (both tests)

- [ ] **Step 8: Commit**

```bash
git add core.py utils.py tests/test_core_run_methods.py
git commit -m "feat(core): collapse MethodSpec to single function"
```

---

## Phase B — Loader: library, distributions, simulations, methods, manifest

### Task B1: Library parsing + callable/distribution resolution + format_float

**Files:**
- Create: `experiments/loader.py`, `experiments/__init__.py` (empty)
- Test: `tests/test_loader.py` (create)

**Interfaces:**
- Produces:
  - `format_float(x: float) -> str` = `f"{float(x):.2f}"`.
  - `load_library(experiments_dir: Path | None = None) -> dict` — parsed `library.yaml` as a plain dict with keys `defaults, designs, enrichments, signals, errors, methods, reductions, analyses, analysis_groups`.
  - `resolve_callable(name: str) -> Any` — `getattr(core, name)`, raising `KeyError` if missing.
  - `resolve_distribution(node: dict) -> Any` — single-key map `{TypeName: kwargs}` → instance from `gibss.distributions`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_loader.py
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from experiments import loader
from gibss.distributions import Normal, PointMass


def test_format_float():
    assert loader.format_float(2.0) == "2.00"
    assert loader.format_float(0.5) == "0.50"


def test_resolve_distribution_normal_and_pointmass():
    n = loader.resolve_distribution({"Normal": {"loc": 2.0, "scale": 0.1,
                                                 "estimate_loc": False, "estimate_scale": False}})
    assert isinstance(n, Normal) and n.loc == 2.0 and n.scale == 0.1
    p = loader.resolve_distribution({"PointMass": {"value": 0.0}})
    assert isinstance(p, PointMass) and p.value == 0.0


def test_resolve_callable_resolves_core_functions():
    assert loader.resolve_callable("run_cox_method").__name__ == "run_cox_method"
    with pytest.raises(KeyError):
        loader.resolve_callable("does_not_exist")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_loader.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'experiments.loader'`)

- [ ] **Step 3: Create experiments/__init__.py and loader.py**

```python
# experiments/__init__.py
```

```python
# experiments/loader.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

import core
from gibss import distributions as _distributions

EXPERIMENTS_DIR = Path(__file__).resolve().parent


def format_float(value: float) -> str:
    return f"{float(value):.2f}"


def resolve_callable(name: str) -> Any:
    if not hasattr(core, name):
        raise KeyError(f"Unknown callable in core: {name!r}")
    return getattr(core, name)


def resolve_distribution(node: dict[str, Any]) -> Any:
    if not isinstance(node, dict) or len(node) != 1:
        raise ValueError(f"Distribution node must be a single-key map, got {node!r}")
    (type_name, ctor_kwargs), = node.items()
    if not hasattr(_distributions, type_name):
        raise KeyError(f"Unknown distribution type: {type_name!r}")
    return getattr(_distributions, type_name)(**(ctor_kwargs or {}))


def load_library(experiments_dir: Path | None = None) -> dict[str, Any]:
    base = Path(experiments_dir) if experiments_dir is not None else EXPERIMENTS_DIR
    data = yaml.safe_load((base / "library.yaml").read_text(encoding="utf-8")) or {}
    for section in ("defaults", "designs", "enrichments", "signals", "errors",
                    "methods", "reductions", "analyses", "analysis_groups"):
        data.setdefault(section, {})
    return data
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_loader.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/__init__.py experiments/loader.py tests/test_loader.py
git commit -m "feat(loader): library parsing + distribution/callable resolution"
```

### Task B2: Simulation resolution

**Files:**
- Modify: `experiments/loader.py`
- Test: `tests/test_loader.py` (append)

**Interfaces:**
- Consumes: `load_library`, `resolve_callable`, `resolve_distribution`.
- Produces: `resolve_simulation(library, design, enrichment, signal, error) -> (core.SimulationSpec, str)` returning the spec and its auto-derived human name `{design}__{enrichment}__{signal}` (+`__{error}` when `error != "gaussian"`). Builds `design_sampler=partial(fn, **arguments)`, `effect_sampler=partial(fn, **arguments)`, `intercept` from enrichment, `f0`/`f1` from signal, `error_sampler` (None for `gaussian`), `base_seed` from defaults.

- [ ] **Step 1: Write the failing test (append)**

```python
def _library_for_tests():
    return {
        "defaults": {"base_seed": 20260501, "replicates_per_batch": 50, "n_batches": 1},
        "designs": {"gaussian_p100": {"function": "gaussian_markov_X",
                                      "arguments": {"n": 500, "p": 100, "rho": 0.9}}},
        "enrichments": {"ser_b2": {"function": "uniform_single_effect",
                                   "arguments": {"causal_effect": 2.0}, "intercept": -2.0}},
        "signals": {"loc_2.0": {"f0": {"PointMass": {"value": 0.0}},
                                "f1": {"Normal": {"loc": 2.0, "scale": 0.1,
                                                  "estimate_loc": False, "estimate_scale": False}}}},
        "errors": {"gaussian": None, "t_df_5": {"function": "t_error_sampler", "arguments": {"df": 5}}},
        "methods": {}, "reductions": {}, "analyses": {}, "analysis_groups": {},
    }


def test_resolve_simulation_builds_spec_and_name():
    lib = _library_for_tests()
    spec, name = loader.resolve_simulation(lib, "gaussian_p100", "ser_b2", "loc_2.0", "gaussian")
    assert name == "gaussian_p100__ser_b2__loc_2.0"
    assert spec.intercept == -2.0
    assert spec.base_seed == 20260501
    assert spec.error_sampler is None
    assert spec.f1.loc == 2.0
    # design/effect samplers are functools.partial of the right callable
    assert spec.design_sampler.func.__name__ == "gaussian_markov_X"
    assert spec.design_sampler.keywords == {"n": 500, "p": 100, "rho": 0.9}


def test_resolve_simulation_nongaussian_error_in_name():
    lib = _library_for_tests()
    spec, name = loader.resolve_simulation(lib, "gaussian_p100", "ser_b2", "loc_2.0", "t_df_5")
    assert name == "gaussian_p100__ser_b2__loc_2.0__t_df_5"
    assert spec.error_sampler is not None
    assert spec.error_sampler.keywords == {"df": 5}
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_loader.py -k resolve_simulation -v`
Expected: FAIL (`AttributeError: module ... has no attribute 'resolve_simulation'`)

- [ ] **Step 3: Implement resolve_simulation (append to loader.py)**

```python
from functools import partial


def _partial_from_entry(entry: dict[str, Any]):
    fn = resolve_callable(entry["function"])
    return partial(fn, **(entry.get("arguments") or {}))


def resolve_simulation(library: dict[str, Any], design: str, enrichment: str,
                       signal: str, error: str) -> tuple[core.SimulationSpec, str]:
    design_entry = library["designs"][design]
    enrich_entry = library["enrichments"][enrichment]
    signal_entry = library["signals"][signal]
    error_entry = library["errors"][error]

    name = f"{design}__{enrichment}__{signal}"
    if error != "gaussian":
        name = f"{name}__{error}"

    error_sampler = None if error_entry is None else _partial_from_entry(error_entry)
    spec = core.SimulationSpec(
        name=name,
        design_sampler=_partial_from_entry(design_entry),
        effect_sampler=_partial_from_entry(enrich_entry),
        intercept=float(enrich_entry["intercept"]),
        f0=resolve_distribution(signal_entry["f0"]),
        f1=resolve_distribution(signal_entry["f1"]),
        base_seed=int(library["defaults"]["base_seed"]),
        error_sampler=error_sampler,
    )
    return spec, name
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_loader.py -k resolve_simulation -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/loader.py tests/test_loader.py
git commit -m "feat(loader): resolve_simulation -> SimulationSpec + name"
```

### Task B3: Method resolution with template/over expansion

**Files:**
- Modify: `experiments/loader.py`
- Test: `tests/test_loader.py` (append)

**Interfaces:**
- Consumes: `resolve_callable`, `resolve_distribution`, `format_float`.
- Produces:
  - `format_over_value(value) -> str` (float→`format_float`, int→`str(int)`, bool→`true`/`false`, None→`none`, str→str).
  - `resolve_distributions_in_kwargs(kwargs) -> kwargs` — replaces any single-key dist-map value with the resolved distribution (recursively shallow over kwargs values).
  - `expand_method(base_name, entry) -> list[core.MethodSpec]` — cartesian over `entry["over"]` (declared order), kwargs = `{**template, **over_combo}` with distributions resolved; name = `{base_name}__{k}={format_over_value(v)}` joined.

- [ ] **Step 1: Write the failing test (append)**

```python
def test_expand_method_cartesian_names_and_kwargs():
    entry = {"function": "run_cox_method", "template": {"time_sign": -1.0},
             "over": {"threshold": [0.0, 2.0], "L": [1, 5]}}
    specs = loader.expand_method("cox_light", entry)
    names = [s.name for s in specs]
    assert names == [
        "cox_light__threshold=0.00__L=1", "cox_light__threshold=0.00__L=5",
        "cox_light__threshold=2.00__L=1", "cox_light__threshold=2.00__L=5",
    ]
    s = specs[2]
    assert s.function.__name__ == "run_cox_method"
    assert s.kwargs == {"time_sign": -1.0, "threshold": 2.0, "L": 1}


def test_expand_method_resolves_distribution_kwargs():
    entry = {"function": "run_twogroup_method",
             "template": {"f1": {"Normal": {"loc": 0.0, "scale": 1.0,
                                            "estimate_loc": True, "estimate_scale": True}}},
             "over": {"L": [1]}}
    specs = loader.expand_method("twogroup", entry)
    assert specs[0].name == "twogroup__L=1"
    from gibss.distributions import Normal
    assert isinstance(specs[0].kwargs["f1"], Normal)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_loader.py -k expand_method -v`
Expected: FAIL (`AttributeError ... 'expand_method'`)

- [ ] **Step 3: Implement (append to loader.py)**

```python
import itertools


def format_over_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "none"
    if isinstance(value, float):
        return format_float(value)
    if isinstance(value, int):
        return str(value)
    return str(value)


def _is_distribution_node(value: Any) -> bool:
    return (isinstance(value, dict) and len(value) == 1
            and next(iter(value)) in ("Normal", "PointMass", "NormalMixture"))


def resolve_distributions_in_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {k: (resolve_distribution(v) if _is_distribution_node(v) else v)
            for k, v in kwargs.items()}


def expand_method(base_name: str, entry: dict[str, Any]) -> list[core.MethodSpec]:
    if "__" in base_name:
        raise ValueError(f"Method base name must not contain '__': {base_name!r}")
    fn = resolve_callable(entry["function"])
    template = dict(entry.get("template") or {})
    over = entry.get("over") or {"_dummy": [None]}
    keys = list(over.keys())
    specs: list[core.MethodSpec] = []
    for combo in itertools.product(*(over[k] for k in keys)):
        over_kwargs = {k: v for k, v in zip(keys, combo) if k != "_dummy"}
        suffix = "".join(f"__{k}={format_over_value(v)}" for k, v in over_kwargs.items())
        kwargs = resolve_distributions_in_kwargs({**template, **over_kwargs})
        specs.append(core.MethodSpec(name=f"{base_name}{suffix}", function=fn, kwargs=kwargs))
    return specs
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_loader.py -k expand_method -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/loader.py tests/test_loader.py
git commit -m "feat(loader): method template/over expansion"
```

### Task B4: Batch specs + manifest emission

**Files:**
- Modify: `experiments/loader.py`
- Test: `tests/test_loader.py` (append)

**Interfaces:**
- Consumes: `resolve_simulation`, `expand_method`, `load_library`, `core.dehydrate_hashed`, `core.HASH_KEY`, `utils.BatchSpec`.
- Produces:
  - `batch_specs_for_simulation(spec, *, replicates_per_batch, n_batches) -> list[utils.BatchSpec]` (relocated from `config_builders`).
  - `all_simulations(library, supercollections) -> dict[name, SimulationSpec]` and `all_methods(library, supercollections) -> dict[name, MethodSpec]` (dedup by name; see C-phase for supercollection plumbing — for now expose `library_methods(library) -> dict[name, MethodSpec]`).
  - `manifest_dict(library, simulations, methods) -> dict` with `{"batches": {...}, "method_specs": {...}}` keyed by `core.dehydrate_hashed(...)[HASH_KEY]`, matching the node shape the snakefile rehydrates.

- [ ] **Step 1: Write the failing test (append)**

```python
def test_library_methods_expands_all_entries():
    lib = _library_for_tests()
    lib["methods"] = {
        "cox_heavy": {"function": "run_cox_method",
                      "template": {"threshold": None, "time_sign": 1.0}, "over": {"L": [1]}},
        "cox_light": {"function": "run_cox_method", "template": {"time_sign": -1.0},
                      "over": {"threshold": [2.0], "L": [1]}},
    }
    methods = loader.library_methods(lib)
    assert set(methods) == {"cox_heavy__L=1", "cox_light__threshold=2.00__L=1"}


def test_manifest_dict_shape():
    lib = _library_for_tests()
    spec, name = loader.resolve_simulation(lib, "gaussian_p100", "ser_b2", "loc_2.0", "gaussian")
    method = loader.expand_method("cox_heavy",
        {"function": "run_cox_method", "template": {"threshold": None, "time_sign": 1.0},
         "over": {"L": [1]}})[0]
    manifest = loader.manifest_dict(lib, {name: spec}, {method.name: method})
    assert set(manifest) == {"batches", "method_specs"}
    (batch_hash, batch_node), = manifest["batches"].items()
    assert batch_node["__spec_hash__"] == batch_hash
    assert batch_node["simulation_spec"]["__spec_hash__"]
    assert list(batch_node["replicates"]) == list(range(50))
    (method_hash, method_node), = manifest["method_specs"].items()
    assert method_node["__spec_hash__"] == method_hash
    assert method_node["fields"]["name"] == "cox_heavy__L=1"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_loader.py -k "library_methods or manifest_dict_shape" -v`
Expected: FAIL

- [ ] **Step 3: Implement (append to loader.py)**

```python
from core import HASH_KEY, dehydrate_hashed
from utils import BatchSpec


def batch_specs_for_simulation(spec, *, replicates_per_batch: int, n_batches: int) -> list[BatchSpec]:
    return [
        BatchSpec(
            name=f"{spec.name}__batch{i}",
            simulation_spec=spec,
            replicates=tuple(range(i * replicates_per_batch, (i + 1) * replicates_per_batch)),
        )
        for i in range(n_batches)
    ]


def library_methods(library: dict[str, Any]) -> dict[str, core.MethodSpec]:
    out: dict[str, core.MethodSpec] = {}
    for base, entry in library["methods"].items():
        for spec in expand_method(base, entry):
            out[spec.name] = spec
    return out


def manifest_dict(library: dict[str, Any], simulations: dict[str, core.SimulationSpec],
                  methods: dict[str, core.MethodSpec]) -> dict[str, Any]:
    defaults = library["defaults"]
    batches: dict[str, Any] = {}
    for spec in simulations.values():
        for batch in batch_specs_for_simulation(
            spec,
            replicates_per_batch=int(defaults["replicates_per_batch"]),
            n_batches=int(defaults["n_batches"]),
        ):
            sim_node = dehydrate_hashed(batch.simulation_spec)
            node = {
                "name": batch.name,
                "simulation_spec": sim_node,
                "replicates": list(batch.replicates),
            }
            node[HASH_KEY] = dehydrate_hashed(batch)[HASH_KEY]
            batches[node[HASH_KEY]] = node
    method_specs: dict[str, Any] = {}
    for spec in methods.values():
        node = dehydrate_hashed(spec)
        method_specs[node[HASH_KEY]] = node
    return {"batches": batches, "method_specs": method_specs}
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_loader.py -k "library_methods or manifest_dict_shape" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/loader.py tests/test_loader.py
git commit -m "feat(loader): batch specs + manifest emission"
```

---

## Phase C — Loader: collections, supercollections, reductions, analyses

### Task C1: Collection expansion (template/over)

**Files:**
- Modify: `experiments/loader.py`
- Test: `tests/test_loader.py` (append)

**Interfaces:**
- Consumes: `resolve_simulation`.
- Produces: `expand_collections(library, sc_name, collections_entry) -> list[dict]`. Each dict: `{"name": str, "alias": str, "simulations": [SimulationSpec, ...]}`. `template` fields given as lists product into the within-collection member simulations; `over` fields product into one collection per combo. Collection name = `{sc_name}__{over-key}={over-value}` (over-values are library keys, verbatim); alias defaults to the joined over-value(s). Supports a single `{template, over}` block, a list of such blocks, and a bare `{name, simulations: [{design,enrichment,signal,error}, ...]}` block.

- [ ] **Step 1: Write the failing test (append)**

```python
def test_expand_collections_within_and_over():
    lib = _library_for_tests()
    lib["signals"]["loc_1.0"] = {"f0": {"PointMass": {"value": 0.0}},
        "f1": {"Normal": {"loc": 1.0, "scale": 0.1, "estimate_loc": False, "estimate_scale": False}}}
    lib["enrichments"]["null_b0"] = {"function": "uniform_single_effect",
                                     "arguments": {"causal_effect": 0.0}, "intercept": -2.0}
    entry = {"template": {"design": "gaussian_p100",
                          "enrichment": ["ser_b2", "null_b0"], "error": "gaussian"},
             "over": {"signal": ["loc_1.0", "loc_2.0"]}}
    colls = loader.expand_collections(lib, "sc", entry)
    assert [c["name"] for c in colls] == ["sc__signal=loc_1.0", "sc__signal=loc_2.0"]
    assert [c["alias"] for c in colls] == ["loc_1.0", "loc_2.0"]
    # within-collection: ser + null pair
    assert {s.name for s in colls[0]["simulations"]} == {
        "gaussian_p100__ser_b2__loc_1.0", "gaussian_p100__null_b0__loc_1.0"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_loader.py -k expand_collections -v`
Expected: FAIL

- [ ] **Step 3: Implement (append to loader.py)**

```python
_SIM_FIELDS = ("design", "enrichment", "signal", "error")


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else [value]


def _expand_block(library: dict[str, Any], sc_name: str, block: dict[str, Any]) -> list[dict]:
    if "simulations" in block:  # explicit one-off
        sims = [resolve_simulation(library, s["design"], s["enrichment"],
                                   s["signal"], s.get("error", "gaussian"))[0]
                for s in block["simulations"]]
        return [{"name": block["name"], "alias": block.get("alias", block["name"]), "simulations": sims}]

    template = dict(block["template"])
    over = block.get("over") or {}
    over_keys = list(over.keys())
    results: list[dict] = []
    for combo in itertools.product(*(over[k] for k in over_keys)) if over_keys else [()]:
        over_map = dict(zip(over_keys, combo))
        fields = {**template, **over_map}
        # within-collection product over any list-valued template field
        member_lists = {f: _as_list(fields.get(f, "gaussian" if f == "error" else None))
                        for f in _SIM_FIELDS}
        sims = []
        for d, e, s, err in itertools.product(member_lists["design"], member_lists["enrichment"],
                                              member_lists["signal"], member_lists["error"]):
            sims.append(resolve_simulation(library, d, e, s, err)[0])
        suffix = "".join(f"__{k}={over_map[k]}" for k in over_keys)
        name = f"{sc_name}{suffix}" if over_keys else sc_name
        alias = block.get("alias") or "__".join(str(over_map[k]) for k in over_keys) or sc_name
        results.append({"name": name, "alias": alias, "simulations": sims})
    return results


def expand_collections(library: dict[str, Any], sc_name: str, collections_entry: Any) -> list[dict]:
    blocks = collections_entry if isinstance(collections_entry, list) else [collections_entry]
    out: list[dict] = []
    for block in blocks:
        out.extend(_expand_block(library, sc_name, block))
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_loader.py -k expand_collections -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/loader.py tests/test_loader.py
git commit -m "feat(loader): collection template/over expansion"
```

### Task C2: Supercollection load + analysis flattening + global accessors

**Files:**
- Modify: `experiments/loader.py`
- Test: `tests/test_loader.py` (append), `tests/fixtures/experiments/` (create a tiny fixture experiments dir)

**Interfaces:**
- Consumes: `load_library`, `expand_collections`, `library_methods`, `expand_method`.
- Produces:
  - `load_config(experiments_dir=None) -> dict` with `{"library": ..., "supercollections": {name: raw_sc}}` (globs `experiments/*.yaml` except `library.yaml`, merges `supercollections`).
  - `resolve_methods_for_sc(library, sc) -> dict[name, MethodSpec]` — selects library method names referenced in `sc["methods"]`, plus inline method defs.
  - `supercollection_collections(library, sc_name, sc) -> list[dict]` (calls `expand_collections`).
  - `flatten_analyses(library, analyses_list) -> list[str]` — expand group shortcuts (in `library["analysis_groups"]`) + one-offs, dedup preserving order; raise if a name is neither group nor analysis, or if group/analysis names collide.
  - `resolve_sc_analyses(config, sc_name) -> list[tuple[str, str]]` — `(analysis_name, args_name)` pairs over the SC's outputs.
  - `all_simulations(config) -> dict[name, SimulationSpec]`, `all_methods(config) -> dict[name, MethodSpec]` (union across SCs, deduped by name).

- [ ] **Step 1: Create the fixture experiments dir**

```yaml
# tests/fixtures/experiments/library.yaml
defaults: {base_seed: 20260501, replicates_per_batch: 2, n_batches: 1}
designs:
  gaussian_p8: {function: gaussian_markov_X, arguments: {n: 30, p: 8, rho: 0.5}}
enrichments:
  ser_b2:  {function: uniform_single_effect, arguments: {causal_effect: 2.0}, intercept: -2.0}
  null_b0: {function: uniform_single_effect, arguments: {causal_effect: 0.0}, intercept: -2.0}
signals:
  loc_2.0: {f0: {PointMass: {value: 0.0}},
            f1: {Normal: {loc: 2.0, scale: 0.1, estimate_loc: false, estimate_scale: false}}}
errors:
  gaussian: null
methods:
  cox_heavy: {function: run_cox_method, template: {threshold: null, time_sign: 1.0}, over: {L: [1]}}
  twogroup:  {function: run_twogroup_method,
              template: {f1: {Normal: {loc: 0.0, scale: 1.0, estimate_loc: true, estimate_scale: true}},
                         n_null_iter: 2, n_intercept_iter: 2},
              over: {L: [1]}}
reductions:
  pip: {function: build_pip_plot_data, needs: {fits: true, sample_metadata: true}}
analyses:
  pip_calibration: {function: render_pip_calibration, requires: [pip]}
  agg_pip_calibration: {function: render_agg_pip_calibration, requires: [pip]}
analysis_groups:
  pip: [pip_calibration, agg_pip_calibration]
```

```yaml
# tests/fixtures/experiments/900_fixture.yaml
supercollections:
  fixture-sc:
    collections:
      template: {design: gaussian_p8, enrichment: [ser_b2, null_b0], error: gaussian}
      over: {signal: [loc_2.0]}
    methods: [cox_heavy__L=1, twogroup__L=1]
    default_args: {max_fdp: 0.5}
    outputs:
      - name: minimal
        method_filter: [twogroup__L=1]
        analyses: [pip]
```

- [ ] **Step 2: Write the failing test (append)**

```python
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "experiments"


def test_load_config_and_accessors():
    cfg = loader.load_config(FIXTURE_DIR)
    assert "fixture-sc" in cfg["supercollections"]
    sims = loader.all_simulations(cfg)
    assert {"gaussian_p8__ser_b2__loc_2.0", "gaussian_p8__null_b0__loc_2.0"} <= set(sims)
    methods = loader.all_methods(cfg)
    assert {"cox_heavy__L=1", "twogroup__L=1"} == set(methods)


def test_flatten_analyses_expands_groups_and_dedups():
    cfg = loader.load_config(FIXTURE_DIR)
    flat = loader.flatten_analyses(cfg["library"], ["pip", "pip_calibration"])
    assert flat == ["pip_calibration", "agg_pip_calibration"]


def test_resolve_sc_analyses_pairs():
    cfg = loader.load_config(FIXTURE_DIR)
    pairs = loader.resolve_sc_analyses(cfg, "fixture-sc")
    assert ("pip_calibration", "minimal") in pairs
    assert ("agg_pip_calibration", "minimal") in pairs
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest tests/test_loader.py -k "load_config or flatten_analyses or resolve_sc_analyses" -v`
Expected: FAIL

- [ ] **Step 4: Implement (append to loader.py)**

```python
def load_config(experiments_dir: Path | None = None) -> dict[str, Any]:
    base = Path(experiments_dir) if experiments_dir is not None else EXPERIMENTS_DIR
    library = load_library(base)
    supercollections: dict[str, Any] = {}
    for path in sorted(base.glob("*.yaml")):
        if path.name == "library.yaml":
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        supercollections.update(data.get("supercollections", {}) or {})
    return {"library": library, "supercollections": supercollections}


def resolve_methods_for_sc(library: dict[str, Any], sc: dict[str, Any]) -> dict[str, core.MethodSpec]:
    lib_methods = library_methods(library)
    out: dict[str, core.MethodSpec] = {}
    for item in sc.get("methods", []):
        if isinstance(item, str):
            out[item] = lib_methods[item]
        else:  # inline def: single-key map base -> entry
            (base, entry), = item.items()
            for spec in expand_method(base, entry):
                out[spec.name] = spec
    return out


def supercollection_collections(library, sc_name, sc) -> list[dict]:
    return expand_collections(library, sc_name, sc["collections"])


def flatten_analyses(library: dict[str, Any], analyses_list: list[str]) -> list[str]:
    groups = library["analysis_groups"]
    analyses = library["analyses"]
    overlap = set(groups) & set(analyses)
    if overlap:
        raise ValueError(f"Analysis/group name collision: {sorted(overlap)}")
    out: list[str] = []
    for item in analyses_list:
        names = groups[item] if item in groups else [item]
        for n in names:
            if n not in analyses:
                raise KeyError(f"Unknown analysis: {n!r}")
            if n not in out:
                out.append(n)
    return out


def resolve_sc_analyses(config: dict[str, Any], sc_name: str) -> list[tuple[str, str]]:
    library = config["library"]
    sc = config["supercollections"][sc_name]
    seen: dict[tuple[str, str], None] = {}
    for output in sc.get("outputs", []):
        for analysis in flatten_analyses(library, output.get("analyses", [])):
            seen[(analysis, output["name"])] = None
    return list(seen.keys())


def all_simulations(config: dict[str, Any]) -> dict[str, core.SimulationSpec]:
    library = config["library"]
    out: dict[str, core.SimulationSpec] = {}
    for sc_name, sc in config["supercollections"].items():
        for coll in supercollection_collections(library, sc_name, sc):
            for spec in coll["simulations"]:
                out[spec.name] = spec
    return out


def all_methods(config: dict[str, Any]) -> dict[str, core.MethodSpec]:
    out: dict[str, core.MethodSpec] = {}
    for sc in config["supercollections"].values():
        out.update(resolve_methods_for_sc(config["library"], sc))
    return out
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/test_loader.py -k "load_config or flatten_analyses or resolve_sc_analyses" -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add experiments/loader.py tests/test_loader.py tests/fixtures/experiments/
git commit -m "feat(loader): supercollection load + analysis flattening + accessors"
```

### Task C3: Reduction & analysis path resolution + method_metadata

**Files:**
- Modify: `experiments/loader.py`
- Test: `tests/test_loader.py` (append)

**Interfaces:**
- Consumes: `manifest_dict`, `all_simulations`, `all_methods`, `core.dehydrate_hashed`, `viz_utils.make_method_display_label` + family label maps.
- Produces:
  - `RESULTS_ROOT = "results"` (module constant).
  - `batch_hashes_for_simulation(library, spec) -> list[str]` — dehydrated batch hashes for a sim's batches.
  - `reduction_scope(library, reduction) -> "fit" | "batch"` (fit if `needs.fits`, else batch).
  - `reduction_output(batch_hash, method_hash, reduction, scope) -> str` — the parquet path.
  - `reduction_inputs(library, manifest, batch_hash, method_hash, reduction) -> list[str]` — upstream file paths per `needs` (`fits.parquet`, `simulations.parquet`, `sample_metadata.parquet`).
  - `collection_method_pairs(config, sc_name) -> dict[coll_name, (alias, list[(batch_hash, method_hash)])]` — restricted to the SC's resolved methods; for a reduction-level `method_filter`, narrowing is applied per reduction at `analysis_inputs` time.
  - `analysis_inputs(config, manifest, sc_name, analysis) -> list[str]` — atomic reduction parquet paths across the SC's collections × (batch,method) × the analysis's `requires`, applying per-reduction `method_filter` (base-name prefix match).
  - `method_metadata(methods: dict[name, MethodSpec]) -> polars.DataFrame` — columns: `method, method_family, L, threshold, is_thresholded, is_oracle, method_display, method_display_base` derived from the spec (`method_family = name.split("__")[0]`, `L`/`threshold` from kwargs).

- [ ] **Step 1: Write the failing test (append)**

```python
def test_reduction_scope_and_paths():
    cfg = loader.load_config(FIXTURE_DIR)
    lib = cfg["library"]
    assert loader.reduction_scope(lib, "pip") == "fit"
    p = loader.reduction_output("BH", "MH", "pip", "fit")
    assert p == "results/by_batch/BH/fits/MH/reductions/pip.parquet"


def test_analysis_inputs_only_required_reductions(tmp_path):
    cfg = loader.load_config(FIXTURE_DIR)
    manifest = loader.manifest_dict(cfg["library"], loader.all_simulations(cfg), loader.all_methods(cfg))
    inputs = loader.analysis_inputs(cfg, manifest, "fixture-sc", "pip_calibration")
    # pip_calibration requires [pip]; every path ends in /reductions/pip.parquet
    assert inputs and all(p.endswith("/reductions/pip.parquet") for p in inputs)


def test_method_metadata_columns():
    cfg = loader.load_config(FIXTURE_DIR)
    methods = loader.all_methods(cfg)
    md = loader.method_metadata(methods)
    assert {"method", "method_family", "L", "threshold", "is_thresholded",
            "is_oracle", "method_display"} <= set(md.columns)
    fams = set(md["method_family"].to_list())
    assert fams == {"cox_heavy", "twogroup"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_loader.py -k "reduction_scope or analysis_inputs or method_metadata" -v`
Expected: FAIL

- [ ] **Step 3: Implement (append to loader.py)**

```python
import polars as pl
from viz_utils import (make_method_display_label, method_family_label_map,
                       method_family_oracle_label_map)

RESULTS_ROOT = "results"


def batch_hashes_for_simulation(library: dict[str, Any], spec: core.SimulationSpec) -> list[str]:
    defaults = library["defaults"]
    out = []
    for batch in batch_specs_for_simulation(
        spec, replicates_per_batch=int(defaults["replicates_per_batch"]),
        n_batches=int(defaults["n_batches"])):
        out.append(dehydrate_hashed(batch)[HASH_KEY])
    return out


def reduction_scope(library: dict[str, Any], reduction: str) -> str:
    needs = library["reductions"][reduction].get("needs", {})
    return "fit" if needs.get("fits") else "batch"


def reduction_output(batch_hash: str, method_hash: str | None, reduction: str, scope: str) -> str:
    if scope == "fit":
        return f"{RESULTS_ROOT}/by_batch/{batch_hash}/fits/{method_hash}/reductions/{reduction}.parquet"
    return f"{RESULTS_ROOT}/by_batch/{batch_hash}/reductions/{reduction}.parquet"


def reduction_inputs(library, manifest, batch_hash, method_hash, reduction) -> list[str]:
    needs = library["reductions"][reduction].get("needs", {})
    paths = []
    if needs.get("fits"):
        paths.append(f"{RESULTS_ROOT}/by_batch/{batch_hash}/fits/{method_hash}/fits.parquet")
    if needs.get("simulations"):
        paths.append(f"{RESULTS_ROOT}/by_batch/{batch_hash}/simulations.parquet")
    if needs.get("sample_metadata"):
        paths.append(f"{RESULTS_ROOT}/by_batch/{batch_hash}/sample_metadata.parquet")
    return paths


def _method_hashes(methods: dict[str, core.MethodSpec]) -> dict[str, str]:
    return {name: dehydrate_hashed(spec)[HASH_KEY] for name, spec in methods.items()}


def collection_method_pairs(config, sc_name) -> dict[str, dict]:
    library = config["library"]
    sc = config["supercollections"][sc_name]
    methods = resolve_methods_for_sc(library, sc)
    mhash = _method_hashes(methods)
    out: dict[str, dict] = {}
    for coll in supercollection_collections(library, sc_name, sc):
        pairs = []
        for spec in coll["simulations"]:
            for bh in batch_hashes_for_simulation(library, spec):
                for mname, mh in mhash.items():
                    pairs.append((bh, mh, mname))
        out[coll["name"]] = {"alias": coll["alias"], "pairs": pairs}
    return out


def analysis_inputs(config, manifest, sc_name, analysis) -> list[str]:
    library = config["library"]
    requires = library["analyses"][analysis].get("requires", [])
    cmp = collection_method_pairs(config, sc_name)
    paths: list[str] = []
    seen = set()
    for reduction in requires:
        scope = reduction_scope(library, reduction)
        mfilter = library["reductions"][reduction].get("needs", {}).get("method_filter")
        for coll in cmp.values():
            for bh, mh, mname in coll["pairs"]:
                if mfilter is not None and not mname.split("__")[0].startswith(mfilter):
                    continue
                mh_arg = None if scope == "batch" else mh
                p = reduction_output(bh, mh_arg, reduction, scope)
                if p not in seen:
                    seen.add(p)
                    paths.append(p)
    return paths


def method_metadata(methods: dict[str, core.MethodSpec]) -> pl.DataFrame:
    label_map = method_family_label_map()
    oracle_map = method_family_oracle_label_map()
    rows = []
    for name, spec in methods.items():
        family = name.split("__")[0]
        L = int(spec.kwargs.get("L", 1))
        threshold = spec.kwargs.get("threshold")
        is_thresholded = threshold is not None
        is_oracle = "oracle" in family
        family_label = label_map.get(family, family)
        oracle_label = oracle_map.get(family, "Oracle")
        suffix = "SER" if L == 1 else f"SuSiE [L={L}]"
        base = f"{family_label} {suffix}"
        rows.append({
            "method": name, "method_family": family, "L": L,
            "threshold": float(threshold) if threshold is not None else None,
            "is_thresholded": is_thresholded, "is_oracle": is_oracle,
            "method_display": make_method_display_label(
                method_label_base=base, threshold=threshold,
                is_thresholded=is_thresholded, is_oracle=is_oracle, oracle_label=oracle_label),
            "method_display_base": make_method_display_label(
                method_label_base=base, threshold=None, is_thresholded=False,
                is_oracle=is_oracle, oracle_label=oracle_label),
        })
    return pl.from_dicts(rows)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_loader.py -k "reduction_scope or analysis_inputs or method_metadata" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/loader.py tests/test_loader.py
git commit -m "feat(loader): reduction/analysis path resolution + method_metadata"
```

---

## Phase D — plot_ready atomic reductions + sample_metadata

### Task D1: Atomic sample_metadata builder

**Files:**
- Modify: `plot_ready.py:115-153` (replace `build_sample_metadata`/`_build_sample_metadata_from_manifest`)
- Test: `tests/test_plot_ready_atomic.py` (create)

**Interfaces:**
- Produces: `build_sample_metadata(batch_hash: str, simulations_df: pl.DataFrame) -> pl.DataFrame` — one row per replicate: `sample_id = f"{batch_hash}::{replicate}"`, `batch_hash`, `replicate`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_plot_ready_atomic.py
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import polars as pl
import plot_ready


def test_build_sample_metadata_atomic():
    sims = pl.from_dicts([{"replicate": 0, "simulation": {}}, {"replicate": 1, "simulation": {}}])
    md = plot_ready.build_sample_metadata("BH", sims)
    assert md["sample_id"].to_list() == ["BH::0", "BH::1"]
    assert md["batch_hash"].to_list() == ["BH", "BH"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_plot_ready_atomic.py::test_build_sample_metadata_atomic -v`
Expected: FAIL (signature mismatch / TypeError)

- [ ] **Step 3: Replace build_sample_metadata, delete _build_sample_metadata_from_manifest**

`plot_ready.py:115-153` →

```python
def build_sample_metadata(batch_hash: str, simulations_df: pl.DataFrame) -> pl.DataFrame:
    rows = [
        {"sample_id": f"{batch_hash}::{int(rep)}", "batch_hash": batch_hash, "replicate": int(rep)}
        for rep in simulations_df["replicate"].to_list()
    ]
    return pl.from_dicts(rows)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_plot_ready_atomic.py::test_build_sample_metadata_atomic -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add plot_ready.py tests/test_plot_ready_atomic.py
git commit -m "refactor(plot_ready): atomic build_sample_metadata"
```

### Task D2: Atomic pip/cs/f1/enrich builders

**Files:**
- Modify: `plot_ready.py:160-204` (`build_pip_plot_data`), `:207-289` (`build_cs_plot_data`), `:292-335` (`build_f1_plot_data`), `:338-392` (`build_enrich_plot_data`)
- Test: `tests/test_plot_ready_atomic.py` (append)

**Interfaces:**
- Produces (all operate on one `(batch, method)` unit):
  - `build_pip_plot_data(fits_df, sample_metadata, simulations_df) -> pl.DataFrame` (simulations_df is the single batch's, not a dict).
  - `build_cs_plot_data(fits_df, sample_metadata, simulations_df) -> pl.DataFrame`.
  - `build_f1_plot_data(fits_df, sim_spec_node) -> pl.DataFrame` — `fits_df` already filtered to one twogroup method's fits; `sim_spec_node` is the batch's dehydrated simulation node (for true f1).
  - `build_enrich_plot_data(fits_df, simulations_df, sim_spec_node) -> pl.DataFrame`.

  Each adds a `batch_hash` column derived from `sample_metadata`/argument, so concat across units is unambiguous.

- [ ] **Step 1: Write the failing test (append)** — uses a recorded fixture fit. Generate the fixture first:

```python
def _make_atomic_fixture(tmp_path):
    """Run one tiny batch+method end-to-end, return (fits_df, sims_df, sample_md, sim_node)."""
    import core, utils
    from experiments import loader
    lib = loader.load_library(Path(__file__).resolve().parent / "fixtures" / "experiments")
    spec, _ = loader.resolve_simulation(lib, "gaussian_p8", "ser_b2", "loc_2.0", "gaussian")
    method = loader.expand_method("cox_heavy", lib["methods"]["cox_heavy"])[0]
    reps = (0, 1)
    sims_df = utils.simulate_batch(spec, replicates=reps)
    bh = core.dehydrate_hashed(utils.BatchSpec(name="b", simulation_spec=spec, replicates=reps))[core.HASH_KEY]
    fits_df = utils.fit_batch_method(spec, method_spec=method, replicates=reps).with_columns(
        pl.lit(bh).alias("batch_hash"))
    sample_md = plot_ready.build_sample_metadata(bh, sims_df)
    return fits_df, sims_df, sample_md, core.dehydrate_hashed(spec)


def test_build_pip_plot_data_atomic(tmp_path):
    fits_df, sims_df, sample_md, _ = _make_atomic_fixture(tmp_path)
    out = plot_ready.build_pip_plot_data(fits_df, sample_md, sims_df)
    assert out["sample_id"].to_list() == [f"{sample_md['batch_hash'][0]}::0",
                                          f"{sample_md['batch_hash'][0]}::1"]
    assert "pip_bin_counts" in out.columns
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_plot_ready_atomic.py::test_build_pip_plot_data_atomic -v`
Expected: FAIL (current `build_pip_plot_data` expects `simulations_by_batch` dict, indexes `[row["batch_hash"]]`)

- [ ] **Step 3: Refactor the four builders to atomic**

For `build_pip_plot_data` (`:160`) and `build_cs_plot_data` (`:207`): change the third parameter from `simulations_by_batch: dict` to `simulations_df: pl.DataFrame`; replace `sim_df = simulations_by_batch[row["batch_hash"]]` with `sim_df = simulations_df`. Keep all per-row logic identical (the join on `(batch_hash, replicate)` still works — `sample_metadata` carries `batch_hash`; if `fits_df` lacks `batch_hash`, the caller adds it).

For `build_f1_plot_data` (`:292`): change signature to `(fits_df, sim_spec_node)`. Remove the `for batch in collection["batches"]` and `for method_spec in collection["method_specs"]` loops + file reads. Derive `true_loc`/`true_scale` from `sim_spec_node["fields"]["f1"]["fields"]`; derive `batch_hash` from `fits_df["batch_hash"][0]`; iterate `fits_df.select(["replicate","method","two_group_state","family_state"]).iter_rows(...)`.

For `build_enrich_plot_data` (`:338`): change signature to `(fits_df, simulations_df, sim_spec_node)`. Replace the per-batch parquet read with the passed `simulations_df`; keep the rest.

(Exact bodies: preserve the row-construction code shown in `plot_ready.py`; only the input plumbing changes.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_plot_ready_atomic.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add plot_ready.py tests/test_plot_ready_atomic.py
git commit -m "refactor(plot_ready): atomic pip/cs/f1/enrich builders"
```

---

## Phase E — generate_plots: analysis renderers + method_filter

### Task E1: Public analysis registry + method_filter foreground

**Files:**
- Modify: `generate_plots.py` (`_PLOT_DISPATCH` → `ANALYSIS_RENDERERS`; `_foreground_methods`; `make_plot`; `_load_supercollection_data`; `_resolve_settings`/`_load_plot_config`)
- Test: `tests/test_generate_plots_filter.py` (create)

**Interfaces:**
- Produces:
  - `ANALYSIS_RENDERERS: dict[str, Callable[[dict, dict], plt.Figure]]` (the existing `_PLOT_DISPATCH`, renamed/public).
  - `_foreground_methods(method_meta, args) -> set[str]` — returns `set(args["method_filter"])` intersected with names present in `method_meta["method"]`.
  - `render_analysis(bundle: dict, args: dict, analysis: str, output_path: str) -> None` — calls `ANALYSIS_RENDERERS[analysis](bundle, args)` then `savefig`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generate_plots_filter.py
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import polars as pl
import generate_plots


def test_foreground_methods_is_name_membership():
    meta = pl.from_dicts([{"method": "twogroup__L=1"}, {"method": "cox_heavy__L=1"}])
    fg = generate_plots._foreground_methods(meta, {"method_filter": ["twogroup__L=1", "absent__L=1"]})
    assert fg == {"twogroup__L=1"}


def test_analysis_registry_has_pip_calibration():
    assert "pip_calibration" in generate_plots.ANALYSIS_RENDERERS
    assert "agg_pip_calibration" in generate_plots.ANALYSIS_RENDERERS
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_generate_plots_filter.py -v`
Expected: FAIL (`ANALYSIS_RENDERERS` undefined / `_foreground_methods` still family-based)

- [ ] **Step 3: Rename dispatch + rewrite `_foreground_methods`**

In `generate_plots.py`: rename the `_PLOT_DISPATCH = {...}` table to `ANALYSIS_RENDERERS = {...}` (keep all entries). Replace `_foreground_methods` (`:118-128`) with:

```python
def _foreground_methods(method_metadata: pl.DataFrame, settings: dict) -> set[str]:
    requested = set(settings.get("method_filter", []))
    present = set(method_metadata["method"].to_list())
    return requested & present
```

Delete `_selected_thresholds` (`:109-115`) usages: in each renderer that passed `selected_thresholds=_selected_thresholds(settings)`, drop that argument (threshold filtering is gone; `viz_utils` expand fns default `selected_thresholds=None`). Delete `_method_family` if present.

- [ ] **Step 4: Add render_analysis + keep make_plot working via loader**

Add:

```python
def render_analysis(bundle: dict, args: dict, analysis: str, output_path: str) -> None:
    fig = ANALYSIS_RENDERERS[analysis](bundle, args)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/test_generate_plots_filter.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add generate_plots.py tests/test_generate_plots_filter.py
git commit -m "feat(generate_plots): analysis registry + name-based method_filter"
```

### Task E2: load_sc_bundle + resolve_args in loader

**Files:**
- Modify: `experiments/loader.py`
- Test: `tests/test_loader.py` (append, integration-style using fixture dir + a built results tree)

**Interfaces:**
- Produces:
  - `resolve_args(config, sc_name, args_name) -> dict` — merge `sc["default_args"]` with the matching output entry's `args`, plus `method_filter` from that entry.
  - `analysis_function(config, analysis) -> Callable` — `generate_plots.ANALYSIS_RENDERERS[analysis]` (import inside the function to avoid a hard matplotlib dep at import time).
  - `analysis_requires(config, analysis) -> list[str]`.
  - `load_sc_bundle(config, sc_name, requires, results_root=RESULTS_ROOT) -> dict` — for each collection, read each `(batch,method)` reduction parquet for the required reductions, concat, tag `collection_name=alias`; concat across collections into `{f"{reduction}_plot_data": df}`; add `method_metadata` (loader-built) and `collection_names` (alias list in order).

- [ ] **Step 1: Write the failing test (append)** — build a tiny results tree, then bundle:

```python
def test_load_sc_bundle_tags_collections(tmp_path):
    import core, utils
    cfg = loader.load_config(FIXTURE_DIR)
    lib = cfg["library"]
    results = tmp_path / "results"
    # materialize + fit + reduce one collection's units for reduction "pip"
    for coll in loader.supercollection_collections(lib, "fixture-sc", cfg["supercollections"]["fixture-sc"]):
        for spec in coll["simulations"]:
            reps = (0, 1)
            bh = core.dehydrate_hashed(utils.BatchSpec(name="b", simulation_spec=spec, replicates=reps))[core.HASH_KEY]
            sims_df = utils.simulate_batch(spec, replicates=reps)
            sample_md = __import__("plot_ready").build_sample_metadata(bh, sims_df)
            for mname, mspec in loader.resolve_methods_for_sc(lib, cfg["supercollections"]["fixture-sc"]).items():
                mh = core.dehydrate_hashed(mspec)[core.HASH_KEY]
                fits = utils.fit_batch_method(spec, method_spec=mspec, replicates=reps).with_columns(pl.lit(bh).alias("batch_hash"))
                red = __import__("plot_ready").build_pip_plot_data(fits, sample_md, sims_df)
                out = results / "by_batch" / bh / "fits" / mh / "reductions" / "pip.parquet"
                out.parent.mkdir(parents=True, exist_ok=True)
                red.write_parquet(out)
    bundle = loader.load_sc_bundle(cfg, "fixture-sc", ["pip"], results_root=str(results))
    assert "pip_plot_data" in bundle and "method_metadata" in bundle
    assert set(bundle["pip_plot_data"]["collection_name"].unique().to_list()) == {"loc_2.0"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_loader.py -k load_sc_bundle -v`
Expected: FAIL

- [ ] **Step 3: Implement (append to loader.py)**

```python
def resolve_args(config, sc_name, args_name) -> dict[str, Any]:
    sc = config["supercollections"][sc_name]
    defaults = dict(sc.get("default_args", {}) or {})
    for output in sc.get("outputs", []):
        if output["name"] == args_name:
            return {**defaults, **(output.get("args") or {}),
                    "method_filter": output.get("method_filter", [])}
    raise KeyError(f"No output named {args_name!r} in supercollection {sc_name!r}")


def analysis_requires(config, analysis) -> list[str]:
    return list(config["library"]["analyses"][analysis].get("requires", []))


def analysis_function(config, analysis):
    import generate_plots
    return generate_plots.ANALYSIS_RENDERERS[analysis]


def load_sc_bundle(config, sc_name, requires, results_root: str = RESULTS_ROOT) -> dict[str, Any]:
    library = config["library"]
    cmp = collection_method_pairs(config, sc_name)
    bundle: dict[str, Any] = {}
    collection_names = [info["alias"] for info in cmp.values()]
    for reduction in requires:
        scope = reduction_scope(library, reduction)
        mfilter = library["reductions"][reduction].get("needs", {}).get("method_filter")
        frames = []
        for info in cmp.values():
            sub = []
            for bh, mh, mname in info["pairs"]:
                if mfilter is not None and not mname.split("__")[0].startswith(mfilter):
                    continue
                mh_arg = None if scope == "batch" else mh
                path = f"{results_root}/{reduction_output(bh, mh_arg, reduction, scope).split('/', 1)[1]}"
                df = pl.read_parquet(path)
                sub.append(df)
            if sub:
                merged = pl.concat(sub, how="diagonal_relaxed").with_columns(
                    pl.lit(info["alias"]).alias("collection_name"))
                frames.append(merged)
        bundle[f"{reduction}_plot_data"] = (
            pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame())
    bundle["method_metadata"] = method_metadata(resolve_methods_for_sc(library, config["supercollections"][sc_name]))
    bundle["collection_names"] = collection_names
    return bundle
```

Note: `reduction_output` returns a path already prefixed with `RESULTS_ROOT`; the `.split('/',1)[1]` re-roots it under the caller's `results_root` for tests. (In the snakefile, `results_root == RESULTS_ROOT`, so this is a no-op.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_loader.py -k load_sc_bundle -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/loader.py tests/test_loader.py
git commit -m "feat(loader): load_sc_bundle + resolve_args + analysis accessors"
```

---

## Phase F — Snakefile, manifest_cache, migration, deletions, integration

### Task F1: manifest_cache repoint + materialize sample_metadata

**Files:**
- Modify: `manifest_cache.py`
- Modify: `twogroup_experiments.snk:191-206` (`materialize_twogroup_experiment_batch`)
- Modify: `utils.py:37-40` (`manifest_dict`)

**Interfaces:**
- Consumes: `loader.load_config`, `loader.manifest_dict`, `loader.all_simulations`, `loader.all_methods`.
- Produces: `manifest_cache.load_manifest_cached()` returns the loader-built manifest; `materialize_*` writes `sample_metadata.parquet`.

- [ ] **Step 1: Rewrite manifest_cache.py**

```python
from __future__ import annotations
import json
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).resolve().parent / "experiments"


def _experiments_mtime() -> float:
    return max((p.stat().st_mtime for p in _EXPERIMENTS_DIR.glob("*.yaml")), default=0.0)


def load_manifest_cached(cache_path: str | Path | None = None) -> dict:
    if cache_path is None:
        cache_path = Path(__file__).resolve().parent / "results" / "manifest_cache.json"
    cache_path = Path(cache_path)
    if cache_path.exists() and _experiments_mtime() <= cache_path.stat().st_mtime:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    from experiments import loader
    cfg = loader.load_config()
    data = loader.manifest_dict(cfg["library"], loader.all_simulations(cfg), loader.all_methods(cfg))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data), encoding="utf-8")
    return data
```

- [ ] **Step 2: Update utils.manifest_dict**

`utils.py:37-40` →

```python
def manifest_dict() -> dict[str, object]:
    from experiments import loader
    cfg = loader.load_config()
    return loader.manifest_dict(cfg["library"], loader.all_simulations(cfg), loader.all_methods(cfg))
```

- [ ] **Step 3: Add sample_metadata output to materialize rule**

`twogroup_experiments.snk:191-206` — add to `output:` block: `sample_metadata=f"{RESULTS_ROOT}/by_batch/{{batch_hash}}/sample_metadata.parquet",` and at the end of the `run:` body:

```python
        from plot_ready import build_sample_metadata
        write_parquet(build_sample_metadata(wildcards.batch_hash, simulations_df), output.sample_metadata)
```

- [ ] **Step 4: Smoke-test the manifest builds**

Run: `uv run python -c "from experiments import loader; cfg=loader.load_config(); m=loader.manifest_dict(cfg['library'], loader.all_simulations(cfg), loader.all_methods(cfg)); print(len(m['batches']), len(m['method_specs']))"`
Expected: prints two non-zero integers (requires `experiments/library.yaml` + the migrated SC files from F3; if run before F3, expect `0 0` or a clean empty result — re-run after F3).

- [ ] **Step 5: Commit**

```bash
git add manifest_cache.py utils.py twogroup_experiments.snk
git commit -m "feat(snk): loader-backed manifest + sample_metadata output"
```

### Task F2: Two generic rules + loader-backed targets in snakefile

**Files:**
- Modify: `twogroup_experiments.snk` — delete the plot-config block (`:36-189`), the 7 collection rules (`:232-439`) and `supercollection_plot` (`:495-509`); add the two generic rules + target aggregators.

**Interfaces:**
- Consumes: `experiments.loader` (`load_config`, `reduction_function` via `library["reductions"][r]["function"]`, `reduction_inputs`, `analysis_inputs`, `analysis_function`, `analysis_requires`, `resolve_args`, `resolve_sc_analyses`, `load_sc_bundle`).
- Produces: rules `atomic_reduction`, `atomic_reduction_batch` (sim-scoped variant), `supercollection_analysis`, plus `all_plots`/`plot_collection` rebuilt from `resolve_sc_analyses`.

- [ ] **Step 1: Replace the config/plot block header (top of file)**

Replace `twogroup_experiments.snk:11-189` (everything from `from core import ...` through the `_batch_sim_node` helper, i.e. the manifest load + all `_load_plot_configs`/`_COLLECTION_*`/`_resolve_collection_*`/`load_collection_yaml`/`load_supercollection`/`_resolve_sc_plot_pairs`/`_method_*` machinery) with:

```python
from core import rehydrate_node
from utils import fit_batch_method, simulate_batch, write_parquet, write_yaml
from experiments import loader
from manifest_cache import load_manifest_cached

_MANIFEST = load_manifest_cached()
_CONFIG = loader.load_config()
BATCH_HASH_TO_INFO = _MANIFEST["batches"]
METHOD_HASH_TO_INFO = _MANIFEST["method_specs"]
SUPERCOLLECTIONS = sorted(_CONFIG["supercollections"].keys())

wildcard_constraints:
    batch_hash=r"[0-9a-f]{64}",
    method_hash=r"[0-9a-f]{64}",
    reduction=r"[A-Za-z0-9_]+",
    analysis=r"[A-Za-z0-9_]+",
    args_name=r"[A-Za-z0-9_\-]+",
    supercollection=r"[A-Za-z0-9_\-\.]+",


def _all_analysis_targets():
    targets = []
    for sc in SUPERCOLLECTIONS:
        for analysis, args_name in loader.resolve_sc_analyses(_CONFIG, sc):
            targets.append(f"{RESULTS_ROOT}/supercollections/{sc}/{analysis}/{args_name}.pdf")
    return targets
```

- [ ] **Step 2: Keep materialize + fit rules; add the reduction rules after `fit_twogroup_experiment_batch_method`**

```python
rule atomic_reduction:
    output:
        f"{RESULTS_ROOT}/by_batch/{{batch_hash}}/fits/{{method_hash}}/reductions/{{reduction}}.parquet"
    input:
        sources=PLOT_READY_SOURCES,
        deps=lambda wc: loader.reduction_inputs(_CONFIG["library"], _MANIFEST,
                                                wc.batch_hash, wc.method_hash, wc.reduction),
    run:
        import plot_ready, polars as pl
        lib = _CONFIG["library"]
        fn = getattr(plot_ready, lib["reductions"][wildcards.reduction]["function"])
        fits = pl.read_parquet(
            f"{RESULTS_ROOT}/by_batch/{wildcards.batch_hash}/fits/{wildcards.method_hash}/fits.parquet"
        ).with_columns(pl.lit(wildcards.batch_hash).alias("batch_hash"))
        sample_md = pl.read_parquet(f"{RESULTS_ROOT}/by_batch/{wildcards.batch_hash}/sample_metadata.parquet")
        sims = pl.read_parquet(f"{RESULTS_ROOT}/by_batch/{wildcards.batch_hash}/simulations.parquet")
        sim_node = BATCH_HASH_TO_INFO[wildcards.batch_hash]["simulation_spec"]
        red = wildcards.reduction
        if red == "pip":
            df = fn(fits, sample_md, sims)
        elif red == "cs":
            df = fn(fits, sample_md, sims)
        elif red == "f1":
            df = fn(fits, sim_node)
        elif red == "enrich":
            df = fn(fits, sims, sim_node)
        else:
            raise KeyError(red)
        write_parquet(df, output[0])
```

(If sim-only reductions are added later, add the analogous `atomic_reduction_batch` rule with output `by_batch/{batch_hash}/reductions/{reduction}.parquet`. None are needed for the 003/000 proof.)

- [ ] **Step 3: Add the analysis rule**

```python
rule supercollection_analysis:
    output:
        f"{RESULTS_ROOT}/supercollections/{{supercollection}}/{{analysis}}/{{args_name}}.pdf"
    input:
        sources=PLOT_RENDER_SOURCES,
        deps=lambda wc: loader.analysis_inputs(_CONFIG, _MANIFEST, wc.supercollection, wc.analysis),
    run:
        import generate_plots
        bundle = loader.load_sc_bundle(_CONFIG, wildcards.supercollection,
                                       loader.analysis_requires(_CONFIG, wildcards.analysis))
        args = loader.resolve_args(_CONFIG, wildcards.supercollection, wildcards.args_name)
        generate_plots.render_analysis(bundle, args, wildcards.analysis, output[0])
```

- [ ] **Step 4: Replace target aggregators**

Replace `all_collections`/`all_supercollections`/`materialize_supercollection`/`twogroup_experiments_target`/`all_null_fits`/`all_plots`/`plot_collection` with:

```python
rule all_plots:
    input:
        _all_analysis_targets(),
```

(Drop `all_null_fits` and its `from config import NULL_METHOD_SPECS` block entirely.)

- [ ] **Step 5: Lint the snakefile (dry run, no execution)**

Run: `uv run snakemake -s twogroup_experiments.snk -n all_plots 2>&1 | tail -30`
Expected: snakemake parses the file and builds a DAG (job counts printed). Errors here mean a wiring bug — fix before continuing. (May require F3 migrated YAMLs to have any targets; if `all_plots` is empty, run with an explicit target path printed by `_all_analysis_targets()`.)

- [ ] **Step 6: Commit**

```bash
git add twogroup_experiments.snk
git commit -m "feat(snk): two generic rules + loader-backed targets"
```

### Task F3: Write library.yaml + migrate 003 and 000

**Files:**
- Create: `experiments/library.yaml`, `experiments/003_loc_snr.yaml`, `experiments/000_t_errors.yaml`

**Interfaces:**
- Consumes: the design samplers/effect samplers/run_* in `core.py`, the reduction builders in `plot_ready.py`, the renderers in `generate_plots.ANALYSIS_RENDERERS`.

- [ ] **Step 1: Write experiments/library.yaml**

Include: `defaults`; designs `hallmark`, `c4`, `gaussian_p100` (`gaussian_markov_X` n=500 p=100 rho=0.9), `uniform_p100` (`uniform_markov_X` n=500 p=100 rho=0.9); enrichments `ser_b2` (causal_effect 2.0, intercept -2.0), `null_b0` (causal_effect 0.0, intercept -2.0); signals `loc_0.5,1.0,1.5,2.0,2.5,3.0` (PointMass 0 + Normal loc=x scale=0.1 fixed) and `scale_2.0` (Normal loc=0 scale=2.0 fixed); errors `gaussian: null`, `t_df_3/5/10/30` (`t_error_sampler` df=N); methods `cox_heavy` (run_cox_method, threshold null, time_sign 1.0, over L:[1]), `cox_light` (run_cox_method, time_sign -1.0, over threshold:[2.0] L:[1]), `logistic_oracle` (run_logistic_method, response_source z, threshold null, over L:[1]), `logistic_threshold` (run_logistic_method, response_source score_threshold, over threshold:[2.0] L:[1]), `twogroup_oracle` (run_twogroup_method, template `{f1: null, n_null_iter: 20, n_intercept_iter: 20}`, over L:[1]), `twogroup` (run_twogroup_method, f1 Normal loc0 scale1 estimate both, over L:[1]), `twogroup_loc_fam` (f1 Normal loc0 scale0.1 estimate_loc true estimate_scale false), `linear_fixed` (run_linear_method, estimate_residual_variance false, over L:[1]); reductions `pip`/`cs` (needs fits+sample_metadata), `f1` (needs fits, method_filter twogroup); analyses + analysis_groups covering at least `pip_calibration, agg_pip_calibration, power_fdp, agg_power_fdp, cs_power_fdp, agg_cs_power_fdp` mapped from `generate_plots.ANALYSIS_RENDERERS` keys; analysis_groups `pip`, `cs`.

Use the exact distribution mini-schema and the `template`/`over` shapes from the spec (spec lines 83-143). Validate names contain no `__`.

- [ ] **Step 2: Write experiments/003_loc_snr.yaml**

One supercollection `003-hallmark-loc-snr`:

```yaml
supercollections:
  003-hallmark-loc-snr:
    collections:
      template: {design: hallmark, enrichment: [ser_b2, null_b0], error: gaussian}
      over: {signal: [loc_0.5, loc_1.0, loc_1.5, loc_2.0, loc_2.5, loc_3.0]}
    methods: [twogroup_oracle__L=1, twogroup__L=1, twogroup_loc_fam__L=1,
              cox_heavy__L=1, cox_light__threshold=2.00__L=1]
    default_args: {min_log_bf: 2.0, max_cs_size: 10000, max_fdp: 0.5}
    outputs:
      - name: minimal-loc
        method_filter: [twogroup_oracle__L=1, twogroup__L=1, twogroup_loc_fam__L=1]
        analyses: [pip, cs]
```

- [ ] **Step 3: Write experiments/000_t_errors.yaml**

One supercollection pooling 4 designs per error level (within-collection design list, over error):

```yaml
supercollections:
  000-all-designs-t-error-loc:
    collections:
      template:
        design: [hallmark, c4, gaussian_p100, uniform_p100]
        enrichment: ser_b2
        signal: loc_2.0
      over: {error: [gaussian, t_df_30, t_df_10, t_df_5, t_df_3]}
    methods: [twogroup_oracle__L=1, cox_heavy__L=1, cox_light__threshold=2.00__L=1]
    default_args: {min_log_bf: 2.0, max_cs_size: 10000, max_fdp: 0.5}
    outputs:
      - name: minimal
        method_filter: [twogroup_oracle__L=1, cox_heavy__L=1, cox_light__threshold=2.00__L=1]
        analyses: [pip, cs]
```

- [ ] **Step 4: Verify loader resolves both configs**

Run:
```bash
uv run python -c "
from experiments import loader
cfg = loader.load_config()
print('SCs:', sorted(cfg['supercollections']))
print('sims:', len(loader.all_simulations(cfg)))
print('methods:', sorted(loader.all_methods(cfg)))
for sc in cfg['supercollections']:
    print(sc, loader.resolve_sc_analyses(cfg, sc))
"
```
Expected: both SCs listed; `methods` includes `cox_light__threshold=2.00__L=1` etc. (proves name formatting matches the YAML references); analysis pairs printed. Any `KeyError` for a method name means a name-format mismatch — fix the YAML reference to match `format_over_value`.

- [ ] **Step 5: Commit**

```bash
git add experiments/library.yaml experiments/003_loc_snr.yaml experiments/000_t_errors.yaml
git commit -m "feat(experiments): library + migrated 003/000 supercollections"
```

### Task F4: Delete the old registry + plot_configs; update viz_utils metadata

**Files:**
- Delete: `config.py`, `config_builders.py`, `config_registry.py`, `plot_configs/`
- Modify: `viz_utils.py:55-75` (`method_metadata_from_method_spec_json`) — only if still referenced; otherwise leave (loader's `method_metadata` supersedes it).

- [ ] **Step 1: Confirm nothing imports the deleted modules**

Run: `grep -rn "import config\b\|from config import\|config_builders\|config_registry\|plot_configs" --include=*.py --include=*.snk . | grep -v "tests/test_plot_configs.py"`
Expected: no hits (besides the obsolete `tests/test_plot_configs.py`, deleted next).

- [ ] **Step 2: Delete files**

```bash
git rm config.py config_builders.py config_registry.py
git rm -r plot_configs
git rm tests/test_plot_configs.py
```

- [ ] **Step 3: Run the full unit suite (excluding snakemake integration)**

Run: `uv run pytest tests/test_loader.py tests/test_core_run_methods.py tests/test_plot_ready_atomic.py tests/test_generate_plots_filter.py -v`
Expected: all PASS. Fix any import errors from deleted modules in the remaining test files (delete or update tests that referenced the old registry).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: delete registry config + plot_configs (replaced by experiments/)"
```

### Task F5: End-to-end integration (tiny replicate count)

**Files:**
- Modify: `experiments/library.yaml` (temporarily set `defaults.replicates_per_batch: 2` via an override env, OR create `experiments/999_smoke.yaml` for the test) — use a dedicated smoke supercollection to avoid touching real configs.
- Create: `experiments/999_smoke.yaml` (tiny: `gaussian_p100`→use a small design; 2 replicates won't apply since defaults are global — instead add a `gaussian_p8` design to library for the smoke SC).

**Interfaces:** end-to-end DAG through `materialize → fit → atomic_reduction → supercollection_analysis`.

- [ ] **Step 1: Add a small design + smoke supercollection**

Add to `experiments/library.yaml` designs: `gaussian_p8: {function: gaussian_markov_X, arguments: {n: 30, p: 8, rho: 0.5}}`. Create `experiments/999_smoke.yaml`:

```yaml
supercollections:
  999-smoke:
    collections:
      template: {design: gaussian_p8, enrichment: ser_b2, error: gaussian}
      over: {signal: [loc_2.0]}
    methods: [cox_heavy__L=1, twogroup__L=1]
    default_args: {min_log_bf: 2.0, max_fdp: 0.5}
    outputs:
      - name: smoke
        method_filter: [twogroup__L=1]
        analyses: [pip_calibration]
```

- [ ] **Step 2: Build one analysis PDF end-to-end**

Run:
```bash
uv run snakemake -s twogroup_experiments.snk -j1 --cores 1 \
  results/supercollections/999-smoke/pip_calibration/smoke.pdf 2>&1 | tail -40
```
Expected: snakemake runs materialize → fit (cox_heavy, twogroup) → atomic_reduction (pip) → supercollection_analysis; the PDF exists.

- [ ] **Step 3: Verify the PDF exists and is non-empty**

Run: `ls -la results/supercollections/999-smoke/pip_calibration/smoke.pdf`
Expected: file present, size > 0.

- [ ] **Step 4: Verify atomic reduction reuse (one parquet per fit)**

Run: `find results/by_batch -path '*/reductions/pip.parquet' | wc -l`
Expected: equals (#batches for the smoke SC) × (#methods passing pip's filter) — for 999-smoke with 1 sim × 2 methods × 1 batch = `2`. Confirms each fit reduced once.

- [ ] **Step 5: Dry-run the real proof configs**

Run: `uv run snakemake -s twogroup_experiments.snk -n results/supercollections/003-hallmark-loc-snr/pip_calibration/minimal-loc.pdf 2>&1 | tail -30`
Expected: DAG resolves; lists materialize/fit/atomic_reduction/analysis jobs for the 003 collections. No execution.

- [ ] **Step 6: Commit**

```bash
git add experiments/library.yaml experiments/999_smoke.yaml
git commit -m "test(e2e): smoke supercollection through the two generic rules"
```

---

## Self-Review

**Spec coverage:**
- Library schema (designs/enrichments/signals/errors/methods/reductions/analyses/analysis_groups) → B1, C2, F3. ✓
- Distribution mini-schema → B1. ✓
- Simulation identity + naming → B2. ✓
- Method template/over + 1:1 + name formatting → B3, Global Constraints. ✓
- MethodSpec collapse + run_* → A1, A2. ✓
- Collection template/over expansion → C1. ✓
- analyses single mixed list + flatten → C2. ✓
- Atomic reductions (fit-level) + sample_metadata from materialize + method_metadata from loader → C3, D1, D2, F1. ✓
- Two generic snakemake rules + dependency-driven inputs → C3, F2. ✓
- load_sc_bundle + resolve_args + render via analysis registry → E1, E2, F2. ✓
- manifest_cache repoint → F1. ✓
- Deletions → F4. ✓
- Migration 003/000 → F3. ✓
- Testing items (expansion, manifest shape, analysis_inputs only-required, e2e, reuse) → B/C tests + F5. ✓

**Placeholder scan:** No "TBD"/"handle edge cases"/"similar to". F3 Step 1 describes the library contents in prose with exact values rather than a 120-line block — acceptable as it enumerates exact functions/args; the structural shapes are given by spec lines 83-143. F2 Step 2 enumerates the per-reduction dispatch explicitly.

**Type consistency:** `MethodSpec(name, function, kwargs)` used consistently (A2, B3, C3). `run_method_spec` returns the row dict (A2) and `fit_batch_method` spreads it (A2). Reduction builders' atomic signatures (D2) match the snakefile dispatch (F2 Step 2) and `load_sc_bundle` reads (E2). `method_metadata` columns (C3) match `_foreground_methods`/renderers' expectations (E1). `collection_method_pairs` shape `{name: {alias, pairs:[(bh,mh,mname)]}}` used by both `analysis_inputs` (C3) and `load_sc_bundle` (E2).

Known follow-ups (out of plan scope, noted): porting 001/002/004-007; cache-reuse hash-injection script; `viz_utils.method_metadata_from_method_spec_json` may be removed once confirmed unreferenced.
