# Deterministic Membership (Well-Specified Cox 009) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic membership mode to the two-group generative model so experiment 009 produces exactly proportional-hazards data for Cox.

**Architecture:** A new `membership` field on `SimulationSpec` switches `z` generation between the current `Bernoulli(sigmoid(logits))` and a deterministic `z = 1[logits > 0]`. The loader plumbs the flag from new `_det` enrichment entries in `library.yaml`; experiment 009 swaps to those enrichments. With binary gene-set `X`, `effect=2, intercept=-1` yields `z = x_causal`, so in-set genes draw `Exp(lambda)` and out-set draw `Exp(1)` — exact PH.

**Tech Stack:** Python, NumPy, pytest (run via `uv run pytest`), Snakemake, YAML.

## Global Constraints

- Run all Python via `uv run` (e.g. `uv run pytest`), never bare `python`/`pip`.
- Default behavior must stay byte-identical: `membership` defaults to `"stochastic"`; existing enrichments (`ser_b2`, `null_b0`, …) are untouched.
- Deterministic clean `z = x_causal` requires `intercept < 0` and `intercept + effect > 0`. Use `intercept=-1.0, effect=2.0`.
- No hash plumbing: `membership` rides inside the enrichment coordinate dict, so only the new `_det` enrichments hash distinct.

---

### Task 1: Core deterministic membership branch

**Files:**
- Modify: `core.py` (`SimulationSpec` dataclass ~line 12-22; `simulate` ~line 116)
- Test: `tests/test_core_run_methods.py`

**Interfaces:**
- Consumes: existing `SimulationSpec`, `core.simulate`, `core._sigmoid`.
- Produces: `SimulationSpec.membership: str = "stochastic"`; when `"deterministic"`, `simulate` sets `z = (logits > 0).astype(int)`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_core_run_methods.py`:

```python
def test_deterministic_membership_pins_z_to_threshold():
    from functools import partial
    from core import SimulationSpec, simulate
    from simulations.distributions import Exponential

    # Fixed 6x3 binary design; deterministic effect sampler picks column 1.
    X = np.array(
        [[0, 0, 1],
         [0, 1, 0],
         [0, 1, 1],
         [0, 0, 0],
         [0, 1, 0],
         [0, 0, 1]],
        dtype=float,
    )
    spec = SimulationSpec(
        design_sampler=lambda rng: X,
        effect_sampler=lambda Xarg, rng: ([1], [2.0]),  # causal column 1, effect 2
        intercept=-1.0,
        f0=Exponential(rate=1.0),
        f1=Exponential(rate=0.5),
        error_sampler=None,
        base_seed=1,
        hash="dethash",
        name="det",
        membership="deterministic",
    )
    sim = simulate(spec, 0)
    logits = sim.intercept + sim.X @ sim.b
    np.testing.assert_array_equal(sim.z, (logits > 0).astype(int))
    # effect=2, intercept=-1 -> z equals the binary causal column exactly
    np.testing.assert_array_equal(sim.z, X[:, 1].astype(int))


def test_membership_defaults_to_stochastic():
    from core import SimulationSpec
    from gibss.distributions import PointMass
    spec = SimulationSpec(
        design_sampler=lambda rng: np.zeros((2, 2)),
        effect_sampler=lambda Xarg, rng: ([], []),
        intercept=-1.0,
        f0=PointMass(0.0),
        f1=PointMass(1.0),
        error_sampler=None,
        base_seed=1,
        hash="h",
        name="n",
    )
    assert spec.membership == "stochastic"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core_run_methods.py::test_deterministic_membership_pins_z_to_threshold tests/test_core_run_methods.py::test_membership_defaults_to_stochastic -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'membership'`.

- [ ] **Step 3: Add the `membership` field**

In `core.py`, the `SimulationSpec` dataclass currently ends:

```python
    base_seed: int
    hash: str
    name: str = ""
```

Change to:

```python
    base_seed: int
    hash: str
    name: str = ""
    membership: str = "stochastic"
```

- [ ] **Step 4: Branch the `z` generation**

In `core.py` `simulate`, replace this line (~116):

```python
    z = rng.binomial(1, _sigmoid(logits)).astype(int)
```

with:

```python
    if simulation_spec.membership == "deterministic":
        z = (logits > 0).astype(int)
    else:
        z = rng.binomial(1, _sigmoid(logits)).astype(int)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_core_run_methods.py::test_deterministic_membership_pins_z_to_threshold tests/test_core_run_methods.py::test_membership_defaults_to_stochastic -v`
Expected: PASS (both).

- [ ] **Step 6: Run the full core test module (no regressions)**

Run: `uv run pytest tests/test_core_run_methods.py -v`
Expected: PASS (all).

- [ ] **Step 7: Commit**

```bash
git add core.py tests/test_core_run_methods.py
git commit -m "feat(core): deterministic membership mode for simulate"
```

---

### Task 2: Loader plumbing, enrichments, and 009 config

**Files:**
- Modify: `experiments/loader.py` (`resolve_simulation_from_coord`)
- Modify: `experiments/library.yaml` (add two enrichments under `enrichments:`)
- Modify: `experiments/009_cox_well_specified.yaml` (swap enrichment list in all 4 supercollections)
- Test: `tests/test_loader.py`

**Interfaces:**
- Consumes: `SimulationSpec.membership` from Task 1; `core.resolve_simulation_from_coord` / `loader.resolve_simulation_from_coord`.
- Produces: enrichment coord key `membership` read via `enrich.get("membership", "stochastic")`; new library enrichments `ser_b2_det`, `null_b0_det`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_loader.py`:

```python
def test_resolve_simulation_membership_flag():
    from experiments import loader
    base = {
        "design": {"function": "degenerate_X", "arguments": {}},
        "signal": {"f0": {"Exponential": {"rate": 1.0}},
                   "f1": {"Exponential": {"rate": 0.5}}},
        "error": None,
        "base_seed": 0,
    }
    det = dict(base, enrichment={
        "function": "uniform_single_effect",
        "arguments": {"causal_effect": 2.0},
        "intercept": -1.0,
        "membership": "deterministic",
    })
    sto = dict(base, enrichment={
        "function": "uniform_single_effect",
        "arguments": {"causal_effect": 2.0},
        "intercept": -2.0,
    })
    assert loader.resolve_simulation_from_coord(det).membership == "deterministic"
    assert loader.resolve_simulation_from_coord(sto).membership == "stochastic"
```

(If `degenerate_X` is not the registered name for the degenerate design, check `experiments/library.yaml` `designs:` and use the matching `function` value.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_loader.py::test_resolve_simulation_membership_flag -v`
Expected: FAIL — `AttributeError: 'SimulationSpec' object has no attribute 'membership'` is impossible after Task 1, so it fails instead on the assertion `== "deterministic"` (resolver does not yet pass the flag, so it stays `"stochastic"`).

- [ ] **Step 3: Plumb the flag in the resolver**

In `experiments/loader.py`, `resolve_simulation_from_coord` builds `core.SimulationSpec(...)`. Add the `membership` argument alongside `intercept`:

```python
    return core.SimulationSpec(
        design_sampler=_partial_from_entry(coord["design"]),
        effect_sampler=_partial_from_entry(enrich),
        intercept=float(enrich["intercept"]),
        membership=enrich.get("membership", "stochastic"),
        f0=resolve_distribution(sig["f0"]),
        f1=resolve_distribution(sig["f1"]),
        error_sampler=None if coord["error"] is None else _partial_from_entry(coord["error"]),
        base_seed=coord["base_seed"],
        hash=sim_hash(coord),
        name="",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_loader.py::test_resolve_simulation_membership_flag -v`
Expected: PASS.

- [ ] **Step 5: Add the `_det` enrichments to the library**

In `experiments/library.yaml`, under `enrichments:` (after `null_enrich`, ~line 265), add:

```yaml
  ser_b2_det:
    function: uniform_single_effect
    arguments:
      causal_effect: 2.0
    intercept: -1.0
    membership: deterministic
  null_b0_det:
    function: uniform_single_effect
    arguments:
      causal_effect: 0.0
    intercept: -1.0
    membership: deterministic
```

- [ ] **Step 6: Swap the enrichment list in experiment 009**

In `experiments/009_cox_well_specified.yaml`, every supercollection's `template` line reads:

```yaml
      template: {design: <design>, enrichment: [ser_b2, null_b0], error: noiseless}
```

Change `enrichment: [ser_b2, null_b0]` to `enrichment: [ser_b2_det, null_b0_det]` in all four supercollections (`009-hallmark-...`, `009-c2-...`, `009-c4-...`, `009-c5-...`).

- [ ] **Step 7: Verify the loader builds 009 end-to-end**

Run:
```bash
uv run python -c "
from experiments import loader
lib = loader.load_library()
spec = loader.resolve_simulation(lib, 'hallmark', 'ser_b2_det', 'exp_lambda_1over2', 'noiseless')
import core
sim = core.simulate(spec, 0)
import numpy as np
logits = sim.intercept + sim.X @ sim.b
assert spec.membership == 'deterministic'
assert np.array_equal(sim.z, (logits > 0).astype(int))
causal = sim.causal_indices[0]
assert np.array_equal(sim.z, np.asarray(sim.X[:, causal]).ravel().astype(int))
print('OK well-specified PH:', int(sim.z.sum()), 'in-set of', sim.z.size)
"
```
Expected: prints `OK well-specified PH: <k> in-set of <n>` with no assertion error.

(If `loader.load_library` / `loader.resolve_simulation` signatures differ, check `experiments/loader.py:60` and the library-load helper and adjust the call.)

- [ ] **Step 8: Run loader + core test modules (no regressions)**

Run: `uv run pytest tests/test_loader.py tests/test_core_run_methods.py -v`
Expected: PASS (all).

- [ ] **Step 9: Commit**

```bash
git add experiments/loader.py experiments/library.yaml experiments/009_cox_well_specified.yaml tests/test_loader.py
git commit -m "feat(009): deterministic-membership enrichments for well-specified Cox"
```

---

### Task 3: Mark existing stochastic sims up-to-date (avoid recompute)

**Files:** none (operational step).

**Interfaces:**
- Consumes: the `core.py` edit from Task 1, which Snakemake treats as a changed code input to every `simulate`/`fit` rule (`twogroup_experiments.snk:56,78`).
- Produces: existing stochastic sim/fit outputs marked current; only new `_det` batches build fresh.

**Rationale:** editing `core.py` bumps a tracked (non-`ancient()`) input, so Snakemake would re-run all twogroup sims. The stochastic path is byte-unchanged and seeds are deterministic, so re-running reproduces identical results — wasted compute only.

- [ ] **Step 1: Dry-run to see what Snakemake would rebuild**

Run: `uv run snakemake -n 2>&1 | tee /tmp/009_dryrun.txt`
Expected: lists `simulate`/`fit` jobs for existing batches (triggered by the `core.py` change) plus the new `_det` batches.

- [ ] **Step 2: Touch existing stochastic outputs as up-to-date**

Mark already-computed outputs current without recomputing (this updates mtimes only for outputs that already exist):

Run: `uv run snakemake --touch --rerun-triggers mtime`
Expected: existing outputs touched; no compute runs. New `_det` outputs do not exist yet, so they are not touched.

- [ ] **Step 3: Confirm only `_det` work remains**

Run: `uv run snakemake -n 2>&1 | tee /tmp/009_dryrun_after.txt`
Expected: remaining jobs are limited to the new `_det` (009 well-specified) batches and their downstream fits/reductions — no stochastic sim recompute.

- [ ] **Step 4: (Optional) Build the new well-specified results**

Run the 009 targets (use the project's normal target/profile, e.g. `uv run snakemake <009 reduction targets> --profile <profile>`).
Expected: only `_det` simulate/fit/reduce jobs execute.

No commit (operational step; nothing tracked changed).

---

## Notes for the implementer

- `Exponential` lives in `simulations/distributions.py`; import as `from simulations.distributions import Exponential`.
- `TwoGroupSimulation` exposes `X`, `b`, `intercept`, `z`, `causal_indices` — recompute `logits = intercept + X @ b` in tests since logits are not stored.
- `sim.X` may be a sparse matrix for gene-set designs; `np.asarray(sim.X[:, causal]).ravel()` normalizes a column to dense 1-D before comparison.
