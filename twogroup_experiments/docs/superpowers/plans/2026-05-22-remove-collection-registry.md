# Remove Collection Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `CollectionSpec` as a registered object; make collections a plot_config-driven concept where a sparse top-level `collections` section declares compute overrides (simulations, method filters), and supercollection entries remain `{name, alias}` — enabling different methods per simulation without YAML churn.

**Architecture:** `config.py` registers all methods and batches explicitly; `plot_config.yaml` gains a sparse top-level `collections` dict (only non-default entries); `twogroup_experiments.snk` resolves each supercollection collection entry by looking up its name in the top-level `collections` dict and filtering the manifest; `resolve_collection_spec` and `materialize_twogroup_experiment_collection_alias` rules are removed; `CollectionSpec` is deleted from `config_registry.py`, `config.py`, and `utils.py`.

**Tech Stack:** Python, Snakemake, PyYAML, pytest

---

## Background

`CollectionSpec` is registered with a hash that is never used as a file path. Collections currently bake a fixed method set in at registration time. To run different methods for different simulations, a new collection must be registered in `config.py`. Moving collection compute specs to `plot_config.yaml` makes experiment setup explicit in one place.

### plot_config.yaml structure (new)

```yaml
# Sparse — only collections that deviate from defaults need an entry.
# Default: simulations=[name], all registered methods, all L, all thresholds.
collections:
  "design=null_enrich__enrichment=null_enrich__signal=loc_0.50":
    method_families: [twogroup, twogroup_oracle, twogroup_oracle_init, twogroup_scale_fam, twogroup_loc_fam]
    L: [1]
  # simulations field optional — default: [name]
  # method_families / L / thresholds optional — default: all in manifest

supercollections:
  hallmark-signal-loc:
    collections:
      - {name: "design=hallmark__enrichment=ser_enrich__signal=loc_0.50", alias: "mu=0.50"}
      # alias is display label scoped to this supercollection — not a global id
```

`alias` stays in supercollection entries (display metadata, supercollection-scoped). Existing entries require zero changes. Only 14 null-enrich entries need new `collections` section entries.

### Resolution in snk

```python
_COLLECTION_DEFS = _PLOT_CONFIG.get("collections", {})  # sparse overrides

for each supercollection collection entry with name N:
    spec = _COLLECTION_DEFS.get(N, {})            # empty → all defaults
    batches  = resolve_batches(spec.get("simulations", [N]))
    methods  = resolve_methods(spec.get("method_families"), spec.get("L"), spec.get("thresholds"))
```

### Key invariants to preserve
- `batch_hash` and `method_hash` drive all file paths (unchanged)
- `manifest.json` structure unchanged
- `results/collections/{name}/plot_ready/*.parquet` structure unchanged
- `load_collection_yaml(name)` still returns `{name, batches: [...], method_specs: [...]}`

### What gets removed
- `resolve_collection_spec` rule — `collection_spec.yaml` (collection specs now in-memory)
- `materialize_twogroup_experiment_collection_alias` rule — `batch_hashes.txt` + symlinks
- `collection_spec` and `batch_hashes` inputs from all collection rules
- `CollectionSpec` registration from `config_registry.py`, `config.py`, `utils.py`
- Collection-level `__spec_hash__` from `build_collection_yaml_node` in `plot_ready.py`

---

## File Map

| File | Change |
|------|--------|
| `config.py` | Register all methods + null_enrich batches explicitly; remove `_register_*_collections`, `COLLECTION_SPECS`, `NULL_ENRICH_COLLECTION_SPECS`, `collection_yaml_node`, `CollectionSpec` import |
| `config_registry.py` | Remove `register_collection`, `register_collection_union`, `_collections_by_name`, `_collection_hashes`, `collections` property |
| `utils.py` | Remove `CollectionSpec` dataclass |
| `twogroup_experiments.snk` | Move `BATCH/METHOD_HASH_TO_INFO` before `_COLLECTION_YAMLS`; add resolution helpers; build `_COLLECTION_YAMLS` from plot_config `collections` + `supercollections`; remove `resolve_collection_spec` and `materialize` rules; strip `collection_spec`/`batch_hashes` inputs |
| `notebooks/plot_config.yaml` | Add top-level `collections:` section with null-enrich compute specs; supercollection entries unchanged |
| `plot_ready.py` | Simplify `build_collection_yaml_node` — drop `CollectionSpec` dep and collection-level `__spec_hash__` |
| `tests/test_twogroup_experiments.py` | Remove stale collection registry tests; add tests for explicit method/batch registration |
| `tests/test_plot_ready.py` | Update `test_build_collection_yaml_node_roundtrip` and `test_union_collection_yaml_nodes_deduplicates` to not expect collection-level `__spec_hash__` |

---

## Task 1: Register all methods and null_enrich batches explicitly in config.py

Currently methods enter the registry only as a side-effect of `register_collection`. This task decouples both.

**Files:**
- Modify: `config.py`
- Modify: `tests/test_twogroup_experiments.py`

- [ ] **Step 1: Write tests**

Add to `tests/test_twogroup_experiments.py`:

```python
def test_all_threshold_sweep_methods_in_registry():
    from config import REGISTRY, THRESHOLD_SWEEP_SER_SPECS, THRESHOLD_SWEEP_SUSIE_SPECS
    registered_hashes = {dehydrate_hashed(m)[HASH_KEY] for m in REGISTRY.methods}
    for spec in THRESHOLD_SWEEP_SER_SPECS + THRESHOLD_SWEEP_SUSIE_SPECS:
        h = dehydrate_hashed(spec)[HASH_KEY]
        assert h in registered_hashes, f"Method {spec.name} not in registry"


def test_null_enrich_simulations_have_registered_batches():
    from config import REGISTRY, NULL_ENRICH_SIMULATION_SPECS
    batch_sim_names = {b.simulation_spec.name for b in REGISTRY.batches}
    for spec in NULL_ENRICH_SIMULATION_SPECS:
        assert spec.name in batch_sim_names, f"No batch for {spec.name}"
```

- [ ] **Step 2: Run tests — confirm both fail**

```bash
cd twogroup_experiments && python -m pytest tests/test_twogroup_experiments.py::test_all_threshold_sweep_methods_in_registry tests/test_twogroup_experiments.py::test_null_enrich_simulations_have_registered_batches -v
```

Expected: both FAIL

- [ ] **Step 3: Register all methods explicitly in config.py**

After `THRESHOLD_SWEEP_SUSIE_SPECS` is defined (around line 268), add:

```python
REGISTRY.register_methods(THRESHOLD_SWEEP_SER_SPECS + THRESHOLD_SWEEP_SUSIE_SPECS)
```

This covers `DEFAULT_SER_SPECS`, `DEFAULT_SUSIE_SPECS`, and full threshold sweep — superset of what any collection needs.

- [ ] **Step 4: Register null_enrich batches explicitly**

Replace the `_register_null_enrich_collections` block (around lines 556-570):

```python
# REMOVE:
def _register_null_enrich_collections() -> tuple[CollectionSpec, ...]:
    return tuple(
        REGISTRY.register_collection(
            name=spec.name,
            simulations=(spec,),
            methods=_NULL_ENRICH_METHOD_SPECS,
            n_batches=N_BATCHES,
            replicates_per_batch=REPLICATES_PER_BATCH,
            batch_builder=batch_specs_for_simulation,
        )
        for spec in NULL_ENRICH_SIMULATION_SPECS
    )
NULL_ENRICH_COLLECTION_SPECS = _register_null_enrich_collections()
```

```python
# ADD:
_NULL_ENRICH_BATCH_SPECS = tuple(
    batch
    for sim in NULL_ENRICH_SIMULATION_SPECS
    for batch in batch_specs_for_simulation(
        sim,
        replicates_per_batch=REPLICATES_PER_BATCH,
        n_batches=N_BATCHES,
    )
)
REGISTRY.register_simulations(NULL_ENRICH_SIMULATION_SPECS)
REGISTRY.register_batches(_NULL_ENRICH_BATCH_SPECS)
```

- [ ] **Step 5: Remove tiny_test register_collection**

```python
# REMOVE:
REGISTRY.register_collection(
    name="tiny_test",
    batches=(TINY_TEST_BATCH,),
    methods=(_logistic_oracle_method_spec(L=1),),
)
# ADD:
REGISTRY.register_batches((TINY_TEST_BATCH,))
```

- [ ] **Step 6: Remove CollectionSpec import and COLLECTION_SPECS from config.py**

- Remove `CollectionSpec` from `from utils import BatchSpec, CollectionSpec` → `from utils import BatchSpec`
- Remove from `__all__`: `"COLLECTION_SPECS"`, `"NULL_ENRICH_COLLECTION_SPECS"`, `"collection_yaml_node"`
- Remove `_register_atomic_collection`, `_register_signal_collections`, `_register_correlation_collections`, `_register_n_feature_collections` functions and their top-level calls
- Remove `SIGNAL_COLLECTION_SPECS`, `CORRELATION_COLLECTION_SPECS`, `N_FEATURE_COLLECTION_SPECS`, `COLLECTION_SPECS` assignments
- Remove `collection_yaml_node` function
- Rename `_NULL_ENRICH_METHOD_SPECS` → `NULL_ENRICH_METHOD_SPECS` (remove leading underscore — useful as reference)

- [ ] **Step 7: Run tests**

```bash
python -m pytest tests/test_twogroup_experiments.py::test_all_threshold_sweep_methods_in_registry tests/test_twogroup_experiments.py::test_null_enrich_simulations_have_registered_batches -v
```

Expected: both PASS

- [ ] **Step 8: Run full test suite**

```bash
python -m pytest tests/ -v 2>&1 | grep -E "PASSED|FAILED|ERROR"
```

Expected: collection registry tests fail (Task 6); all others pass.

- [ ] **Step 9: Commit**

```bash
git add config.py tests/test_twogroup_experiments.py
git commit -m "refactor(config): register methods and batches explicitly, remove register_collection"
```

---

## Task 2: Update snk to build _COLLECTION_YAMLS from plot_config

**Files:**
- Modify: `twogroup_experiments.snk`

- [ ] **Step 1: Move BATCH_HASH_TO_INFO and METHOD_HASH_TO_INFO before _COLLECTION_YAMLS**

In the snk, `BATCH_HASH_TO_INFO` and `METHOD_HASH_TO_INFO` are currently defined at line ~80, after `_COLLECTION_YAMLS` at line ~38. Move both assignments to immediately after the `configfile` directive (line 22) so they're available when `_COLLECTION_YAMLS` is built.

- [ ] **Step 2: Replace import and _COLLECTION_YAMLS block**

Replace:

```python
from config import COLLECTION_SPECS, collection_yaml_node
...
_COLLECTION_YAMLS: dict = {
    collection.name: collection_yaml_node(collection.name)
    for collection in COLLECTION_SPECS
}
COLLECTION_ALIASES = sorted(collection.name for collection in COLLECTION_SPECS)
```

With:

```python
import re as _re

def _method_family(name: str) -> str:
    return _re.sub(r"_L\d+$", "", name)

def _resolve_collection_batches(simulations: list) -> list:
    sim_set = set(simulations)
    return [
        {**node, HASH_KEY: h}
        for h, node in BATCH_HASH_TO_INFO.items()
        if node["simulation_spec"]["fields"]["name"] in sim_set
    ]

def _resolve_collection_methods(method_families, L, thresholds) -> list:
    result = []
    for h, node in METHOD_HASH_TO_INFO.items():
        fields = node["fields"]
        kwargs = fields.get("kwargs", {})
        family = _method_family(fields["name"])
        if method_families is not None and family not in method_families:
            continue
        if L is not None and kwargs.get("L") not in L:
            continue
        thresh = kwargs.get("threshold")
        if thresh is not None and thresholds is not None and thresh not in thresholds:
            continue
        result.append({**node, HASH_KEY: h})
    return result

_COLLECTION_DEFS = _PLOT_CONFIG.get("collections", {})

_COLLECTION_YAMLS: dict[str, dict] = {}
for _sc in _PLOT_CONFIG.get("supercollections", {}).values():
    for _coll in _sc.get("collections", []):
        _name = _coll["name"]
        if _name in _COLLECTION_YAMLS:
            continue
        _spec = _COLLECTION_DEFS.get(_name, {})
        _sims = _spec.get("simulations", [_name])
        _COLLECTION_YAMLS[_name] = {
            "name": _name,
            "batches": _resolve_collection_batches(_sims),
            "method_specs": _resolve_collection_methods(
                _spec.get("method_families"),
                _spec.get("L"),
                _spec.get("thresholds"),
            ),
        }

COLLECTION_ALIASES = sorted(_COLLECTION_YAMLS.keys())
```

- [ ] **Step 3: Verify resolution correctness**

```bash
python -c "
import yaml, json, re
from pathlib import Path

manifest = json.loads(Path('results/manifest.json').read_text())
BATCH_HASH_TO_INFO = manifest['batches']
METHOD_HASH_TO_INFO = manifest['method_specs']
plot_config = yaml.safe_load(Path('notebooks/plot_config.yaml').read_text()) or {}

def method_family(n): return re.sub(r'_L\d+$', '', n)

def resolve_batches(sims):
    sim_set = set(sims)
    return [h for h, n in BATCH_HASH_TO_INFO.items()
            if n['simulation_spec']['fields']['name'] in sim_set]

def resolve_methods(families, L, thresholds):
    result = []
    for h, node in METHOD_HASH_TO_INFO.items():
        fields = node['fields']
        kwargs = fields.get('kwargs', {})
        fam = method_family(fields['name'])
        if families is not None and fam not in families: continue
        if L is not None and kwargs.get('L') not in L: continue
        thresh = kwargs.get('threshold')
        if thresh is not None and thresholds is not None and thresh not in thresholds: continue
        result.append(h)
    return result

defs = plot_config.get('collections', {})
yamls = {}
for sc in plot_config.get('supercollections', {}).values():
    for coll in sc.get('collections', []):
        name = coll['name']
        if name in yamls: continue
        spec = defs.get(name, {})
        yamls[name] = {
            'batches': resolve_batches(spec.get('simulations', [name])),
            'method_specs': resolve_methods(spec.get('method_families'), spec.get('L'), spec.get('thresholds')),
        }

print(f'Resolved {len(yamls)} collections')

hallmark = 'design=hallmark__enrichment=ser_enrich__signal=loc_0.50'
print(f'hallmark: {len(yamls[hallmark][\"method_specs\"])} methods, {len(yamls[hallmark][\"batches\"])} batches')
assert len(yamls[hallmark]['method_specs']) > 0
assert len(yamls[hallmark]['batches']) > 0
print('OK')
"
```

Expected: resolves collections, hallmark has methods and batches.

- [ ] **Step 4: Commit**

```bash
git add twogroup_experiments.snk
git commit -m "refactor(snk): build _COLLECTION_YAMLS from plot_config collections + supercollections"
```

---

## Task 3: Add top-level collections section to plot_config for null-enrich

**Files:**
- Modify: `notebooks/plot_config.yaml`

Existing supercollection entries (`{name, alias}`) require zero changes.

- [ ] **Step 1: Add collections section at top of plot_config.yaml**

Insert before the `settings:` key at the top of the file:

```yaml
collections:
  "design=null_enrich__enrichment=null_enrich__signal=loc_0.50":
    method_families: [twogroup, twogroup_oracle, twogroup_oracle_init, twogroup_scale_fam, twogroup_loc_fam]
    L: [1]
  "design=null_enrich__enrichment=null_enrich__signal=loc_1.00":
    method_families: [twogroup, twogroup_oracle, twogroup_oracle_init, twogroup_scale_fam, twogroup_loc_fam]
    L: [1]
  "design=null_enrich__enrichment=null_enrich__signal=loc_1.50":
    method_families: [twogroup, twogroup_oracle, twogroup_oracle_init, twogroup_scale_fam, twogroup_loc_fam]
    L: [1]
  "design=null_enrich__enrichment=null_enrich__signal=loc_2.00":
    method_families: [twogroup, twogroup_oracle, twogroup_oracle_init, twogroup_scale_fam, twogroup_loc_fam]
    L: [1]
  "design=null_enrich__enrichment=null_enrich__signal=loc_2.50":
    method_families: [twogroup, twogroup_oracle, twogroup_oracle_init, twogroup_scale_fam, twogroup_loc_fam]
    L: [1]
  "design=null_enrich__enrichment=null_enrich__signal=loc_3.00":
    method_families: [twogroup, twogroup_oracle, twogroup_oracle_init, twogroup_scale_fam, twogroup_loc_fam]
    L: [1]
  "design=null_enrich__enrichment=null_enrich__signal=scale_0.75":
    method_families: [twogroup, twogroup_oracle, twogroup_oracle_init, twogroup_scale_fam, twogroup_loc_fam]
    L: [1]
  "design=null_enrich__enrichment=null_enrich__signal=scale_1.00":
    method_families: [twogroup, twogroup_oracle, twogroup_oracle_init, twogroup_scale_fam, twogroup_loc_fam]
    L: [1]
  "design=null_enrich__enrichment=null_enrich__signal=scale_1.50":
    method_families: [twogroup, twogroup_oracle, twogroup_oracle_init, twogroup_scale_fam, twogroup_loc_fam]
    L: [1]
  "design=null_enrich__enrichment=null_enrich__signal=scale_1.75":
    method_families: [twogroup, twogroup_oracle, twogroup_oracle_init, twogroup_scale_fam, twogroup_loc_fam]
    L: [1]
  "design=null_enrich__enrichment=null_enrich__signal=scale_2.00":
    method_families: [twogroup, twogroup_oracle, twogroup_oracle_init, twogroup_scale_fam, twogroup_loc_fam]
    L: [1]
  "design=null_enrich__enrichment=null_enrich__signal=scale_3.00":
    method_families: [twogroup, twogroup_oracle, twogroup_oracle_init, twogroup_scale_fam, twogroup_loc_fam]
    L: [1]
  "design=null_enrich__enrichment=null_enrich__signal=scale_4.00":
    method_families: [twogroup, twogroup_oracle, twogroup_oracle_init, twogroup_scale_fam, twogroup_loc_fam]
    L: [1]
  "design=null_enrich__enrichment=null_enrich__signal=scale_5.00":
    method_families: [twogroup, twogroup_oracle, twogroup_oracle_init, twogroup_scale_fam, twogroup_loc_fam]
    L: [1]
```

- [ ] **Step 2: Verify null-enrich filtering**

```bash
python -c "
import yaml, json, re
from pathlib import Path

manifest = json.loads(Path('results/manifest.json').read_text())
METHOD_HASH_TO_INFO = manifest['method_specs']
plot_config = yaml.safe_load(Path('notebooks/plot_config.yaml').read_text()) or {}

def method_family(n): return re.sub(r'_L\d+$', '', n)

defs = plot_config.get('collections', {})
null_name = 'design=null_enrich__enrichment=null_enrich__signal=loc_0.50'
spec = defs[null_name]
families = spec['method_families']
L = spec['L']

methods = [
    (METHOD_HASH_TO_INFO[h]['fields']['name'], METHOD_HASH_TO_INFO[h]['fields']['kwargs']['L'])
    for h in METHOD_HASH_TO_INFO
    if method_family(METHOD_HASH_TO_INFO[h]['fields']['name']) in families
    and METHOD_HASH_TO_INFO[h]['fields']['kwargs'].get('L') in L
]
print('null_enrich methods:', set(n for n, _ in methods))
assert all('twogroup' in n for n, _ in methods), 'Non-twogroup method!'
assert all(l == 1 for _, l in methods), 'L != 1!'
print('OK')
"
```

Expected: only twogroup family, all L=1.

- [ ] **Step 3: Commit**

```bash
git add notebooks/plot_config.yaml
git commit -m "config(plot): add collections section with null-enrich method filters"
```

---

## Task 4: Remove resolve_collection_spec and materialize rules; strip redundant inputs

**Files:**
- Modify: `twogroup_experiments.snk`

- [ ] **Step 1: Remove rule resolve_collection_spec**

Delete the entire block:
```python
rule resolve_collection_spec:
    input:
        manifest=f"{RESULTS_ROOT}/manifest.json",
    output:
        f"{RESULTS_ROOT}/collections/{{collection_alias}}/collection_spec.yaml",
    run:
        collection = load_collection_yaml(wildcards.collection_alias)
        write_yaml(collection, output[0])
```

- [ ] **Step 2: Remove rule materialize_twogroup_experiment_collection_alias**

Delete the entire block (lines 136-212) — eliminates `batch_hashes.txt` and the symlink tree under `collections/{alias}/batches/`.

- [ ] **Step 3: Strip collection_spec and batch_hashes inputs from collection rules**

For each rule below, remove the listed input lines:

`rule collection_method_metadata` — remove:
```python
collection_spec=rules.resolve_collection_spec.output,
batch_hashes=f"{RESULTS_ROOT}/collections/{{collection_alias}}/batch_hashes.txt",
```

`rule collection_simulation_metadata` — remove:
```python
collection_spec=rules.resolve_collection_spec.output,
```
(No file inputs remain — correct; reads from in-memory `load_collection_yaml()`.)

`rule collection_sample_metadata` — remove:
```python
collection_spec=rules.resolve_collection_spec.output,
batch_hashes=f"{RESULTS_ROOT}/collections/{{collection_alias}}/batch_hashes.txt",
```

`rule collection_pip_plot_data` — remove:
```python
collection_spec=rules.resolve_collection_spec.output,
```

`rule collection_cs_plot_data` — remove:
```python
collection_spec=rules.resolve_collection_spec.output,
```

- [ ] **Step 4: Simplify twogroup_experiments_target**

Replace current inputs (which depend on `batch_hashes` and `collection_spec`) with direct dependency on all 5 plot_ready parquets:

```python
rule twogroup_experiments_target:
    input:
        lambda wildcards: expand(
            f"{RESULTS_ROOT}/collections/{wildcards.collection_alias}/plot_ready/{{name}}",
            name=[
                "method_metadata.parquet",
                "simulation_metadata.parquet",
                "sample_metadata.parquet",
                "pip_plot_data.parquet",
                "cs_plot_data.parquet",
            ],
        ),
    output:
        f"{RESULTS_ROOT}/collections/{{collection_alias}}/plot_ready/out.txt",
    shell:
        "echo ran > {output[0]}"
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/ -v 2>&1 | grep -E "PASSED|FAILED|ERROR"
```

Expected: `test_collection_plot_ready_snakemake_rules_are_declared` passes; collection registry tests fail (Task 6).

- [ ] **Step 6: Commit**

```bash
git add twogroup_experiments.snk
git commit -m "refactor(snk): remove resolve_collection_spec and materialize rules, strip redundant inputs"
```

---

## Task 5: Remove CollectionSpec from config_registry, utils, plot_ready

**Files:**
- Modify: `config_registry.py`
- Modify: `utils.py`
- Modify: `plot_ready.py`

- [ ] **Step 1: Remove collection machinery from config_registry.py**

From `__init__`, remove:
```python
self._collections_by_name: dict[str, CollectionSpec] = {}
self._collection_hashes: dict[str, CollectionSpec] = {}
```

Remove `collections` property, `register_collection` method, `register_collection_union` method.

Remove `CollectionSpec` from import:
```python
# was:
from utils import BatchSpec, CollectionSpec
# becomes:
from utils import BatchSpec
```

- [ ] **Step 2: Remove CollectionSpec from utils.py**

Delete lines 34-38:
```python
@dataclass(frozen=True)
class CollectionSpec:
    name: str
    batches: tuple[BatchSpec, ...]
    method_specs: tuple[MethodSpec, ...]
```

- [ ] **Step 3: Simplify build_collection_yaml_node in plot_ready.py**

Replace the function body — drop `CollectionSpec` instantiation and collection-level hash:

```python
def build_collection_yaml_node(
    name: str,
    batch_nodes: list[dict],
    method_nodes: list[dict],
) -> dict:
    return {
        "name": name,
        "batches": batch_nodes,
        "method_specs": method_nodes,
    }
```

Remove `CollectionSpec` from import:
```python
# was:
from utils import attach_spec_metadata, CollectionSpec
# becomes:
from utils import attach_spec_metadata
```

Check if `dehydrate_hashed` is used elsewhere in `plot_ready.py`; remove from imports if `build_collection_yaml_node` was its only use.

- [ ] **Step 4: Verify no remaining CollectionSpec references**

```bash
grep -rn "CollectionSpec\|register_collection\|collection_yaml_node" \
  --include="*.py" --include="*.snk" . \
  | grep -v "__pycache__"
```

Expected: only in test files (addressed Task 6) and possibly `notebooks/collections.py` (check — `build_collection_yaml_node` still exists, just without collection-level hash).

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/ -v 2>&1 | grep -E "PASSED|FAILED|ERROR"
```

Expected: only the 5 stale collection registry tests fail.

- [ ] **Step 6: Commit**

```bash
git add config_registry.py utils.py plot_ready.py
git commit -m "refactor: remove CollectionSpec from config_registry, utils, plot_ready"
```

---

## Task 6: Update tests

**Files:**
- Modify: `tests/test_twogroup_experiments.py`
- Modify: `tests/test_plot_ready.py`

- [ ] **Step 1: Remove stale collection registry tests from test_twogroup_experiments.py**

Delete:
- `test_collectionspec_owns_batches_and_methods`
- `test_config_registry_register_collection_accumulates_unique_specs`
- `test_config_registry_register_collection_is_idempotent_for_duplicates`
- `test_config_registry_register_collection_union_reuses_batches_and_methods`
- `test_config_registry_register_collection_union_raises_for_unknown_collection`

Remove `CollectionSpec` from the import block.

- [ ] **Step 2: Update test_build_collection_yaml_node_roundtrip**

```python
def test_build_collection_yaml_node_roundtrip():
    import json
    from pathlib import Path

    manifest = json.loads(
        (Path(__file__).parent.parent / "results" / "manifest.json").read_text()
    )
    batch_hash = next(iter(manifest["batches"]))
    method_hash = next(iter(manifest["method_specs"]))

    result = plot_ready.build_collection_yaml_node(
        name="test_collection",
        batch_nodes=[manifest["batches"][batch_hash]],
        method_nodes=[manifest["method_specs"][method_hash]],
    )

    assert result["name"] == "test_collection"
    assert len(result["batches"]) == 1
    assert len(result["method_specs"]) == 1
    assert "__spec_hash__" not in result
```

- [ ] **Step 3: Update test_union_collection_yaml_nodes_deduplicates**

```python
def test_union_collection_yaml_nodes_deduplicates():
    import json
    from pathlib import Path

    manifest = json.loads(
        (Path(__file__).parent.parent / "results" / "manifest.json").read_text()
    )
    batch_hashes = list(manifest["batches"].keys())[:2]
    method_hash = next(iter(manifest["method_specs"]))
    method_node = manifest["method_specs"][method_hash]

    node_a = plot_ready.build_collection_yaml_node(
        name="a",
        batch_nodes=[manifest["batches"][batch_hashes[0]]],
        method_nodes=[method_node],
    )
    node_b = plot_ready.build_collection_yaml_node(
        name="b",
        batch_nodes=[manifest["batches"][h] for h in batch_hashes],
        method_nodes=[method_node],
    )

    result = plot_ready.union_collection_yaml_nodes("union", [node_a, node_b])

    assert result["name"] == "union"
    assert len(result["batches"]) == 2
    assert len(result["method_specs"]) == 1
    assert "__spec_hash__" not in result
```

- [ ] **Step 4: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_twogroup_experiments.py tests/test_plot_ready.py
git commit -m "test: remove stale collection registry tests, update plot_ready node tests"
```

---

## Self-Review

**Spec coverage:**
- ✅ Top-level `collections` section in plot_config — sparse, only non-defaults
- ✅ `alias` stays in supercollection entries — zero changes to existing 158 entries
- ✅ Collection entries support `simulations` (union), `method_families`, `L`, `thresholds` — all optional
- ✅ Null-enrich entries defined in `collections` section (14 entries)
- ✅ `CollectionSpec` removed from `utils.py`, `config_registry.py`, `config.py`
- ✅ All methods registered explicitly via `THRESHOLD_SWEEP_SER_SPECS + THRESHOLD_SWEEP_SUSIE_SPECS`
- ✅ null_enrich batches registered explicitly
- ✅ `resolve_collection_spec` rule and `collection_spec.yaml` eliminated
- ✅ `materialize_twogroup_experiment_collection_alias` rule and symlinks eliminated
- ✅ `build_collection_yaml_node` simplified — no `CollectionSpec`, no collection-level hash
- ✅ Tests updated

**Watch out:**
- `BATCH_HASH_TO_INFO`/`METHOD_HASH_TO_INFO` must be defined before `_COLLECTION_YAMLS` construction in snk — verify ordering in Task 2 Step 1.
- `notebooks/collections.py` uses `build_collection_yaml_node` and `union_collection_yaml_nodes` — both remain compatible (same signature, just no `__spec_hash__` in output). Check if any notebook code reads `result["__spec_hash__"]`.
- `SIGNAL_COLLECTION_SPECS`, `CORRELATION_COLLECTION_SPECS`, `N_FEATURE_COLLECTION_SPECS` — grep for any external imports before removing in Task 1 Step 6.
