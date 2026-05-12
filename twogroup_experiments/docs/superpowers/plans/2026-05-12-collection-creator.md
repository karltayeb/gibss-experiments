# Collection Creator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace config-baked collection definitions with file-based discovery — one yaml per collection in `results/collections/` — and add a Marimo notebook to build, batch-generate, and compose collections from the manifest.

**Architecture:** `results/collections/{name}.yaml` is the source of truth for each collection (a dehydrated `CollectionSpec` dict). Snakemake discovers collections by globbing that directory. A new `notebooks/collections.py` Marimo notebook writes these yamls by browsing the manifest.

**Tech Stack:** Python, Marimo, Polars, PyYAML, Snakemake, existing `core.py` (`HASH_KEY`, `rehydrate_node`, `dehydrate_hashed`, `CollectionSpec`), `utils.py` (`BatchSpec`, `write_yaml`, `symlink_output`)

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `results/collections/` | Create dir | Stores one yaml per collection |
| `plot_ready.py` | Modify | Add `build_collection_yaml_node`, `union_collection_yaml_nodes` |
| `twogroup_experiments.snk` | Modify | Glob-based discovery, `load_collection_yaml` helper, update all collection rules |
| `config.py` | Modify | Drop `collections` from `manifest_dict()` |
| `notebooks/collections.py` | Create | Marimo notebook: manifest loader, build, batch-generate, compose |
| `tests/test_plot_ready.py` | Modify | Tests for the two new helpers |

---

## Task 1: Migrate existing collection yamls and create directory

**Files:**
- Create: `results/collections/` (directory)
- Migrate: 3 existing `results/by_alias/*/collection_spec.yaml` files

- [ ] **Step 1: Create the collections directory**

```bash
mkdir -p results/collections
```

- [ ] **Step 2: Copy the three existing collection specs**

```bash
for alias in hallmark__ser_enrich__loc hallmark__ser_enrich__scale hallmark__ser__all__ser_fits; do
  cp results/by_alias/$alias/collection_spec.yaml results/collections/$alias.yaml
done
```

- [ ] **Step 3: Verify the yamls are valid**

```bash
python3 -c "
import yaml, pathlib
for p in pathlib.Path('results/collections').glob('*.yaml'):
    c = yaml.safe_load(p.read_text())
    print(p.name, '| batches:', len(c['batches']), '| methods:', len(c['method_specs']))
"
```

Expected output: three lines, each with a non-zero batch and method count.

- [ ] **Step 4: Commit**

```bash
git add results/collections/
git commit -m "feat: create results/collections/ with migrated collection yamls"
```

---

## Task 2: Add collection builder helpers to `plot_ready.py` and test them

**Files:**
- Modify: `plot_ready.py`
- Modify: `tests/test_plot_ready.py`

- [ ] **Step 1: Write the failing tests**

Add to the bottom of `tests/test_plot_ready.py`:

```python
def test_build_collection_yaml_node_roundtrip():
    import yaml, json
    from pathlib import Path

    manifest = json.loads(
        (Path(__file__).parent.parent / "results" / "manifest.json").read_text()
    )
    # pick one batch and one method from the manifest
    batch_hash = next(iter(manifest["batches"]))
    method_hash = next(iter(manifest["method_specs"]))
    batch_node = manifest["batches"][batch_hash]
    method_node = manifest["method_specs"][method_hash]

    result = plot_ready.build_collection_yaml_node(
        name="test_collection",
        batch_nodes=[batch_node],
        method_nodes=[method_node],
    )

    assert result["name"] == "test_collection"
    assert len(result["batches"]) == 1
    assert len(result["method_specs"]) == 1
    assert "__spec_hash__" in result


def test_union_collection_yaml_nodes_deduplicates():
    import json
    from pathlib import Path

    manifest = json.loads(
        (Path(__file__).parent.parent / "results" / "manifest.json").read_text()
    )
    batch_hashes = list(manifest["batches"].keys())[:2]
    method_hash = next(iter(manifest["method_specs"]))
    batch_nodes = [manifest["batches"][h] for h in batch_hashes]
    method_node = manifest["method_specs"][method_hash]

    # two collections sharing one batch and the same method
    node_a = plot_ready.build_collection_yaml_node(
        name="a", batch_nodes=batch_nodes[:1], method_nodes=[method_node]
    )
    node_b = plot_ready.build_collection_yaml_node(
        name="b", batch_nodes=batch_nodes, method_nodes=[method_node]
    )

    result = plot_ready.union_collection_yaml_nodes("union", [node_a, node_b])

    assert result["name"] == "union"
    assert len(result["batches"]) == 2   # deduped
    assert len(result["method_specs"]) == 1  # deduped
    assert "__spec_hash__" in result
```

- [ ] **Step 2: Run to confirm they fail**

```bash
uv run pytest tests/test_plot_ready.py::test_build_collection_yaml_node_roundtrip tests/test_plot_ready.py::test_union_collection_yaml_nodes_deduplicates -v
```

Expected: `AttributeError: module 'plot_ready' has no attribute 'build_collection_yaml_node'`

- [ ] **Step 3: Implement the two helpers in `plot_ready.py`**

Add after the existing imports at the top of `plot_ready.py`:

```python
from core import HASH_KEY, CollectionSpec, rehydrate_node, dehydrate_hashed
```

Then add these two functions anywhere after the existing helpers (e.g., before `available_plot_ready_collections`):

```python
def build_collection_yaml_node(
    name: str,
    batch_nodes: list[dict[str, Any]],
    method_nodes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a dehydrated CollectionSpec dict from manifest batch/method nodes."""
    from utils import BatchSpec
    batches = tuple(rehydrate_node(b) for b in batch_nodes)
    methods = tuple(rehydrate_node(m) for m in method_nodes)
    collection = CollectionSpec(name=name, batches=batches, method_specs=methods)
    return dehydrate_hashed(collection)


def union_collection_yaml_nodes(
    name: str,
    collection_nodes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Union multiple dehydrated collection specs, deduplicating by __spec_hash__."""
    seen_batches: dict[str, dict[str, Any]] = {}
    seen_methods: dict[str, dict[str, Any]] = {}
    for node in collection_nodes:
        for batch in node["batches"]:
            seen_batches.setdefault(batch[HASH_KEY], batch)
        for method in node["method_specs"]:
            seen_methods.setdefault(method[HASH_KEY], method)
    return build_collection_yaml_node(
        name=name,
        batch_nodes=list(seen_batches.values()),
        method_nodes=list(seen_methods.values()),
    )
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/test_plot_ready.py -v
```

Expected: all tests pass, including the two new ones.

- [ ] **Step 5: Commit**

```bash
git add plot_ready.py tests/test_plot_ready.py
git commit -m "feat: add build_collection_yaml_node and union_collection_yaml_nodes"
```

---

## Task 3: Rework Snakemake collection discovery and loading

**Files:**
- Modify: `twogroup_experiments.snk`

The goal: replace every `config["collections"][wildcards.collection_alias]` with `load_collection_yaml(wildcards.collection_alias)` and switch `COLLECTION_ALIASES` from config keys to a glob.

- [ ] **Step 1: Replace `COLLECTION_ALIASES` and add the loader helper**

Find these lines near the top of `twogroup_experiments.snk`:

```python
configfile: RESULTS_ROOT + "/manifest.json"

COLLECTION_ALIASES = tuple(config["collections"].keys())
```

Replace with:

```python
configfile: RESULTS_ROOT + "/manifest.json"

COLLECTION_ALIASES = glob_wildcards(f"{RESULTS_ROOT}/collections/{{name}}.yaml").name


def load_collection_yaml(collection_alias: str) -> dict:
    import yaml
    path = f"{RESULTS_ROOT}/collections/{collection_alias}.yaml"
    with open(path) as fh:
        return yaml.safe_load(fh)
```

- [ ] **Step 2: Update `materialize_twogroup_experiment_collection_alias` input lambdas**

In the rule `materialize_twogroup_experiment_collection_alias`, every lambda references `config["collections"][wildcards.collection_alias]`. Replace all five occurrences in the `input:` block with `load_collection_yaml(wildcards.collection_alias)`:

```python
rule materialize_twogroup_experiment_collection_alias:
    input:
        batch_specs=lambda wildcards: expand(
            f"{RESULTS_ROOT}/by_batch/{{batch_hash}}/batch_spec.yaml",
            batch_hash=[
                batch[HASH_KEY]
                for batch in load_collection_yaml(wildcards.collection_alias)["batches"]
            ],
        ),
        simulation_specs=lambda wildcards: expand(
            f"{RESULTS_ROOT}/by_batch/{{batch_hash}}/simulation_spec.yaml",
            batch_hash=[
                batch[HASH_KEY]
                for batch in load_collection_yaml(wildcards.collection_alias)["batches"]
            ],
        ),
        simulations=lambda wildcards: expand(
            f"{RESULTS_ROOT}/by_batch/{{batch_hash}}/simulations.parquet",
            batch_hash=[
                batch[HASH_KEY]
                for batch in load_collection_yaml(wildcards.collection_alias)["batches"]
            ],
        ),
        fits=lambda wildcards: [
            f"{RESULTS_ROOT}/by_batch/{batch[HASH_KEY]}/fits/{method_spec[HASH_KEY]}/fits.parquet"
            for batch in load_collection_yaml(wildcards.collection_alias)["batches"]
            for method_spec in load_collection_yaml(wildcards.collection_alias)["method_specs"]
        ],
        method_specs=lambda wildcards: [
            f"{RESULTS_ROOT}/by_batch/{batch[HASH_KEY]}/fits/{method_spec[HASH_KEY]}/method_spec.yaml"
            for batch in load_collection_yaml(wildcards.collection_alias)["batches"]
            for method_spec in load_collection_yaml(wildcards.collection_alias)["method_specs"]
        ],
    output:
        batch_hashes=f"{RESULTS_ROOT}/by_alias/{{collection_alias}}/batch_hashes.txt",
        collection_spec=f"{RESULTS_ROOT}/by_alias/{{collection_alias}}/collection_spec.yaml",
        batches_dir=directory(f"{RESULTS_ROOT}/by_alias/{{collection_alias}}/batches"),
```

- [ ] **Step 3: Update the `run:` block of the same rule**

Replace the two `config["collections"][...]` reads and the `write_yaml(collection, output.collection_spec)` line:

```python
    run:
        from pathlib import Path

        collection = load_collection_yaml(wildcards.collection_alias)
        batch_hashes = [batch[HASH_KEY] for batch in collection["batches"]]
        batch_hashes_text = "".join(f"{batch_hash}\n" for batch_hash in batch_hashes)
        write_text(batch_hashes_text, output.batch_hashes)
        symlink_output(
            f"{RESULTS_ROOT}/collections/{wildcards.collection_alias}.yaml",
            output.collection_spec,
        )
        # The remaining lines of the run: block are unchanged — they create batches_dir,
        # mkdir per batch, and symlink batch_spec.yaml / simulation_spec.yaml /
        # simulations.parquet / fits.parquet / method_spec.yaml using the local
        # `collection` variable (now loaded from yaml). Only the two lines above change.
```

The rest of the `run:` block (creating `batches_dir`, symlinking per-batch files) uses the local `collection` variable, which is now loaded from the yaml — no other changes needed there.

- [ ] **Step 4: Update the 9 `collection_*_plot_ready` rule `run:` blocks**

Each of these rules has a line like:
```python
collection = config["collections"][wildcards.collection_alias]
```

Replace every occurrence with:
```python
collection = load_collection_yaml(wildcards.collection_alias)
```

Rules to update (9 total):
- `collection_method_metadata`
- `collection_simulation_metadata`
- `collection_sample_metadata`
- `collection_pip_calibration_plot_ready`
- `collection_power_fdp_plot_ready`
- `collection_causal_pip_plot_ready`
- `collection_cs_raw_plot_ready`
- `collection_cs_size_histogram_plot_ready`
- `collection_ser_log_bf_histogram_plot_ready`

- [ ] **Step 5: Verify with a dry-run**

```bash
uv run snakemake --snakefile twogroup_experiments.snk --dry-run \
    results/by_alias/hallmark__ser_enrich__loc/plot_ready/out.txt 2>&1 | tail -15
```

Expected: dry-run shows the target would be satisfied (all outputs up-to-date), no errors about missing collections.

- [ ] **Step 6: Commit**

```bash
git add twogroup_experiments.snk
git commit -m "feat: snakemake discovers collections from results/collections/ glob"
```

---

## Task 4: Drop collections from `manifest_dict()`

**Files:**
- Modify: `config.py`

- [ ] **Step 1: Remove the collections block from `manifest_dict()`**

In `config.py`, find `manifest_dict()`. It has a section:

```python
manifest: dict[str, object] = {
    "simulation_specs": {},
    "method_specs": {},
    "batches": {},
    "collections": {},
}
```

And later:

```python
for collection in REGISTRY.collections:
    collection_batches = [...]
    collection_method_specs = [...]
    collection_node = {...}
    collections[collection.name] = {...}
```

Remove the `"collections": {}` key from the initial dict, remove the `collections = manifest["collections"]` assignment, the `assert isinstance(collections, dict)` line, and the entire `for collection in REGISTRY.collections:` loop. Also remove `collections` from the return value (it's inlined as part of `manifest`).

Result — `manifest_dict()` initial dict becomes:

```python
manifest: dict[str, object] = {
    "simulation_specs": {},
    "method_specs": {},
    "batches": {},
}
```

- [ ] **Step 2: Regenerate the manifest**

```bash
uv run python config.py
```

Expected: `Manifest written to results/manifest_....json` with no errors.

- [ ] **Step 3: Verify the manifest no longer has a collections key**

```bash
python3 -c "
import json
m = json.load(open('results/manifest.json'))
print('keys:', list(m.keys()))
assert 'collections' not in m
print('OK')
"
```

- [ ] **Step 4: Copy new manifest to `results/manifest.json`**

```bash
cp results/manifest_$(date +%Y_%m_%d).json results/manifest.json
```

- [ ] **Step 5: Confirm snakemake dry-run still works**

```bash
uv run snakemake --snakefile twogroup_experiments.snk --dry-run \
    results/by_alias/hallmark__ser_enrich__loc/plot_ready/out.txt 2>&1 | tail -5
```

Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add config.py results/manifest.json
git commit -m "feat: drop collections from manifest, source of truth is results/collections/"
```

---

## Task 5: Build `notebooks/collections.py` — manifest loader + build mode

**Files:**
- Create: `notebooks/collections.py`

The notebook reads `results/manifest.json` and exposes a UI to hand-pick batches × methods and write a collection yaml.

- [ ] **Step 1: Create the notebook skeleton with manifest loader**

Create `notebooks/collections.py`:

```python
import marimo

__generated_with = "0.23.5"
app = marimo.App(width="columns")

with app.setup:
    import json
    import sys
    from pathlib import Path

    import marimo as mo

    parent_dir = str(Path(__file__).parent.parent)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    import plot_ready
    from utils import write_yaml


@app.cell(hide_code=True)
def title_cell():
    _md = mo.md("# Collection Creator")
    _md
    return


@app.cell
def manifest_cell():
    _manifest_path = Path(__file__).parent.parent / "results" / "manifest.json"
    manifest = json.loads(_manifest_path.read_text())
    mo.md(f"Loaded manifest: **{len(manifest['batches'])} batches**, **{len(manifest['method_specs'])} methods**, **{len(manifest['simulation_specs'])} simulations**")
    return (manifest,)


@app.cell
def sim_index_cell(manifest):
    """Parse simulation dimensions from batch nodes for filter UI."""
    def _parse_sim_name(name: str) -> dict:
        parts = name.split("__")
        d = {"name": name, "design": parts[0] if parts else ""}
        if len(parts) > 1:
            d["regime"] = parts[1]
        if len(parts) > 2:
            param_part = parts[2]
            idx = param_part.rfind("_")
            if idx > 0:
                d["parameter"] = param_part[:idx]
                try:
                    d["value"] = float(param_part[idx + 1:])
                except ValueError:
                    d["parameter"] = param_part
            else:
                d["parameter"] = param_part
        return d

    def _parse_method_node(node: dict) -> dict:
        fields = node.get("fields", {})
        name = fields.get("name", "")
        kwargs = fields.get("kwargs", {})
        return {
            "hash": node["__spec_hash__"],
            "name": name,
            "L": kwargs.get("L"),
            "threshold": kwargs.get("threshold"),
            "family": name.rsplit("_L", 1)[0] if "_L" in name else name,
        }

    # batch hash → {batch_node, sim_dims}
    batch_index = {}
    for _bh, _bn in manifest["batches"].items():
        _sim_name = _bn["simulation_spec"]["fields"]["name"]
        batch_index[_bh] = {**_bn, "sim_dims": _parse_sim_name(_sim_name)}

    method_index = {
        mh: _parse_method_node(mn)
        for mh, mn in manifest["method_specs"].items()
    }

    return batch_index, method_index
```

- [ ] **Step 2: Add mode selector and build mode filters**

Append to `notebooks/collections.py`:

```python
@app.cell
def mode_cell():
    mode_tabs = mo.ui.tabs({
        "Build": mo.md(""),
        "Batch-generate": mo.md(""),
        "Compose": mo.md(""),
    })
    mode_tabs
    return (mode_tabs,)


@app.cell
def build_filters_cell(batch_index, method_index, mode_tabs):
    mo.stop(mode_tabs.value != "Build")

    _all_designs = sorted({v["sim_dims"].get("design", "") for v in batch_index.values() if v["sim_dims"].get("design")})
    _all_regimes = sorted({v["sim_dims"].get("regime", "") for v in batch_index.values() if v["sim_dims"].get("regime")})
    _all_params = sorted({v["sim_dims"].get("parameter", "") for v in batch_index.values() if v["sim_dims"].get("parameter")})
    _all_families = sorted({v["family"] for v in method_index.values()})

    design_filter = mo.ui.multiselect(options=_all_designs, value=_all_designs, label="Design")
    regime_filter = mo.ui.multiselect(options=_all_regimes, value=_all_regimes, label="Regime")
    param_filter = mo.ui.multiselect(options=_all_params, value=_all_params, label="Parameter")
    family_filter = mo.ui.multiselect(options=_all_families, value=_all_families, label="Method family")

    mo.hstack([design_filter, regime_filter, param_filter, family_filter])
    return design_filter, family_filter, param_filter, regime_filter


@app.cell
def build_selectors_cell(batch_index, design_filter, family_filter, method_index, mode_tabs, param_filter, regime_filter):
    mo.stop(mode_tabs.value != "Build")
    import polars as pl

    _filtered_batches = {
        bh: bv for bh, bv in batch_index.items()
        if bv["sim_dims"].get("design") in design_filter.value
        and bv["sim_dims"].get("regime", "") in (regime_filter.value or [""])
        and bv["sim_dims"].get("parameter", "") in (param_filter.value or [""])
    }
    _filtered_methods = {
        mh: mv for mh, mv in method_index.items()
        if mv["family"] in family_filter.value
    }

    _batch_options = {bh: bv["name"] for bh, bv in _filtered_batches.items()}
    _method_options = {mh: f"{mv['name']} (L={mv['L']})" for mh, mv in _filtered_methods.items()}

    batch_multiselect = mo.ui.multiselect(
        options=_batch_options, value=list(_batch_options.keys()), label="Batches"
    )
    method_multiselect = mo.ui.multiselect(
        options=_method_options, value=list(_method_options.keys()), label="Methods"
    )
    collection_name_input = mo.ui.text(placeholder="my_collection_name", label="Collection name")

    mo.vstack([
        mo.hstack([batch_multiselect, method_multiselect]),
        collection_name_input,
        mo.md(f"**{len(batch_multiselect.value)} batches × {len(method_multiselect.value)} methods selected**"),
    ])
    return batch_multiselect, collection_name_input, method_multiselect


@app.cell
def build_write_cell(batch_multiselect, collection_name_input, manifest, method_multiselect, mode_tabs):
    mo.stop(mode_tabs.value != "Build")
    mo.stop(not collection_name_input.value.strip())

    _collections_dir = Path(__file__).parent.parent / "results" / "collections"
    _collections_dir.mkdir(parents=True, exist_ok=True)
    _out_path = _collections_dir / f"{collection_name_input.value.strip()}.yaml"

    def _do_write(_btn):
        _batch_nodes = [manifest["batches"][h] for h in batch_multiselect.value]
        _method_nodes = [manifest["method_specs"][h] for h in method_multiselect.value]
        _node = plot_ready.build_collection_yaml_node(
            name=collection_name_input.value.strip(),
            batch_nodes=_batch_nodes,
            method_nodes=_method_nodes,
        )
        write_yaml(_node, str(_out_path))

    write_btn = mo.ui.button(label=f"Write {_out_path.name}", on_click=_do_write)
    write_btn
    return (write_btn,)
```

- [ ] **Step 3: Start the notebook and verify the Build tab renders without error**

```bash
uv run marimo edit notebooks/collections.py
```

Open the notebook in a browser, select the Build tab, confirm filters and selectors render, confirm the write button appears when a name is entered.

- [ ] **Step 4: Write a test collection using the UI**

Enter name `test_build`, keep default selections, click Write. Verify:
```bash
python3 -c "
import yaml, pathlib
c = yaml.safe_load(pathlib.Path('results/collections/test_build.yaml').read_text())
print('name:', c['name'], '| batches:', len(c['batches']), '| methods:', len(c['method_specs']))
assert c['name'] == 'test_build'
print('OK')
"
```

Then delete the test file: `rm results/collections/test_build.yaml`

- [ ] **Step 5: Commit**

```bash
git add notebooks/collections.py
git commit -m "feat: collections notebook with manifest loader and build mode"
```

---

## Task 6: Add batch-generate mode to `notebooks/collections.py`

**Files:**
- Modify: `notebooks/collections.py`

- [ ] **Step 1: Add the batch-generate cell**

Append to `notebooks/collections.py`:

```python
@app.cell
def batchgen_cell(batch_index, manifest, method_index, mode_tabs):
    mo.stop(mode_tabs.value != "Batch-generate")

    _all_method_options = {
        mh: f"{mv['name']} (L={mv['L']})"
        for mh, mv in method_index.items()
    }

    batchgen_method_select = mo.ui.multiselect(
        options=_all_method_options,
        value=list(_all_method_options.keys()),
        label="Methods to include in every generated collection",
    )

    # one collection per unique simulation name (grouping all its batches)
    sim_to_batches: dict[str, list[str]] = {}
    for _bh, _bv in batch_index.items():
        _sname = _bv["simulation_spec"]["fields"]["name"]
        sim_to_batches.setdefault(_sname, []).append(_bh)

    _preview_rows = [
        {"collection": sim_name, "n_batches": len(batch_hashes)}
        for sim_name, batch_hashes in sorted(sim_to_batches.items())
    ]

    import polars as _pl
    _preview_df = _pl.from_dicts(_preview_rows)

    mo.vstack([
        batchgen_method_select,
        mo.md(f"Will generate **{len(sim_to_batches)} collection yamls**, one per simulation."),
        mo.ui.table(_preview_df, selection=None),
    ])
    return batchgen_method_select, sim_to_batches


@app.cell
def batchgen_write_cell(batchgen_method_select, manifest, method_index, mode_tabs, sim_to_batches):
    mo.stop(mode_tabs.value != "Batch-generate")

    _collections_dir = Path(__file__).parent.parent / "results" / "collections"

    def _do_batchgen(_btn):
        _collections_dir.mkdir(parents=True, exist_ok=True)
        _method_nodes = [manifest["method_specs"][h] for h in batchgen_method_select.value]
        for _sim_name, _batch_hashes in sim_to_batches.items():
            _batch_nodes = [manifest["batches"][h] for h in _batch_hashes]
            _node = plot_ready.build_collection_yaml_node(
                name=_sim_name,
                batch_nodes=_batch_nodes,
                method_nodes=_method_nodes,
            )
            write_yaml(_node, str(_collections_dir / f"{_sim_name}.yaml"))

    batchgen_write_btn = mo.ui.button(
        label=f"Generate {len(_sim_to_batches)} collection yamls",
        on_click=_do_batchgen,
    )
    batchgen_write_btn
    return (batchgen_write_btn,)
```

- [ ] **Step 2: Verify the Batch-generate tab renders**

Open `notebooks/collections.py` in marimo, switch to Batch-generate tab. Confirm the preview table and button appear.

- [ ] **Step 3: Commit**

```bash
git add notebooks/collections.py
git commit -m "feat: collections notebook batch-generate mode"
```

---

## Task 7: Add compose mode to `notebooks/collections.py`

**Files:**
- Modify: `notebooks/collections.py`

- [ ] **Step 1: Add the compose cell**

Append to `notebooks/collections.py`:

```python
@app.cell
def compose_cell(mode_tabs):
    mo.stop(mode_tabs.value != "Compose")

    _collections_dir = Path(__file__).parent.parent / "results" / "collections"
    _existing = {
        p.stem: str(p)
        for p in sorted(_collections_dir.glob("*.yaml"))
    }

    compose_select = mo.ui.multiselect(
        options=_existing,
        label="Collections to union",
    )
    compose_name_input = mo.ui.text(placeholder="my_union_collection", label="Union collection name")

    if compose_select.value:
        import yaml as _yaml
        _nodes = [_yaml.safe_load(open(p)) for p in compose_select.value]
        _all_batches = {b["__spec_hash__"] for n in _nodes for b in n["batches"]}
        _all_methods = {m["__spec_hash__"] for n in _nodes for m in n["method_specs"]}
        _preview_md = mo.md(
            f"Union: **{len(_all_batches)} batches**, **{len(_all_methods)} methods** (after dedup)"
        )
    else:
        _preview_md = mo.md("Select collections above to preview the union.")

    mo.vstack([compose_select, compose_name_input, _preview_md])
    return compose_name_input, compose_select


@app.cell
def compose_write_cell(compose_name_input, compose_select, mode_tabs):
    mo.stop(mode_tabs.value != "Compose")
    mo.stop(not compose_name_input.value.strip())
    mo.stop(not compose_select.value)

    import yaml as _yaml
    _collections_dir = Path(__file__).parent.parent / "results" / "collections"

    def _do_compose(_btn):
        _nodes = [_yaml.safe_load(open(p)) for p in compose_select.value]
        _union_node = plot_ready.union_collection_yaml_nodes(
            name=compose_name_input.value.strip(),
            collection_nodes=_nodes,
        )
        _out = _collections_dir / f"{compose_name_input.value.strip()}.yaml"
        write_yaml(_union_node, str(_out))

    compose_write_btn = mo.ui.button(
        label=f"Write union to {compose_name_input.value.strip()}.yaml",
        on_click=_do_compose,
    )
    compose_write_btn
    return (compose_write_btn,)
```

- [ ] **Step 2: Verify compose mode end-to-end**

Open `notebooks/collections.py`, switch to Compose. Select two existing collections (e.g., `hallmark__ser_enrich__loc` and `hallmark__ser_enrich__scale`), enter name `hallmark__ser_enrich__union`, click Write. Verify:

```bash
python3 -c "
import yaml, pathlib
c = yaml.safe_load(pathlib.Path('results/collections/hallmark__ser_enrich__union.yaml').read_text())
print('name:', c['name'], '| batches:', len(c['batches']), '| methods:', len(c['method_specs']))
"
```

Then clean up: `rm results/collections/hallmark__ser_enrich__union.yaml`

- [ ] **Step 3: Verify snakemake dry-run picks up a new collection**

```bash
cp results/collections/hallmark__ser_enrich__loc.yaml results/collections/dry_run_test.yaml
uv run snakemake --snakefile twogroup_experiments.snk --dry-run \
    results/by_alias/dry_run_test/plot_ready/out.txt 2>&1 | tail -5
rm results/collections/dry_run_test.yaml
```

Expected: dry-run shows jobs would be scheduled for the `dry_run_test` alias.

- [ ] **Step 4: Final test suite run**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add notebooks/collections.py
git commit -m "feat: collections notebook compose mode — complete collection creator"
```
