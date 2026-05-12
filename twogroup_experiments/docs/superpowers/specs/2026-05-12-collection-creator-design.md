# Collection Creator Design

**Date:** 2026-05-12
**Status:** Approved

## Problem

Collections (batch × method groupings) are currently defined in `config.py` and baked into `manifest.json`. Adding or changing a collection requires editing Python source and regenerating the manifest. There is no lightweight post-hoc way to compose new collections from already-computed batches and methods.

## Goals

- Define collections without touching `config.py` or the manifest.
- Support three composition patterns: hand-picked batch × method selection, bulk per-simulation generation, and union of existing collections.
- One file on disk = one registered collection. Depositing a yaml is sufficient to register it with Snakemake.

## Architecture

### New directory: `results/collections/`

One yaml per collection. Each file is a self-contained dehydrated `CollectionSpec` — same structure as the existing per-alias `collection_spec.yaml` (produced by `dehydrate_hashed(collection_spec)`):

```yaml
__spec_hash__: <hash>
name: hallmark__ser_enrich__loc
batches:
  - __spec_hash__: <batch_hash>
    name: hallmark__ser_enrich__loc_1.00__batch0
    simulation_spec: { ... }
    replicates: [0, 1, ..., 49]
  - ...
method_specs:
  - __spec_hash__: <method_hash>
    fields: { name: logistic_threshold_L1, kwargs: { L: 1 } }
  - ...
```

Depositing a file here is the only registration step — no manifest change, no config edit.

### Manifest changes

`manifest_dict()` in `config.py` drops the `"collections"` key. The manifest retains `simulation_specs`, `method_specs`, and `batches`. The notebook reads the manifest to discover available atoms for building collections.

### New notebook: `notebooks/collections.py`

Marimo notebook with four sections:

**Manifest loader** — loads `results/manifest.json`, parses simulation metadata from `simulation_spec["fields"]` (structured keys: design, regime, parameter dimensions, values) and method metadata from `method_spec["fields"]` (family, L, threshold). Stateless data-prep for downstream cells. Does not parse dimension names from the string name — reads structured fields directly.

**Build mode** — hand-pick one collection:
- Simulation filters: multiselects for design, regime, parameter, value → shows matching batch count
- Method multiselect: grouped by family, shows method name + L
- Collection name text input
- Preview: table of selected batches and methods
- Write button: constructs `CollectionSpec`, calls `dehydrate_hashed`, writes `results/collections/{name}.yaml`

**Batch-generate mode** — bulk generate one collection per simulation spec:
- Method multiselect (same controls as Build)
- Preview: table of N collection names to be written (named after each simulation spec)
- Confirm + write all button: writes N yamls at once

**Compose mode** — union existing collections:
- Multiselect of existing files in `results/collections/`
- Preview: union batch count + method count
- Collection name text input
- Write button: unions `batches` and `method_specs` lists (dedup by `__spec_hash__`), reconstructs a `CollectionSpec` via `rehydrate_node`, calls `dehydrate_hashed` to get a fresh hash, writes yaml

### Snakemake changes (`twogroup_experiments.snk`)

**Collection discovery** — replace static alias list:

```python
# before
COLLECTION_ALIASES = tuple(config["collections"].keys())

# after
COLLECTION_ALIASES = glob_wildcards("results/collections/{name}.yaml").name
```

**Collection loading helper** — add once at top of snakefile:

```python
def load_collection_yaml(collection_alias):
    import yaml
    with open(f"results/collections/{collection_alias}.yaml") as f:
        return yaml.safe_load(f)
```

**All collection rules** — every rule that currently reads `config["collections"][wildcards.collection_alias]` switches to `load_collection_yaml(wildcards.collection_alias)`. Affected rules: `collection_alias` and all 9 `collection_*_plot_ready` rules. No other rule logic changes — the yaml format matches what the rules already expect.

**`collection_alias` rule** — the `collection_spec` output becomes a symlink to the source file:

```python
symlink_output(
    f"results/collections/{wildcards.collection_alias}.yaml",
    output.collection_spec,
)
```

This preserves backward compatibility for anything reading `results/by_alias/{alias}/collection_spec.yaml`.

## Data Flow

```
manifest.json (sim/method/batch atoms)
       │
       ▼
notebooks/collections.py
       │  writes
       ▼
results/collections/{name}.yaml   ←── source of truth
       │
       ├── symlinked to ──► results/by_alias/{alias}/collection_spec.yaml
       │
       └── discovered by snakemake (glob_wildcards)
                  │
                  ▼
           collection_alias rule → plot_ready rules → dashboard.py
```

## Testing

- Notebook writes a valid dehydrated yaml: load it back, verify `__spec_hash__`, batch count, method count.
- Compose union: verify dedup (overlapping batches/methods appear once).
- Snakemake dry-run after depositing a new yaml: confirm the new alias appears in the DAG.
- Symlink: verify `by_alias/{alias}/collection_spec.yaml` resolves to the correct content.
