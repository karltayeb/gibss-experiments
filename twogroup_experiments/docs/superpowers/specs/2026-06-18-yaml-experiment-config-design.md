# YAML-driven experiment config

**Date:** 2026-06-18
**Status:** Design approved, pending spec review

## Problem

The simulation/method configuration is registry-driven and has grown burdensome:
`config.py` (~900 lines), `config_builders.py`, `config_registry.py` imperatively
construct thousands of `SimulationSpec`/`MethodSpec`/`BatchSpec` objects from dense
parameter grids, register them with dedup/conflict logic, and emit `manifest.json`.
A separate `plot_configs/` tree (`main.yaml` + numbered files) then *selects* among
the registered universe via name references and method-family filters.

This is hard to read, hard to extend, and the simulation universe is decoupled from
the experiments that actually use it.

## Goal

Replace the registry layer with a declarative, four-part YAML config that directly
expresses the simulations each experiment needs. A simulation is the combination of:

1. **design** — covariate matrix sampler (`design_sampler`)
2. **enrichment** — which features are causal + intercept (`effect_sampler` + `intercept`)
3. **signal** — null/non-null effect distributions (`f0`, `f1`)
4. **error** — noise sampler (`error_sampler`, `None` = standard normal)

Each is a named library entry of `function` + `arguments`. A simulation references one
of each by name. Methods are similarly `function` + `kwargs` library entries.

Not a goal: reproducing the full dense grids of the old registry. The config expresses
the simulations experiments actually consume, with compact expansion for sweeps.

## Non-goals / decisions

- **Clean break on hashes.** New construction yields new content hashes; existing
  `results/by_batch/<hash>/` results are orphaned (not migrated). Cache reuse is a
  fully separable post-hoc concern (see Future work) that does not intersect loader
  development, so it is out of scope here.
- **`core.py` stays.** `SimulationSpec`, `MethodSpec`, `BatchSpec`, the
  `dehydrate/rehydrate/spec_hash` machinery, `simulate`, all `fit_*`/`summarize_*`
  methods, and all sampler functions remain. The loader reuses
  `core.dehydrate_hashed` so manifest node shapes are unchanged and the snakefile
  rehydration path is untouched.
- The numbered-file split is preserved (renamed `plot_configs/` → `experiments/`).
- **No coexistence.** Done on a new branch. `config.py` and the registry are deleted
  on that branch; the snakefile switches over fully. Experiments not yet ported are
  simply unavailable until ported — acceptable on a feature branch.

## File layout

```
experiments/
  library.yaml        # shared: defaults, designs, enrichments, signals,
                      #         errors, methods, plot_type_groups
  003_loc_snr.yaml    # supercollections: {...}   (proof migration)
  000_t_errors.yaml   # supercollections: {...}   (proof migration)
  ...                 # remaining 001,002,004-007 ported in follow-ups
```

`library.yaml` is the single shared library. Numbered files each hold
`supercollections:`. The loader globs `experiments/*.yaml`, treats `library.yaml`
as the library, and merges `supercollections` from the rest.

## Schemas

### Distribution mini-schema

Single-key map `{<TypeName>: {<ctor-kwargs>}}`, resolved against
`gibss.distributions`:

```yaml
{PointMass: {value: 0.0}}
{Normal: {loc: 2.0, scale: 0.1, estimate_loc: false, estimate_scale: false}}
{NormalMixture: {weights: [...], locs: [...], scales: [...]}}
```

Used in `signals` (`f0`/`f1`) and in method `kwargs` (e.g. two-group `f1` init).

### library.yaml

```yaml
defaults:
  base_seed: 20260501
  replicates_per_batch: 50
  n_batches: 1

designs:                                    # -> design_sampler = partial(fn, **arguments)
  hallmark:      {function: hallmark_gene_sets_X, arguments: {}}
  c4:            {function: c4_gene_sets_X, arguments: {}}
  gaussian_p100: {function: gaussian_markov_X, arguments: {n: 500, p: 100, rho: 0.9}}
  uniform_p100:  {function: uniform_markov_X,  arguments: {n: 500, p: 100, rho: 0.9}}

enrichments:                                # -> effect_sampler = partial(fn, **arguments) + intercept
  ser_b2:  {function: uniform_single_effect, arguments: {causal_effect: 2.0}, intercept: -2.0}
  null_b0: {function: uniform_single_effect, arguments: {causal_effect: 0.0}, intercept: -2.0}

signals:                                    # -> f0, f1
  loc_2.0:
    f0: {PointMass: {value: 0.0}}
    f1: {Normal: {loc: 2.0, scale: 0.1, estimate_loc: false, estimate_scale: false}}
  scale_2.0:
    f0: {PointMass: {value: 0.0}}
    f1: {Normal: {loc: 0.0, scale: 2.0, estimate_loc: false, estimate_scale: false}}

errors:                                     # -> error_sampler (null => None => standard normal)
  gaussian: null
  t_df_5:   {function: t_error_sampler, arguments: {df: 5}}

methods:                                    # -> MethodSpec(name, fit, summarize, kwargs)
  cox_heavy_L1:
    fit: fit_cox_method
    summarize: summarize_cox_method
    kwargs: {threshold: null, time_sign: 1.0, L: 1}
  twogroup_L1:
    fit: fit_twogroup_method
    summarize: summarize_twogroup_method
    kwargs:
      f1: {Normal: {loc: 0.0, scale: 1.0, estimate_loc: true, estimate_scale: true}}
      L: 1
      n_null_iter: 20
      n_intercept_iter: 20

plot_type_groups:                           # unchanged from current main.yaml
  pip: [pip_calibration, agg_pip_calibration, power_fdp, agg_power_fdp]
  cs:  [...]
```

`function`/`fit`/`summarize` names resolve to importable top-level callables in
`core.py` (same `module:qualname` contract `core._callable_path` already enforces).

### supercollection

Each supercollection has exactly three concerns: **collections**, **methods**,
**plots**.

```yaml
supercollections:
  003-hallmark-loc-snr:
    collections:
      - template:
          design: hallmark
          enrichment: [ser_b2, null_b0]     # plain list => joint, WITHIN one collection
          error: gaussian
        over:
          signal: [loc_0.5, loc_1.0, loc_1.5, loc_2.0, loc_2.5, loc_3.0]
        # => 6 collections; each = {ser_b2, null_b0} x that signal
        # alias defaults to the over-value key (e.g. loc=2.00 style label)

    methods: [cox_heavy_L1, twogroup_L1]     # select from library
    # inline method defs also allowed here (same schema as library.methods),
    # merged into method scope for this supercollection only.

    default_plot_args:                       # shared numeric knobs (was default_settings)
      min_log_bf: 2.0
      max_cs_size: 10000
      max_fdp: 0.5

    plots:
      - method_filter: [twogroup_L1]         # subset of this SC's methods, by name
        plot_args: {thresholds: [2.0]}       # overrides default_plot_args
        plot_type_groups: [pip, cs]
        # plot_types: [...] also allowed (explicit list, like current)
```

#### Collection expansion semantics

A `collections` entry is a `{template, over}` block (or a list of such blocks; a bare
explicit `{name, simulations:[...]}` form is also accepted for one-offs).

- **Within-collection (joint):** any field in `template` (design/enrichment/signal/
  error) given as a list expands by cartesian product into the *member simulations of
  a single collection* — the set analyzed jointly (e.g. the ser+null pair).
- **Across-collections:** `over` lists one or more fields; the cartesian product of
  `over` values yields *one collection per combination*. The `over` axis is the
  supercollection's comparison axis, so its value drives the collection **alias**.

Generated collection **name** (the `results/collections/<name>/` path key) is
deterministic and globally unique: `{supercollection}__{over-key}={over-value}`
(joined for multi-key `over`). An optional `name_template`/`alias_template` may
override. `alias` defaults to the over-value(s).

#### Simulation identity & naming

Each resolved `(design, enrichment, signal, error)` builds a `SimulationSpec`; its
content hash (`core.dehydrate_hashed`) is the `batch_hash` path key, exactly as today.
A human-readable simulation name is auto-derived from the component library keys
(`{design}__{enrichment}__{signal}` + `__{error}` when non-`gaussian`) and stored in
spec metadata for debugging/plot labels. Names are no longer load-bearing for path
resolution (hashes are).

#### Aggregation levels (and how `agg_` plots fit)

Two distinct aggregation levels exist today and map directly onto this model:

- **Level 1 — within-collection pooling.** The snakefile `collection_*` rules build
  one `plot_ready` bundle per collection by pooling *all member simulations'* fits.
  This is expressed by **template lists**: `enrichment: [ser_b2, null_b0]` or
  `design: [hallmark, c4, gaussian_p100, uniform_p100]` produces a single collection
  whose members are pooled. (The old `t-error-agg-*` collections — one collection
  listing 4 designs — are exactly this.)
- **Level 2 — across-collection (`agg_` plot types).** `make_plot` loads every
  collection in the supercollection, tags each with its alias as `collection_name`,
  then non-`agg_` plots facet by collection (`facet_by_simulation=True`,
  `collection_names=[...]`) while `agg_` plots pool across collections
  (`group_by` drops `collection_name` / `aggregate_across_collections=True`).

Therefore **the `over:` axis is the facet/aggregation dimension**: `over: {signal:
[loc_0.5..3.0]}` yields 6 collections; the non-`agg_` variant facets the 6, the `agg_`
variant collapses them into one panel. Whatever is placed in `over` is what `agg_`
pools. No structural change to the plot layer's aggregation logic is required.

## Loader (`experiments/loader.py`)

Pure-Python module, no snakemake dependency, unit-testable:

- `load_library()` — parse `library.yaml`; build resolver tables for designs,
  enrichments, signals, errors, methods.
- `resolve_distribution(node)` — distribution mini-schema → gibss distribution.
- `resolve_simulation(design, enrichment, signal, error)` → `SimulationSpec`
  (builds `partial(fn, **arguments)` samplers, `f0`/`f1`, intercept, `base_seed`).
- `resolve_method(name_or_inline)` → `MethodSpec`.
- `expand_collection(entry)` → `[(collection_name, alias, [SimulationSpec])]`.
- `load_supercollections()` — parse numbered files; resolve collections + methods +
  plots into in-memory structures.
- `manifest_dict()` → `{"batches": {...}, "method_specs": {...}}` keyed by hash with
  `core.dehydrate_hashed` node shapes — drop-in for the current snakefile.
- Collection/supercollection accessors replacing `_load_plot_configs`,
  `_COLLECTION_YAMLS`, `_resolve_collection_batches`, `_resolve_collection_methods`,
  `load_collection_yaml`, `load_supercollection`, `_resolve_sc_plot_pairs`.

`BatchSpec`s are built per simulation from `defaults.replicates_per_batch`/`n_batches`
(reusing `config_builders.batch_specs_for_simulation` logic, relocated into the loader).

## Snakefile changes

- `manifest_cache.py`: repoint from `config.py` to `experiments/loader.py`
  (mtime trigger watches `experiments/` instead of `config.py`).
- Replace the plot-config block (`_load_plot_configs`, `_COLLECTION_DEFS`,
  `_METHOD_COLLECTIONS`, `_resolve_collection_*`, `load_collection_yaml`,
  `load_supercollection`, `_resolve_sc_plot_pairs`, `_method_passes`,
  `_method_family`) with loader-backed equivalents that consume the new
  collection/method-filter model.
- `all_null_fits`: replace `from config import NULL_METHOD_SPECS` with a loader query.
- Content-addressed rules (`materialize_*`, `fit_*`) unchanged — still hash-keyed via
  the manifest and `core.rehydrate_node`.

### generate_plots.py

`generate_plots.py` reads `plot_configs/` directly and must be ported to the loader:

- `_load_plot_config`, `_resolve_settings`, `_load_supercollection_data` →
  loader-backed supercollection accessors (collections list, aliases, plot bundles).
- `_foreground_methods` currently filters by `method_families` + `thresholds`; it
  becomes **explicit-name membership** against the plot entry's `method_filter`
  (intersected with the supercollection's `methods`). `plot_args` replaces the merged
  `default_settings`/named-`settings` dict (`min_log_bf`, `max_cs_size`, `max_fdp`,
  `thresholds`, …).
- `agg_`/non-`agg_` dispatch and the per-plot-type render functions are unchanged;
  only how `combined_data` and `settings` are assembled changes.

## Deletions

- `config.py`, `config_builders.py`, `config_registry.py` (after both proof configs
  migrate and the snakefile no longer imports them).
- `plot_configs/` once all numbered files are ported (kept during the proof phase).
- Unused module-level `MethodSpec` constants in `core.py` (`COX_HEAVY`, etc.) and the
  `build_*_registry`/alias helpers, if nothing else references them after the cutover
  (verify before removing).

## Migration (proof scope)

Port two supercollection files exercising the tricky paths:

- `003_loc_snr` — within-collection ser+null pairing via `enrichment: [ser_b2, null_b0]`
  and across-collection `over: {signal: [loc_*]}`.
- `000_t_errors` — non-trivial `error` sampler entries (`t_df_*`).

Plus the `library.yaml` entries they need (hallmark/c4/gaussian_p100/uniform_p100
designs; ser_b2/null_b0 enrichments; loc/scale signals; t-error errors; the method set).
Remaining files (001, 002, 004-007) ported in follow-ups; `config.py` retained until then.

## Testing

- Loader unit tests: distribution parsing; within vs across (`over`) expansion;
  product correctness; manifest node shape matches `core.dehydrate_hashed`; hash
  stability across repeated loads; collection name/alias derivation + uniqueness.
- Snakemake dry-run (`-n`) on the migrated `003`/`000` supercollections: DAG resolves,
  expected `by_batch`/`collections` targets enumerated.
- One end-to-end tiny fit (single replicate) through `materialize` → `fit` to confirm
  rehydration works against loader-emitted manifest nodes.

## Future work (not in this spec)

**Cache reuse via hash injection.** A standalone post-hoc script, independent of loader
development: for each new spec, rebuild its dehydrated semantic node and look it up in
the existing `results/manifest_cache.json`. Where an old hash with an equivalent
semantic node exists, symlink/rename the old `results/by_batch/<old_hash>/` directory
to the new hash to skip recompute. (Where the new YAML resolves to byte-identical
partials/distributions, the hash is already identical and no mapping is needed.) This
touches only result directories + the cache file, never the loader or schema.
