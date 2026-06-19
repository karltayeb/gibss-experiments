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
- **`core.py` mostly stays.** `SimulationSpec`, `BatchSpec`, the
  `dehydrate/rehydrate/spec_hash` machinery, `simulate`, all samplers, and the
  `fit_*`/`summarize_*` helpers remain. The loader reuses `core.dehydrate_hashed` so
  the simulation/batch manifest node shapes and the snakefile rehydration path are
  unchanged. The one change: `MethodSpec` collapses `fit_function`+`summarize_function`
  into a single `function` (see "MethodSpec collapse").
- The numbered-file split is preserved (renamed `plot_configs/` → `experiments/`).
- **No coexistence.** Done on a new branch. `config.py` and the registry are deleted
  on that branch; the snakefile switches over fully. Experiments not yet ported are
  simply unavailable until ported — acceptable on a feature branch.

## File layout

```
experiments/
  library.yaml        # shared: defaults, designs, enrichments, signals,
                      #         errors, methods, reductions, analyses, analysis_groups
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

methods:                                    # -> one MethodSpec per (template x over) combo
  cox_heavy:
    function: run_cox_method                 # single entrypoint -> summary row
    template: {threshold: null, time_sign: 1.0}
    over: {L: [1]}                           # -> cox_heavy__L=1
  cox_light:
    function: run_cox_method
    template: {time_sign: -1.0}
    over: {threshold: [0.0, 1.0, 2.0, 3.0, 4.0], L: [1, 5]}
    # -> cox_light__threshold=2.00__L=1, ... (10 distinct methods)
  twogroup:
    function: run_twogroup_method
    template:
      f1: {Normal: {loc: 0.0, scale: 1.0, estimate_loc: true, estimate_scale: true}}
      n_null_iter: 20
      n_intercept_iter: 20
    over: {L: [1, 5]}                        # -> twogroup__L=1, twogroup__L=5

reductions:                                 # (sims + fits) -> compact cacheable parquet
  sample_metadata: {function: build_sample_metadata, needs: {simulations: true}}
  method_metadata: {function: build_method_metadata, needs: {fits: true}}
  pip: {function: build_pip_plot_data,
        needs: {fits: true, simulations: true, reductions: [sample_metadata]}}
  cs:  {function: build_cs_plot_data,
        needs: {fits: true, simulations: true, reductions: [sample_metadata]}}
  f1:  {function: build_f1_plot_data, needs: {fits: true, method_filter: twogroup}}

analyses:                                   # reduction bundle -> artifact (PDF for now)
  pip_calibration:     {function: render_pip_calibration, requires: [pip, method_metadata]}
  agg_pip_calibration: {function: render_agg_pip_calibration, requires: [pip, method_metadata]}
  power_fdp:           {function: render_power_fdp, requires: [pip, method_metadata]}

analysis_groups:                            # was plot_type_groups
  pip: [pip_calibration, agg_pip_calibration, power_fdp, agg_power_fdp]
  cs:  [...]
```

All `function` names (designs, enrichments, errors, methods) resolve to importable
top-level callables in `core.py` (same `module:qualname` contract `core._callable_path`
already enforces). A method's `function` is a single entrypoint returning the summary
row dict — see Method expansion.

### supercollection

Each supercollection has exactly three concerns: **collections**, **methods**,
**outputs** (analyses — plots and, later, tables/stats).

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

    methods: [cox_heavy__L=1, cox_light__threshold=2.00__L=1, twogroup__L=1]
    # generated method names (see Method expansion). inline method defs also allowed
    # here (same schema as library.methods), merged into method scope for this SC only.

    default_args:                            # shared numeric knobs (was default_settings)
      min_log_bf: 2.0
      max_cs_size: 10000
      max_fdp: 0.5

    outputs:                                 # was plots
      - name: minimal                        # -> args_name in the output path
        method_filter: [twogroup__L=1, cox_light__threshold=2.00__L=1]  # subset, by name
        args: {max_fdp: 0.5}                 # overrides default_args
        analysis_groups: [pip, cs]
        # analyses: [...] also allowed (explicit list)
```

#### Method expansion semantics

A `methods` library entry is `{function, template, over}`. `function` is a single
method entrypoint (`run_<method>(simulation, **kwargs) -> summary-row dict`); fit and
summarize are no longer separate spec fields (they were always paired, same kwargs,
same rule — see "MethodSpec collapse" below). All keys in `template` and `over` are
passed as **kwargs to `function`**. `over` is the cartesian sweep: each combination of
`over` values yields **one distinct named `MethodSpec`**, so name ↔ method is 1:1.
`template` holds the shared kwargs.

Generated method name: `{base}__{over-key}={over-value}` for each `over` key
(joined). A method with no real sweep uses a single-value `over` (e.g.
`over: {L: [1]}`).

Unlike collections, methods have no "joint" axis: a method is one fit. Therefore
**lists inside `template` are literal kwarg values, not an expansion** — only `over`
expands. (Collections expand `template` lists into joint members; methods do not.)

`threshold` is now just one such kwarg. There is no special threshold/family
machinery: distinct thresholds are distinct named methods. `threshold` still appears
in the cox/logistic summary row → `method_metadata` as **passive metadata** for series
labels/ordering, but it is never a filter dimension. `method_filter` selects methods
purely by name.

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

- **Level 1 — within-collection pooling.** `load_sc_bundle` concatenates the atomic
  reductions of *all of a collection's `(batch, method)` members* under one
  `collection_name` tag, so the collection is treated as a single pooled unit. This is
  expressed by **template lists**: `enrichment: [ser_b2, null_b0]` or
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

## Reductions + analyses layer

Generalizes the fixed `collection_*_plot_data` rules + plot types into two pluggable,
data-driven stages. Same `function`+args pattern as the rest of the config.

**Reductions are fit-level (atomic), not collection-level.** A reduction runs over a
single `by_batch` atomic unit and is content-addressed there, so a `(batch, method)`
fit shared by many collections is reduced **once** and reused everywhere. Collections
are **purely logical** — a set of `(batch, method)` pairs + alias — and are assembled
by an in-memory concat at analysis time. There is no `collections/.../reductions/` dir
and no collection-level cache.

- **reduction** (atomic): `function(inputs) -> dataframe`, written next to the atomic
  unit. Declares `needs`, which also fixes its **scope**:
  - `simulations: true` → **per-batch** scope, output
    `by_batch/<batch_hash>/reductions/<reduction>.parquet`.
  - `fits: true` → **per-(batch,method)** scope, output
    `by_batch/<batch_hash>/fits/<method_hash>/reductions/<reduction>.parquet`. (Whether
    a method participates at all is the analysis-level `method_filter`; a reduction-
    level `method_filter`, e.g. f1 = twogroup-only, restricts which methods this
    reduction is even defined for.)
  - `reductions: [name, ...]` → consume sibling atomic reductions at the same unit
    (e.g. `pip` needs `sample_metadata` for the same batch). Same generic rule,
    different wildcard, so edges resolve automatically.
  Reductions are **parameter-free by name**: a variant is a new named reduction (no
  args encoded in the path; consistent with 1:1 methods).
- **analysis** (per *supercollection*): `function(bundle, args, output)` writes an
  artifact. Declares `requires: [reduction, ...]`. Output is **PDF for now**; an
  `output: pdf|csv|json` format field (tables/stats as peer artifacts) is a noted
  future extension, not built here.

*Escape hatch (not built):* if a future reduction needs cross-fit context within a
collection (e.g. ranking methods against each other), add a second declared
`scope: collection` tier (map atomic, then reduce). None exist today — all current
reductions are atomic + concat.

### Mapping to snakemake (two generic rules)

Snakemake rules are static; reductions/analyses are dynamic from YAML. Two generic
rules keyed by name-wildcards, with lambda input-functions consulting the loader,
replace the 8 hardcoded rules.

```python
# fit-scoped reduction (sim-scoped reduction is the analogous rule without method_hash)
rule atomic_reduction:
    output: f"{ROOT}/by_batch/{{batch_hash}}/fits/{{method_hash}}/reductions/{{reduction}}.parquet"
    input:
        sources = PLOT_READY_SOURCES,
        deps = lambda wc: reduction_inputs(wc.batch_hash, wc.method_hash, wc.reduction)
    run:
        fn = loader.reduction_function(wc.reduction)
        write_parquet(fn(load_reduction_inputs(wc.batch_hash, wc.method_hash, wc.reduction)), output[0])

rule supercollection_analysis:
    output: f"{ROOT}/supercollections/{{sc}}/{{analysis}}/{{args_name}}.pdf"
    input:
        sources = PLOT_RENDER_SOURCES,
        deps = lambda wc: analysis_inputs(wc.sc, wc.analysis)
    run:
        fn   = loader.analysis_function(wc.analysis)
        args = loader.resolve_args(wc.sc, wc.analysis, wc.args_name)
        bundle = load_sc_bundle(wc.sc, loader.analysis_requires(wc.analysis))  # gather+concat+tag
        fn(bundle, args=args, output=output[0])
```

- `reduction_inputs(...)` → the atomic unit's `simulations.parquet`/`fits.parquet`
  (per `needs`) + sibling-reduction parquets at the same unit.
- `analysis_inputs(sc, analysis)` → for each collection in the SC, for each
  `(batch, method)` in the collection (methods ∩ analysis `method_filter`), for each
  reduction in `analysis.requires`: the atomic reduction parquet. **This is the
  dependency-driven materialization** — only the reductions a requested analysis
  declares get built, and each atomic reduction is shared across every collection/SC
  that references that fit.
- `load_sc_bundle` (in-memory, no rule): for each collection, gather its atomic
  reduction parquets, concat, tag `collection_name=alias`; then concat across
  collections. Replaces `_load_supercollection_data`. agg vs non-agg analyses
  pool/facet this bundle (see Aggregation levels).
- `args_name` is the output entry's `name`; `resolve_args` merges SC `default_args`
  with the entry's `args`. `method_filter` is render-time foreground selection.

Wildcard constraints: `reduction`/`analysis` ∈ `[A-Za-z0-9_]+`, `args_name` ∈
`[A-Za-z0-9_\-]+`. Targets enumerated by `_resolve_sc_analyses(sc)` →
`(analysis, args_name)` pairs (expanding `analysis_groups`); path
`supercollections/<sc>/<analysis>/<args_name>.pdf`.

## Loader (`experiments/loader.py`)

Pure-Python module, no snakemake dependency, unit-testable:

- `load_library()` — parse `library.yaml`; build resolver tables for designs,
  enrichments, signals, errors, methods.
- `resolve_distribution(node)` — distribution mini-schema → gibss distribution.
- `resolve_simulation(design, enrichment, signal, error)` → `SimulationSpec`
  (builds `partial(fn, **arguments)` samplers, `f0`/`f1`, intercept, `base_seed`).
- `resolve_methods(entry)` → `[MethodSpec]` via `template`×`over` expansion (1:1
  name↔method; `template`+`over` keys → kwargs to `function`); `resolve_method(
  name_or_inline)` resolves a reference/inline def.
- `expand_collection(entry)` → `[(collection_name, alias, [SimulationSpec])]`.
- `load_supercollections()` — parse numbered files; resolve collections + methods +
  plots into in-memory structures.
- `manifest_dict()` → `{"batches": {...}, "method_specs": {...}}` keyed by hash with
  `core.dehydrate_hashed` node shapes — drop-in for the current snakefile.
- Collection/supercollection accessors replacing `_load_plot_configs`,
  `_COLLECTION_YAMLS`, `_resolve_collection_batches`, `_resolve_collection_methods`,
  `load_collection_yaml`, `load_supercollection`, `_resolve_sc_plot_pairs`.
- Reduction/analysis layer (above): `reduction_function`, `reduction_inputs`,
  `analysis_function`, `analysis_requires`, `analysis_inputs`, `resolve_args`,
  `load_sc_bundle`, `_resolve_sc_analyses`.

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
- Replace the 7 `collection_*_plot_data`/metadata rules + `supercollection_plot` with
  the **two generic rules** `atomic_reduction` and `supercollection_analysis`
  (see Reductions + analyses layer). `twogroup_experiments_target`/
  `materialize_supercollection`/`all_plots` rebuild their target lists from
  `_resolve_sc_analyses` (analysis × args_name) instead of the fixed 7-parquet +
  plot-type enumeration.

### generate_plots.py

`generate_plots.py` reads `plot_configs/` directly and must be ported to the loader.
Its per-plot-type render functions (`_make_*`) become the **analysis functions**
referenced by `analyses` (`render_pip_calibration`, …); their bodies are largely
unchanged — only how `combined_data`/`args` reach them changes.

- `make_plot` → the `supercollection_analysis` rule body: `load_sc_bundle` builds
  `combined_data` (replacing `_load_supercollection_data`); `resolve_args` builds
  `args` (replacing `_load_plot_config`/`_resolve_settings`); dispatch is by
  `analysis` name (replacing the `_PLOT_DISPATCH` table, or keep the table keyed by
  analysis name).
- `_foreground_methods` currently filters by `method_families` + `thresholds`; it
  becomes **explicit-name membership** against the output entry's `method_filter`
  (intersected with the supercollection's `methods`). The `method_families`/
  `thresholds`/`is_thresholded` filtering and `_method_family` name-stripping are
  deleted (threshold is now passive metadata, methods are 1:1 by name). `args`
  replaces the merged `default_settings`/named-`settings` dict (`min_log_bf`,
  `max_cs_size`, `max_fdp`, …) and no longer carries `thresholds`.
- `agg_`/non-`agg_` remain distinct analyses (separate render fns), pooling/faceting
  the combined bundle as today.

## core.py changes (MethodSpec collapse)

`fit_*` and `summarize_*` are always paired, take the same kwargs, and run
back-to-back in the same rule (`fit_batch_method` → `run_method_spec` then
`summarize_method_spec`); `fit_obj` is never persisted. So the spec carries one
callable, not two:

- Add four thin entrypoints to `core.py`:
  `run_cox_method`, `run_logistic_method`, `run_twogroup_method`, `run_linear_method`,
  each `run_X(simulation, **kwargs)` = `summarize_X(fit_X(simulation, **kwargs),
  simulation, **kwargs)`. Existing `fit_*`/`summarize_*` stay as internal helpers.
- `MethodSpec`: replace `fit_function` + `summarize_function` with a single
  `function`. Drop `summarize_method_spec`; `run_method_spec(method_spec, simulation)`
  returns the summary row (`{"method": name, **function(simulation, **kwargs)}`).
- `utils.fit_batch_method`: `row = run_method_spec(method_spec, simulation)` (drop the
  separate summarize call).
- Delete the unused module-level `MethodSpec` constants (`COX_HEAVY`, etc.) — the
  library replaces them.

The method node in the manifest loses `summarize_function`; `plot_ready`'s
`is_thresholded` read is reworked alongside the threshold-machinery removal.

## Deletions

- `config.py`, `config_builders.py`, `config_registry.py` — deleted on the branch
  (no coexistence; snakefile + `manifest_cache.py` no longer import them).
- `plot_configs/` — replaced by `experiments/` (deleted on the branch; only the ported
  `003`/`000` supercollections exist until the rest are ported in follow-ups).
- The 7 `collection_*_plot_data`/metadata snakemake rules + `supercollection_plot` +
  `_resolve_sc_plot_pairs` + the fixed `PLOT_TYPES` list — replaced by the two generic
  rules + `_resolve_sc_analyses`.
- Unused module-level `MethodSpec` constants in `core.py` (`COX_HEAVY`, etc.) and the
  `build_*_registry`/alias helpers, if nothing else references them after the cutover
  (verify before removing).

## Migration (proof scope)

Port two supercollection files exercising the tricky paths:

- `003_loc_snr` — within-collection ser+null pairing via `enrichment: [ser_b2, null_b0]`
  and across-collection `over: {signal: [loc_*]}`.
- `000_t_errors` — non-trivial `error` sampler entries (`t_df_*`).

Plus the `library.yaml` entries they need (hallmark/c4/gaussian_p100/uniform_p100
designs; ser_b2/null_b0 enrichments; loc/scale signals; t-error errors; the method set;
the reductions + analyses these supercollections use). Remaining files (001, 002,
004-007) ported in follow-ups; on this branch they are simply absent (no coexistence).

## Testing

- Loader unit tests: distribution parsing; within vs across (`over`) expansion;
  product correctness; manifest node shape matches `core.dehydrate_hashed`; hash
  stability across repeated loads; collection name/alias derivation + uniqueness.
- Method expansion: `template`×`over` → expected distinct names + kwargs.
- Reduction/analysis resolution: `reduction_inputs` returns correct sims/fits/sibling
  paths per `needs`; `analysis_inputs` returns only `requires` reductions (assert an
  unrequested reduction is NOT in the DAG — the dependency-driven win).
- Snakemake dry-run (`-n`) on the migrated `003`/`000` supercollections: DAG resolves,
  expected `by_batch` reduction + `supercollections` targets enumerated.
- One end-to-end tiny fit (single replicate) through `materialize` → `fit` →
  `atomic_reduction` → `supercollection_analysis` to confirm rehydration + the two
  generic rules work against loader-emitted manifest nodes.
- Atomic reduction reuse: a fit shared by two collections produces exactly one
  reduction parquet (one job), referenced by both.

## Future work (not in this spec)

**Cache reuse via hash injection.** A standalone post-hoc script, independent of loader
development: for each new spec, rebuild its dehydrated semantic node and look it up in
the existing `results/manifest_cache.json`. Where an old hash with an equivalent
semantic node exists, symlink/rename the old `results/by_batch/<old_hash>/` directory
to the new hash to skip recompute. (Where the new YAML resolves to byte-identical
partials/distributions, the hash is already identical and no mapping is needed.) This
touches only result directories + the cache file, never the loader or schema.
