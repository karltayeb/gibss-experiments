# Pipeline v2: config-hash, module reorg, code fingerprints, predicate filters

**Date:** 2026-06-19
**Status:** Design note (addendum to `2026-06-18-yaml-experiment-config-design.md`). Not yet planned/implemented.

This refines the YAML-config pipeline (v1, implemented on branch `yaml-experiment-config`) to fix code-change invalidation, simplify the spec serialization, and make reduction/analysis scoping declarative. The four-stage path is unchanged: **simulate → fit → reduce → analyze**, driven by **four generic snakemake rules** (one per stage).

## Motivation

v1 issues surfaced in review/use:
- **Code-change invalidation is broken.** Generic rules take a whole-file `input.sources` (`plot_ready.py`, `generate_plots.py`). Adding a reduction edits the shared file → *all* reductions rerun (over-trigger); a helper change outside the listed files → *no* rerun (under-trigger). `simulate`/`fit` declare *no* source inputs at all → editing a sampler or solver reruns nothing (stale data on the expensive stages).
- **`rehydrate_node` + the dehydrate/canonicalize machinery (~250 lines of `core.py`)** exists to reconstruct live spec objects in the rule process. In a YAML framework the YAML *is* the serialization — the rule can re-resolve from config instead.
- **Reduction `needs` and string `method_filter`** are heavier/fragiler than necessary.

## 1. Config-hash + drop rehydrate

The content hash (`batch_hash`/`method_hash`, the `by_batch/<hash>/` path key) should encode **config identity only** — function *paths* + arguments + distribution params + `base_seed` — **not** code bodies. (If code changed the hash, every path would re-shard on a one-line fix.) Code-body invalidation is snakemake's job (§3), not the hash's.

- **Hash:** `spec_hash = sha256(canonical_json(resolved_coordinate))`, where the coordinate is the resolved library sub-dicts: `{design: {function, arguments}, enrichment: {function, arguments, intercept}, signal: {f0, f1}, error: {function, arguments}|null, base_seed}` for sims; `{function, kwargs}` (distributions as their `{Normal: {...}}` mini-schema) for methods. This is the same data `dehydrate_spec` produced, minus the partial/callable graph wrapping.
- **Reconstruction:** the manifest stores the **YAML coordinate** per batch (`design/enrichment/signal/error` keys + `replicates`) and per method (base + over-combo, or the generated name). The rule re-resolves: `loader.resolve_simulation(library, *coord)` / `loader.resolve_method(...)`. The rule process already does `from experiments import loader`, so this is trivial.
- **Delete from `core.py`:** `dehydrate_node`, `rehydrate_node`, `canonicalize_node`, `_dehydrate_constructed_instance`, `dehydrate_spec`, `rehydrate_spec`, `dehydrate_simulation_semantics`, `build_hash_registry`, `build_alias_registry`, `dehydrate_hashed`. **Keep** `canonical_json_bytes`, `spec_hash`, `_callable_path` (path validation).
- **Touch:** `loader` (new config-hash + coordinate manifest, drop dehydrate calls), snakefile rules (re-resolve), `reductions` that read the old dehydrated node shape (`build_f1`/`build_enrich` read `sim_spec_node["fields"]["f1"]["fields"]` → read the signal coordinate's `f1` params instead).

Clean break on hashes (already accepted on this branch; nothing to preserve).

## 1a. Two representations: config coordinate vs resolved Spec

v1 conflated two roles in `SimulationSpec`/`MethodSpec`: the runtime carrier **and** the serialization/hash unit (hence all the dehydrate machinery). v2 splits them:

- **Config coordinate (declarative, plain data)** — the YAML keys + resolved library sub-dicts. This is the hashed, stored, manifest-persisted **identity**. No class needed; it's dicts/YAML.
- **`SimulationSpec` (resolved, executable)** — built by the loader *from* the coordinate, holding live objects (`partial`s, distributions). Consumed by `core.simulate`. **Transient** (built in the rule process, discarded), with **no** hashing/serialization role.

A "spec" object earns its place in proportion to how **composite** the stage is:

- **`MethodSpec` — dropped.** A method is just `function(**kwargs)` + a name; the dataclass bundled three values and added no behavior. The fit site resolves inline:
  ```python
  fn = resolve_callable(coord.function)
  row = {"method": coord.name, **fn(simulation, **resolve_kwargs(coord))}   # kwargs: mini-schema dists resolved
  ```
- **`SimulationSpec` — kept**, but solely as the **resolved parameter-object** for the simulate engine. Its value is that `core.simulate` takes *resolved objects*, never a config blob — so the engine stays ignorant of YAML/library/resolution and is testable from hand-built samplers/distributions (as `tests/test_core_run_methods.py` already does). At ~8 fields, a bundle beats a long arg list. (The decoupling comes from "resolved args, not config" — the dataclass is just ergonomics.)

### `SimulationSpec` in v2

```python
@dataclass                              # not frozen; no hashing/serialization role
class SimulationSpec:
    design_sampler: Callable            # partial(gaussian_markov_X, n=..., p=..., rho=...)
    effect_sampler: Callable            # partial(uniform_single_effect, causal_effect=2.0)
    intercept: float
    f0: Any                             # PointMass / Normal / ...
    f1: Any
    error_sampler: Callable | None      # None == standard normal
    base_seed: int
    hash: str                           # config-hash from the loader: seeding + by_batch path key
    name: str = ""                      # human label for logs/debug only
```

Instance for `hallmark__ser_b2__loc_2.0`:
```python
SimulationSpec(
    design_sampler = partial(hallmark_gene_sets_X),
    effect_sampler = partial(uniform_single_effect, causal_effect=2.0),
    intercept      = -2.0,
    f0             = PointMass(0.0),
    f1             = Normal(loc=2.0, scale=0.1, estimate_loc=False, estimate_scale=False),
    error_sampler  = None,
    base_seed      = 20260501,
    hash           = "9846f5a5...",     # = sha256(canonical_json(coordinate))
    name           = "hallmark__ser_b2__loc_2.0",
)
```

The coordinate it resolves from (the hashed/stored config — *not* the Spec):
```python
{
  "design":     {"function": "hallmark_gene_sets_X", "arguments": {}},
  "enrichment": {"function": "uniform_single_effect", "arguments": {"causal_effect": 2.0}, "intercept": -2.0},
  "signal":     {"f0": {"PointMass": {"value": 0.0}},
                 "f1": {"Normal": {"loc": 2.0, "scale": 0.1, "estimate_loc": false, "estimate_scale": false}}},
  "error":      null,
  "base_seed":  20260501,
}
```

Connection:
```python
def resolve_simulation(library, design, enrichment, signal, error) -> SimulationSpec:
    coord = coordinate(library, design, enrichment, signal, error)
    return SimulationSpec(
        design_sampler = _partial(coord["design"]),
        effect_sampler = _partial(coord["enrichment"]),
        intercept      = coord["enrichment"]["intercept"],
        f0             = resolve_distribution(coord["signal"]["f0"]),
        f1             = resolve_distribution(coord["signal"]["f1"]),
        error_sampler  = None if coord["error"] is None else _partial(coord["error"]),
        base_seed      = coord["base_seed"],
        hash           = spec_hash(coord),
        name           = sim_name(design, enrichment, signal, error),
    )

def simulate(spec: SimulationSpec, replicate: int):           # pure execution; no YAML, no dehydrate
    rng = np.random.default_rng(replicate_seed(spec.base_seed, spec.hash, replicate))
    X = spec.design_sampler(rng)
    causal_idx, causal_eff = spec.effect_sampler(X, rng)
    # ... f0/f1.sample(...), error_sampler ...
```

Key change from v1: `hash` is a **field set by the loader from the coordinate**; `simulate` uses `spec.hash` for seeding instead of v1's `simulation_hash(spec)` (which dehydrated the spec). No dehydrate call anywhere; `SimulationSpec` is no longer frozen/hashable.

## 2. Module reorganization

Organize the run-code into cohesive per-item (or per-family) modules so each item's code surface is a single file (precise fingerprints, §3). Re-export from `core` so the YAML `function:` contract (`resolve_callable = getattr(core, name)`) still resolves; `inspect.getfile(fn)` follows the real definition.

```
simulations/   designs.py, effects.py, errors.py      # samplers (gaussian_markov_X, uniform_single_effect, t_error_sampler, ...)
fits/          cox.py, logistic.py, twogroup.py, linear.py   # run_*/fit_*/summarize_*
reductions/    pip.py, cs.py, f1.py, enrich.py
analyses/      pip.py, cs.py, logbf.py, f1.py          # the _make_*/render_* fns, grouped by family
core.py        simulate(), SimulationSpec (resolved bundle; MethodSpec dropped), spec_hash, re-exports
```

- Re-export: `from simulations.designs import *`, `from fits.cox import *`, etc. in `core` (or a thin `core/__init__`), preserving the flat `getattr(core, name)` lookup.
- Granularity knob = module organization: one reduction per module → per-reduction precision; renderers grouped by family → per-family precision (avoids 38 one-renderer files while staying far better than whole-file).

## 3. Code fingerprints (the rerun fix) — generic rules preserved

Keep **four generic rules**. Each carries a `params` code-fingerprint over the file(s) defining the user-code it runs; snakemake's `params` rerun-trigger (v9) reruns only the items whose file changed, and adding an item doesn't perturb others' fingerprints.

```python
def code_fingerprint(*fns) -> str:
    return sha256(b"".join(Path(inspect.getfile(f)).read_bytes() for f in fns)).hexdigest()
```

| Stage | rule `params.code` over | invalidation granularity |
|---|---|---|
| simulate | this sim's `design`+`effect`+`error` sampler files (+ `core.simulate`) | per sampler-set |
| fit | this method's function file (`fits/<family>.py`) | per method family |
| reduce | `reductions/<name>.py` | per reduction |
| analyze | `analyses/<family>.py` | per analysis family |

Decisions:
- **File-hash, not transitive AST inference.** A decorator that walks the body to hash referenced globals/helpers was considered and rejected: distinguishing own-code vs third-party, serializing constant values, and AST edge cases make it fragile, and its failure mode is *silent under-hash* (stale cache) — the very bug being fixed. The file boundary captures body + in-file constants/helpers automatically, with zero introspection and trivial debuggability ("did the file change? then it reruns"). `code_fingerprint` is the thin glue; the precision lever is where code lives (§2).
- **Third-party (gibss/numpy) is opaque.** Don't hash their source. Pin gibss version (or `--forcerun` on the rare editable-gibss edit). Fingerprints cover our files only.
- Drop the coarse whole-file `input.sources`; the `params.code` fingerprint replaces it.

### Uniform entrypoints (kills the per-name dispatch)

With the reorg, give each stage a uniform entrypoint signature so the generic rule body is `fn = resolve(...); fn(ctx)` with **no `if reduction == "pip" ...` branching** (the wart the v1 final review flagged):
- **reduce:** every builder is `build(ctx) -> df`, where `ctx` exposes `fits`, `sims`, `sample_metadata`, `sim_coordinate`; the builder reads what it needs.
- **fit:** already uniform — `run_<method>(simulation, **kwargs) -> row`.
- **simulate:** uniform via `core.simulate(spec, replicate)`.
- **analyze:** already uniform — `render(bundle, args) -> figure`.

Adding an item becomes purely: a new module + a library entry. No rule edit.

## 4. Reductions lose `needs`

Every reduction is fit-scoped and its input set is fixed (`fits`, `simulations`, `sample_metadata`). So:
- **Drop the `needs` toggle dict.** The generic `reduce` rule always wires `(fits, simulations, sample_metadata)` as inputs and passes them in `ctx`. (This also fixes the v1 latent bug where pip/cs read `simulations.parquet` without declaring it.)
- **Scope:** hardwire fit-scope (per-`(batch, method)`). The only thing `needs.fits` also encoded was fit-vs-batch scope; no sim-only reduction exists today (the one such thing, `sample_metadata`, is a materialize output). If a pure simulation-summary (sim-only, per-batch) reduction ever appears, reintroduce a single `scope: batch` field then — the spec's existing escape hatch.

A reduction library entry collapses to `{function}` (+ optional `method_filter`, §5).

> Analyses keep `requires: [reduction, ...]` — that genuinely varies (pip analyses need `pip`, cs need `cs`) and drives which reductions get built.

## 5. Predicate filters

Replace string/group filtering with importable boolean predicates, resolved via `resolve_callable`, applied at **DAG-construction time** (so they see *config*, not materialized data).

- **`method_filter(method_spec: MethodSpec) -> bool`** on a **reduction** — which methods the reduction is valid for (f1/enrich read `two_group_state`, `None` for non-twogroup). Replaces `startswith("twogroup")`. Applied wherever the loader enumerates `(batch, method)` pairs (`collection_method_pairs`, `analysis_inputs`, `load_sc_bundle`).
  ```yaml
  reductions:
    f1:     {function: build_f1_plot_data, method_filter: is_twogroup}
    enrich: {function: build_enrich_plot_data, method_filter: is_twogroup}
  ```
- **`simulation_filter(sim_descriptor) -> bool`** on an **analysis** — which sims to pool into this plot. Excluding null sims is a *plotting* choice, not a reduction one (the `pip` reduction is identical for null/non-null; `causal_pip`/`mass_above_causal` are just meaningless on nulls). This replaces the old `pip_non_null`/`cs_non_null` plot_type_group hack. Applied in `analysis_inputs` (depend only on passing sims' reductions) and `load_sc_bundle` (pool only passing sims).
  ```yaml
  analyses:
    pip_calibration: {requires: [pip]}                                  # all sims (null = calibration/FDP arm)
    causal_pip:      {requires: [pip], simulation_filter: has_causal}   # non-null only
  ```

### Predicate input contracts
- `method_filter(method_spec)` — `MethodSpec` (`.function`, `.kwargs`, `.name`).
- `simulation_filter(sim_descriptor)` — the **resolved coordinate dict**, NOT the `SimulationSpec` (whose `effect_sampler` is a `partial` you'd have to introspect):
  ```python
  {design, enrichment: {function, arguments, intercept}, signal: {f0, f1}, error}
  ```
  e.g. `def has_causal(s): return s["enrichment"]["arguments"].get("causal_effect", 0) != 0` — declarative, no partial-poking.

### Boundaries
- Predicates decide on **config-knowable** properties only (null vs non-null, signal kind, error type). Data-dependent selection ("causal MAF > x") can't be a DAG-time filter — that'd be a post-hoc filter inside the analysis on the loaded bundle.
- Predicates must be **cheap + pure** (run once per item at DAG build).

## Open decisions

- **Output-level `method_filter`** (the render-time foreground name-list in a supercollection `output`) is a *different* thing from the reduction-level predicate — it picks which methods to *draw*. Keep it an explicit name list, or unify it to a predicate too? (Lean: keep as list — explicit foreground is usually hand-picked.)
- **Analysis family granularity** for `analyses/<family>.py` — group by required reduction (pip/cs/logbf/f1), or finer? Coarser = simpler, slightly more rerun on a family edit.

## Carried-over small fixes (from v1 review)

- Rule names: the four generic rules read as `simulate` / `fit` / `reduce` / `analyze` (drop `materialize_twogroup_experiment_batch` jargon).
- `notebooks/dashboard.py` + `scripts/symlink_plots.py` still reference deleted `config`/`plot_configs` — port or remove.
- Remaining experiments `001/002/004-007` ported to the new format.

## Not in scope

- Cache-reuse hash-injection (v1 spec "Future work") — independent.
- Data-dependent filters (post-hoc, inside analyses).
