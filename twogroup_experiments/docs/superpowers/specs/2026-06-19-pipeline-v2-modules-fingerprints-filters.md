# Pipeline v2: config-hash, module layout, native code-tracking, predicate filters

**Date:** 2026-06-19
**Status:** Design note (addendum to `2026-06-18-yaml-experiment-config-design.md`). Converged; not yet planned/implemented.

Refines the YAML-config pipeline (v1, implemented on branch `yaml-experiment-config`) to: hash on config identity (drop the dehydrate/rehydrate serializer), reorganize run-code into cohesive modules so snakemake tracks code changes natively, and make reduction/analysis scoping declarative via predicates. The four-stage path is unchanged: **simulate → fit → reduce → analyze**.

## Motivation (v1 issues)

- **Code-change invalidation broken.** Generic rules took a whole-file `input.sources` (`plot_ready.py`, …): adding a reduction reran *all* reductions (over-trigger); a helper change outside the listed files reran *nothing* (under-trigger). `simulate`/`fit` declared *no* source inputs → editing a sampler or solver reran nothing (silent-stale on the expensive stages).
- **`rehydrate_node` + the dehydrate/canonicalize machinery (~250 lines of `core.py`)** existed to reconstruct live spec objects in the rule process. In a YAML framework the YAML *is* the serialization — the rule re-resolves from config instead.
- **Reduction `needs` and string `method_filter`** heavier/fragiler than necessary.
- **`MethodSpec`** is a degenerate `function`+`kwargs` bundle adding no behavior.

## 1. Data model: two representations, two tiers

**Two representations**
- **Config coordinate (declarative data).** The YAML keys + resolved library sub-dicts. This is the hashed, manifest-stored **identity**. No class — dicts/YAML.
- **`SimulationSpec` (resolved, executable).** Built by the loader from the coordinate, holding live objects (`partial`s, distributions). Consumed by `core.simulate`. Transient (built in the rule process, discarded); **no** hashing/serialization role.

**Two tiers (flat, not nested)**
- **reduce** — atomic, per `(batch, method)`. Pure function of *one* batch's data (`fits` + `simulations` + `sample_metadata`). Content-addressed under `by_batch/`; a fit shared by many collections is reduced once and reused. **Reductions do not compose** (no reduction→reduction).
- **analyze** — per supercollection. Consumes a **set** of reductions (`requires: [...]`, many-to-many) **plus loader-supplied metadata** (method_metadata, aliases, coordinates). **Never reads raw sims/fits** — if a plot needs sim/fit-level data, add a (possibly trivial) reduction. `analyses/` is a *sibling* of `reductions/`, not nested.

## 2. Config-hash + drop rehydrate

The content hash (`batch_hash`/`method_hash`, the `by_batch/<hash>/` path key) encodes **config identity only** — function *paths* + arguments + distribution params + `base_seed` — **not** code bodies. (Code in the hash would re-shard every path on a one-line fix.) Code-body invalidation is snakemake's job (§4), not the hash's.

- **Hash:** `spec_hash = sha256(canonical_json(coordinate))`. For sims the coordinate is `{design: {function, arguments}, enrichment: {function, arguments, intercept}, signal: {f0, f1}, error: {function, arguments}|null, base_seed}`; for methods `{function, kwargs}` (distributions as `{Normal: {...}}` mini-schema).
- **Reconstruction:** the manifest stores the **coordinate** per batch (`design/enrichment/signal/error` keys + `replicates`) and per method (generated name + base/over, or just the resolvable name). The rule re-resolves via `loader.resolve_simulation(...)` / `loader.resolve_method(...)` (the rule process already imports `loader`).
- **Delete from `core.py`:** `dehydrate_node`, `rehydrate_node`, `canonicalize_node`, `_dehydrate_constructed_instance`, `dehydrate_spec`, `rehydrate_spec`, `dehydrate_simulation_semantics`, `build_hash_registry`, `build_alias_registry`, `dehydrate_hashed`. **Keep** `canonical_json_bytes`, `spec_hash`, `_callable_path` (path validation).

Clean break on hashes (already accepted on this branch).

### `SimulationSpec` (v2) and `MethodSpec` (dropped)

A "spec" object earns its place in proportion to how composite the stage is.

- **`MethodSpec` — dropped.** A method is just `function(**kwargs)` + a name; the fit site resolves inline:
  ```python
  fn  = resolve_callable(coord.function)
  row = {"method": coord.name, **fn(simulation, **resolve_kwargs(coord))}
  ```
- **`SimulationSpec` — kept**, solely as the resolved parameter-object that keeps `core.simulate` ignorant of YAML/library/resolution (testable from hand-built samplers, as `tests/test_core_run_methods.py` does). Not frozen; carries a loader-set `hash` field used for seeding + as the `by_batch` path key.

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
    hash: str                           # config-hash from the loader: seeding + path key
    name: str = ""                      # human label for logs/debug only

def simulate(spec: SimulationSpec, replicate: int):    # pure execution; no YAML, no dehydrate
    rng = np.random.default_rng(replicate_seed(spec.base_seed, spec.hash, replicate))
    X = spec.design_sampler(rng)
    causal_idx, causal_eff = spec.effect_sampler(X, rng)
    # ... f0/f1.sample(...), error_sampler ...
```

## 3. Module layout

Organize run-code into **cohesive modules** — a function plus the helpers it depends on share a file (so the file boundary captures the real code surface; §4). Re-export from `core` so the YAML `function:` contract (`resolve_callable = getattr(core, name)`) still resolves; `inspect.getfile(fn)` follows the real definition.

```
simulations/
  design/    markov.py     # gaussian_markov_X + uniform_markov_X (shared AR(1) core)
             genesets.py   # hallmark + c4 (shared load_gene_sets)
  effect/    effects.py    # uniform_single_effect (+ future)
  error/     errors.py     # t_error_sampler
             # NO signal/ — signals are distribution DATA (config), resolved by shared
             #              resolve_distribution; editing a signal = config change = hash rerun
fits/        cox.py logistic.py twogroup.py linear.py     # run_<method> + fit_/summarize_
reductions/  pip.py cs.py f1.py enrich.py
analyses/    pip.py cs.py logbf.py f1.py                  # grouped by family (38 renderers; per-renderer not worth it)
core.py      simulate(), SimulationSpec, spec_hash, resolve_distribution, re-exports
```

Principle: **distinct code → distinct file; shared code → same file; config-only differences (args, signal params) ride the hash, not a code file.** `ser_b2`/`null_b0` share `uniform_single_effect`; `t_df_3..30` share `t_error_sampler` — one file each, differences are args. `gaussian`/`uniform` markov share the AR(1) core → one file (splitting them would hide the `uniform→gaussian` dependency from the file-mtime trigger). Granularity knob = file organization.

## 4. Rules + native code-tracking — split by how the output is keyed

Code-change invalidation is **native snakemake** (mtime on the right files), enabled by §3's layout. The mechanism differs by stage because the output key differs:

**Name-keyed stages (reduce, analyze) → loop-generated per-item rules.** Their paths carry a readable discriminator, so generate one rule per library item; the static `script:` path gives native script-mtime tracking for free — no `code` lambda, no fingerprint helper.
```python
for r in LIBRARY["reductions"]:
    rule:
        name:   f"reduce_{r}"
        output: f"{ROOT}/by_batch/{{batch_hash}}/fits/{{method_hash}}/reductions/{r}.parquet"
        input:  deps = lambda wc, r=r: reduction_inputs(wc, r)     # early-bind r (late-binding closure trap)
        script: f"reductions/{r}.py"                                # static path -> native mtime tracking
```
- **Early-bind the loop var (`r=r`) in EVERY deferred lambda** — without it all generated rules close over the final `r`. `output`/`script` f-strings are evaluated eagerly, so they're safe; only lambdas need the snapshot. Same for the analyze loop.
- Edit `reductions/pip.py` → only `reduce_pip` reruns → only analyses requiring `pip` rerun. Add `reductions/new.py` → nothing else reruns. Data-driven (loop over the library), explicit, idiomatic.

**Hash-keyed stages (simulate, fit) → one generic rule each.** Their paths are content hashes with no per-item name to route on, so they stay generic and dispatch via the manifest. Code-tracking via a per-item **`code` lambda input** (native mtime on the resolved module file(s)):
```python
rule fit:
    output: f"{ROOT}/by_batch/{{batch_hash}}/fits/{{method_hash}}/fits.parquet"
    input:
        code = lambda wc: (method_code_files(MANIFEST["methods"][wc.method_hash]["name"], LIBRARY)
                           + simulation_code_files(MANIFEST["batches"][wc.batch_hash]["coordinate"], LIBRARY)),
    run:
        b      = MANIFEST["batches"][wc.batch_hash]
        spec   = resolve_simulation(LIBRARY, *b["coordinate"])
        method = resolve_method(LIBRARY, MANIFEST["methods"][wc.method_hash]["name"])
        write_parquet(fit_batch_method(spec, method, replicates=b["replicates"]), output.fits)
```
- `*_code_files` map an item → its defining module file(s) via `inspect.getfile`:
  ```python
  def _file(fn): return inspect.getfile(getattr(fn, "func", fn))   # unwrap partials
  def simulation_code_files(coord, lib):
      s = resolve_simulation(lib, *coord); fns = [core.simulate, s.design_sampler, s.effect_sampler]
      if s.error_sampler is not None: fns.append(s.error_sampler)
      return sorted({_file(f) for f in fns})
  def method_code_files(name, lib): return [_file(resolve_method(lib, name).function)]
  ```
- **fit depends on sim code too** (it re-simulates to obtain `X`, which `simulations.parquet` omits) — so a sampler bugfix correctly reruns sims *and* fits.
- **Precision matters most on fit** (expensive gibss models) — this is where the `code` lambda earns its keep; per-item rules aren't possible (hash-keyed).
- simulate: same mechanism; cheap, so coarse would also be acceptable.

**Limitations (accept/handle explicitly):** native mtime is **per-file** — editing `design/markov.py` reruns both gaussian and uniform sims (correct — shared code). **Imports aren't followed**: shared libs like `viz_utils` (used by analyses) aren't tracked unless listed — if a `viz_utils` change should retrigger analyses, add it to that analysis module's effective deps (or list `viz_utils.py`). Third-party (`gibss`/`numpy`) is opaque — pin versions / `--forcerun` on the rare editable edit.

This drops the v1 coarse `sources` **and** the proposed custom `code_fingerprint` — all native.

## 5. Reductions: drop `needs`

Every reduction is fit-scoped with a fixed input set (`fits`, `simulations`, `sample_metadata`). So:
- **Drop the `needs` toggle dict.** The reduce rule always wires `(fits, simulations, sample_metadata)` and passes a uniform `ctx`; the builder reads what it needs. (Also fixes the v1 latent bug where pip/cs read `simulations.parquet` without declaring it.)
- **Scope:** hardwire fit-scope (per-`(batch, method)`). No sim-only reduction exists today (`sample_metadata` is a materialize output). If a pure sim-summary reduction ever appears, add a `scope: batch` field then (escape hatch).
- A reduction entry collapses to `{function}` (+ optional `method_filter`, §6).
- **Uniform entrypoint:** every builder is `build(ctx) -> df` (`ctx` exposes `fits`, `sims`, `sample_metadata`, `sim_coordinate`) — no per-name dispatch.

## 6. Predicate filters

Importable boolean predicates (resolved via `resolve_callable`), applied at **DAG-construction time** (they see *config*, not materialized data).

- **`method_filter(method_spec: MethodSpec) -> bool`** on a **reduction** — which methods it's valid for (f1/enrich read `two_group_state`, `None` for non-twogroup). Replaces `startswith("twogroup")`. Applied where the loader enumerates `(batch, method)` pairs.
  ```yaml
  reductions:
    f1:     {function: build_f1_plot_data, method_filter: is_twogroup}
    enrich: {function: build_enrich_plot_data, method_filter: is_twogroup}
  ```
- **`simulation_filter(sim_descriptor) -> bool`** on an **analysis** — which sims to pool into the plot (e.g. exclude null sims from `causal_pip`/`mass_above_causal`; the `pip` reduction is identical for null/non-null — exclusion is a *plotting* choice). Replaces the old `pip_non_null`/`cs_non_null` group hack. Applied in `analysis_inputs` (depend only on passing sims' reductions) and `load_sc_bundle` (pool only passing sims).
  ```yaml
  analyses:
    pip_calibration: {requires: [pip]}                                 # all sims (null = calibration/FDP arm)
    causal_pip:      {requires: [pip], simulation_filter: has_causal}  # non-null only
  ```

**Predicate input contracts** (config-knowable only; cheap + pure; run once per item at DAG build):
- `method_filter(method_spec)` — `MethodSpec` (`.function`, `.kwargs`, `.name`).
- `simulation_filter(sim_descriptor)` — the **resolved coordinate dict** (not the `SimulationSpec`):
  ```python
  {design, enrichment: {function, arguments, intercept}, signal: {f0, f1}, error}
  # e.g. def has_causal(s): return s["enrichment"]["arguments"].get("causal_effect", 0) != 0
  ```
Data-dependent selection ("causal MAF > x") can't be a DAG-time filter — that's a post-hoc filter inside the analysis on the loaded bundle.

## Worked example (003 `hallmark__ser_b2__loc_2.0`, method `twogroup__L=1`, reduction `pip`)

| job | `code` deps (what reruns it) | rule form |
|---|---|---|
| simulate `…loc_2.0` | `core.py`, `design/genesets.py`, `effect/effects.py` | generic + `code` input |
| fit `…twogroup__L=1` | `fits/twogroup.py` + the simulate deps (re-simulates) | generic + `code` input |
| reduce `pip` | `reductions/pip.py` | loop-generated `reduce_pip`, `script:` |
| analyze `pip_calibration` | `analyses/pip.py` (+ `viz_utils` if listed) | loop-generated `analyze_pip_calibration`, `script:` |

Edit `reductions/pip.py` → only pip reductions + pip-requiring analyses rerun. Edit `fits/twogroup.py` → only twogroup fits (+ their reductions/analyses). Change `gaussian_p100`'s `rho` (config) → only that sim reruns via the hash, no code file touched.

## Carried-over fixes (from v1 review)

- Rule names read as `simulate` / `fit` / `reduce_<name>` / `analyze_<name>` (drop `materialize_twogroup_experiment_batch` jargon).
- `notebooks/dashboard.py` + `scripts/symlink_plots.py` still reference deleted `config`/`plot_configs` — port or remove.
- Port remaining experiments `001/002/004-007` to the new format.

## Open decisions

- **Output-level `method_filter`** (render-time foreground name-list in a supercollection `output`) — keep as an explicit list (lean) or unify to a predicate.
- **Analysis family granularity** — group `analyses/<family>.py` by required reduction (lean), or finer.

## Not in scope

- Cache-reuse hash-injection (v1 spec "Future work") — independent.
- Data-dependent (post-hoc) analysis filters.

## Limitation (resolved by Fix 1, 2026-06-20)

`load_sc_bundle` now receives the analysis's `simulation_filter` (wired in each `analyses/*.py` script block), so sim-filtered analyses (e.g. `causal_pip`) are safe in any group; prior to that fix, a sim-filtered analysis would attempt to read parquet for sims the DAG never built (FileNotFoundError).
