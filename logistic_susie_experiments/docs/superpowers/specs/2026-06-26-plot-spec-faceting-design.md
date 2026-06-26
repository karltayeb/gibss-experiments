# Plot-spec faceting: separate "what gets fit" from "how it's plotted"

**Date:** 2026-06-26
**Status:** design approved, pre-implementation
**Scope:** `logistic_susie_experiments` plotting layer (loader, viz, Snakefile analyze path, experiment yaml).

## Problem

Today the `over:` field in a collection does double duty: it enumerates *what gets
fit* (design × enrichment) **and** becomes the only faceting handle for plots (via
`collection_name`). Aesthetics are hard-coded in renderers (color = method family;
facet = simulation/collection), so isolating a contrast requires authoring a
`method_filter` list per figure, and the `center` contrast can't get distinct
colors (two methods share a family base). Every analysis is also duplicated into a
per-sim and an `agg_*` (pooled) variant.

## Principle

Separate the two concerns:

- **Fit spec** = *what gets fit*: batches × methods, each already a coordinate in
  `MANIFEST` (`manifest_cache.json`). `over:` survives **only** as a fit
  enumerator.
- **Plot spec** = *how it's sliced*: declares faceting + aesthetics + filter over
  **dimensions derived from those coordinates**.

## Data model: dimensions as row columns

Every dimension a plot can reference becomes a column on each plot_data row. The
data is already in `MANIFEST`, keyed by `method` (method coordinate) and
`batch_hash` (batch coordinate) — both consumed today by the `fit` rule. We carry
them through to the bundle.

### `viz_dims.py` (new) — single source of truth

Two pure functions map a raw coordinate to clean **semantic** dims plus flattened
**raw** args:

```python
def method_dims(coord: dict) -> dict:
    k = coord.get("kwargs", {})
    fn = coord.get("function", "")
    impl = k.get("impl", "")
    if "irls" in fn or impl == "irls":
        family = "irls"
    elif "globaljj" in fn or impl == "globaljj":
        family = "globaljj"
    else:
        family = impl or fn
    semantic = {
        "family": family,
        "step": "one_step" if k.get("n_outer") == 1 else "converged",
        "prior": "fixed" if k.get("estimate_prior_variance") is False else "eb",
        "center": bool(k.get("center", False)),
        "cadence": k.get("ser_cadence", "block"),
        "L": int(k.get("L", 1)),
        "function": fn,                              # method type
    }
    raw = {f"m_{key}": v for key, v in k.items()}    # m_n_outer, m_prior_variance, ...
    return {**semantic, **raw}


def sim_dims(coord: dict) -> dict:
    d, e = coord["design"], coord["enrichment"]
    b = float(e["arguments"].get("causal_effect", 0.0))
    design_name = {"gaussian_markov_X": "gaussian",
                   "c4_gene_sets_X": "c4"}.get(d["function"], d["function"])
    semantic = {
        "design": design_name,
        "intercept": float(e["intercept"]),
        "b": b,
        "signal": b != 0.0,                          # null vs signal
    }
    raw = {f"d_{key}": v for key, v in d["arguments"].items()}    # d_n, d_p, d_rho
    raw |= {f"e_{key}": v for key, v in e["arguments"].items()}   # e_causal_effect
    return {**semantic, **raw}
```

Adding a new knob = one line here. No name/alias parsing, no per-spec re-derivation.

### Attachment

`load_sc_bundle` joins these onto every plot_data row using `MANIFEST`
(`method_dims(MANIFEST["methods"][mh])`, `sim_dims(MANIFEST["batches"][bh]["coordinate"])`).
`collection_name` and `over:`-as-facet are removed from the plot path.

A row goes from `{method, batch_hash, sample_id, <metrics>}` to additionally carry
`{family, step, prior, center, cadence, L, function, design, intercept, b, signal,
m_*, d_*, e_*}`.

## Plot spec schema

`plots:` replaces `outputs:`. One entry per figure:

```yaml
plots:
  step:   {analysis: pip_calibration, color: step,   linestyle: family, facet_col: design, filter: {prior: fixed, center: false}}
  prior:  {analysis: pip_calibration, color: prior,  linestyle: family, facet_col: design, filter: {step: converged, center: false}}
  center: {analysis: pip_calibration, color: center, facet_col: design, filter: {family: irls, step: converged, prior: fixed}}
  family: {analysis: pip_calibration, color: family, facet_col: design, filter: {step: converged, prior: fixed, center: false}}
```

Channels (all optional, all reference dim column names): `color`, `linestyle`,
`facet_row`, `facet_col`. Plus `analysis` (required) and `filter`.

### Filter semantics (this iteration)

- scalar value → equality (`center: false`)
- list value → membership (`prior: [eb, fixed]`)
- references any dim column (semantic or raw `m_*`/`d_*`/`e_*`).

Comparison operators (`{">=": 50}`) and expression escape hatches are **out of
scope**; the filter-apply is one function, so they are a backward-compatible
addition later (scalar/list syntax unchanged).

## Renderer architecture

Split each analysis's *geometry* from a shared *facet/aesthetic engine*.

### Per-analysis hooks

Each analysis exposes:

- `aggregate(rows) -> stats`: pool a set of rows into the plotted statistic
  (calibration sums bin-counts; power_fdp sweeps thresholds). **Must be
  grouping-invariant** — valid at any grouping, since the engine decides the
  grouping.
- `draw(ax, stats, *, color, linestyle, label)`: render one series on an axis.

### Generic driver (written once)

```
apply filter
-> split rows into facet grid by facet_row x facet_col
   -> within each cell, split into series by color x linestyle
      -> pool remaining rows via analysis.aggregate
      -> analysis.draw(ax, stats, color, linestyle, label)
build subplot grid, assign colors/linestyles from the channel dims' distinct
values, one shared legend, per-cell facet titles.
```

**Grouping rule:** a row lands in `facet[row,col] -> series[color,linestyle]`;
anything not on a channel and not filtered is pooled by `aggregate`. Putting
`design` on `facet_col` disaggregates by design; leaving it off pools across
designs. Colors come from a palette keyed on the `color` dim's distinct values
(not the old family-name map), so `center` (true/false) gets two colors.

## Agg elimination

Pooling-by-default makes `agg_*` redundant: the pooled view is a spec with no sim
dim on any channel; the per-sim view facets by a sim dim (`facet_col: design`).
Remove all `agg_*` analyses and `agg_*` analysis_groups from `library.yaml`. One
analysis defines geometry; agg/per-sim is a plot-spec choice. Reductions are
unchanged (still per-row).

## Migration (full, single path — no phasing)

- All experiments (`000`, `001`, `002`) convert `outputs:` → `plots:`. Remove
  `method_filter` and `collection_name`-as-facet.
- All 16 analyses get the hook form and lose their `agg_*` twin:
  - pip: `pip_calibration`, `power_fdp`, `causal_pip`, `mass_above_causal`
  - cs: `causal_rank`, `preceding_posterior_mass_ecdf`, `cs_dot_summary`,
    `cs_calibrated_dot`, `cs_size_power`, `cs_radius_power`, `cs_power_fdp`,
    `cs_power_size_coverage_trace`, `cs_coverage_trace`, `cs_coverage_size`,
    `cs_coverage_radius`, `cs_calibration`
- One Snakefile analyze path (no `outputs:`/`plots:` branch). The analyze rule
  resolves a plot spec instead of an output args set.
- `simulation_filter` on analyses (e.g. `has_causal`) folds into spec `filter`
  (`signal: true`) where applicable; keep the predicate only if a filter can't
  express it.

### Caveats to resolve during implementation

- Verify each analysis's `aggregate` is genuinely grouping-invariant. Flagged
  risk: dot-summary / per-method scalar summaries (`cs_dot_summary`,
  `cs_calibrated_dot`) — adapt or mark unsupported under arbitrary pooling.
- `plot_browser.py` axes ("Comparison" = old args_name) map to plot-spec names;
  confirm `symlink_plots.py` path layout still holds (`{sc}/{analysis}/{plot}.pdf`).

## Out of scope

- Filter comparison operators / expression filters.
- New analyses or metrics.
- twogroup_experiments (shares the pattern; migrate separately if desired).
- Marker/size aesthetic channels (color + linestyle + 2 facets suffice now).
