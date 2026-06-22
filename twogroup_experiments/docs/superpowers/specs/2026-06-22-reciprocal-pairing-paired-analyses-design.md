# Reciprocal-rate pairing for 009 (paired analyses)

Date: 2026-06-22

## Problem

Experiment 009 sweeps a log-symmetric lambda (hazard-ratio) grid of four
reciprocal pairs: `2/3 & 3/2`, `3/4 & 4/3`, `4/5 & 5/4`, `9/10 & 10/9`. The
plots currently treat each lambda as a separate collection (8 facet rows in
non-agg plots; 8 collections aggregated in agg plots), so a reciprocal pair
(equal-strength depletion vs enrichment) is not visually paired.

We want a uniform treatment across all 009 plots (agg and non-agg):

- **Facet by `max(rate, 1/rate)`** — each reciprocal pair shares one panel.
- **Color by `rate < 1`** — depletion (rate<1) vs enrichment (rate>1) as two
  series.

This must be a **minimal change that reuses existing rendering** — no rewrite
of the ~25 existing `_make_*` analysis functions.

## Approach

A new "paired" analysis family that wraps every existing analysis: it
transforms the loaded bundle (`combined_data`), then calls the existing
`_make_*` renderer unchanged. The single-method nature of 009
(`cox_reversed__L=1`) lets us repurpose the method/color dimension to encode
depletion/enrichment.

### Bundle facts (verified)

`loader.load_sc_bundle` returns:
- `<reduction>_plot_data` frames with a `collection_name` column = the
  collection **alias** (e.g. `lambda=2/3`) and a `method` column
  (`cox_reversed__L=1`).
- `method_metadata` (DataFrame) built from the SC's real methods.
- `collection_names` = list of aliases.

`generate_plots.render_analysis` calls `ANALYSIS_RENDERERS[analysis](bundle,
settings)`. `foreground_methods(meta, settings) = settings["method_filter"] &
meta.method`. Renderers facet by `collection_name` and color by `method`
(family = `method.split("__")[0]`).

### The transform — `pair_reciprocal(bundle)`

Returns a new bundle:
1. Parse numeric `rate` from each `collection_name` alias (strip `lambda=`,
   evaluate the `a/b` fraction).
2. For every `<reduction>_plot_data` frame:
   - `method` → `"depletion"` if `rate < 1` else `"enrichment"`.
   - `collection_name` → pair label = formatted `max(rate, 1/rate)`
     (e.g. `lambda*=3/2`).
   - leave `threshold` null (cox_reversed is threshold-free).
3. Replace `method_metadata` with two rows (`depletion`, `enrichment`):
   `threshold=null`, `is_thresholded=False`,
   `method_display`/`method_display_base` = `Depletion` / `Enrichment`.
4. `collection_names` → sorted unique pair labels.

### The wrapper

```python
def _paired(make_fn):
    def renderer(bundle, settings):
        b = pair_reciprocal(bundle)
        s = {**settings, "method_filter": ["depletion", "enrichment"]}
        return make_fn(b, s)
    return renderer
```

Overriding `method_filter` keeps `foreground_methods` non-empty (it now selects
the two sign pseudo-methods).

### Why this works uniformly

- **Non-agg** renderers facet by `collection_name` → now the pair label (one
  row per pair), color by `method` → now the sign. Two curves per panel.
- **Agg** renderers ignore `collection_names` and aggregate by `method` → now
  aggregate depletion vs enrichment across all collections. One depletion + one
  enrichment curve.

No `_make_*` function changes.

## Components

1. **`analyses/paired.py`** (new):
   - `pair_reciprocal(bundle)` transform.
   - `_paired(make_fn)` factory.
   - `RENDERERS = {f"{name}_paired": _paired(fn) for name, fn in
     {**pip.RENDERERS, **cs.RENDERERS, **logbf.RENDERERS}.items()}` — auto-wraps
     every analysis.
   - the `if "snakemake" in globals():` dispatch block, mirroring
     `analyses/pip.py` (so `analyses/paired.py` works as the script).
2. **`analyses/__init__.py`**: add `**paired.RENDERERS` to `ANALYSIS_RENDERERS`.
3. **`experiments/loader.py`** — suffix-aware lookups so paired analyses need no
   duplicated library entries:
   - `_base_analysis(name) = name[:-7] if name.endswith("_paired") else name`.
   - `analysis_family`: if `name.endswith("_paired")` return `"paired"`.
   - `analysis_requires` / `analysis_simulation_filter`: look up
     `_base_analysis(name)`.
   - `flatten_analyses`: validate via `_base_analysis(n)` so `_paired` names
     pass the "known analysis" check.
4. **`viz_utils.py`** — map entries:
   - `method_family_color_map`: `depletion`, `enrichment` (two distinct colors).
   - `method_family_label_map`: `depletion` → `Depletion`, `enrichment` →
     `Enrichment`.
5. **`experiments/library.yaml`** — one analysis group listing the paired
   versions of 009's analyses:
   ```yaml
   paired_009:
   - <each analysis in pip/cs/logbf/pip_non_null/cs_non_null>_paired
   ```
6. **`experiments/009_cox_well_specified.yaml`**: `analyses` anchor →
   `[paired_009]`.

## Testing

- **Transform unit test:** feed a synthetic bundle (two collections
  `lambda=2/3`, `lambda=3/2`, one method) to `pair_reciprocal`; assert both map
  to the same pair label, `method` becomes `depletion`/`enrichment`
  respectively, and `method_metadata` has the two sign rows.
- **Rate parsing:** `lambda=9/10` → 0.9 → depletion; `lambda=10/9` → enrichment;
  pair labels equal for reciprocals.
- **Registration:** `generate_plots.ANALYSIS_RENDERERS` contains
  `causal_pip_paired`; `loader.analysis_family("causal_pip_paired") ==
  "paired"`; `loader.analysis_requires(cfg, "causal_pip_paired")` equals the
  base `causal_pip` requires.
- **Render smoke:** a paired renderer on the synthetic bundle returns a figure
  with the two sign series (legend `Depletion`, `Enrichment`), no exception.

## Out of scope

- Applying paired analyses to experiments other than 009.
- Changing any existing `_make_*` renderer or the generic facet/color logic.
- Multi-method paired plots (009 is single-method by construction).

## Caveats

- The pseudo-method trick discards the real method identity in the plot
  (acceptable: 009 is single-method `cox_reversed`). Color/legend encode
  depletion vs enrichment.
- Rate parsing assumes aliases of the form `lambda=a/b`; if an alias is not a
  parseable fraction the transform should error loudly rather than mislabel.
