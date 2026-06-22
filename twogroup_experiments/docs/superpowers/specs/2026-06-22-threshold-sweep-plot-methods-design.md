# Threshold-sweep plot methods (003 minimal-loc-threshold-sweep)

Date: 2026-06-22

## Problem

The `minimal-loc-threshold-sweep` output in `experiments/003_loc_snr.yaml`
renders `causal_pip` and `mass_above_causal` across the cox/logistic threshold
grid. We want this plot to:

1. Add the reversed right-censored cox (`cox_reversed_censored`) as a swept line.
2. Add threshold-free reference methods (uncensored cox, uncensored
   cox-reversed, twogroup variants) as horizontal lines.
3. Connect a method's thresholded points with a line.
4. One legend entry per method (color denotes method/family); horizontal lines
   denote threshold-free methods — no per-threshold legend entries.

## Background finding

Requirements 3 and 4 are already implemented. `_plot_causal_pip_on_ax` and
`_plot_mass_above_causal_on_ax` (`viz_utils.py`) draw any method with a
non-null `threshold` as a line+markers over the threshold axis (colored by
family), and any threshold-null method as a horizontal `axhline`. The legend
labels by `method_display_base` (one per family), colored by `method_color`.
`method_color` and `method_metadata` both derive family as
`name.split("__")[0]`.

So this work is config plus color/label map entries — **no viz code change**.

The uncensored endpoints must be drawn as horizontal (threshold-free) lines.
Mathematically "uncensored cox" = cox at threshold 0 and "uncensored
cox-reversed" = cox-reversed at threshold ∞, but to render as a horizontal the
method must carry `threshold = null` (so `is_thresholded` is false). The
reversed uncensored already exists as the threshold-null `cox_reversed`; the
forward uncensored needs a new threshold-null entry.

## Design

### Components

1. **`experiments/library.yaml` — add `cox_uncensored`:**
   ```yaml
   cox_uncensored:
     function: run_cox_method
     template:
       time_sign: -1.0
       threshold: null
     over:
       L:
       - 1
   ```
   Produces `cox_uncensored__L=1` (all events, time_sign -1) → threshold-null →
   horizontal. The reversed uncensored is the existing `cox_reversed__L=1`.

2. **`viz_utils.py` — color and label map entries (shared family colors):**
   - `method_family_color_map`:
     - `cox_reversed_censored`: `#E69F00` (same orange as `cox_reversed`)
     - `cox_uncensored`: `#009E73` (same green as `cox`)
   - `method_family_label_map`:
     - `cox_reversed_censored`: `Cox reversed (censored)`
     - `cox_uncensored`: `Cox (uncensored)`

   Existing `cox` and `cox_reversed` map entries are unchanged, so other
   experiments are unaffected.

3. **`experiments/003_loc_snr.yaml` — sweep output method sets:**
   - Swept lines (thresholded, grid 0.5–3.5): `cox`, `cox_reversed_censored`,
     `logistic_threshold`.
   - Horizontals (threshold-free): `cox_uncensored__L=1`, `cox_reversed__L=1`,
     `twogroup_oracle__L=1`, `twogroup__L=1`, `twogroup_loc_fam__L=1`.
   - Add every method above to the supercollection `methods` list (so they are
     fit) and to the `minimal-loc-threshold-sweep` output's `method_filter`.
   - The `minimal-loc` output and other 003 settings are unchanged.

### Resulting plot

- Cox family (green): solid line = censored sweep (`cox`), dashed horizontal =
  uncensored (`cox_uncensored`).
- Cox-reversed family (orange): solid line = censored sweep
  (`cox_reversed_censored`), dashed horizontal = uncensored (`cox_reversed`).
- Logistic (blue): solid line = `logistic_threshold` sweep.
- Twogroup variants (reds): horizontal lines.
- One legend entry per method; color = family.

## Testing

- **Map resolution:** `viz_utils.method_color("cox_reversed_censored__threshold=2.00__L=1")`
  returns the cox-reversed orange and `method_color("cox_uncensored__L=1")`
  returns the cox green; `method_family_label_map()` contains the two new
  families with the specified labels.
- **003 config resolution:** `loader.method_metadata` over the 003 sweep
  methods yields the expected `is_thresholded` split — `cox`,
  `cox_reversed_censored`, `logistic_threshold` thresholded (lines);
  `cox_uncensored`, `cox_reversed`, the three twogroup variants non-thresholded
  (horizontals). The `minimal-loc-threshold-sweep` output's resolved method set
  matches.

## Out of scope

- Wiring these methods into other experiments.
- Logistic uncensored reference (not requested).
- Any change to the `causal_pip` / `mass_above_causal` rendering code.

## Naming note

`cox_uncensored` is a naming convenience for "cox with threshold 0"; the
`cox`/`cox_reversed` base names remain the (threshold-suffixed) censored
sweeps, consistent with the deferred full rename in the right-censored-cox
spec.
