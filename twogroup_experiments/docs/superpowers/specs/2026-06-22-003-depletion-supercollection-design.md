# 003 depletion supercollection (mirror of enrichment)

Date: 2026-06-22

## Problem

Experiment 003 currently studies only an *enrichment* scenario: the causal gene
set is more likely to contain signals than the background
(`ser_b2` = intercept −2, causal_effect +2 → background P(signal)=0.12, in-set
0.50). We want a mirrored *depletion* scenario where the causal set is *less*
likely to contain signals, with the same low (mostly-null) background.

## Generative model recap

`logits = intercept + causal_effect · x_causal`, `z ~ Bernoulli(sigmoid(logits))`.

| scenario | enrichment def | background (x=0) | in-set (x=1) | Δlogit | null counterpart |
|---|---|---|---|---|---|
| enrichment (existing) | `ser_b2` (−2, +2) | 0.12 | 0.50 | +2 | `null_b0` (−2, 0) → 0.12 |
| depletion (new) | `ser_bneg2` (−2, −2) | 0.12 | 0.018 | −2 | `null_b0` (−2, 0) → 0.12 |

`(-2, -2)` keeps the same low background and the same null collection as
enrichment, flipping only the set effect (logit-symmetric around the −2
background). This preserves the "most background genes are null" property and is
magnitude-matched (|Δlogit| = 2) to enrichment. Depletion (0.12 → 0.018) is an
inherently subtler signal than enrichment (0.12 → 0.50) — expected, not a flaw.

## Design

### Components

1. **`experiments/library.yaml` — new enrichment `ser_bneg2`:**
   ```yaml
   ser_bneg2:
     function: uniform_single_effect
     arguments:
       causal_effect: -2.0
     intercept: -2.0
   ```
   (`uniform_single_effect` returns the single causal index for any nonzero
   effect, so a negative effect works unchanged.) The depletion null reuses the
   existing `null_b0` (−2, 0).

2. **`experiments/003_loc_snr.yaml` — add a depletion twin supercollection.**
   Mirror `003-hallmark-loc-snr` exactly, swapping only the collection
   enrichment list to `[ser_bneg2, null_b0]`:
   - Name: `003-hallmark-loc-snr-depletion`.
   - Same design (`hallmark`), same `error: gaussian`, same loc signal grid
     (`loc_0.5 … loc_3.0`).
   - Same `methods: *sweep_methods`, same `default_args`.
   - Same two outputs: `minimal-loc` (`method_filter: *base_methods`,
     `analyses: [pip, cs, logbf, pip_non_null, cs_non_null]`) and
     `minimal-loc-threshold-sweep` (`method_filter: *sweep_methods`,
     `analyses: [threshold_sweep]`).

   To avoid duplication, factor the shared loc-signal list, `default_args`, and
   the two-output block into `_anchors` and reference them from both
   supercollections. The existing enrichment SC keeps its current behavior
   byte-for-byte (same resolved methods/outputs/analyses).

### Not in scope

- Paired (reciprocal) analyses — those are 009-specific; 003 uses the regular
  analyses.
- Any change to the enrichment SC's resolved configuration.
- Other experiments.

## Testing

- **Enrichment def:** `loader.resolve_simulation(lib, "hallmark", "ser_bneg2",
  <signal>, "gaussian")` yields a spec with `intercept == -2.0` and the causal
  effect `-2.0` (verify via `b` having a single `-2.0` entry after `simulate`,
  or via the enrichment library entry).
- **Depletion SC resolves:** `loader.load_config()` contains
  `003-hallmark-loc-snr-depletion`; its collections use enrichment
  `ser_bneg2`/`null_b0`; `resolve_sc_analyses` for it matches the enrichment SC's
  analysis set (same outputs).
- **Enrichment SC unchanged:** the resolved methods + `resolve_sc_analyses` for
  `003-hallmark-loc-snr` are identical to before the change (guard the
  anchor refactor).
- **Manifest builds:** `manifest_dict` succeeds with both SCs present.

## Naming note

`ser_bneg2` mirrors the `ser_b2` convention (`b = -2`). The depletion direction
is encoded by the negative `causal_effect`; the shared `intercept: -2.0` keeps
the background identical to enrichment.
