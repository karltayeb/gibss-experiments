"""Generate experiments/019_logistic_calibrated.yaml from betas.json.

Two supercollections sharing the same content-addressed simulations:
  * 019-logistic-cheap : gIBSS + global-JJ, both L=1 and L=10, 50 reps (rpb=10, n_batches=5)
  * 019-logistic-cavi  : CAVI-cf,          both L=1 and L=10, 10 reps (rpb=10, n_batches=1)
Batch 0 (reps 0-9) is content-identical across both, so CAVI reuses those simulations.

Each supercollection has one collection PER DESIGN (betas are design-specific). Enrichment
entries are inline dicts (loader accepts name-or-dict); labels are display-only (stripped
from the content hash). Signal=binary + error=noiseless makes z the regressed response.

Cells per design: 3 intercepts x 4 T single-effect (12) + 3 nulls + 3 L* x 4 T x 4 gap
multi (48) = 63; x3 designs = 189 distinct simulation cells.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "experiments" / "019_logistic_calibrated.yaml"

TARGETS = [4, 8, 16, 32]          # E[LRT] rungs (~ logBF {2,4,8,16})
INTERCEPTS = [-3, -2, -1]         # single-effect sweep
LSTARS = [2, 3, 5]                # multi-effect causal counts
GAPS = [2, 5, 10, 20]            # spaced_index_effect gaps
MULTI_B0 = -2                     # multi-effect fixed intercept

DESIGNS = [
    ("gaussian", "gaussian_n500", "{function: gaussian_markov_X, arguments: {n: 500, p: 256, rho: 0.9}}"),
    ("bin1000", "binary_n1000_q50", "{function: binary_markov_X, arguments: {n: 1000, p: 256, corr: 0.8, density: 0.5}}"),
    ("bin10000", "binary_n10000_q05", "{function: binary_markov_X, arguments: {n: 10000, p: 256, corr: 0.8, density: 0.05}}"),
]


def _fmt_i(b0: int) -> str:
    return f"i{abs(b0)}" if b0 < 0 else f"i{b0}"  # i3 = intercept -3 (avoid '-')


def enrichment_entries(short: str, profile: str, betas: dict) -> list[str]:
    """Inline YAML enrichment dicts for one design's collection."""
    tbl = betas[profile]
    rows: list[str] = []
    # single-effect ladder
    for b0 in INTERCEPTS:
        row = tbl[str(b0)]
        for t, beta in zip(TARGETS, row):
            rows.append(
                f"- {{label: ser_{short}_{_fmt_i(b0)}_T{t}, function: uniform_single_effect, "
                f"arguments: {{causal_effect: {beta:.4f}}}, intercept: {float(b0)}}}"
            )
    # matched nulls (b=0), one per intercept
    for b0 in INTERCEPTS:
        rows.append(
            f"- {{label: null_{short}_{_fmt_i(b0)}, function: uniform_single_effect, "
            f"arguments: {{causal_effect: 0.0}}, intercept: {float(b0)}}}"
        )
    # multi-effect: equal-strength causals at b0=-2, placed by gap
    row_m = tbl[str(MULTI_B0)]
    for lstar in LSTARS:
        for t, beta in zip(TARGETS, row_m):
            effects = ", ".join(f"{beta:.4f}" for _ in range(lstar))
            for g in GAPS:
                rows.append(
                    f"- {{label: mc{lstar}_{short}_T{t}_g{g}, function: spaced_index_effect, "
                    f"arguments: {{causal_effects: [{effects}], gap: {g}}}, intercept: {float(MULTI_B0)}}}"
                )
    return rows


def collections_block(betas: dict, indent: str) -> str:
    blocks = []
    for short, profile, design_anchor in DESIGNS:
        entries = enrichment_entries(short, profile, betas)
        entry_lines = "\n".join(f"{indent}        {e}" for e in entries)
        blocks.append(
            f"{indent}- template: {{design: {design_anchor}, signal: binary, error: noiseless}}\n"
            f"{indent}  over:\n"
            f"{indent}    enrichment:\n{entry_lines}"
        )
    return "\n".join(blocks)


def pilot_block(betas: dict, indent: str) -> str:
    """A tiny content-subset (Gaussian design only) for smoke-testing the pipeline.

    Reuses the full run's enrichment entries verbatim (label stripped from the hash),
    so pilot fits are content-identical to full-run cells and get reused, not recomputed.
    A couple of single-effect rungs + null + one multi cell -> exercises L=1, L=10,
    spaced_index_effect, and the null path.
    """
    prof = "gaussian_n500"
    row = betas[prof]
    b2 = row["-2"]
    e = []
    for t, beta in zip(TARGETS, b2):           # b0=-2 ladder (all four rungs)
        e.append(f"- {{label: ser_gaussian_i2_T{t}, function: uniform_single_effect, "
                 f"arguments: {{causal_effect: {beta:.4f}}}, intercept: -2.0}}")
    e.append("- {label: null_gaussian_i2, function: uniform_single_effect, "
             "arguments: {causal_effect: 0.0}, intercept: -2.0}")
    beta16 = b2[TARGETS.index(16)]
    eff = ", ".join(f"{beta16:.4f}" for _ in range(3))
    e.append(f"- {{label: mc3_gaussian_T16_g10, function: spaced_index_effect, "
             f"arguments: {{causal_effects: [{eff}], gap: 10}}, intercept: -2.0}}")
    lines = "\n".join(f"{indent}        {x}" for x in e)
    design = "{function: gaussian_markov_X, arguments: {n: 500, p: 256, rho: 0.9}}"
    return (f"{indent}- template: {{design: {design}, signal: binary, error: noiseless}}\n"
            f"{indent}  over:\n{indent}    enrichment:\n{lines}")


def main() -> None:
    betas = json.loads((HERE / "betas.json").read_text())
    colls = collections_block(betas, "      ")
    pilot = pilot_block(betas, "      ")
    n_cells = len(DESIGNS) * (len(INTERCEPTS) * len(TARGETS) + len(INTERCEPTS)
                              + len(LSTARS) * len(TARGETS) * len(GAPS))
    text = f"""\
# 019_logistic_calibrated: logistic-SuSiE approximations (gIBSS / CAVI-cf / global-JJ), Q2,
# centered, shared Gaussian intercept, EB prior variance (capped at 100). See
# notes/logistic_simulation_design.md and analysis/logistic_susie_simulations/.
#
# Signal axis T = E[LRT] (~ 2 x E[logBF]); betas calibrated per (design, intercept) in
# calibration.py (profiled intercept), stored in betas.json. This file is GENERATED by
# generate_experiment.py -- edit that, not this.
#
# TWO supercollections share the same {n_cells} content-addressed simulation cells:
#   cheap: gIBSS + global-JJ, 50 reps (rpb=10 x n_batches=5)
#   cavi:  CAVI-cf,           10 reps (rpb=10 x n_batches=1); bump n_batches later for more.
# Both carry the L=1 and L=10 method variants; multi-effect cells also receive the (cheap)
# L=1 fits, which the analysis simply does not feature.
_anchors:
  cheap_methods: &cheap [logistic_q2_ser_gibss, logistic_q2_ser_globaljj, logistic_q2_L10_gibss, logistic_q2_L10_globaljj]
  cavi_methods: &cavi [logistic_q2_ser_cavi, logistic_q2_L10_cavi]
  default_args: &default_args {{min_log_bf: 2.0, max_cs_size: 10000, max_fdp: 0.5}}

supercollections:
  019-logistic-cheap:
    replicates_per_batch: 10
    n_batches: 5
    collections:
{colls}
    methods: *cheap
    default_args: *default_args
    outputs:
      - {{name: logistic_cheap, method_filter: *cheap, analyses: [pip, cs]}}

  019-logistic-cavi:
    replicates_per_batch: 10
    n_batches: 1
    collections:
{colls}
    methods: *cavi
    default_args: *default_args
    outputs:
      - {{name: logistic_cavi, method_filter: *cavi, analyses: [pip, cs]}}

  # --- pilot: Gaussian-only content subset, 10 reps, for smoke-testing the .done chain
  # (fits -> reductions -> analyses) before the full run. Cells are reused by 019 above.
  019-logistic-pilot-cheap:
    replicates_per_batch: 10
    n_batches: 1
    collections:
{pilot}
    methods: *cheap
    default_args: *default_args
    outputs:
      - {{name: pilot_cheap, method_filter: *cheap, analyses: [pip, cs]}}

  019-logistic-pilot-cavi:
    replicates_per_batch: 10
    n_batches: 1
    collections:
{pilot}
    methods: *cavi
    default_args: *default_args
    outputs:
      - {{name: pilot_cavi, method_filter: *cavi, analyses: [pip, cs]}}
"""
    OUT.write_text(text)
    print(f"wrote {OUT} ({n_cells} cells/supercollection)")


if __name__ == "__main__":
    main()
