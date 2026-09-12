"""Generate experiments/021_poisson_calibrated.yaml from betas.json.

The Poisson companion to 019's generator. ONE supercollection (Poisson CAVI folds are
closed-form/cheap, so no cheap-vs-cavi split): all 20 method arms at 50 reps.

Baseline-rate axis lambda0 in {0.1, 1, 10}; the enrichment `intercept` is the baseline
log-rate b0 = log(lambda0). Betas are calibrated per (design, lambda0, T) in calibration.py
and stored in betas.json (keyed betas[profile][lambda0][T]). Enrichment entries are inline
dicts; labels are display-only (stripped from the content hash).

Cells per design: 3 rates x 4 T single-effect (12) + 3 nulls + 3 L* x 4 T x 4 gap multi (48)
= 63; x3 designs = 189 distinct simulation cells (same shape as 019).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "experiments" / "021_poisson_calibrated.yaml"

# Batches of `replicates_per_batch` (10) reps for the full `021-poisson` SC. Start at 1 (10 reps)
# to shake out the SLURM path; bump to 5 (50 reps) later -- the loader takes the max n_batches
# across sharing SCs, so scaling up only ADDS batches 1..N-1 and reuses batch 0 (bare sim_hash).
N_BATCHES = 5

TARGETS = [4, 8, 16, 32]          # E[LRT] rungs (~ logBF {2,4,8,16})
LAMBDA0 = [0.1, 1.0, 10.0]        # baseline mean count; b0 = log(lambda0) = {-2.30, 0, +2.30}
LSTARS = [2, 3, 5]                # multi-effect causal counts
GAPS = [2, 5, 10, 20]            # spaced_index_effect gaps
MULTI_LAMBDA0 = 1.0               # multi-effect fixed baseline rate (b0 = 0)

DESIGNS = [
    ("gaussian", "gaussian_n500", "{function: gaussian_markov_X, arguments: {n: 500, p: 256, rho: 0.9}}"),
    ("bin1000", "binary_n1000_q50", "{function: binary_markov_X, arguments: {n: 1000, p: 256, corr: 0.8, density: 0.5}}"),
    ("bin10000", "binary_n10000_q05", "{function: binary_markov_X, arguments: {n: 10000, p: 256, corr: 0.8, density: 0.05}}"),
]

# method arms: EB (headline) + fixed unit-prior (_pv1) twins. See library.yaml.
_EB = [
    "poisson_q2_ser_gibss", "poisson_q1_ser_gibss", "poisson_q2_ser_cavi", "poisson_q1_ser_cavi",
    "poisson_q2_ser_score", "poisson_q2_L10_gibss", "poisson_q1_L10_gibss", "poisson_q2_L10_cavi",
    "poisson_q1_L10_cavi", "poisson_q2_L10_score",
]
METHODS = _EB + [m + "_pv1" for m in _EB]


def _key(lam0: float) -> str:
    return f"{lam0:g}"                       # betas.json lambda0 key: 0.1 / 1 / 10


def _rlab(lam0: float) -> str:
    return "r" + f"{lam0:g}".replace(".", "p")   # rate label: r0p1 / r1 / r10 (no dot)


def enrichment_entries(short: str, profile: str, betas: dict) -> list[str]:
    """Inline YAML enrichment dicts for one design's collection."""
    tbl = betas[profile]
    rows: list[str] = []
    # single-effect ladder, one block per baseline rate
    for lam0 in LAMBDA0:
        b0 = math.log(lam0)
        for t in TARGETS:
            beta = tbl[_key(lam0)][str(t)]
            rows.append(
                f"- {{label: ser_{short}_{_rlab(lam0)}_T{t}, function: uniform_single_effect, "
                f"arguments: {{causal_effect: {beta:.4f}}}, intercept: {b0:.4f}}}"
            )
    # matched nulls (b=0), one per rate
    for lam0 in LAMBDA0:
        b0 = math.log(lam0)
        rows.append(
            f"- {{label: null_{short}_{_rlab(lam0)}, function: uniform_single_effect, "
            f"arguments: {{causal_effect: 0.0}}, intercept: {b0:.4f}}}"
        )
    # multi-effect: equal-strength causals at lambda0=1 (b0=0), placed by gap
    b0_m = math.log(MULTI_LAMBDA0)
    tbl_m = tbl[_key(MULTI_LAMBDA0)]
    for lstar in LSTARS:
        for t in TARGETS:
            beta = tbl_m[str(t)]
            effects = ", ".join(f"{beta:.4f}" for _ in range(lstar))
            for g in GAPS:
                rows.append(
                    f"- {{label: mc{lstar}_{short}_T{t}_g{g}, function: spaced_index_effect, "
                    f"arguments: {{causal_effects: [{effects}], gap: {g}}}, intercept: {b0_m:.4f}}}"
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
    """Gaussian-only content subset for smoke-testing the .done chain before the full run.
    Reuses the full run's enrichment entries verbatim (label stripped from the hash), so pilot
    fits are content-identical to full-run cells and get reused. lambda0=1 ladder + null + one
    multi cell -> exercises L=1, L=10, spaced_index_effect, and the null path."""
    prof = "gaussian_n500"
    tbl = betas[prof][_key(1.0)]
    e = []
    for t in TARGETS:
        e.append(f"- {{label: ser_gaussian_r1_T{t}, function: uniform_single_effect, "
                 f"arguments: {{causal_effect: {tbl[str(t)]:.4f}}}, intercept: 0.0}}")
    e.append("- {label: null_gaussian_r1, function: uniform_single_effect, "
             "arguments: {causal_effect: 0.0}, intercept: 0.0}")
    beta16 = tbl["16"]
    eff = ", ".join(f"{beta16:.4f}" for _ in range(3))
    e.append(f"- {{label: mc3_gaussian_T16_g10, function: spaced_index_effect, "
             f"arguments: {{causal_effects: [{eff}], gap: 10}}, intercept: 0.0}}")
    lines = "\n".join(f"{indent}        {x}" for x in e)
    design = "{function: gaussian_markov_X, arguments: {n: 500, p: 256, rho: 0.9}}"
    return (f"{indent}- template: {{design: {design}, signal: binary, error: noiseless}}\n"
            f"{indent}  over:\n{indent}    enrichment:\n{lines}")


def main() -> None:
    betas = json.loads((HERE / "betas.json").read_text())["betas"]
    colls = collections_block(betas, "      ")
    pilot = pilot_block(betas, "      ")
    methods_yaml = "[" + ", ".join(METHODS) + "]"
    n_cells = len(DESIGNS) * (len(LAMBDA0) * len(TARGETS) + len(LAMBDA0)
                              + len(LSTARS) * len(TARGETS) * len(GAPS))
    text = f"""\
# 021_poisson_calibrated: Poisson-SuSiE approximations (gIBSS Q1/Q2, CAVI-cf Q2, CAVI-selfnorm
# Q1, score), centered, shared Gaussian intercept. Well-specified counts y ~ Poisson(exp(b0 +
# X beta)); the response is simulation.y_count. NO Jaakkola-Jordan arm (logistic-only). Every
# arm has an EB prior variance twin (capped 100) and a fixed unit-prior `_pv1` twin.
#
# Baseline-rate axis lambda0 = e^{{b0}} in {{0.1, 1, 10}} (b0 in {{-2.30, 0, +2.30}}): a DISCRETENESS
# axis, not power -- betas are calibrated per (design, lambda0) to E[LRT] = T in {{4,8,16,32}} in
# analysis/poisson_susie_simulations/calibration.py (betas.json), so detectability is matched and
# only the count discreteness varies with the rate. This file is GENERATED by
# generate_experiment.py -- edit that, not this.
#
# ONE supercollection (Poisson CAVI folds are closed-form/cheap): all {len(METHODS)} arms at
# {N_BATCHES * 10} reps (rpb=10 x n_batches={N_BATCHES}) over {n_cells} content-addressed simulation cells per
# design x3 designs. Bump n_batches (in generate_experiment.py) to scale reps; batch 0 is reused.
_anchors:
  methods: &methods {methods_yaml}
  default_args: &default_args {{min_log_bf: 2.0, max_cs_size: 10000, max_fdp: 0.5}}

supercollections:
  021-poisson:
    replicates_per_batch: 10
    n_batches: {N_BATCHES}
    collections:
{colls}
    methods: *methods
    default_args: *default_args
    outputs:
      - {{name: poisson, method_filter: *methods, analyses: [pip, cs]}}

  # --- pilot: Gaussian-only content subset, 10 reps, for smoke-testing the .done chain
  # (fits -> reductions -> analyses) before the full run. Cells are reused by 021 above.
  021-poisson-pilot:
    replicates_per_batch: 10
    n_batches: 1
    collections:
{pilot}
    methods: *methods
    default_args: *default_args
    outputs:
      - {{name: pilot, method_filter: *methods, analyses: [pip, cs]}}
"""
    OUT.write_text(text)
    print(f"wrote {OUT} ({n_cells} cells/design, {len(METHODS)} methods)")


if __name__ == "__main__":
    main()
