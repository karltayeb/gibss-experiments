"""Generate experiments/025_poisson_rate.yaml: the 022-024 protocol for Poisson SuSiE, with the
baseline rate as the axis.

Question: does the background rate lambda0 separate gIBSS from exact CAVI? gIBSS plugs in the mean
offset, so it drops the Jensen term sum_i lambda_i (E e^{o_i} - e^{E o_i}), which scales with the
rate. Arms (all Q2, EB prior capped at 100): CAVI-Q2 (exact closed-form offset fold), gIBSS-Q2
(plug-in mean offset), gIBSS-Laplace (order-1 quadrature SER, plug-in offset, reduced to Q2),
score (one Newton step from the intercept-only null).

Grid per design: lambda0 in {0.01, 0.1, 1, 10, 100} (intercept b0 = log lambda0) x T = E[LRT] in
{8, 12, 16, 20, 24} x gap in {8, 64}, L* = 3 equal causals, fit L = 5; plus one null per lambda0.
beta per (design, lambda0, T) from betas_rate.json (calibrate_rate.py, one causal column).

Designs:
  gaussian  gaussian_markov_X(n=500, p=256, rho=0.9) (023's). Causals at columns 0, gap, 2 gap.
  binary    binary_markov_X(n=1000, p=256, corr=0.8, density=0.1) (022's chain at m = 100).
  block     binary_attrition_X(n=1000, p=256, corr=0.8, density=0.5, block_size=8, drop=0.45)
            (024's), causals at depth 3 (m ~ 83, close to binary's 100) in blocks gap/8 apart, so
            their decoys are their own ancestors/descendants.

Supercollections per design D: 025-rate-D (200 reps), 025-rate-D-pilot (its batch 0, 10 reps),
025-rate-D-ser (one causal at column 64 + depth, L=1 SER arms, 200 reps) and 025-rate-D-ser-pilot.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "experiments" / "025_poisson_rate.yaml"

P = 256
BATCHES = 20                    # 200 reps
LSTAR = 3
GAPS = [8, 64]
SER_GAP = 64
METHODS = ["poisson_q2_L5_cavi", "poisson_q2_L5_gibss", "poisson_q2_L5_gibss_laplace", "poisson_q2_L5_score"]
SER_METHODS = ["poisson_q2_ser_cavi", "poisson_q2_ser_gibss", "poisson_q2_ser_gibss_laplace", "poisson_q2_ser_score"]


def _rlab(lam0: str) -> str:
    return "r" + lam0.replace(".", "p")          # r0p01 / r1 / r10 / r100


def _yl(xs):
    return "[" + ", ".join(xs) + "]"


def _designs(meta: dict) -> dict[str, tuple[str, int]]:
    """design key -> (YAML design entry, causal base index)."""
    return {
        "gaussian": (f"{{function: gaussian_markov_X, arguments: {{n: {meta['gauss_n']}, p: {P}, rho: 0.9}}}}", 0),
        "binary": (f"{{function: binary_markov_X, arguments: {{n: {meta['bin_n']}, p: {P}, corr: 0.8, "
                   f"density: {meta['bin_density']}}}}}", 0),
        "block": (f"{{function: binary_attrition_X, arguments: {{n: {meta['bin_n']}, p: {P}, corr: 0.8, "
                  f"density: 0.5, block_size: 8, drop: 0.45}}}}", meta["block_depth"]),
    }


def _effect(label: str, betas: list[float], gap: int, base: int, b0: float) -> str:
    eff = ", ".join(f"{b:.4f}" for b in betas)
    base_s = f", base_index: {base}" if base else ""
    return (f"- {{label: {label}, function: spaced_index_effect, "
            f"arguments: {{causal_effects: [{eff}], gap: {gap}{base_s}}}, intercept: {b0:.4f}}}")


def _null(label: str, b0: float) -> str:
    return (f"- {{label: {label}, function: uniform_single_effect, "
            f"arguments: {{causal_effect: 0.0}}, intercept: {b0:.4f}}}")


def _collection(design: str, rows: list[str], ind: str = "      ") -> str:
    lines = "\n".join(f"{ind}        {e}" for e in rows)
    return (f"{ind}- template: {{design: {design}, signal: binary, error: noiseless}}\n"
            f"{ind}  over:\n{ind}    enrichment:\n{lines}")


def _sc(name: str, collection: str, methods: list[str], batches: int) -> str:
    out = name.replace("-", "_")
    return f"""\
  {name}:
    replicates_per_batch: 10
    n_batches: {batches}
    collections:
{collection}
    methods: {_yl(methods)}
    default_args: *default_args
    outputs:
      - {{name: {out}, method_filter: {_yl(methods)}, analyses: [pip, cs]}}
"""


def main() -> None:
    cal = json.loads((HERE / "betas_rate.json").read_text())
    meta, betas = cal["_meta"], cal["betas"]
    lams = [f"{l:g}" for l in meta["lambda0"]]
    targets = [str(t) for t in meta["targets"]]
    blocks = []
    n_multi = n_ser = 0
    for key, (design, base) in _designs(meta).items():
        multi, ser = [], []
        for lam0 in lams:
            b0 = math.log(float(lam0))
            for t in targets:
                beta = betas[key][lam0][t]
                for g in GAPS:
                    multi.append(_effect(f"mc{LSTAR}_{_rlab(lam0)}_T{t}_g{g}", [beta] * LSTAR, g, base, b0))
                ser.append(_effect(f"ser_{_rlab(lam0)}_T{t}", [beta], SER_GAP, base, b0))
            multi.append(_null(f"null_{_rlab(lam0)}", b0))
            ser.append(_null(f"null_{_rlab(lam0)}", b0))
        n_multi, n_ser = len(multi), len(ser)
        mc, sc = _collection(design, multi), _collection(design, ser)
        sc_name = f"025-rate-{key}"
        blocks += [f"  # {key}: {n_multi} cells x {len(METHODS)} L=5 arms, {BATCHES * 10} reps; pilot = batch 0.",
                   _sc(sc_name, mc, METHODS, BATCHES), _sc(sc_name + "-pilot", mc, METHODS, 1),
                   f"  # {key} single effect: {n_ser} cells x {len(SER_METHODS)} L=1 SER arms.",
                   _sc(sc_name + "-ser", sc, SER_METHODS, BATCHES), _sc(sc_name + "-ser-pilot", sc, SER_METHODS, 1)]
    header = f"""\
# 025_poisson_rate: Poisson SuSiE, gIBSS-Q2 vs CAVI-Q2 vs score along the background rate.
# y ~ Poisson(exp(b0 + X beta)), b0 = log lambda0, lambda0 in {{{", ".join(lams)}}}; T = E[LRT] in
# {{{", ".join(targets)}}}; L* = {LSTAR} causals at gap in {{{", ".join(map(str, GAPS))}}}, fit L = 5; one null per
# lambda0. beta per (design, lambda0, T) from analysis/poisson_susie_simulations/betas_rate.json.
# Designs: gaussian (AR1 n={meta['gauss_n']} rho 0.9), binary (Markov n={meta['bin_n']} density
# {meta['bin_density']}), block (024's attrition design, causals at depth {meta['block_depth']}, m ~
# {meta['bin_n'] * meta['block_density']:.0f}). -ser SCs: one causal at column {SER_GAP} + depth, L=1 SER arms.
# GENERATED by analysis/poisson_susie_simulations/generate_rate_experiment.py -- edit that, not this.
_anchors:
  default_args: &default_args {{min_log_bf: 2.0, max_cs_size: 10000, max_fdp: 0.5}}

supercollections:
"""
    OUT.write_text(header + "\n".join(blocks))
    print(f"wrote {OUT}: 3 designs x ({n_multi} multi + {n_ser} ser) cells")


if __name__ == "__main__":
    main()
