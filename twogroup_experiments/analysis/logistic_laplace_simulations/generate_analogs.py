"""Generate the two 022 analogs: experiments/023_logistic_laplace_gaussian.yaml and
experiments/024_logistic_laplace_nested.yaml.

Both keep 022's protocol fixed (n=1000, p=256, b0=-2, L=5, L*=3 at gap in {8, 64}, T in {8, 16},
EB prior, the six Q2-scored arms CAVI-Q2 / gIBSS-Q2 / gIBSS-Laplace / global-JJ / local-JJ / score,
200 reps) and swap the design + its axis:

  023 GAUSSIAN  gaussian_markov_X(n=500, p, rho=0.95): a small illustrative dense design. Every column is N(0,1): dense,
                every row informative, no set size. Axis = signal T in {4, 8, 12, 16, 20, 24} at fixed
                rho (beta per T, betas_gaussian_n500.json): the quantity 022 varies within each
                panel, extended upward to where the per-row curvature spread strains Laplace /
                the JJ bound / score's one-step. Causal-causal latent correlation rho**gap
                (0.43 at gap 8, ~0 at gap 64: correlated vs independent causals). One null.
  024 NESTED    binary_attrition_X(n, p, corr=0.8, density=0.5, block_size=8, drop=0.45): 32
                blocks of 8 nested sets shrinking from ~500 to ~8 members. Axis = causal DEPTH
                d in 0..7 (expected size 500 * 0.55**d = 500/275/151/83/46/25/14/8, the full
                breadth of 022's m in one matrix), beta per (d, T) from betas_nested_n1000.json. Causals sit at depth d in
                blocks gap/8 apart (spaced_index_effect base_index=d), so gap 8 = adjacent
                blocks, gap 64 = 8 blocks apart. Each causal's decoys are its own ancestors and
                descendants at other sizes - the heterogeneity 022's homogeneous-m cells cannot
                show. The design is fixed, so there is ONE null cell.

Supercollections per experiment: <sc> (full grid), <sc>-nocavi (same cells minus CAVI, run
first), <sc>-pilot (the full grid's cells at 10 reps = batch 0, for a local smoke / first look).
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent / "experiments"

N, P, B0 = 1000, 256, -2.0
BATCHES = 20                    # 200 reps
TARGETS = [8, 16]
LSTAR = 3
GAPS = [8, 64]
METHODS = ["logistic_q2_L5_cavi", "logistic_q2_L5_gibss", "logistic_q2_L5_gibss_laplace",
           "logistic_q2_L5_globaljj", "logistic_q2_L5_localjj", "logistic_q2_L5_score"]
NOCAVI = [m for m in METHODS if not m.endswith("_cavi")]

GAUSS_RHO = 0.9
GAUSS_N = 500                   # small, illustrative; not tied to 019
GAUSS_TARGETS = [4, 8, 12, 16, 20, 24]
NESTED = dict(corr=0.8, density=0.5, block_size=8, drop=0.45)
DEPTHS = list(range(8))


def _yl(xs):
    return "[" + ", ".join(xs) + "]"


def _multi(label, beta, gap, base=0):
    eff = ", ".join(f"{beta:.4f}" for _ in range(LSTAR))
    base_s = f", base_index: {base}" if base else ""
    return (f"- {{label: {label}, function: spaced_index_effect, "
            f"arguments: {{causal_effects: [{eff}], gap: {gap}{base_s}}}, intercept: {B0:.1f}}}")


def _null(label):
    return (f"- {{label: {label}, function: uniform_single_effect, "
            f"arguments: {{causal_effect: 0.0}}, intercept: {B0:.1f}}}")


def _collection(design, rows, ind="      "):
    lines = "\n".join(f"{ind}        {e}" for e in rows)
    return (f"{ind}- template: {{design: {design}, signal: binary, error: noiseless}}\n"
            f"{ind}  over:\n{ind}    enrichment:\n{lines}")


def _sc(name, collections, methods, batches, out_name):
    return f"""\
  {name}:
    replicates_per_batch: 10
    n_batches: {batches}
    collections:
{collections}
    methods: {_yl(methods)}
    default_args: *default_args
    outputs:
      - {{name: {out_name}, method_filter: {_yl(methods)}, analyses: [pip, cs]}}
"""


def _file(header, sc, cells, n_cells):
    return f"""\
{header}
_anchors:
  default_args: &default_args {{min_log_bf: 2.0, max_cs_size: 10000, max_fdp: 0.5}}

supercollections:
  # FULL GRID: {n_cells} cells x {len(METHODS)} arms, {BATCHES * 10} reps.
{_sc(sc, cells, METHODS, BATCHES, sc.split('-', 1)[1].replace('-', '_'))}
  # the full grid's cells WITHOUT CAVI-Q2 (content-identical; run first, CAVI is most of the compute).
{_sc(sc + "-nocavi", cells, NOCAVI, BATCHES, sc.split('-', 1)[1].replace('-', '_') + "_nocavi")}
  # PILOT: the full grid's cells at 10 reps (= the full grid's batch 0), all five arms.
{_sc(sc + "-pilot", cells, METHODS, 1, sc.split('-', 1)[1].replace('-', '_') + "_pilot")}"""


def gaussian() -> None:
    b = json.loads((HERE / "betas_gaussian_n500.json").read_text())["betas"]
    design = f"{{function: gaussian_markov_X, arguments: {{n: {GAUSS_N}, p: {P}, rho: {GAUSS_RHO}}}}}"
    rows = [_multi(f"mc{LSTAR}_T{t}_g{g}", b[str(t)], g) for t in GAUSS_TARGETS for g in GAPS]
    rows.append(_null("null"))
    header = f"""\
# 023_logistic_laplace_gaussian: the 022 protocol on a small illustrative AR1 GAUSSIAN design (rho={GAUSS_RHO}).
# Same six Q2-scored arms (CAVI-Q2 / gIBSS-Q2 / gIBSS-Laplace / global-JJ / local-JJ / score), n={GAUSS_N}, p={P},
# b0={B0:g}, L=5, L*={LSTAR} at gap in {{{", ".join(map(str, GAPS))}}} (causal-causal latent corr rho**gap),
# EB prior (cap 100).
#
# Axis: signal T = E[LRT] in {{{", ".join(map(str, GAUSS_TARGETS))}}} at fixed rho. Every column is N(0,1) so
# there is no set-size axis; beta per T from analysis/logistic_laplace_simulations/
# betas_gaussian_n500.json (MC inversion). The dense control for 022's sparse-set findings:
# large beta is where per-row curvature spread strains Laplace, the JJ bound and score. One
# null. GENERATED by generate_analogs.py -- edit that, not this."""
    n_cells = len(GAUSS_TARGETS) * len(GAPS) + 1
    (EXP / "023_logistic_laplace_gaussian.yaml").write_text(_file(header, "023-laplace-gaussian", _collection(design, rows), n_cells))
    print(f"wrote 023 ({n_cells} cells)")


def nested() -> None:
    b = json.loads((HERE / "betas_nested_n1000.json").read_text())
    sizes, betas = b["_meta"]["expected_sizes"], b["betas"]
    args = ", ".join(f"{k}: {v}" for k, v in NESTED.items())
    design = f"{{function: binary_attrition_X, arguments: {{n: {N}, p: {P}, {args}}}}}"
    rows = []
    for d in DEPTHS:
        for t in TARGETS:
            for g in GAPS:
                rows.append(_multi(f"mc{LSTAR}_d{d}_m{int(float(sizes[str(d)]))}_T{t}_g{g}", betas[str(d)][str(t)], g, base=d))
    rows.append(_null("null"))
    header = f"""\
# 024_logistic_laplace_nested: the 022 protocol on the NESTED attrition design. Same five Q2-scored
# arms (CAVI-Q2 / gIBSS-Q2 / gIBSS-Laplace / global-JJ / score), n={N}, p={P}, b0={B0:g}, L=5,
# L*={LSTAR} at gap in {{{", ".join(map(str, GAPS))}}}, T in {{{", ".join(map(str, TARGETS))}}}, EB prior (cap 100).
#
# Design: binary_attrition_X({args}): Markov roots (phi {NESTED['corr']}, ~500 members)
# expanded into 32 blocks of {NESTED['block_size']} nested sets shrinking by {1 - NESTED['drop']:g}x per step (500 -> 8).
# Axis: causal DEPTH d in {{{", ".join(map(str, DEPTHS))}}} = expected size {{{", ".join(str(int(float(sizes[str(d)]))) for d in DEPTHS)}}} (022's m), beta
# per (d, T) from analysis/logistic_laplace_simulations/betas_nested_n1000.json. Causals sit at
# depth d in blocks gap/8 apart (gap 8 = adjacent blocks, 64 = 8 apart); their decoys are their
# own ancestors/descendants at other sizes. Fixed design -> one null cell. GENERATED by
# generate_analogs.py -- edit that, not this."""
    n_cells = len(DEPTHS) * len(TARGETS) * len(GAPS) + 1
    (EXP / "024_logistic_laplace_nested.yaml").write_text(_file(header, "024-laplace-nested", _collection(design, rows), n_cells))
    print(f"wrote 024 ({n_cells} cells)")


if __name__ == "__main__":
    gaussian()
    nested()
