"""Generate experiments/022_logistic_laplace.yaml.

022 asks when gIBSS-Laplace (the simplest logistic SuSiE: Laplace SER + plug-in offset) is
good enough, against the other cheap logistic-SuSiE approximations. All arms are in Q2 and
scored on the same exact Q2 ELBO:

  gIBSS-Laplace  order-1 quad SER (Laplace evidence), plug-in mean offset, reduced to Q2
  gIBSS-Q2       Gaussian-VI SER (GH over b), plug-in mean offset
  global-JJ      Jaakkola-Jordan quadratic bound, one shared tilt (jj_fixed)
  score          linear approximation: one Newton step from the intercept-only null
  CAVI-Q2        Gaussian-VI SER, exact offset fold (cf)

Axis: expected set size m of a sparse 0/1 design (corr=0.8 between adjacent sets, b0=-2), with
beta calibrated per m to E[LRT] = T (calibrate.py). Detectability is matched along the axis;
small m forces large beta and a skewed per-set likelihood (Laplace's worst case, score's
overshoot), large m is near-Gaussian.

Supercollections (cells are content-addressed; labels are display-only):
  * 022-laplace              FULL GRID. n=1000, L=5, L*=3 causals at gap in {8, 64} (binary
                             phi between causals ~0.47 / ~0.05), T in {8,16}, m in {5,10,50,100,200,400},
                             + one null per m. CAVI-Q2 / gIBSS-Q2 / gIBSS-Laplace / global-JJ / score,
                             200 reps.
  * 022-laplace-ser          FULL GRID, single effect. n=1000, L=1 (SER arms), one causal,
                             T in {8,16}, m in {5,10,50,100,200,400}, + one null per m (the null
                             cells are content-identical to the full grid's, so their sims are
                             shared). All five arms, 200 reps. At L=1 gIBSS-Q2 == CAVI-Q2 by
                             construction (no other effects to fold): an identity check.
  * 022-laplace-pilot        n=10000 pilot: T in {8,16} single-effect + null + L*=3 gap-10 T=8,
                             m in {5,10,30,100}; all five arms at L=1 and L=10; 10 reps.
  * 022-laplace-pilot-indep  pilot gap-10 multi cells at corr=0 (L=10, no CAVI), 20 reps.
  * 022-laplace-pilot-gapmax pilot multi cells at gap 85, T in {8,16} (L=10, no CAVI), 20 reps.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "experiments" / "022_logistic_laplace.yaml"

P, CORR, B0 = 256, 0.8, -2.0

# ---- full grid (n=1000) -------------------------------------------------------------------
FULL_N = 1000
FULL_BATCHES = 20                     # 200 reps
FULL_SET_SIZES = [5, 10, 50, 100, 200, 400]
FULL_TARGETS = [8, 16]
FULL_LSTAR = 3
FULL_GAPS = [8, 64]                   # binary phi between causals ~0.47 / ~0.05
FULL_METHODS = [
    "logistic_q2_L5_cavi", "logistic_q2_L5_gibss", "logistic_q2_L5_gibss_laplace",
    "logistic_q2_L5_globaljj", "logistic_q2_L5_score",
]
NOCAVI_METHODS = [m for m in FULL_METHODS if not m.endswith("_cavi")]
SER_METHODS = [
    "logistic_q2_ser_cavi", "logistic_q2_ser_gibss", "logistic_q2_ser_gibss_laplace",
    "logistic_q2_ser_globaljj", "logistic_q2_ser_score",
]

# ---- pilot (n=10000) ----------------------------------------------------------------------
PILOT_N = 10000
PILOT_SET_SIZES = [5, 10, 30, 100]
PILOT_TARGETS = [8, 16]
PILOT_MULTI = (3, 10, 8)              # (L*, gap, T)
PILOT_METHODS = [
    "logistic_q2_ser_gibss_laplace", "logistic_q2_ser_gibss", "logistic_q2_ser_globaljj",
    "logistic_q2_ser_cavi", "logistic_q2_ser_score",
    "logistic_q2_L10_gibss_laplace", "logistic_q2_L10_gibss", "logistic_q2_L10_globaljj",
    "logistic_q2_L10_cavi", "logistic_q2_L10_score",
]
CONTROL_METHODS = [m for m in PILOT_METHODS if "_L10_" in m and not m.endswith("_cavi")]
CONTROL_BATCHES = 2                   # 20 reps
INDEP_CORR = 0.0
GAPMAX = (P - 1) // PILOT_MULTI[0]    # 85: causals at 85, 170, 255
GAPMAX_TARGETS = [8, 16]


def _design(n: int, m: int, corr: float = CORR) -> str:
    return (f"{{function: binary_markov_X, arguments: "
            f"{{n: {n}, p: {P}, corr: {corr}, density: {m / n:g}}}}}")


def _single(m: int, t: int, beta: float) -> str:
    return (f"- {{label: ser_m{m}_T{t}, function: uniform_single_effect, "
            f"arguments: {{causal_effect: {beta:.4f}}}, intercept: {B0:.1f}}}")


def _null(m: int) -> str:
    return (f"- {{label: null_m{m}, function: uniform_single_effect, "
            f"arguments: {{causal_effect: 0.0}}, intercept: {B0:.1f}}}")


def _multi(m: int, lstar: int, gap: int, beta: float, t: int) -> str:
    effects = ", ".join(f"{beta:.4f}" for _ in range(lstar))
    return (f"- {{label: mc{lstar}_m{m}_T{t}_g{gap}, function: spaced_index_effect, "
            f"arguments: {{causal_effects: [{effects}], gap: {gap}}}, intercept: {B0:.1f}}}")


def _collections(n: int, set_sizes: list[int], rows_for, indent: str, corr: float = CORR) -> str:
    """One collection per set size (the design density differs per m)."""
    blocks = []
    for m in set_sizes:
        lines = "\n".join(f"{indent}        {e}" for e in rows_for(m))
        blocks.append(
            f"{indent}- template: {{design: {_design(n, m, corr)}, signal: binary, error: noiseless}}\n"
            f"{indent}  over:\n"
            f"{indent}    enrichment:\n{lines}"
        )
    return "\n".join(blocks)


def _yaml_list(xs: list[str]) -> str:
    return "[" + ", ".join(xs) + "]"


def main() -> None:
    full_b = json.loads((HERE / "betas_n1000.json").read_text())["betas"]
    pilot_b = json.loads((HERE / "betas.json").read_text())["betas"]
    lstar, gap_p, t_p = PILOT_MULTI

    def full_rows(m):
        tbl = full_b[str(m)]
        return ([_multi(m, FULL_LSTAR, g, tbl[str(t)], t) for t in FULL_TARGETS for g in FULL_GAPS]
                + [_null(m)])

    def ser_rows(m):
        tbl = full_b[str(m)]
        return [_single(m, t, tbl[str(t)]) for t in FULL_TARGETS] + [_null(m)]

    def pilot_rows(m):
        tbl = pilot_b[str(m)]
        return ([_single(m, t, tbl[str(t)]) for t in PILOT_TARGETS] + [_null(m)]
                + [_multi(m, lstar, gap_p, tbl[str(t_p)], t_p)])

    def indep_rows(m):
        return [_multi(m, lstar, gap_p, pilot_b[str(m)][str(t_p)], t_p)]

    def gapmax_rows(m):
        return [_multi(m, lstar, GAPMAX, pilot_b[str(m)][str(t)], t) for t in GAPMAX_TARGETS]

    n_full = len(FULL_SET_SIZES) * (len(FULL_TARGETS) * len(FULL_GAPS) + 1)
    n_ser = len(FULL_SET_SIZES) * (len(FULL_TARGETS) + 1)
    ind = "      "
    text = f"""\
# 022_logistic_laplace: when is gIBSS-Laplace good enough? Logistic-SuSiE approximations in Q2,
# all scored on the exact Q2 ELBO: gIBSS-Laplace (order-1 SER, plug-in offset, reduced to Q2),
# gIBSS-Q2 (Gaussian-VI SER, plug-in offset), global-JJ (jj_fixed bound), score (one Newton step
# at the null); CAVI-Q2 (exact cf offset fold). Centered, shared Gaussian
# intercept, EB prior variance (cap 100).
#
# Axis: expected set size m of a sparse 0/1 design (p={P}, corr={CORR} between adjacent sets,
# b0={B0:g}), beta calibrated per m to E[LRT] = T (analysis/logistic_laplace_simulations/
# calibrate.py -> betas_n1000.json for the full grid, betas.json for the n=10000 pilot). This file
# is GENERATED by generate_experiment.py -- edit that, not this.
_anchors:
  default_args: &default_args {{min_log_bf: 2.0, max_cs_size: 10000, max_fdp: 0.5}}

supercollections:
  # FULL GRID: n={FULL_N}, L=5, L*={FULL_LSTAR} at gap in {{{", ".join(map(str, FULL_GAPS))}}}, T in {{{", ".join(map(str, FULL_TARGETS))}}},
  # m in {{{", ".join(map(str, FULL_SET_SIZES))}}} + one null per m ({n_full} cells), {FULL_BATCHES * 10} reps.
  022-laplace:
    replicates_per_batch: 10
    n_batches: {FULL_BATCHES}
    collections:
{_collections(FULL_N, FULL_SET_SIZES, full_rows, ind)}
    methods: {_yaml_list(FULL_METHODS)}
    default_args: *default_args
    outputs:
      - {{name: laplace, method_filter: {_yaml_list(FULL_METHODS)}, analyses: [pip, cs]}}

  # the full grid's cells WITHOUT CAVI-Q2 (content-identical, so its fits are the full grid's):
  # run first, since CAVI-Q2 is ~2/3 of the compute.
  022-laplace-nocavi:
    replicates_per_batch: 10
    n_batches: {FULL_BATCHES}
    collections:
{_collections(FULL_N, FULL_SET_SIZES, full_rows, ind)}
    methods: {_yaml_list(NOCAVI_METHODS)}
    default_args: *default_args
    outputs:
      - {{name: laplace_nocavi, method_filter: {_yaml_list(NOCAVI_METHODS)}, analyses: [pip, cs]}}

  # FULL GRID, SINGLE EFFECT: n={FULL_N}, L=1 (SER arms), one causal, T in {{{", ".join(map(str, FULL_TARGETS))}}},
  # m in {{{", ".join(map(str, FULL_SET_SIZES))}}} + one null per m ({n_ser} cells), {FULL_BATCHES * 10} reps.
  022-laplace-ser:
    replicates_per_batch: 10
    n_batches: {FULL_BATCHES}
    collections:
{_collections(FULL_N, FULL_SET_SIZES, ser_rows, ind)}
    methods: {_yaml_list(SER_METHODS)}
    default_args: *default_args
    outputs:
      - {{name: laplace_ser, method_filter: {_yaml_list(SER_METHODS)}, analyses: [pip, cs]}}

  # PILOT (n={PILOT_N}): T in {{{", ".join(map(str, PILOT_TARGETS))}}} single-effect + null + mc{lstar} gap {gap_p} T={t_p},
  # m in {{{", ".join(map(str, PILOT_SET_SIZES))}}}; all five arms at L=1 and L=10; 10 reps.
  022-laplace-pilot:
    replicates_per_batch: 10
    n_batches: 1
    collections:
{_collections(PILOT_N, PILOT_SET_SIZES, pilot_rows, ind)}
    methods: {_yaml_list(PILOT_METHODS)}
    default_args: *default_args
    outputs:
      - {{name: pilot, method_filter: {_yaml_list(PILOT_METHODS)}, analyses: [pip, cs]}}

  # pilot control: the mc{lstar} gap-{gap_p} cells at corr={INDEP_CORR}, L=10 arms (no CAVI), {CONTROL_BATCHES * 10} reps.
  022-laplace-pilot-indep:
    replicates_per_batch: 10
    n_batches: {CONTROL_BATCHES}
    collections:
{_collections(PILOT_N, PILOT_SET_SIZES, indep_rows, ind, corr=INDEP_CORR)}
    methods: {_yaml_list(CONTROL_METHODS)}
    default_args: *default_args
    outputs:
      - {{name: indep, method_filter: {_yaml_list(CONTROL_METHODS)}, analyses: [pip, cs]}}

  # pilot control: mc{lstar} at gap {GAPMAX} (causals ~uncorrelated), T in {{{", ".join(map(str, GAPMAX_TARGETS))}}},
  # L=10 arms (no CAVI), {CONTROL_BATCHES * 10} reps.
  022-laplace-pilot-gapmax:
    replicates_per_batch: 10
    n_batches: {CONTROL_BATCHES}
    collections:
{_collections(PILOT_N, PILOT_SET_SIZES, gapmax_rows, ind)}
    methods: {_yaml_list(CONTROL_METHODS)}
    default_args: *default_args
    outputs:
      - {{name: gapmax, method_filter: {_yaml_list(CONTROL_METHODS)}, analyses: [pip, cs]}}
"""
    OUT.write_text(text)
    print(f"wrote {OUT} (full grid {n_full} cells x {len(FULL_METHODS)} arms; ser {n_ser} cells x {len(SER_METHODS)} arms)")


if __name__ == "__main__":
    main()
