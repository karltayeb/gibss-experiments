"""Is an arm's CS coverage worse than the reference's (CAVI-Q2)? A direct, paired check.

Every arm sees the same data, and every arm fits L components. On each replicate, assign the
arm's L components to the reference's L components one-to-one (Hungarian) maximising the
Bhattacharyya coefficient BC = sum_j sqrt(alpha_a[j] alpha_r[j]) between their inclusion
vectors, which matches components that are "about the same signal" whether or not their 95%
sets agree. No leftovers: declaration (component log BF >= 2) is then part of the
classification, not a reason to drop a set.

Per matched pair we record: declared in arm / ref, covers a causal in arm / ref, BC, and the
arm's alpha on the causal the reference CS contains (the "excluded but still has evidence"
diagnostic for arm-miss pairs).

Tables, pooled over m within T x gap:
  1. both declared: both / arm miss / ref miss / neither, exact McNemar on the discordant
     counts; paired coverage difference with a rep-cluster bootstrap CI.
  2. declaration disagreement: arm-only declared (false/total), ref-only declared (covering/total).
  3. arm-miss diagnostic: the arm's alpha on the causal that the reference CS covers.

    uv run python analysis/logistic_laplace_simulations/cs_vs_cavi.py [SC] [REF]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment
from scipy.stats import binomtest

_HERE = os.path.dirname(os.path.abspath(__file__))
_TG_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _TG_ROOT)
sys.path.insert(0, _HERE)
import experiments.loader as loader  # noqa: E402
import laplacelib as R  # noqa: E402


def _reps(path: str) -> dict[int, dict]:
    """replicate -> {alpha (L x p), lbf (L), cov (L bool), causal (list), cs (list of lists)}."""
    out = {}
    df = pl.read_parquet(path, columns=["replicate", "credible_sets", "single_effects"])
    for r in df.iter_rows(named=True):
        se, cs = r["single_effects"], r["credible_sets"]
        out[int(r["replicate"])] = {
            "alpha": np.array([e["alpha"] for e in se]),
            "lbf": np.array([e["ser_log_bf"] for e in se]),
            "cov": np.array([bool(c["causal_in_cs"]) for c in cs]),
            "cs": [list(c["cs"]) for c in cs],
            "causal": list(cs[0]["causal_indices"]),
        }
    return out


def pairs_frame(sc: str, ref: str) -> pl.DataFrame:
    cfg = loader.load_config()
    cells: dict[str, dict] = {}
    for coll in loader.collection_method_pairs(cfg, sc).values():
        for bh, mh, mname, mcoord, scoord in coll["pairs"]:
            c = cells.setdefault(bh, {"meta": R.cell_meta(scoord), "paths": {}})
            c["paths"][R._method_key(mname)] = f"{_TG_ROOT}/results/by_batch/{bh}/fits/{mh}/fits.parquet"
    rows = []
    for bh, c in cells.items():
        if c["meta"]["null"] or not os.path.exists(c["paths"].get(ref, "")):
            continue
        refr = _reps(c["paths"][ref])
        for arm, path in c["paths"].items():
            if arm == ref or not os.path.exists(path):
                continue
            for rep, a in _reps(path).items():
                if rep not in refr:
                    continue
                b = refr[rep]
                causal = b["causal"]
                bc = np.sqrt(a["alpha"]) @ np.sqrt(b["alpha"]).T          # L x L affinity
                ia, ib = linear_sum_assignment(-bc)
                for la, lb in zip(ia, ib):
                    # the causal the reference component's CS contains (if any)
                    ref_hit = [j for j in causal if j in b["cs"][lb]]
                    j_ref = ref_hit[0] if ref_hit else None
                    rows.append({
                        "T": c["meta"]["T"], "gap": c["meta"]["gap"], "m": c["meta"]["m"],
                        "batch_hash": bh, "rep": rep, "arm": arm, "bc": float(bc[la, lb]),
                        "a_decl": bool(a["lbf"][la] >= R.MIN_LOG_BF),
                        "r_decl": bool(b["lbf"][lb] >= R.MIN_LOG_BF),
                        "a_cov": bool(a["cov"][la]), "r_cov": bool(b["cov"][lb]),
                        "a_lbf": float(a["lbf"][la]), "r_lbf": float(b["lbf"][lb]),
                        "a_size": len(a["cs"][la]), "r_size": len(b["cs"][lb]),
                        # arm's alpha on the causal the ref CS covers; ref's own for scale
                        "a_alpha_j": float(a["alpha"][la, j_ref]) if j_ref is not None else None,
                        "r_alpha_j": float(b["alpha"][lb, j_ref]) if j_ref is not None else None,
                        "a_rank_j": int((a["alpha"][la] > a["alpha"][la, j_ref]).sum()) + 1
                                    if j_ref is not None else None,
                    })
    return pl.DataFrame(rows, infer_schema_length=None)


def _hdr(cols):
    return "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols)


def main(sc: str = "022-laplace", ref: str = "cavi") -> None:
    P = pairs_frame(sc, ref)
    arms = [m for m in R.METHODS if m in P["arm"].unique().to_list()]
    panels = P.select("T", "gap").unique().sort(["T", "gap"]).rows()
    rng = np.random.default_rng(0)
    RL = R.METHOD_LABEL[ref]
    n_reps = P.select("batch_hash", "rep").unique().height
    print(f"**{sc}**, reference = {RL}, {n_reps} signal reps where the reference is fit. "
          f"Components matched one-to-one per rep by Bhattacharyya affinity of alpha; pooled over m "
          f"within T x gap.\n")

    print("**1. Both declared.** *both* = both CSs contain a causal; *arm miss* = arm's misses while "
          f"{RL}'s covers; *ref miss* = reverse; *neither*; *p* = exact McNemar on the two discordant "
          "counts; *BC* = median affinity of these pairs; *dcov* = arm minus reference coverage over "
          "declared CSs, rep-cluster bootstrap 95% CI.\n")
    print(_hdr(["T", "gap", "arm", "both", "arm miss", "ref miss", "neither", "p", "BC", "dcov [95% CI]"]))
    for T, gap in panels:
        for arm in arms:
            g = P.filter(pl.col("T") == T, pl.col("gap") == gap, pl.col("arm") == arm)
            bd = g.filter(pl.col("a_decl"), pl.col("r_decl"))
            both = bd.filter(pl.col("a_cov"), pl.col("r_cov")).height
            am = bd.filter(~pl.col("a_cov"), pl.col("r_cov")).height
            rm = bd.filter(pl.col("a_cov"), ~pl.col("r_cov")).height
            nei = bd.filter(~pl.col("a_cov"), ~pl.col("r_cov")).height
            p = binomtest(am, am + rm).pvalue if am + rm else 1.0
            per = (g.group_by("batch_hash", "rep")
                     .agg((pl.col("a_decl") & pl.col("a_cov")).sum().alias("x"), pl.col("a_decl").sum().alias("n"),
                          (pl.col("r_decl") & pl.col("r_cov")).sum().alias("x0"), pl.col("r_decl").sum().alias("n0")))
            x, n, x0, n0 = (per[k].to_numpy() for k in ("x", "n", "x0", "n0"))
            diff = x.sum() / n.sum() - x0.sum() / n0.sum()
            boots = []
            for _ in range(2000):
                i = rng.integers(0, len(x), len(x))
                boots.append(x[i].sum() / max(n[i].sum(), 1) - x0[i].sum() / max(n0[i].sum(), 1))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            print(f"| {T} | {gap} | {R.METHOD_LABEL[arm]} | {both} | {am} | {rm} | {nei} | {p:.2g} | "
                  f"{bd['bc'].median():.2f} | {diff:+.3f} [{lo:+.3f}, {hi:+.3f}] |")

    print(f"\n**2. Declaration disagreement.** *arm only* = arm declares, {RL} does not (false/total: "
          f"these are the arm's extra CSs); *ref only* = {RL} declares, arm does not (covering/total: "
          "signals the arm missed); *BC* = median affinity of the ref-only pairs.\n")
    print(_hdr(["T", "gap", "arm", "arm only false/total", "ref only covering/total", "BC (ref only)"]))
    for T, gap in panels:
        for arm in arms:
            g = P.filter(pl.col("T") == T, pl.col("gap") == gap, pl.col("arm") == arm)
            ao = g.filter(pl.col("a_decl"), ~pl.col("r_decl"))
            ro = g.filter(~pl.col("a_decl"), pl.col("r_decl"))
            print(f"| {T} | {gap} | {R.METHOD_LABEL[arm]} | {ao.filter(~pl.col('a_cov')).height}/{ao.height} | "
                  f"{ro.filter(pl.col('r_cov')).height}/{ro.height} | "
                  f"{'-' if ro.height == 0 else f'{ro['bc'].median():.2f}'} |")

    print(f"\n**3. Arm-miss diagnostic** (both declared, {RL} covers, arm misses): the arm's alpha on "
          f"the causal that {RL}'s CS contains. *median alpha*; *>0.05* = fraction with alpha above "
          "0.05 (excluded from the 95% set but still carrying evidence); *rank* = median rank of that "
          "causal within the arm's component; *size a/r* = median CS sizes; *lbf a/r* = median "
          "component log BF.\n")
    print(_hdr(["T", "gap", "arm", "n", "median alpha", ">0.05", ">0.01", "rank", "size a/r", "lbf a/r"]))
    for T, gap in panels:
        for arm in arms:
            g = P.filter(pl.col("T") == T, pl.col("gap") == gap, pl.col("arm") == arm,
                         pl.col("a_decl"), pl.col("r_decl"), ~pl.col("a_cov"), pl.col("r_cov"))
            if g.height == 0:
                print(f"| {T} | {gap} | {R.METHOD_LABEL[arm]} | 0 | - | - | - | - | - | - |")
                continue
            a = g["a_alpha_j"].to_numpy()
            print(f"| {T} | {gap} | {R.METHOD_LABEL[arm]} | {g.height} | {np.median(a):.3f} | "
                  f"{(a > 0.05).mean():.2f} | {(a > 0.01).mean():.2f} | {g['a_rank_j'].median():.0f} | "
                  f"{g['a_size'].median():.0f}/{g['r_size'].median():.0f} | "
                  f"{g['a_lbf'].median():.1f}/{g['r_lbf'].median():.1f} |")


if __name__ == "__main__":
    main(*sys.argv[1:])
