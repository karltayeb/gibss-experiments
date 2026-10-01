"""Is an arm's CS coverage worse than CAVI-Q2's? A direct, paired check.

On each replicate every arm sees the same data. Match each declared CS of the arm to the
CAVI-Q2 CS on that replicate with the largest Jaccard overlap (greedy one-to-one, overlap
required). Then:

  * matched pairs: a 2x2 of (arm covers, CAVI covers). The discordant counts
    `arm miss / CAVI cover` vs `arm cover / CAVI miss` feed an exact McNemar test
    (two-sided binomial). Excess of the first = the arm's matched CSs are too narrow.
  * unmatched: CSs only the arm declares (and only CAVI declares), with their false rate.
    Excess false arm-only CSs = over-declaration.
  * paired coverage difference: per-rep (covered, declared) counts, rep-cluster bootstrap
    CI on arm coverage minus CAVI coverage.

    uv run python analysis/logistic_laplace_simulations/cs_vs_cavi.py [SC] [REF]
"""
from __future__ import annotations

import sys

import numpy as np
import polars as pl
from scipy.stats import binomtest

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import cs_paired as P  # noqa: E402
import cs_tables as C  # noqa: E402
import laplacelib as R  # noqa: E402


def declared_sets(df: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for r in df.iter_rows(named=True):
        keep = [i for i, b in enumerate(r["lbf"]) if b >= R.MIN_LOG_BF]
        rows.append({**{k: r[k] for k in P.KEY + ["method", "null"]},
                     "sets": [sorted(r["css"][i]) for i in keep],
                     "covers": [bool(r["covers"][i]) for i in keep]})
    return pl.DataFrame(rows, infer_schema_length=None)


def match(a_sets, a_cov, b_sets, b_cov):
    """Greedy one-to-one Jaccard matching. Returns matched (a_cov, b_cov) pairs and leftovers."""
    pairs = []
    a_sets = [set(s) for s in a_sets]
    b_sets = [set(t) for t in b_sets]
    for i, s in enumerate(a_sets):
        for j, t in enumerate(b_sets):
            inter = len(s & t)
            if inter:
                pairs.append((inter / len(s | t), i, j))
    pairs.sort(reverse=True)
    used_a, used_b, out = set(), set(), []
    for _, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i); used_b.add(j)
        out.append((a_cov[i], b_cov[j]))
    a_only = [a_cov[i] for i in range(len(a_sets)) if i not in used_a]
    b_only = [b_cov[j] for j in range(len(b_sets)) if j not in used_b]
    return out, a_only, b_only


def main(sc="022-laplace", ref="cavi"):
    d = declared_sets(C.load(sc)).filter(~pl.col("null"))
    refd = (d.filter(pl.col("method") == ref)
              .select(*P.KEY, pl.col("sets").alias("r_sets"), pl.col("covers").alias("r_cov")))
    j = d.filter(pl.col("method") != ref).join(refd, on=P.KEY, how="inner")
    arms = [m for m in R.METHODS if m != ref and m in j["method"].unique().to_list()]
    rng = np.random.default_rng(0)
    print(f"**{sc}**, reference = {R.METHOD_LABEL[ref]}; only reps where the reference is fit. "
          f"Pooled over m within T x gap.\n")
    print("Columns: *matched* = one-to-one Jaccard-matched CS pairs; *both* = both cover; "
          "*arm miss* = arm's CS misses the causal while the reference's matched CS covers it; "
          "*ref miss* = the reverse; *p* = exact McNemar (two-sided binomial on the two discordant "
          "counts); *arm-only false* = CSs only the arm declared, shown as false/total; *ref-only "
          "false* likewise; *dcov* = arm coverage minus reference coverage with a rep-cluster "
          "bootstrap 95% CI.\n")
    hdr = ["T", "gap", "arm", "matched", "both", "arm miss", "ref miss", "p", "arm-only false",
           "ref-only false", "dcov [95% CI]"]
    print("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
    for T, gap in j.select("T", "gap").unique().sort(["T", "gap"]).iter_rows():
        for arm in arms:
            g = j.filter(pl.col("T") == T, pl.col("gap") == gap, pl.col("method") == arm)
            both = am = rm = neither = 0
            ao, ro = [], []
            rep_x, rep_n, rep_x0, rep_n0 = [], [], [], []
            for r in g.iter_rows(named=True):
                pairs, a_only, r_only = match(r["sets"], r["covers"], r["r_sets"], r["r_cov"])
                for a, b in pairs:
                    if a and b: both += 1
                    elif b: am += 1
                    elif a: rm += 1
                    else: neither += 1
                ao += a_only; ro += r_only
                rep_x.append(sum(r["covers"])); rep_n.append(len(r["covers"]))
                rep_x0.append(sum(r["r_cov"])); rep_n0.append(len(r["r_cov"]))
            p = binomtest(am, am + rm).pvalue if am + rm else 1.0
            x, n, x0, n0 = map(np.array, (rep_x, rep_n, rep_x0, rep_n0))
            diff = x.sum() / n.sum() - x0.sum() / n0.sum()
            boots = []
            for _ in range(2000):
                i = rng.integers(0, len(x), len(x))
                boots.append(x[i].sum() / max(n[i].sum(), 1) - x0[i].sum() / max(n0[i].sum(), 1))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            print(f"| {T} | {gap} | {R.METHOD_LABEL[arm]} | {both+am+rm+neither} | {both} | {am} | {rm} | "
                  f"{p:.3g} | {sum(1 for c in ao if not c)}/{len(ao)} | {sum(1 for c in ro if not c)}/{len(ro)} | "
                  f"{diff:+.3f} [{lo:+.3f}, {hi:+.3f}] |")


if __name__ == "__main__":
    main(*sys.argv[1:])
