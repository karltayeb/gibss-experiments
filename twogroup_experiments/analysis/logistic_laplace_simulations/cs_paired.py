"""Paired comparison of the 022 arms against a reference arm on the SAME simulated data.

Every arm is fit to the same (batch, rep), so the honest uncertainty for an arm-vs-arm
difference is the spread of the PER-REP difference, not two pooled proportions. Reports, per
cell and arm: mean paired difference +- SE over reps for power (fraction of causals captured)
and false CSs per fit (declared minus covering); a rep-cluster bootstrap CI for coverage; and
the fraction of reps whose declared CS collection is IDENTICAL to the reference's.

    uv run python analysis/logistic_laplace_simulations/cs_paired.py [SC] [REF]
"""
from __future__ import annotations

import sys

import numpy as np
import polars as pl

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import cs_tables as C  # noqa: E402
import laplacelib as R  # noqa: E402

KEY = ["T", "gap", "m", "batch_hash", "replicate"]


def per_rep(df: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for r in df.iter_rows(named=True):
        keep = [i for i, b in enumerate(r["lbf"]) if b >= R.MIN_LOG_BF]
        css = frozenset(frozenset(r["css"][i]) for i in keep)
        hit = set().union(*(set(r["css"][i]) for i in keep)) & set(r["causal"]) if keep else set()
        rows.append({**{k: r[k] for k in KEY + ["method", "null", "Lstar"]},
                     "power": len(hit) / max(r["Lstar"], 1), "n_declared": len(keep),
                     "n_false": sum(1 for i in keep if not r["covers"][i]),
                     "n_cover": sum(1 for i in keep if r["covers"][i]), "css": css})
    return pl.DataFrame(rows, infer_schema_length=None)


def fmt_pm(m, se):
    return f"{m:+.3f} ± {se:.3f}"


def main(sc="022-laplace", ref="gibss"):
    d = per_rep(C.load(sc)).filter(~pl.col("null"))
    others = [m for m in R.METHODS if m != ref and m in d["method"].unique().to_list()]
    refd = d.filter(pl.col("method") == ref).select(*KEY, pl.col("power").alias("p0"),
                                                     pl.col("n_false").alias("f0"), pl.col("css").alias("c0"))
    out = []
    for cell, g in d.filter(pl.col("method") != ref).join(refd, on=KEY, how="inner") \
                     .group_by(["T", "gap", "m", "method"], maintain_order=True):
        dp = (g["power"] - g["p0"]).to_numpy()
        dfl = (g["n_false"] - g["f0"]).to_numpy()
        same = np.mean([a == b for a, b in zip(g["css"], g["c0"])])
        out.append({"T": cell[0], "gap": cell[1], "m": cell[2], "method": cell[3], "n": len(dp),
                    "d_power": dp.mean(), "se_power": dp.std(ddof=1) / np.sqrt(len(dp)),
                    "d_false": dfl.mean(), "se_false": dfl.std(ddof=1) / np.sqrt(len(dfl)),
                    "same_cs": same})
    t = pl.DataFrame(out)
    print(f"**{sc}**, reference = {R.METHOD_LABEL[ref]}. Per cell: paired per-rep differences, "
          f"mean ± SE over reps (n reps where both arms are fit).\n")
    for metric, label in [("d_power", "power (arm minus ref)"), ("d_false", "false CSs per fit (arm minus ref)")]:
        se = metric.replace("d_", "se_")
        print(f"\n**{label}**\n")
        present = [m for m in R.METHODS if m in others]
        hdr = ["T", "gap", "m"] + [R.METHOD_LABEL[m] for m in present]
        print("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
        for T, gap, m in t.select("T", "gap", "m").unique().sort(["T", "gap", "m"]).iter_rows():
            cells = []
            for mth in present:
                x = t.filter(pl.col("T") == T, pl.col("gap") == gap, pl.col("m") == m, pl.col("method") == mth)
                cells.append("-" if x.height == 0 else fmt_pm(x[metric][0], x[se][0]))
            print(f"| {T} | {gap} | {m} | " + " | ".join(cells) + " |")
    print(f"\n**identical declared-CS collection** (fraction of reps where the arm's declared CSs "
          f"equal {R.METHOD_LABEL[ref]}'s exactly)\n")
    present = [m for m in R.METHODS if m in others]
    hdr = ["T", "gap", "m"] + [R.METHOD_LABEL[m] for m in present]
    print("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
    for T, gap, m in t.select("T", "gap", "m").unique().sort(["T", "gap", "m"]).iter_rows():
        cells = []
        for mth in present:
            x = t.filter(pl.col("T") == T, pl.col("gap") == gap, pl.col("m") == m, pl.col("method") == mth)
            cells.append("-" if x.height == 0 else f"{x['same_cs'][0]:.2f}")
        print(f"| {T} | {gap} | {m} | " + " | ".join(cells) + " |")

    # coverage: rep-cluster bootstrap CI (resample reps, recompute x/n), per arm, pooled over m
    print("\n**coverage, rep-cluster bootstrap 95% CI** (pooled over m within T x gap; resampling "
          "reps, not CSs, since CSs within a fit are not independent)\n")
    rng = np.random.default_rng(0)
    hdr = ["T", "gap"] + [R.METHOD_LABEL[m] for m in R.METHODS if m in d["method"].unique().to_list()]
    print("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
    for T, gap in d.select("T", "gap").unique().sort(["T", "gap"]).iter_rows():
        cells = []
        for mth in R.METHODS:
            g = d.filter(pl.col("T") == T, pl.col("gap") == gap, pl.col("method") == mth)
            if g.height == 0:
                cells.append("-"); continue
            x, n = g["n_cover"].to_numpy(), g["n_declared"].to_numpy()
            boots = []
            for _ in range(2000):
                i = rng.integers(0, len(x), len(x))
                boots.append(x[i].sum() / max(n[i].sum(), 1))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            cells.append(f"{x.sum()/n.sum():.3f} [{lo:.3f}, {hi:.3f}]")
        print(f"| {T} | {gap} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(*sys.argv[1:])
