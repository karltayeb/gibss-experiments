"""Declared-CS tables for the 022 full grid: 95% CS coverage (x/n), CS size, power.

Lean loader: reads only `credible_sets` + the per-component `ser_log_bf` from each
fits.parquet (no per-feature posteriors), so it stays small on the full grid.

    uv run python analysis/logistic_laplace_simulations/cs_tables.py [SC]

Columns: *coverage* = declared CSs (component log BF >= 2) containing a causal, as x/n;
*size* = median declared-CS size (features); *power* = mean over reps of the fraction of
the L*=3 causals captured by some declared CS; *CS/fit* = declared CSs per fit;
*reps* = fitted replicates (CAVI may be partial while its arm is still running).
"""
from __future__ import annotations

import os
import sys

import polars as pl

_HERE = os.path.dirname(os.path.abspath(__file__))
_TG_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _TG_ROOT)
sys.path.insert(0, _HERE)
import experiments.loader as loader  # noqa: E402
import laplacelib as R  # noqa: E402


def load(sc: str) -> pl.DataFrame:
    cfg = loader.load_config()
    frames, seen = [], set()
    for coll in loader.collection_method_pairs(cfg, sc).values():
        for bh, mh, mname, mcoord, scoord in coll["pairs"]:
            if (bh, mh) in seen:
                continue
            seen.add((bh, mh))
            f = f"{_TG_ROOT}/results/by_batch/{bh}/fits/{mh}/fits.parquet"
            if not os.path.exists(f):
                continue
            meta = R.cell_meta(scoord)
            df = (pl.read_parquet(f, columns=["replicate", "credible_sets", "single_effects"])
                    .with_columns(
                        pl.col("single_effects").list.eval(pl.element().struct.field("ser_log_bf"))
                          .alias("lbf"),
                        pl.col("credible_sets").list.eval(pl.element().struct.field("cs_size"))
                          .alias("sizes"),
                        pl.col("credible_sets").list.eval(pl.element().struct.field("causal_in_cs"))
                          .alias("covers"),
                        pl.col("credible_sets").list.eval(pl.element().struct.field("cs")).alias("css"),
                        pl.col("credible_sets").list.first().struct.field("causal_indices").alias("causal"),
                    ).drop("credible_sets", "single_effects"))
            frames.append(df.with_columns(
                pl.lit(R._method_key(mname)).alias("method"), pl.lit(bh).alias("batch_hash"),
                **{k: pl.lit(v) for k, v in meta.items()}))
    return pl.concat(frames, how="diagonal_relaxed")


def declared(df: pl.DataFrame) -> pl.DataFrame:
    """One row per fit with declared-CS lists and the number of causals detected."""
    rows = []
    for r in df.iter_rows(named=True):
        keep = [i for i, b in enumerate(r["lbf"]) if b >= R.MIN_LOG_BF]
        hit = set()
        for i in keep:
            hit |= set(r["css"][i]) & set(r["causal"])
        rows.append({"sizes": [r["sizes"][i] for i in keep],
                     "covers": [r["covers"][i] for i in keep],
                     "n_detected": len(hit), "n_declared": len(keep)})
    return pl.concat([df.drop("lbf", "sizes", "covers", "css"), pl.DataFrame(rows)], how="horizontal")


def table(d: pl.DataFrame, rows: list[str], value: str, fmt) -> str:
    present = [m for m in R.METHODS if m in d["method"].unique().to_list()]
    header = rows + [R.METHOD_LABEL[m] for m in present]
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for combo in d.select(rows).unique().sort(rows).iter_rows():
        g = d
        for c, v in zip(rows, combo):
            g = g.filter(pl.col(c) == v)
        cells = []
        for m in present:
            x = g.filter(pl.col("method") == m)
            cells.append("-" if x.height == 0 else fmt(x.row(0, named=True)))
        out.append("| " + " | ".join([str(v) for v in combo] + cells) + " |")
    return "\n".join(out)


def main(sc: str = "022-laplace") -> None:
    d = declared(load(sc))
    key = ["T", "gap", "m", "method"]
    sig = d.filter(~pl.col("null"))
    cs = (sig.explode(["sizes", "covers"])
             .group_by(key)
             .agg(pl.col("covers").drop_nulls().sum().alias("x"),
                  pl.col("covers").drop_nulls().len().alias("n"),
                  pl.col("sizes").drop_nulls().median().alias("size")))
    fit = (sig.group_by(key)
              .agg((pl.col("n_detected") / pl.col("Lstar")).mean().alias("power"),
                   pl.col("n_declared").mean().alias("cs_per_fit"), pl.len().alias("reps")))
    tab = cs.join(fit, on=key, how="full", coalesce=True)
    reps = sig.group_by("method").agg(pl.len().alias("fits")).sort("method")
    print(f"**{sc}**: fits per arm over signal cells: "
          + ", ".join(f"{R.METHOD_LABEL[m]} {n}" for m, n in reps.iter_rows()) + "\n")
    print(__doc__.split("Columns:")[1].strip(), "\n")
    for T in sorted(tab["T"].unique().to_list()):
        for gap in sorted(tab["gap"].unique().to_list()):
            t = tab.filter(pl.col("T") == T, pl.col("gap") == gap)
            print(f"\n### T = {T}, gap = {gap}\n")
            print("**coverage** (declared CSs containing a causal, x/n)\n")
            print(table(t, ["m"], "x", lambda r: f"{int(r['x'])}/{int(r['n'])} ({r['x']/r['n']:.2f})"
                        if r["n"] else "0/0"), "\n")
            print("**size** (median declared-CS size)\n")
            print(table(t, ["m"], "size", lambda r: "-" if r["size"] is None else f"{r['size']:.0f}"), "\n")
            print("**power** (fraction of the 3 causals captured by a declared CS)\n")
            print(table(t, ["m"], "power", lambda r: f"{r['power']:.2f}"), "\n")
            print("**CS/fit** (declared CSs per fit; reps in parentheses)\n")
            print(table(t, ["m"], "cs_per_fit", lambda r: f"{r['cs_per_fit']:.2f} ({int(r['reps'])})"), "\n")
    nul = (d.filter(pl.col("null")).group_by(["m", "method"])
             .agg(pl.col("n_declared").mean().alias("false_cs"), pl.len().alias("reps")))
    print("\n### Null cells (beta = 0): false declared CSs per fit (reps in parentheses)\n")
    print(table(nul, ["m"], "false_cs", lambda r: f"{r['false_cs']:.2f} ({int(r['reps'])})"))


if __name__ == "__main__":
    main(*sys.argv[1:])
