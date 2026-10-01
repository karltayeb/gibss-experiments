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

import numpy as np
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
            fit_L = int((mcoord.get("kwargs") or {}).get("L", 1) or 1)
            df = (pl.read_parquet(f, columns=["replicate", "credible_sets", "single_effects", "q2_elbo"])
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
                pl.lit(fit_L).alias("fit_L"),
                **{k: (pl.lit(None, dtype=pl.Int64) if v is None else pl.lit(v)) for k, v in meta.items()}))
    return pl.concat(frames, how="diagonal_relaxed")


def eq(col: str, v):
    """Null-safe equality for cell axes (gap is null on single-causal cells)."""
    return pl.col(col).is_null() if v is None else pl.col(col) == v


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


def elbo_table(d: pl.DataFrame, ref: str = "cavi", n_boot: int = 2000, seed: int = 0,
               rows: tuple[str, ...] = ("T", "gap", "m"), digits: int = 3) -> str:
    """Exact Q2 ELBO of each arm minus the reference's on the SAME replicate (nats), mean with a
    rep-resampled 95% CI. One row per combination of `rows` (m is shown as theta = m/n), pooled
    over the other cell axes; columns arms. Signal cells only."""
    rng = np.random.default_rng(seed)
    sig = d.filter(~pl.col("null"))
    arms = [m for m in R.METHODS if m != ref and m in sig["method"].unique().to_list()]
    key = ["T", "gap", "m", "batch_hash", "replicate"]
    sig = sig.with_columns(pl.col("gap").fill_null(-1))   # nulls don't join; single-causal cells
    refd = sig.filter(pl.col("method") == ref).select(*key, pl.col("q2_elbo").alias("ref_elbo"))
    j = (sig.filter(pl.col("method") != ref).join(refd, on=key, how="inner")
            .with_columns(pl.when(pl.col("gap") < 0).then(None).otherwise(pl.col("gap")).alias("gap"))
            .with_columns((pl.col("q2_elbo") - pl.col("ref_elbo")).alias("d")))
    rows = list(rows)
    n = float(sig["n"][0])
    hdr = ["theta" if r == "m" else r for r in rows] + [R.METHOD_LABEL[a] for a in arms]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    fmt = f"{{:+.{digits}f}}"
    for combo in sig.select(rows).unique().sort(rows).iter_rows():
        g = j
        for c, v in zip(rows, combo):
            g = g.filter(eq(c, None if (c == "gap" and v == -1) else v))
        cells = []
        for a in arms:
            x = g.filter(pl.col("method") == a)["d"].to_numpy()
            if len(x) == 0:
                cells.append("-"); continue
            b = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
            lo, hi = np.percentile(b, [2.5, 97.5])
            cells.append(f"{fmt.format(x.mean())} [{fmt.format(lo)}, {fmt.format(hi)}]")
        label = [f"{v / n:g}" if c == "m" else ("-" if (c == "gap" and v == -1) else str(v)) for c, v in zip(rows, combo)]
        lines.append("| " + " | ".join(label + cells) + " |")
    return "\n".join(lines)


def table(d: pl.DataFrame, rows: list[str], value: str, fmt) -> str:
    present = [m for m in R.METHODS if m in d["method"].unique().to_list()]
    header = rows + [R.METHOD_LABEL[m] for m in present]
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for combo in d.select(rows).unique().sort(rows).iter_rows():
        g = d
        for c, v in zip(rows, combo):
            g = g.filter(eq(c, v))
        cells = []
        for m in present:
            x = g.filter(pl.col("method") == m)
            cells.append("-" if x.height == 0 else fmt(x.row(0, named=True)))
        out.append("| " + " | ".join([str(v) for v in combo] + cells) + " |")
    return "\n".join(out)


def pooled_summary(d: pl.DataFrame, ref: str | None = "cavi", n_boot: int = 2000,
                   seed: int = 0) -> str:
    """Headline table per T x gap (pooled over m). With `ref`, that arm is shown in absolute
    terms and every other arm as its difference from `ref` (arm minus ref). Intervals resample
    replicates jointly across arms (the independent unit; CSs within a fit are not)."""
    rng = np.random.default_rng(seed)
    sig = d.filter(~pl.col("null"))
    present = [m for m in R.METHODS if m in sig["method"].unique().to_list()]
    if ref not in present:
        ref = None
    lines = ["| T | gap | arm | coverage [95% CI] | size (IQR) | power [95% CI] | CS/fit |",
             "|---|---|---|---|---|---|---|"]
    for T, gap in sig.select("T", "gap").unique().sort(["T", "gap"]).iter_rows():
        g = sig.filter(eq("T", T), eq("gap", gap)).sort(["batch_hash", "replicate"])
        stat = {}
        for mth in present:
            h = g.filter(pl.col("method") == mth)
            stat[mth] = dict(
                x=h.select(pl.col("covers").list.sum())["covers"].to_numpy().astype(float),
                n=h.select(pl.col("sizes").list.len())["sizes"].to_numpy().astype(float),
                pw=(h["n_detected"] / h["Lstar"]).to_numpy(),
                size=h.explode("sizes")["sizes"].drop_nulls().quantile(0.5),
                q1=h.explode("sizes")["sizes"].drop_nulls().quantile(0.25),
                q3=h.explode("sizes")["sizes"].drop_nulls().quantile(0.75))
        n_rep = len(stat[present[0]]["x"])
        idx = rng.integers(0, n_rep, (n_boot, n_rep))
        def cov(s, i=None):
            x, n = (s["x"], s["n"]) if i is None else (s["x"][i].sum(1), s["n"][i].sum(1))
            return x.sum() / n.sum() if i is None else x / np.maximum(n, 1)
        def pw(s, i=None):
            return s["pw"].mean() if i is None else s["pw"][i].mean(1)
        for mth in present:
            s_ = stat[mth]
            if ref and mth != ref:
                r_ = stat[ref]
                c, cb = cov(s_) - cov(r_), cov(s_, idx) - cov(r_, idx)
                p, pb = pw(s_) - pw(r_), pw(s_, idx) - pw(r_, idx)
                fmt, label = "{:+.3f}", R.METHOD_LABEL[mth]
            else:
                c, cb, p, pb = cov(s_), cov(s_, idx), pw(s_), pw(s_, idx)
                fmt, label = "{:.3f}", R.METHOD_LABEL[mth]
            c_lo, c_hi = np.percentile(cb, [2.5, 97.5])
            p_lo, p_hi = np.percentile(pb, [2.5, 97.5])
            lines.append(f"| {T} | {'-' if gap is None else gap} | {label} | {fmt.format(c)} [{fmt.format(c_lo)}, {fmt.format(c_hi)}] | "
                         f"{s_['size']:.0f} ({s_['q1']:.0f}-{s_['q3']:.0f}) | {fmt.format(p)} [{fmt.format(p_lo)}, {fmt.format(p_hi)}] | "
                         f"{s_['n'].mean():.2f} |")
    return "\n".join(lines)


def delta_by_m(d: pl.DataFrame, ref: str = "cavi", n_boot: int = 2000, seed: int = 0) -> str:
    """Per cell (T, gap, m): reference arm absolute, other arms as arm minus reference, for
    coverage and power, joint rep-resampled 95% CIs. Wide: one row per cell, one column per arm."""
    rng = np.random.default_rng(seed)
    sig = d.filter(~pl.col("null"))
    present = [m for m in R.METHODS if m in sig["method"].unique().to_list()]
    others = [m for m in present if m != ref]
    out = []
    for metric in ("coverage", "power"):
        hdr = ["T", "gap", "m", R.METHOD_LABEL[ref]] + [R.METHOD_LABEL[m] for m in others]
        lines = [f"\n**{metric}** (arm minus {R.METHOD_LABEL[ref]}, 95% CI resamples replicates)\n",
                 "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        for T, gap, m in sig.select("T", "gap", "m").unique().sort(["T", "gap", "m"]).iter_rows():
            g = sig.filter(eq("T", T), eq("gap", gap), pl.col("m") == m).sort(["batch_hash", "replicate"])
            st = {}
            for mth in present:
                h = g.filter(pl.col("method") == mth)
                if metric == "coverage":
                    x = h.select(pl.col("covers").list.sum())["covers"].to_numpy().astype(float)
                    n = h.select(pl.col("sizes").list.len())["sizes"].to_numpy().astype(float)
                    st[mth] = (x, n)
                else:
                    st[mth] = ((h["n_detected"] / h["Lstar"]).to_numpy(), None)
            n_rep = len(st[ref][0])
            idx = rng.integers(0, n_rep, (n_boot, n_rep))
            def est(a, i=None):
                x, n = a
                if n is None:
                    return x.mean() if i is None else x[i].mean(1)
                return x.sum() / n.sum() if i is None else x[i].sum(1) / np.maximum(n[i].sum(1), 1)
            cells = [f"{est(st[ref]):.2f}"]
            for mth in others:
                dlt = est(st[mth]) - est(st[ref])
                b = est(st[mth], idx) - est(st[ref], idx)
                lo, hi = np.percentile(b, [2.5, 97.5])
                cells.append(f"{dlt:+.2f} [{lo:+.2f}, {hi:+.2f}]")
            lines.append(f"| {T} | {gap} | {m} | " + " | ".join(cells) + " |")
        out.append("\n".join(lines))
    return "\n".join(out)


def compact_vs_m(d: pl.DataFrame, arm: str, ref: str = "cavi", metric: str = "coverage",
                 n_boot: int = 2000, seed: int = 0) -> str:
    """One row per T x gap, m across columns: `arm` minus `ref` with a joint rep-resampled
    95% CI (both arms recomputed on the same resampled replicates)."""
    rng = np.random.default_rng(seed)
    sig = d.filter(~pl.col("null"))
    ms = sorted(sig["m"].unique().to_list())
    lines = ["| T | gap | " + " | ".join(f"m={m}" for m in ms) + " |", "|---|---|" + "---|" * len(ms)]
    for T, gap in sig.select("T", "gap").unique().sort(["T", "gap"]).iter_rows():
        cells = []
        for m in ms:
            st = {}
            for mth in (arm, ref):
                h = (sig.filter(pl.col("T") == T, pl.col("gap") == gap, pl.col("m") == m, pl.col("method") == mth)
                        .sort(["batch_hash", "replicate"]))
                if metric == "coverage":
                    st[mth] = (h.select(pl.col("covers").list.sum())["covers"].to_numpy().astype(float),
                               h.select(pl.col("sizes").list.len())["sizes"].to_numpy().astype(float))
                else:
                    st[mth] = ((h["n_detected"] / h["Lstar"]).to_numpy(), None)
            n_rep = len(st[ref][0])
            idx = rng.integers(0, n_rep, (n_boot, n_rep))
            def est(a, i=None):
                x, n = a
                if n is None:
                    return x.mean() if i is None else x[i].mean(1)
                return x.sum() / n.sum() if i is None else x[i].sum(1) / np.maximum(n[i].sum(1), 1)
            dlt = est(st[arm]) - est(st[ref])
            lo, hi = np.percentile(est(st[arm], idx) - est(st[ref], idx), [2.5, 97.5])
            cells.append(f"{dlt:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        lines.append(f"| {T} | {gap} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(sc: str = "022-laplace") -> None:
    d = declared(load(sc))
    print(f"## Pooled over m within T x gap\n")
    print("*coverage* = declared CSs containing a causal; *size* = median declared-CS size with its "
          "interquartile range; "
          "*power* = mean over reps of the fraction of the 3 causals captured by a declared CS; "
          "*CS/fit* = declared CSs per fit. CAVI-Q2 in absolute terms; other arms as arm minus CAVI-Q2 "
          "on the same replicates. 95% CIs resample replicates.\n")
    print(pooled_summary(d, ref="cavi"), "\n")
    print("## global-JJ minus CAVI-Q2 across m (95% CI resamples replicates)\n")
    for metric in ("coverage", "power"):
        print(f"**{metric}**\n")
        print(compact_vs_m(d, "globaljj", "cavi", metric), "\n")
    print("## Per cell, delta to CAVI-Q2\n")
    print(delta_by_m(d, ref="cavi"), "\n")
    print("## Per cell, absolute\n")
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
            t = tab.filter(eq("T", T), eq("gap", gap))
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
