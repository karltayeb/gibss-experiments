"""Wall-clock runtime of each arm relative to CAVI-Q2, for 022/023/024 at L=5 and L=1.

`fit_seconds` (stored per replicate in fits.parquet) times the `fit_glm_susie` call alone: no
simulation, no ELBO scoring, no reduction. Three rules make the arms comparable:

* ONE MACHINE. Every timing here comes from Midway caslake, one core per SLURM job (the profile
  sets no --cpus-per-task). The 022 full-grid fits ran on a laptop under parallel load, except
  local-JJ, so 022 reads the `-timing` supercollections (batch 0 rerun on Midway) from a mirror
  directory instead of results/.
* STEADY STATE. Each fit job is a fresh process, so the first replicate of a batch also pays JAX
  compilation. Position 0 is dropped from the ratios and reported on its own (`compile_overhead`).
  Positions 1..9 are flat (no per-replicate recompile).
* PAIRED. An arm's time is divided by CAVI-Q2's time on the same replicate (same data), and the
  ratio is summarized on the log scale (geometric mean).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

_HERE = os.path.dirname(os.path.abspath(__file__))
_TG_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _TG_ROOT not in sys.path:
    sys.path.insert(0, _TG_ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import experiments.loader as loader  # noqa: E402
import laplacelib as R  # noqa: E402

RESULTS = os.path.join(_TG_ROOT, "results")
# Midway copies of the 022 -timing fits (same hashes as the laptop-run fits in results/, which
# the accuracy notebooks read, so they live apart).
MIRROR = os.path.join(_TG_ROOT, "results", "midway_timing")

# (experiment, design, L, supercollection, results root)
SOURCES = [
    ("022", "binary", 5, "022-laplace-timing", MIRROR),
    ("022", "binary", 1, "022-laplace-ser-timing", MIRROR),
    ("023", "gaussian", 5, "023-laplace-gaussian", RESULTS),
    ("023", "gaussian", 1, "023-laplace-gaussian-ser", RESULTS),
    ("024", "nested", 5, "024-laplace-nested", RESULTS),
    ("024", "nested", 1, "024-laplace-nested-ser", RESULTS),
]
ARMS = [m for m in R.METHODS if m != "cavi"]
_PAIR = ["exp", "L", "batch_hash", "rep"]


def fit_paths(sc: str, root: str) -> list[tuple[str, str, str, dict]]:
    """(path, batch_hash, method key, cell meta) for every fit of a supercollection."""
    cfg = loader.load_config()
    out, seen = [], set()
    for coll in loader.collection_method_pairs(cfg, sc).values():
        for bh, mh, mname, _mcoord, scoord in coll["pairs"]:
            if (bh, mh) in seen:
                continue
            seen.add((bh, mh))
            out.append((f"{root}/by_batch/{bh}/fits/{mh}/fits.parquet", bh, R._method_key(mname),
                        R.cell_meta(scoord)))
    return out


def load(cache: str | None = None) -> pl.DataFrame:
    """One row per fit replicate: experiment, L, arm, cell coordinates, position in its batch,
    fit_seconds and IBSS sweep count. Missing fits are skipped (and counted in `n_missing`)."""
    if cache and os.path.exists(cache):
        return pl.read_parquet(cache)
    rows, missing = [], 0
    for exp, design, L, sc, root in SOURCES:
        for path, bh, method, meta in fit_paths(sc, root):
            if not os.path.exists(path):
                missing += 1
                continue
            d = pl.read_parquet(path, columns=["replicate", "fit_seconds", "fit_summary"])
            for pos, r in enumerate(d.iter_rows(named=True)):
                rows.append({"exp": exp, "design": design, "L": L, "sc": sc, "method": method,
                             "m": meta["m"], "T": meta["T"], "depth": meta["depth"],
                             "gap": meta["gap"], "null": meta["null"], "batch_hash": bh,
                             "rep": int(r["replicate"]), "pos": pos,
                             "secs": float(r["fit_seconds"]),
                             "n_iter": int(r["fit_summary"]["n_iter"])})
    df = pl.DataFrame(rows, infer_schema_length=None)
    if missing:
        print(f"runtime.load: {missing} fits missing", file=sys.stderr)
    if cache:
        df.write_parquet(cache)
    return df


def paired(df: pl.DataFrame) -> pl.DataFrame:
    """Steady-state replicates (position >= 1), each arm joined to CAVI-Q2 on the same replicate.
    `ratio` = arm seconds / CAVI seconds; `sweep_ratio` = the same per IBSS sweep."""
    steady = df.filter(pl.col("pos") >= 1)
    cavi = steady.filter(pl.col("method") == "cavi").select(
        *_PAIR, pl.col("secs").alias("cavi_secs"), pl.col("n_iter").alias("cavi_iter"))
    return (steady.filter(pl.col("method") != "cavi").join(cavi, on=_PAIR, how="inner")
            .with_columns((pl.col("secs") / pl.col("cavi_secs")).alias("ratio"),
                          ((pl.col("secs") / pl.col("n_iter"))
                           / (pl.col("cavi_secs") / pl.col("cavi_iter"))).alias("sweep_ratio")))


def _boot_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def speedup(pf: pl.DataFrame, by: list[str], col: str = "ratio") -> pl.DataFrame:
    """Geometric-mean speedup over CAVI-Q2 (CAVI seconds / arm seconds) per group, with a 95%
    interval from resampling replicates. `speedup` > 1 means the arm is faster than CAVI."""
    out = []
    for key, g in pf.group_by(by, maintain_order=True):
        lr = -np.log(g[col].to_numpy())
        lo, hi = _boot_ci(lr)
        out.append({**dict(zip(by, key)), "speedup": float(np.exp(lr.mean())),
                    "lo": float(np.exp(lo)), "hi": float(np.exp(hi)), "n": len(lr)})
    return pl.DataFrame(out)


def absolute(df: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Median steady-state seconds and median IBSS sweeps per group (all arms incl. CAVI)."""
    return (df.filter(pl.col("pos") >= 1).group_by(by, maintain_order=True)
            .agg(pl.col("secs").median().alias("secs"), pl.col("n_iter").median().alias("sweeps"),
                 pl.len().alias("n")))


def compile_overhead(df: pl.DataFrame) -> pl.DataFrame:
    """First-replicate cost above steady state: per batch, position-0 seconds minus the median of
    positions 1..9, then the median over batches."""
    steady = (df.filter(pl.col("pos") >= 1).group_by("exp", "L", "method", "batch_hash")
              .agg(pl.col("secs").median().alias("steady")))
    return (df.filter(pl.col("pos") == 0).join(steady, on=["exp", "L", "method", "batch_hash"])
            .with_columns((pl.col("secs") - pl.col("steady")).alias("overhead"))
            .group_by("exp", "L", "method").agg(pl.col("overhead").median(),
                                                pl.col("steady").median())
            .sort("exp", "L", "method"))


def fmt_ci(tab: pl.DataFrame) -> pl.DataFrame:
    """Speedup as '7.1 (6.9, 7.3)'."""
    return tab.with_columns(pl.format("{} ({}, {})", pl.col("speedup").round(1),
                                      pl.col("lo").round(1), pl.col("hi").round(1)).alias("cell"))


def axis_col(exp: str) -> str:
    return {"022": "m", "023": "T", "024": "depth"}[exp]


_AXIS_LABEL = {"022": "expected set size $m$ (022, binary)", "023": "signal $T$ (023, Gaussian)",
               "024": "causal depth $d$ (024, nested)"}


def speedup_by_axis_figure(pf: pl.DataFrame):
    """Speedup over CAVI-Q2 against each experiment's sweep axis (non-null cells): columns are
    the experiments, rows L=5 and L=1, log y, one line per arm; dashed line = CAVI-Q2."""
    import matplotlib.pyplot as plt
    exps = ["022", "023", "024"]
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.3), sharey=True)
    for i, L in enumerate([5, 1]):
        for j, exp in enumerate(exps):
            ax, col = axes[i, j], axis_col(exp)
            sub = pf.filter((pl.col("exp") == exp) & (pl.col("L") == L) & ~pl.col("null"))
            if sub.height:
                tab = speedup(sub, ["method", col]).sort(col)
                for m in ARMS:
                    t = tab.filter(pl.col("method") == m)
                    if t.height == 0:
                        continue
                    x = t[col].to_numpy()
                    ax.errorbar(x, t["speedup"], yerr=[t["speedup"] - t["lo"], t["hi"] - t["speedup"]],
                                color=R.METHOD_COLOR[m], marker=R.METHOD_MARKER[m], ms=3.5, lw=1,
                                capsize=0, label=R.METHOD_LABEL[m])
            ax.axhline(1.0, color=R.METHOD_COLOR["cavi"], ls="--", lw=0.8)
            ax.set_yscale("log")
            if exp == "022":
                ax.set_xscale("log")
            ax.grid(True, which="major", alpha=0.25)
            if i == 1:
                ax.set_xlabel(_AXIS_LABEL[exp], fontsize=7.5)
            if j == 0:
                ax.set_ylabel(f"L = {L}\nspeedup over CAVI-Q2", fontsize=8)
            ax.tick_params(labelsize=7)
    handles = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(handles.values(), handles.keys(), loc="upper center", ncol=max(len(handles), 1),
               fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig
