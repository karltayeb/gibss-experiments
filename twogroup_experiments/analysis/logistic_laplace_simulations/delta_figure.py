"""Figure: each approximation arm minus CAVI-Q2, against expected set size m.

Rows = coverage of declared CSs, power (fraction of causals captured by a declared CS).
Columns = T x gap panels. Bands = 95% CI from resampling replicates jointly for both arms,
so the interval is on the difference. Zero line = identical to CAVI-Q2.

    uv run python analysis/logistic_laplace_simulations/delta_figure.py [SC] [REF]
writes figures/delta_vs_<REF>.{pdf,png} next to this file.
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import cs_tables as C  # noqa: E402
import laplacelib as R  # noqa: E402

METRIC_LABEL = {"coverage": "coverage", "power": "power"}


def delta_frame(d: pl.DataFrame, ref: str = "cavi", n_boot: int = 2000, seed: int = 0) -> pl.DataFrame:
    """Per (metric, T, gap, m, arm): arm minus ref with joint rep-resampled 95% CI."""
    rng = np.random.default_rng(seed)
    sig = d.filter(~pl.col("null"))
    arms = [m for m in R.METHODS if m != ref and m in sig["method"].unique().to_list()]
    rows = []
    for T, gap, m in sig.select("T", "gap", "m").unique().sort(["T", "gap", "m"]).iter_rows():
        g = sig.filter(pl.col("T") == T, pl.col("gap") == gap, pl.col("m") == m).sort(["batch_hash", "replicate"])
        per = {}
        for mth in arms + [ref]:
            h = g.filter(pl.col("method") == mth)
            per[mth] = {
                "coverage": (h.select(pl.col("covers").list.sum())["covers"].to_numpy().astype(float),
                             h.select(pl.col("sizes").list.len())["sizes"].to_numpy().astype(float)),
                "power": ((h["n_detected"] / h["Lstar"]).to_numpy(), None),
            }
        n_rep = len(per[ref]["power"][0])
        idx = rng.integers(0, n_rep, (n_boot, n_rep))

        def est(a, i=None):
            x, n = a
            if n is None:
                return x.mean() if i is None else x[i].mean(1)
            return x.sum() / n.sum() if i is None else x[i].sum(1) / np.maximum(n[i].sum(1), 1)

        for metric in ("coverage", "power"):
            for mth in arms:
                dlt = est(per[mth][metric]) - est(per[ref][metric])
                b = est(per[mth][metric], idx) - est(per[ref][metric], idx)
                lo, hi = np.percentile(b, [2.5, 97.5])
                rows.append({"metric": metric, "T": T, "gap": gap, "m": m, "method": mth,
                             "delta": float(dlt), "lo": float(lo), "hi": float(hi)})
    return pl.DataFrame(rows)


def draw(df: pl.DataFrame, *, ref: str = "cavi", ax_w: float = 1.75, ax_h: float = 1.9):
    panels = df.select("T", "gap").unique().sort(["T", "gap"]).rows()
    metrics = ["coverage", "power"]
    arms = [m for m in R.METHODS if m in df["method"].unique().to_list()]
    m_ticks = sorted(df["m"].unique().to_list())
    fig, axes = plt.subplots(len(metrics), len(panels), figsize=(ax_w * len(panels) + 0.5, ax_h * len(metrics) + 0.5),
                             sharex=True, sharey="row", squeeze=False)
    for i, metric in enumerate(metrics):
        for j, (T, gap) in enumerate(panels):
            ax = axes[i][j]
            g = df.filter(pl.col("metric") == metric, pl.col("T") == T, pl.col("gap") == gap)
            ax.axhline(0, color="0.55", lw=0.8, ls="--", zorder=0)
            for mth in arms:
                s = g.filter(pl.col("method") == mth).sort("m")
                x = s["m"].to_numpy() * R._DODGE[mth]
                ax.fill_between(x, s["lo"].to_numpy(), s["hi"].to_numpy(), color=R.METHOD_COLOR[mth],
                                alpha=0.16, lw=0, zorder=1)
                ax.plot(x, s["delta"].to_numpy(), color=R.METHOD_COLOR[mth], marker=R.METHOD_MARKER[mth],
                        ms=4, lw=1.4, mec="white", mew=0.6, label=R.METHOD_LABEL[mth], zorder=2)
            ax.set_xscale("log")
            ax.set_xticks(m_ticks)
            ax.set_xticklabels([str(v) for v in m_ticks], fontsize=7)
            ax.minorticks_off()
            ax.tick_params(axis="y", labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="0.92", lw=0.6)
            if i == 0:
                ax.set_title(f"T = {T}, gap = {gap}", fontsize=8.5)
            if j == 0:
                ax.set_ylabel(METRIC_LABEL[metric], fontsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=8)
    fig.supxlabel("expected set size m", fontsize=8.5, y=0.035)
    fig.supylabel(f"arm minus {R.METHOD_LABEL[ref]}", fontsize=8.5)
    fig.tight_layout(rect=(0.01, 0.0, 1, 0.95))
    return fig


def main(sc: str = "022-laplace", ref: str = "cavi") -> None:
    d = C.declared(C.load(sc))
    df = delta_frame(d, ref=ref)
    out = os.path.join(_HERE, "figures")
    os.makedirs(out, exist_ok=True)
    df.write_parquet(os.path.join(out, f"delta_vs_{ref}.parquet"))
    fig = draw(df, ref=ref)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"delta_vs_{ref}.{ext}"), dpi=200, bbox_inches="tight")
    print(os.path.join(out, f"delta_vs_{ref}.png"))


if __name__ == "__main__":
    main(*sys.argv[1:])
