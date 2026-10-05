"""Figure: each approximation arm minus CAVI-Q2, against expected set size m.

Rows = coverage of declared CSs, power (fraction of causals captured by a declared CS), and
CS size as the ratio of mean declared-CS size (arm over reference, log axis, 1 = same size).
Columns = T x gap panels. Bands = 95% CI from resampling replicates jointly for both arms,
so the interval is on the difference. Zero line = identical to CAVI-Q2.

    uv run python analysis/logistic_laplace_simulations/delta_figure.py [SC] [REF]
writes figures/delta_vs_<REF>.{pdf,png} next to this file. REF=none gives the absolute
version: every arm's own coverage (0.95 nominal line), power and mean CS size,
each with its own rep-resampled 95% CI, written to figures/absolute_vs_m.{pdf,png}.
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

METRIC_LABEL = {"coverage": "coverage", "power": "power", "size": "CS size ratio"}
METRIC_REF = {"coverage": 0.0, "power": 0.0, "size": 1.0}   # "same as reference" line
ABS_LABEL = {"coverage": "coverage", "power": "power", "size": "mean CS size"}
ABS_REF = {"coverage": 0.95, "power": None, "size": None}   # nominal line
# column headers: signal strength (T = expected LRT of the causal vs null) over causal correlation
SIGNAL_LABEL = {8: "moderate signal", 16: "strong signal"}
# binary phi between causals at the design's adjacent-column rho = 0.8 (laplacelib / brief.typ)
GAP_LABEL = {8: "correlated\ncausals (r = 0.47)", 64: "weakly correlated\ncausals (r = 0.05)",
             None: "single causal"}
X_LABEL = r"design density, $\theta = P(X = 1)$"


def delta_frame(d: pl.DataFrame, ref: str | None = "cavi", n_boot: int = 2000, seed: int = 0) -> pl.DataFrame:
    """Per (metric, T, gap, m, arm): arm minus ref with joint rep-resampled 95% CI; with
    ref=None, each arm's own value with its own rep-resampled CI (size = mean declared-CS size)."""
    rng = np.random.default_rng(seed)
    sig = d.filter(~pl.col("null"))
    arms = [m for m in R.METHODS if m != ref and m in sig["method"].unique().to_list()]
    rows = []
    for T, gap, m in sig.select("T", "gap", "m").unique().sort(["T", "gap", "m"]).iter_rows():
        g = sig.filter(C.eq("T", T), C.eq("gap", gap), pl.col("m") == m).sort(["batch_hash", "replicate"])
        per = {}
        for mth in arms + ([ref] if ref else []):
            h = g.filter(pl.col("method") == mth)
            n_decl = h.select(pl.col("sizes").list.len())["sizes"].to_numpy().astype(float)
            per[mth] = {
                "coverage": (h.select(pl.col("covers").list.sum())["covers"].to_numpy().astype(float), n_decl),
                "power": ((h["n_detected"] / h["Lstar"]).to_numpy(), None),
                # per-rep sum of sizes / count -> mean declared-CS size; ratio of means with ref
                "size": (h.select(pl.col("sizes").list.sum())["sizes"].fill_null(0).to_numpy().astype(float), n_decl),
            }
        n_rep = len(per[arms[0]]["power"][0])
        idx = rng.integers(0, n_rep, (n_boot, n_rep))

        def est(a, i=None):
            x, n = a
            if n is None:
                return x.mean() if i is None else x[i].mean(1)
            return x.sum() / n.sum() if i is None else x[i].sum(1) / np.maximum(n[i].sum(1), 1)

        for metric in ("coverage", "power", "size"):
            for mth in arms:
                if ref and metric == "size":     # ratio of mean sizes
                    dlt = est(per[mth][metric]) / est(per[ref][metric])
                    b = est(per[mth][metric], idx) / est(per[ref][metric], idx)
                elif ref:
                    dlt = est(per[mth][metric]) - est(per[ref][metric])
                    b = est(per[mth][metric], idx) - est(per[ref][metric], idx)
                else:
                    dlt, b = est(per[mth][metric]), est(per[mth][metric], idx)
                lo, hi = np.percentile(b, [2.5, 97.5])
                rows.append({"metric": metric, "T": T, "gap": gap, "m": m, "theta": m / float(g["n"][0]),
                             "method": mth, "delta": float(dlt), "lo": float(lo), "hi": float(hi)})
    return pl.DataFrame(rows)


# log2-axis ticks for the ratio (ref) / absolute metrics drawn on a log scale
LOG_TICKS = {"size": [0.25, 0.35, 0.5, 0.71, 1, 1.41, 2]}
ABS_LOG_TICKS = {"size": [1, 2, 4, 8, 16, 32]}


def draw(df: pl.DataFrame, *, ref: str | None = "cavi", ax_w: float | None = None, ax_h: float = 1.75,
         metrics: list[str] | None = None, labels: dict | None = None, refs: dict | None = None,
         log_ticks: dict | None = None, ylabel: str | None = None):
    """Rows = `metrics` (default the CS metrics), columns = T x gap panels, x = design density.
    `labels`/`refs` give the row label and the "no difference" (ref) or nominal (absolute) line
    per metric; `log_ticks` lists metrics drawn on a log2 axis with their ticks. Other modules
    (posterior_figure) pass their own metric specs and get the same layout."""
    if labels is None:
        labels = METRIC_LABEL if ref else ABS_LABEL
    if refs is None:
        refs = METRIC_REF if ref else ABS_REF
    if log_ticks is None:
        log_ticks = LOG_TICKS if ref else ABS_LOG_TICKS
    if metrics is None:
        metrics = ["coverage", "power", "size"]
    panels = df.select("T", "gap").unique().sort(["T", "gap"]).rows()
    if ax_w is None:
        ax_w = 1.5 if len(panels) > 2 else 2.4   # a 2-panel (single-causal) figure can afford wider axes
    arms = [m for m in R.METHODS if m in df["method"].unique().to_list()]
    x_ticks = sorted(df["theta"].unique().to_list())
    fig, axes = plt.subplots(len(metrics), len(panels), figsize=(ax_w * len(panels) + 0.6, ax_h * len(metrics) + 0.9),
                             sharex=True, sharey="row", squeeze=False)
    for i, metric in enumerate(metrics):
        for j, (T, gap) in enumerate(panels):
            ax = axes[i][j]
            g = df.filter(pl.col("metric") == metric, C.eq("T", T), C.eq("gap", gap))
            if refs[metric] is not None:
                ax.axhline(refs[metric], color="0.55", lw=0.8, ls="--", zorder=0)
            for mth in arms:
                s = g.filter(pl.col("method") == mth).sort("theta")
                x = s["theta"].to_numpy() * R._DODGE[mth]
                ax.fill_between(x, s["lo"].to_numpy(), s["hi"].to_numpy(), color=R.METHOD_COLOR[mth],
                                alpha=0.16, lw=0, zorder=1)
                ax.plot(x, s["delta"].to_numpy(), color=R.METHOD_COLOR[mth], marker=R.METHOD_MARKER[mth],
                        ms=4, lw=1.4, mec="white", mew=0.6, label=R.METHOD_LABEL[mth], zorder=2)
            if metric in log_ticks:
                ax.set_yscale("log", base=2)
                ticks = log_ticks[metric]
                ax.set_yticks(ticks)
                ax.set_yticklabels([f"{t:g}" for t in ticks])
            ax.set_xscale("log")
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([f"{v:g}" for v in x_ticks], fontsize=6.5, rotation=45, ha="right",
                               rotation_mode="anchor")
            ax.minorticks_off()
            ax.tick_params(axis="y", labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="0.92", lw=0.6)
            if i == 0:
                ax.set_title(GAP_LABEL.get(gap, f"gap = {gap}"), fontsize=7, pad=4)
            if j == 0:
                ax.set_ylabel(labels[metric], fontsize=8)
    handles, labels_ = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=len(labels_), frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=8)
    fig.supxlabel(X_LABEL, fontsize=8.5, y=0.02)
    if ylabel is None:
        ylabel = f"metrics, relative to {R.METHOD_LABEL[ref]}" if ref else "metrics"
    fig.supylabel(ylabel, fontsize=8.5)
    # vertical offsets in inches so the legend / super-header band is the same whatever the row count
    fig_h = fig.get_size_inches()[1]
    fig.tight_layout(rect=(0.01, 0.0, 1, 1 - 0.62 / fig_h))
    # super headers: one per signal strength, spanning that T's columns, with a rule beneath
    fig.canvas.draw()
    from matplotlib.lines import Line2D
    for T in sorted({t for t, _ in panels}):
        cols = [j for j, (t, _) in enumerate(panels) if t == T]
        left = axes[0][cols[0]].get_position().x0
        right = axes[0][cols[-1]].get_position().x1
        top = axes[0][cols[0]].get_position().y1
        fig.text((left + right) / 2, top + 0.46 / fig_h, f"{SIGNAL_LABEL.get(T, f'T = {T}')} (T = {T})",
                 ha="center", va="bottom", fontsize=8.5)
        fig.add_artist(Line2D([left, right], [top + 0.43 / fig_h] * 2, color="0.3", lw=0.8))
    return fig


def main(sc: str = "022-laplace", ref: str = "cavi") -> None:
    ref = None if ref.lower() == "none" else ref
    d = C.declared(C.load(sc))
    df = delta_frame(d, ref=ref)
    out = os.path.join(_HERE, "figures")
    os.makedirs(out, exist_ok=True)
    base = f"delta_vs_{ref}" if ref else "absolute_vs_m"
    stem = base if sc == "022-laplace" else f"{sc.removeprefix('022-laplace-')}_{base}"
    df.write_parquet(os.path.join(out, f"{stem}.parquet"))
    fig = draw(df, ref=ref)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{stem}.{ext}"), dpi=200, bbox_inches="tight")
    print(os.path.join(out, f"{stem}.png"))


if __name__ == "__main__":
    main(*sys.argv[1:])
