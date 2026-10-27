"""Figure: credible sets of every arm relative to a reference arm (default linear_int).

The 021/025 `relative_cs` layout. Rows = coverage of declared sets (arm minus reference),
power (share of causal sites inside a declared set, arm minus reference) and CS size (ratio
of mean declared-set size, arm over reference, log2 axis). Columns = (mu0, phi) panels;
x = |beta|; pooled over k_causal and libsize_sd unless `facet` says otherwise. Declared =
95% set of a component with SER log BF >= 2. Bands = 95% CI from resampling replicates
jointly for both arms (paired: same simulated data).

    uv run python analysis/relative_cs.py [COLLECTION] [L] [REF]
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))

MIN_LOG_BF = 2.0
METRICS = ["coverage", "power", "size"]
METRIC_LABEL = {"coverage": "coverage", "power": "power", "size": "CS size ratio"}
METRIC_REF = {"coverage": 0.0, "power": 0.0, "size": 1.0}
ARM_ORDER = ["linear_int", "linear_log", "nb_true", "nb_est", "nb_score_rss", "nb_score_rss_true", "nb_mle_rss", "nb_cavi", "nb_cavi_true"]
PALETTE = {"linear_int": "#0072B2", "linear_log": "#56B4E9", "nb_true": "#D55E00", "nb_est": "#E69F00",
           "nb_score_rss": "#009E73", "nb_score_rss_true": "#CC79A7", "nb_mle_rss": "#117733", "nb_cavi": "#000000", "nb_cavi_true": "#999999"}
PANEL = ("mu0", "phi")
XAXIS = "beta_abs"


def per_rep(cs: pl.DataFrame, fit: pl.DataFrame, L: str) -> pl.DataFrame:
    """One row per (cell, arm, rep): declared-set count, covering count, size sum, causal hits."""
    decl = pl.col("ser_log_bf") >= MIN_LOG_BF
    d = (cs.filter(pl.col("L") == L)
           .group_by("cell", "arm", "rep")
           .agg(decl.sum().alias("n_decl"),
                (decl & pl.col("covered")).sum().alias("n_cover"),
                pl.when(decl).then(pl.col("cs_size")).otherwise(0).sum().alias("size_sum"),
                pl.when(decl).then(pl.col("n_causal_in_cs")).otherwise(0).sum().alias("hits")))
    coords = fit.filter(pl.col("L") == L).select("cell", "arm", "rep", "k_causal", *PANEL, XAXIS).unique()
    out = coords.join(d, on=["cell", "arm", "rep"], how="left").fill_null(0)
    return out.with_columns(pl.min_horizontal(pl.col("hits"), pl.col("k_causal")).alias("hits"))


def delta_frame(rep: pl.DataFrame, ref: str, arms: list[str], n_boot: int = 2000, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in rep.group_by(*PANEL, XAXIS, maintain_order=True):
        wide = {}
        for a in arms + [ref]:
            h = g.filter(pl.col("arm") == a).sort("cell", "rep")
            wide[a] = {c: h[c].to_numpy().astype(float) for c in ("n_decl", "n_cover", "size_sum", "hits", "k_causal")}
        n_rep = len(wide[ref]["n_decl"])
        if n_rep == 0 or any(len(w["n_decl"]) != n_rep for w in wide.values()):
            continue
        idx = rng.integers(0, n_rep, (n_boot, n_rep))

        def est(w, metric, i=None):
            """Pooled estimate; NaN where nothing was declared (coverage / size undefined)."""
            take = (lambda a: a) if i is None else (lambda a: a[i])
            ax = None if i is None else 1
            if metric == "power":
                return take(w["hits"]).sum(axis=ax) / np.maximum(take(w["k_causal"]).sum(axis=ax), 1)
            num = take(w["n_cover"] if metric == "coverage" else w["size_sum"]).sum(axis=ax)
            den = take(w["n_decl"]).sum(axis=ax).astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)

        for metric in METRICS:
            for a in arms:
                if metric == "size":
                    pt = est(wide[a], metric) / est(wide[ref], metric)
                    bs = est(wide[a], metric, idx) / est(wide[ref], metric, idx)
                else:
                    pt = est(wide[a], metric) - est(wide[ref], metric)
                    bs = est(wide[a], metric, idx) - est(wide[ref], metric, idx)
                bs = np.asarray(bs, dtype=float)
                lo, hi = (np.nanpercentile(bs, [2.5, 97.5]) if np.isfinite(bs).any() else (np.nan, np.nan))
                rows.append({**dict(zip((*PANEL, XAXIS), key)), "metric": metric, "arm": a,
                             "delta": float(pt), "lo": float(lo), "hi": float(hi), "n_rep": n_rep,
                             "n_decl_ref": int(wide[ref]["n_decl"].sum()), "n_decl_arm": int(wide[a]["n_decl"].sum())})
    return pl.DataFrame(rows).sort(*PANEL, "metric", "arm", XAXIS)


def draw(df: pl.DataFrame, ref: str, arms: list[str], L: str):
    panels = df.select(*PANEL).unique().sort(*PANEL).rows()
    fig, axes = plt.subplots(len(METRICS), len(panels), figsize=(min(6.5, 1.6 * len(panels) + 0.8), 1.75 * len(METRICS) + 1.0),
                             sharex=True, sharey="row", squeeze=False)
    xs = sorted(df[XAXIS].unique().to_list())
    dodge = np.linspace(-0.03, 0.03, len(arms))
    for i, metric in enumerate(METRICS):
        for j, panel in enumerate(panels):
            ax = axes[i][j]
            ax.axhline(METRIC_REF[metric], color="0.55", lw=0.8, ls="--", zorder=0)
            for k, a in enumerate(arms):
                s = df.filter(pl.col("metric") == metric, pl.col("arm") == a,
                              *[pl.col(c) == v for c, v in zip(PANEL, panel)]).sort(XAXIS)
                x = np.array([xs.index(v) for v in s[XAXIS].to_list()], dtype=float) + dodge[k]
                ax.fill_between(x, s["lo"].to_numpy(), s["hi"].to_numpy(), color=PALETTE[a], alpha=0.16, lw=0, zorder=1)
                ax.plot(x, s["delta"].to_numpy(), color=PALETTE[a], marker="o", ms=3.5, lw=1.2, mec="white", mew=0.5, label=a, zorder=2)
            if metric == "size":
                ax.set_yscale("log", base=2)
                ticks = [0.25, 0.5, 1, 2, 4]
                ax.set_yticks(ticks); ax.set_yticklabels([f"{t:g}" for t in ticks]); ax.set_ylim(0.2, 5)
            else:
                ax.set_ylim(-1.05, 1.05)
            ax.set_xticks(range(len(xs))); ax.set_xticklabels([f"{v:g}" for v in xs], fontsize=7)
            ax.minorticks_off(); ax.tick_params(axis="y", labelsize=7)
            ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="0.92", lw=0.6)
            if i == 0:
                ax.set_title(", ".join(f"{c}={v:g}" for c, v in zip(PANEL, panel)), fontsize=7.5, pad=4)
            if j == 0:
                ax.set_ylabel(METRIC_LABEL[metric], fontsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=7)
    fig.supxlabel("|beta| (log fold change per allele)", fontsize=8.5, y=0.02)
    fig.supylabel(f"relative to {ref}  (L = {L})", fontsize=8.5)
    fig.tight_layout(rect=(0.01, 0.0, 1, 1 - 0.45 / fig.get_size_inches()[1]))
    return fig


def relative_cs(cs: pl.DataFrame, fit: pl.DataFrame, *, L: str, ref: str = "linear_int", arms=None):
    """(delta frame, figure) for one L. `arms` defaults to every arm present except `ref`."""
    present = [a for a in ARM_ORDER if a in set(fit["arm"]) and a != ref]
    arms = present if arms is None else [a for a in arms if a in present]
    df = delta_frame(per_rep(cs, fit, L), ref, arms)
    return df, draw(df, ref, arms, L)


def main(collection: str = "pilot", L: str = "5", ref: str = "linear_int") -> None:
    root = os.path.join(os.path.dirname(_HERE), "results", "collections", collection)
    cs = pl.read_parquet(os.path.join(root, "cs.parquet")); fit = pl.read_parquet(os.path.join(root, "fit.parquet"))
    df, fig = relative_cs(cs, fit, L=L, ref=ref)
    out = os.path.join(_HERE, "figures"); os.makedirs(out, exist_ok=True)
    stem = os.path.join(out, f"relative_cs_{collection}_L{L}_vs_{ref}")
    df.write_parquet(stem + ".parquet"); fig.savefig(stem + ".png", dpi=200, bbox_inches="tight")
    with pl.Config(tbl_rows=200, tbl_width_chars=200):
        print(df.select(*PANEL, XAXIS, "metric", "arm", "delta", "lo", "hi", "n_rep", "n_decl_ref", "n_decl_arm"))
    print("wrote", stem + ".png")


if __name__ == "__main__":
    main(*sys.argv[1:])
