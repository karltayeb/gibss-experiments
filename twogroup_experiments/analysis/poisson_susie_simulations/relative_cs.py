"""Figure: credible sets of gIBSS-Q2 and score relative to CAVI-cf-Q2 across the 021 rate axis.

021's single-causal cells (L* = 1) carry the baseline-rate axis lambda0 in {0.1, 1, 10}, with beta
calibrated per (design, lambda0) to E[LRT] = T. Rows = coverage of declared CSs (arm minus
CAVI), power (fraction of reps whose causal is in a declared CS, arm minus CAVI) and CS size
(ratio of mean declared-CS size, arm over CAVI, log axis). Columns = designs. x = lambda0.
Pooled over T in {4, 8, 16, 32}. Declared = 95% CS of a component with SER log BF >= 2.
Bands = 95% CI from resampling replicates jointly for both arms (paired: same simulated data).

    uv run python analysis/poisson_susie_simulations/relative_cs.py [FIT_L]

FIT_L = 10 (default; the other nine components feed the offset, so gIBSS can differ from CAVI)
or 1 (no offset: gIBSS-Q2 == CAVI-Q2 by construction). Writes figures/relative_cs_L<FIT_L>.{pdf,png}.
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import resultslib as R  # noqa: E402

REF = "q2_cavi"
ARMS = ["q2_gibss", "q2_score"]
MIN_LOG_BF = 2.0
METRICS = ["coverage", "power", "size"]
METRIC_LABEL = {"coverage": "coverage", "power": "power", "size": "CS size ratio"}
METRIC_REF = {"coverage": 0.0, "power": 0.0, "size": 1.0}
DODGE = {"q2_gibss": 0.94, "q2_score": 1.06}


def per_rep(cs: pl.DataFrame, fit_L: int) -> pl.DataFrame:
    """One row per (cell, method, replicate): declared-CS count, covering count, size sum, detected."""
    d = cs.filter(pl.col("Lstar") == 1, pl.col("fit_L") == fit_L,
                  pl.col("method").is_in(ARMS + [REF]))
    decl = (pl.col("ser_log_bf") >= MIN_LOG_BF)
    return (d.group_by("design", "design_n", "lambda0", "T", "method", "sample_id")
             .agg(decl.sum().alias("n_decl"),
                  (decl & pl.col("captured95")).sum().alias("n_cover"),
                  pl.when(decl).then(pl.col("cs_size95")).otherwise(0).sum().alias("size_sum"))
             .with_columns((pl.col("n_cover") > 0).cast(pl.Float64).alias("detected")))


def delta_frame(rep: pl.DataFrame, n_boot: int = 2000, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (design, n, lam0), g in rep.group_by("design", "design_n", "lambda0"):
        wide = {}
        for mth in ARMS + [REF]:
            h = g.filter(pl.col("method") == mth).sort("T", "sample_id")
            wide[mth] = {c: h[c].to_numpy().astype(float) for c in ("n_decl", "n_cover", "size_sum", "detected")}
        n_rep = len(wide[REF]["detected"])
        assert all(len(w["detected"]) == n_rep for w in wide.values()), "unpaired reps"
        idx = rng.integers(0, n_rep, (n_boot, n_rep))

        def est(w, metric, i=None):
            take = (lambda a: a) if i is None else (lambda a: a[i])
            ax = None if i is None else 1
            if metric == "power":
                return take(w["detected"]).mean(axis=ax)
            num = take(w["n_cover"] if metric == "coverage" else w["size_sum"]).sum(axis=ax)
            return num / np.maximum(take(w["n_decl"]).sum(axis=ax), 1)

        for metric in METRICS:
            for mth in ARMS:
                if metric == "size":
                    pt = est(wide[mth], metric) / est(wide[REF], metric)
                    bs = est(wide[mth], metric, idx) / est(wide[REF], metric, idx)
                else:
                    pt = est(wide[mth], metric) - est(wide[REF], metric)
                    bs = est(wide[mth], metric, idx) - est(wide[REF], metric, idx)
                lo, hi = np.nanpercentile(bs, [2.5, 97.5])
                rows.append({"design": design, "design_n": n, "lambda0": lam0, "metric": metric,
                             "method": mth, "delta": float(pt), "lo": float(lo), "hi": float(hi),
                             "n_rep": n_rep,
                             "n_decl_ref": int(wide[REF]["n_decl"].sum()),
                             "n_decl_arm": int(wide[mth]["n_decl"].sum())})
    return pl.DataFrame(rows).sort("design_n", "metric", "method", "lambda0")


def draw(df: pl.DataFrame, fit_L: int):
    designs = df.select("design_n", "design").unique().sort("design_n").rows()
    fig, axes = plt.subplots(len(METRICS), len(designs), figsize=(2.1 * len(designs) + 0.7, 1.75 * len(METRICS) + 1.0),
                             sharex=True, sharey="row", squeeze=False)
    for i, metric in enumerate(METRICS):
        for j, (n, design) in enumerate(designs):
            ax = axes[i][j]
            ax.axhline(METRIC_REF[metric], color="0.55", lw=0.8, ls="--", zorder=0)
            for mth in ARMS:
                s = df.filter(pl.col("design_n") == n, pl.col("metric") == metric,
                              pl.col("method") == mth).sort("lambda0")
                x = s["lambda0"].to_numpy() * DODGE[mth]
                ax.fill_between(x, s["lo"].to_numpy(), s["hi"].to_numpy(), color=R.METHOD_COLOR[mth],
                                alpha=0.16, lw=0, zorder=1)
                ax.plot(x, s["delta"].to_numpy(), color=R.METHOD_COLOR[mth], marker="o", ms=4, lw=1.4,
                        mec="white", mew=0.6, label=R.METHOD_LABEL[mth], zorder=2)
            if metric == "size":
                ax.set_yscale("log", base=2)
                ticks = [0.5, 0.71, 1, 1.41, 2]
                ax.set_yticks(ticks)
                ax.set_yticklabels([f"{t:g}" for t in ticks])
            ax.set_xscale("log")
            lams = sorted(df["lambda0"].unique().to_list())
            ax.set_xticks(lams)
            ax.set_xticklabels([f"{v:g}" for v in lams], fontsize=7)
            ax.minorticks_off()
            ax.tick_params(axis="y", labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="0.92", lw=0.6)
            if i == 0:
                ax.set_title(design, fontsize=7.5, pad=4)
            if j == 0:
                ax.set_ylabel(METRIC_LABEL[metric], fontsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=8)
    fig.supxlabel(r"baseline rate $\lambda_0$ (mean count), $\beta$ calibrated to matched $E[\mathrm{LRT}]$",
                  fontsize=8.5, y=0.02)
    fig.supylabel(f"relative to {R.METHOD_LABEL[REF]}  (single causal, fit L = {fit_L})", fontsize=8.5)
    fig.tight_layout(rect=(0.01, 0.0, 1, 1 - 0.35 / fig.get_size_inches()[1]))
    return fig


def main(fit_L: str = "10") -> None:
    fit_L = int(fit_L)
    cs = R.load_cs("021-poisson", os.path.join(os.path.dirname(os.path.dirname(_HERE)), "results"))
    df = delta_frame(per_rep(cs, fit_L))
    out = os.path.join(_HERE, "figures")
    os.makedirs(out, exist_ok=True)
    stem = os.path.join(out, f"relative_cs_L{fit_L}")
    df.write_parquet(stem + ".parquet")
    with pl.Config(tbl_rows=100, tbl_cols=20, tbl_width_chars=200):
        print(df.select("design_n", "lambda0", "metric", "method", "delta", "lo", "hi", "n_rep",
                        "n_decl_ref", "n_decl_arm"))
    fig = draw(df, fit_L)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    print(stem + ".png")


if __name__ == "__main__":
    main(*sys.argv[1:])
