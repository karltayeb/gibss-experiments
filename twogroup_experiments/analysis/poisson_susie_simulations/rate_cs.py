"""025 figures: gIBSS-Q2 and score relative to CAVI-Q2 along the background rate lambda0.

Per design (gaussian / binary / block), one figure: rows = coverage of declared CSs (arm minus
CAVI), power (fraction of causals captured by a declared CS, arm minus CAVI), CS size (ratio of
mean declared-CS size, arm over CAVI, log axis) and ELBO (arm minus CAVI, nats, common Q2 ELBO).
Columns = signal T. x = lambda0 (log). The ELBO row omits score: score's Q2 ELBO collapses
at low rate (~-1e15 nats at lambda0 = 0.01), which the per-cell parquet still records. Colour = arm; solid = gap 8 (correlated causals), dashed =
gap 64. The -ser variant (one causal, L=1) has no gap and no ELBO row (gIBSS-Q2 == CAVI-Q2 at L=1).
Declared = 95% CS of a component with SER log BF >= 2. Bands = 95% CI from resampling replicates
jointly for both arms (paired on the simulated data).

    uv run python analysis/poisson_susie_simulations/rate_cs.py [gaussian|binary|block|all] [ser]

Writes figures/rate_<design>[_ser].{pdf,png}; per-cell numbers to the matching .parquet.
"""
from __future__ import annotations

import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

_HERE = os.path.dirname(os.path.abspath(__file__))
_TG = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _TG)
from experiments import loader

RESULTS = os.path.join(_TG, "results")
CACHE = os.path.join(_HERE, ".cache")
REF = "cavi"
ARMS = ["smoothed", "taylor2", "gibss", "laplace", "score"]
LABEL = {"cavi": "CAVI-Q2", "gibss": "gIBSS-Q2", "laplace": "gIBSS-Laplace", "smoothed": "gIBSS-GO", "taylor2": "Taylor2", "score": "score"}
COLOR = {"smoothed": "#000000", "taylor2": "#7F7F7F", "gibss": "#0072B2", "laplace": "#009E73", "score": "#CC79A7"}
GAP_STYLE = {8: "-", 64: "--", None: "-"}
DODGE = {"smoothed": 0.84, "taylor2": 0.92, "gibss": 1.0, "laplace": 1.08, "score": 1.16}
MIN_LOG_BF = 2.0
IDX95 = 94                                              # CS_BETA_GRID index of 0.95
DESIGN_TITLE = {"gaussian": "AR1 Gaussian (n=500, rho=0.9)",
                "binary": "binary Markov (n=1000, density 0.1)",
                "block": "nested block design (n=1000, causals at depth 3)"}
_BETAS = None


def _meta(design: str, scoord: dict) -> dict:
    """Cell coordinates from the enrichment arguments (labels are stripped from the hash). T is
    recovered from beta via betas_rate.json."""
    global _BETAS
    if _BETAS is None:
        import json
        with open(os.path.join(_HERE, "betas_rate.json")) as fh:
            _BETAS = json.load(fh)["betas"]
    e = scoord["enrichment"]
    lam0 = float(f"{math.exp(float(e['intercept'])):.4g}")
    args = e.get("arguments") or {}
    if e["function"] == "uniform_single_effect":
        return {"lambda0": lam0, "T": None, "gap": None, "null": True}
    beta = float(args["causal_effects"][0])
    tab = _BETAS[design][f"{lam0:g}"]
    T = min(tab, key=lambda t: abs(tab[t] - beta))
    gap = int(args["gap"]) if len(args["causal_effects"]) > 1 else None
    return {"lambda0": lam0, "T": int(T), "gap": gap, "null": False}


def _pairs(sc: str):
    design = sc.split("-")[2]
    seen = set()
    for coll in loader.collection_method_pairs(loader.load_config(), sc).values():
        for bh, mh, mname, _mcoord, scoord in coll["pairs"]:
            if (bh, mh) not in seen:
                seen.add((bh, mh))
                yield bh, mh, mname.rsplit("_", 1)[-1], _meta(design, scoord)


def per_rep(sc: str) -> pl.DataFrame:
    """One row per (cell, method, replicate): declared count, covering count, size sum, number of
    causals captured, Lstar. Cached on the set of reduction files present."""
    os.makedirs(CACHE, exist_ok=True)
    pairs = list(_pairs(sc))
    present = [p for p in pairs if os.path.exists(f"{RESULTS}/by_batch/{p[0]}/fits/{p[1]}/reductions/cs.parquet")]
    cache = os.path.join(CACHE, f"rate_rep_{sc}_{len(present)}.parquet")
    if os.path.exists(cache):
        return pl.read_parquet(cache)
    frames = []
    for bh, mh, method, meta in present:
        d = pl.read_parquet(f"{RESULTS}/by_batch/{bh}/fits/{mh}/reductions/cs.parquet",
                            columns=["sample_id", "ser_log_bf", "cs_sizes", "causal_indices", "mass_above_causal"])
        decl = pl.col("ser_log_bf") >= MIN_LOG_BF
        # causal k is captured by a component when it sits inside that component's 95% CS
        d = d.with_columns(pl.col("cs_sizes").list.get(IDX95).alias("size95"),
                           pl.struct("causal_indices", "mass_above_causal").map_elements(
                               lambda r: [c for c, m in zip(r["causal_indices"], r["mass_above_causal"]) if m < 0.95],
                               return_dtype=pl.List(pl.Int64)).alias("captured"))
        r = (d.group_by("sample_id")
              .agg(decl.sum().alias("n_decl"),
                   (decl & (pl.col("captured").list.len() > 0)).sum().alias("n_cover"),
                   pl.when(decl).then(pl.col("size95")).otherwise(0).sum().alias("size_sum"),
                   pl.col("captured").filter(decl).explode().drop_nulls().n_unique().alias("n_detected"),
                   pl.col("causal_indices").first().list.len().alias("Lstar"))
              .with_columns(pl.lit(bh).alias("batch_hash"), pl.lit(method).alias("method"),
                            **{k: pl.lit(v) for k, v in meta.items()}))
        frames.append(r)
    out = pl.concat(frames, how="diagonal_relaxed")
    out.write_parquet(cache)
    return out


def elbo_rep(sc: str) -> pl.DataFrame:
    """One row per (cell, method, replicate) with the common Q2 ELBO. Cached like per_rep."""
    pairs = list(_pairs(sc))
    present = [p for p in pairs if os.path.exists(f"{RESULTS}/by_batch/{p[0]}/fits/{p[1]}/fits.parquet")]
    cache = os.path.join(CACHE, f"rate_elbo_{sc}_{len(present)}.parquet")
    if os.path.exists(cache):
        return pl.read_parquet(cache)
    frames = []
    for bh, mh, method, meta in present:
        d = pl.read_parquet(f"{RESULTS}/by_batch/{bh}/fits/{mh}/fits.parquet", columns=["replicate", "q2_elbo"])
        frames.append(d.with_columns(pl.lit(bh).alias("batch_hash"), pl.lit(method).alias("method"),
                                     **{k: pl.lit(v) for k, v in meta.items()}))
    out = pl.concat(frames, how="diagonal_relaxed")
    out.write_parquet(cache)
    return out


def _boot(rng, n, n_boot=2000):
    return rng.integers(0, n, (n_boot, n))


def delta_frame(rep: pl.DataFrame, elbo: pl.DataFrame | None, seed: int = 0) -> pl.DataFrame:
    """Arm minus CAVI per (lambda0, T, gap) with paired rep-resampled 95% CIs."""
    rng = np.random.default_rng(seed)
    rows = []
    # arms fit in this supercollection (gIBSS-GO and Taylor2 are L=5 only, so the -ser SCs lack them)
    arms = [m for m in ARMS if m in set(rep["method"].unique().to_list())]
    sig = rep.filter(~pl.col("null"))
    for (lam0, T, gap), g in sig.group_by("lambda0", "T", "gap"):
        w = {m: g.filter(pl.col("method") == m).sort("batch_hash", "sample_id") for m in arms + [REF]}
        keys = w[REF].select("batch_hash", "sample_id")
        w = {m: keys.join(h, on=["batch_hash", "sample_id"], how="inner") for m, h in w.items()}
        common = w[REF].select("batch_hash", "sample_id")
        for m in arms:
            common = common.join(w[m].select("batch_hash", "sample_id"), on=["batch_hash", "sample_id"])
        w = {m: common.join(h, on=["batch_hash", "sample_id"], how="left") for m, h in w.items()}
        n = common.height
        if n == 0:
            continue
        idx = _boot(rng, n)
        a = {m: {c: h[c].to_numpy().astype(float) for c in ("n_decl", "n_cover", "size_sum", "n_detected", "Lstar")}
             for m, h in w.items()}

        def est(x, metric, i=None):
            s = (lambda v: v.sum()) if i is None else (lambda v: v[i].sum(1))
            if metric == "power":
                return s(x["n_detected"]) / s(x["Lstar"])
            num = x["n_cover"] if metric == "coverage" else x["size_sum"]
            return s(num) / np.maximum(s(x["n_decl"]), 1)

        for metric in ("coverage", "power", "size"):
            for m in arms:
                op = np.divide if metric == "size" else np.subtract
                pt = op(est(a[m], metric), est(a[REF], metric))
                bs = op(est(a[m], metric, idx), est(a[REF], metric, idx))
                lo, hi = np.nanpercentile(bs, [2.5, 97.5])
                rows.append({"lambda0": lam0, "T": T, "gap": gap, "metric": metric, "method": m,
                             "delta": float(pt), "lo": float(lo), "hi": float(hi), "n_rep": n,
                             "ref_value": float(est(a[REF], metric)),
                             "n_decl_ref": int(a[REF]["n_decl"].sum()), "n_decl_arm": int(a[m]["n_decl"].sum())})
    if elbo is not None and not elbo.is_empty():
        e = elbo.filter(~pl.col("null"))
        wide = (e.pivot(on="method", index=["lambda0", "T", "gap", "batch_hash", "replicate"], values="q2_elbo")
                 .drop_nulls(arms + [REF]))
        for (lam0, T, gap), g in wide.group_by("lambda0", "T", "gap"):
            idx = _boot(rng, g.height)
            for m in arms:
                d = (g[m] - g[REF]).to_numpy()
                lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
                rows.append({"lambda0": lam0, "T": T, "gap": gap, "metric": "elbo", "method": m,
                             "delta": float(d.mean()), "lo": float(lo), "hi": float(hi), "n_rep": g.height,
                             "ref_value": float(g[REF].mean()), "n_decl_ref": None, "n_decl_arm": None})
    return pl.DataFrame(rows).sort("metric", "method", "gap", "T", "lambda0", nulls_last=True)


ROW_LABEL = {"coverage": "coverage", "power": "power", "size": "CS size ratio", "elbo": "ELBO vs\nCAVI (nats)"}
ROW_REF = {"coverage": 0.0, "power": 0.0, "size": 1.0, "elbo": 0.0}


def draw(df: pl.DataFrame, title: str):
    metrics = [m for m in ("coverage", "power", "size", "elbo") if m in df["metric"].unique().to_list()]
    Ts = sorted(t for t in df["T"].unique().to_list() if t is not None)
    gaps = sorted(df["gap"].unique().to_list(), key=lambda v: (v is None, v))
    lams = sorted(df["lambda0"].unique().to_list())
    fig, axes = plt.subplots(len(metrics), len(Ts), figsize=(1.12 * len(Ts) + 0.6, 1.4 * len(metrics) + 1.0),
                             sharex=True, sharey="row", squeeze=False)
    for i, metric in enumerate(metrics):
        for j, T in enumerate(Ts):
            ax = axes[i][j]
            ax.axhline(ROW_REF[metric], color="0.55", lw=0.8, ls=":", zorder=0)
            # score's Q2 ELBO collapses (to ~-1e15 at lambda0 = 0.01): ELBO row is gIBSS arms only
            for m in ([a for a in ARMS if a != "score"] if metric == "elbo" else ARMS):
                for gap in gaps:
                    s = df.filter(pl.col("metric") == metric, pl.col("method") == m, pl.col("T") == T,
                                  (pl.col("gap") == gap) if gap is not None else pl.col("gap").is_null()).sort("lambda0")
                    if s.is_empty():
                        continue
                    x = s["lambda0"].to_numpy() * DODGE[m]
                    ax.fill_between(x, s["lo"].to_numpy(), s["hi"].to_numpy(), color=COLOR[m], alpha=0.12, lw=0)
                    ax.plot(x, s["delta"].to_numpy(), color=COLOR[m], ls=GAP_STYLE[gap], marker="o", ms=3,
                            lw=1.2, mec="white", mew=0.5)
            if metric == "size":
                ax.set_yscale("log", base=2)
                ax.set_yticks([0.5, 0.71, 1, 1.41, 2])
                ax.set_yticklabels(["0.5", "0.71", "1", "1.41", "2"])
            ax.set_xscale("log")
            ax.set_xticks(lams)
            ax.set_xticklabels([rf"$10^{{{k}}}$" if k % 2 == 0 else "" for k in (round(math.log10(v)) for v in lams)],
                               fontsize=6.5)
            ax.minorticks_off()
            ax.tick_params(axis="y", labelsize=6.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="0.92", lw=0.6)
            if i == 0:
                ax.set_title(f"T = {T}", fontsize=8)
            if j == 0:
                ax.set_ylabel(ROW_LABEL[metric], fontsize=8)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COLOR[m], marker="o", ms=3, lw=1.2, label=LABEL[m]) for m in ARMS]
    if len(gaps) > 1:
        handles += [Line2D([], [], color="0.3", ls=GAP_STYLE[g], lw=1.2,
                           label=("correlated causals (gap 8)" if g == 8 else "independent causals (gap 64)"))
                    for g in gaps]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, 1.0))
    fig.supxlabel(r"background rate $\lambda_0$ (mean count; $\beta$ calibrated to $E[\mathrm{LRT}] = T$)",
                  fontsize=8, y=0.01)
    fig.supylabel(f"relative to {LABEL[REF]}", fontsize=8)
    fig_h = fig.get_size_inches()[1]
    fig.tight_layout(rect=(0.01, 0.0, 1, 1 - 0.5 / fig_h))
    top = max(ax.get_position().y1 for ax in axes[0])
    fig.suptitle(title, fontsize=8.5, y=top + 0.3 / fig_h, va="bottom")
    return fig


def absolute_table(sc: str) -> pl.DataFrame:
    """CAVI-Q2's own coverage, power and mean declared-CS size per (lambda0, gap), pooled over T."""
    rep = per_rep(sc).filter(~pl.col("null"), pl.col("method") == REF)
    return (rep.group_by("lambda0", "gap")
               .agg((pl.col("n_cover").sum() / pl.col("n_decl").sum()).alias("coverage"),
                    (pl.col("n_detected").sum() / pl.col("Lstar").sum()).alias("power"),
                    (pl.col("size_sum").sum() / pl.col("n_decl").sum()).alias("mean CS size"),
                    pl.col("n_decl").sum().alias("declared CSs"),
                    pl.len().alias("reps"))
               .sort("gap", "lambda0", nulls_last=True)
               .with_columns(pl.col("lambda0").map_elements(lambda v: f"{v:g}", return_dtype=pl.String)))


def figure(design: str, ser: bool = False):
    sc = f"025-rate-{design}" + ("-ser" if ser else "")
    df = delta_frame(per_rep(sc), None if ser else elbo_rep(sc))
    return draw(df, DESIGN_TITLE[design] + (", one causal, L = 1" if ser else ", 3 causals, L = 5"))


def run(design: str, ser: bool) -> str:
    sc = f"025-rate-{design}" + ("-ser" if ser else "")
    rep = per_rep(sc)
    elbo = None if ser else elbo_rep(sc)
    df = delta_frame(rep, elbo)
    out = os.path.join(_HERE, "figures")
    os.makedirs(out, exist_ok=True)
    stem = os.path.join(out, f"rate_{design}" + ("_ser" if ser else ""))
    df.write_parquet(stem + ".parquet")
    title = DESIGN_TITLE[design] + (", one causal, L = 1" if ser else ", 3 causals, L = 5")
    fig = draw(df, title)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return stem + ".png"


def main(which: str = "all", ser: str = "") -> None:
    designs = ["gaussian", "binary", "block"] if which == "all" else [which]
    for d in designs:
        print(run(d, ser == "ser"))


if __name__ == "__main__":
    main(*sys.argv[1:])


# ------------------------------------------------------------------------------ per scenario
SCEN = ["design", "L", "lambda0", "T", "gap"]
DESIGN_ORDER = ["binary", "block", "gaussian"]
DESIGN_LABEL = {"binary": "Binary", "block": "Binary-Block", "gaussian": "Gaussian"}
DESIGN_MARKER = {"binary": "o", "block": "s", "gaussian": "^"}
CAVI_COLOR = "#D55E00"
_ABS = [("coverage", "coverage", False), ("power", "power", False), ("size", "mean CS size", True),
        ("n_cs", "declared CSs per fit", False)]
_REL = [("d_coverage", "− CAVI-Q2", False), ("d_power", "− CAVI-Q2", False), ("r_size", "÷ CAVI-Q2", True),
        ("d_n_cs", "− CAVI-Q2", False)]


def scenario_frame() -> pl.DataFrame:
    """One row per (signal scenario, arm): coverage, power, mean size and number per fit of
    declared 95% CSs, CAVI-Q2's values in the same scenario (`*_cavi`), and the arm's difference
    from them (size as a ratio). A scenario is one cell (design, L, lambda0, T, gap); every arm
    sees the same simulated replicates."""
    frames = []
    for d in DESIGN_ORDER:
        for L, suffix in ((5, ""), (1, "-ser")):
            frames.append(per_rep(f"025-rate-{d}{suffix}").with_columns(
                pl.lit(d).alias("design"), pl.lit(L).alias("L")))
    reps = pl.concat(frames, how="diagonal_relaxed").filter(~pl.col("null"))
    df = (reps.group_by(SCEN + ["method"])
          .agg((pl.col("n_cover").sum() / pl.col("n_decl").sum()).alias("coverage"),
               (pl.col("n_detected").sum() / pl.col("Lstar").sum()).alias("power"),
               (pl.col("size_sum").sum() / pl.col("n_decl").sum()).alias("size"),
               pl.col("n_decl").mean().alias("n_cs"))
          .with_columns(pl.col(c).fill_nan(None) for c in ("coverage", "size")))
    ref = df.filter(pl.col("method") == REF).select(
        SCEN + [pl.col(c).alias(f"{c}_cavi") for c, _, _ in _ABS])
    return (df.filter(pl.col("method") != REF).join(ref, on=SCEN, how="inner", nulls_equal=True)
            .with_columns((pl.col("coverage") - pl.col("coverage_cavi")).alias("d_coverage"),
                          (pl.col("power") - pl.col("power_cavi")).alias("d_power"),
                          (pl.col("size") / pl.col("size_cavi")).alias("r_size"),
                          (pl.col("n_cs") - pl.col("n_cs_cavi")).alias("d_n_cs")))


def _log_axis(ax):
    """Log y with plain tick labels at 1, 2, 5 x powers of ten."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def vs_cavi_figure(sf: pl.DataFrame, seed: int = 0):
    """All designs pooled. For each L (5, then 1) two columns: CAVI-Q2's own coverage, power,
    mean CS size and declared CSs per fit per scenario, then each arm minus CAVI-Q2 (size: over).
    One point per scenario; colour = arm, marker = design. Within each arm, scenarios sit left to
    right by lambda0 (jittered), with a bar at each rate's median. CAVI columns share y per row,
    as do the arm columns (the analog of truth.vs_cavi_figure for 022-024)."""
    rng = np.random.default_rng(seed)
    ms_by_L = {L: [m for m in ARMS if m in set(sf.filter(pl.col("L") == L)["method"].unique().to_list())]
               for L in (5, 1)}
    Ls = [L for L in (5, 1) if ms_by_L[L]]
    widths = []
    for L in Ls:
        widths += [1.25, len(ms_by_L[L]) + 0.6]
    fig, axes = plt.subplots(len(_REL), 2 * len(Ls), figsize=(6.5, 1.7 * len(_REL) + 1.4),
                             squeeze=False, layout="constrained", gridspec_kw={"width_ratios": widths})
    for i in range(len(_REL)):
        for c in range(2, 2 * len(Ls)):
            axes[i, c].sharey(axes[i, c % 2])
    lams = sorted(sf["lambda0"].unique().to_list())
    offs = dict(zip(lams, np.linspace(-0.34, 0.34, len(lams))))
    half = 0.34 / max(len(lams) - 1, 1)                 # half the spacing between rate slots

    def strip(ax, x, df, c, color):
        for lam in lams:
            g = df.filter(pl.col("lambda0") == lam)
            ys = []
            for d in DESIGN_ORDER:
                y = g.filter(pl.col("design") == d)[c].drop_nulls().to_numpy().astype(float)
                y = y[np.isfinite(y)]
                ys.append(y)
                ax.scatter(x + offs[lam] + rng.uniform(-0.45, 0.45, y.size) * half, y, s=7,
                           marker=DESIGN_MARKER[d], facecolor="none", edgecolor=color, linewidth=0.5,
                           alpha=0.7, zorder=2)
            y = np.concatenate(ys)
            if y.size:
                ax.plot([x + offs[lam] - 0.8 * half, x + offs[lam] + 0.8 * half], [np.median(y)] * 2,
                        color=color, lw=1.6, solid_capstyle="butt", zorder=3)

    ref = sf.unique(subset=SCEN, keep="first")   # one row per scenario carries CAVI-Q2's values
    last = len(_REL) - 1
    for b, L in enumerate(Ls):
        ca, cr = 2 * b, 2 * b + 1
        dref, d_L, ms = ref.filter(pl.col("L") == L), sf.filter(pl.col("L") == L), ms_by_L[L]
        for i, ((acol, alabel, alog), (rcol, rlabel, ratio)) in enumerate(zip(_ABS, _REL)):
            ax = axes[i, ca]
            strip(ax, 0, dref, f"{acol}_cavi", CAVI_COLOR)
            if alog and b == 0:
                _log_axis(ax)
            if acol == "coverage":
                ax.axhline(0.95, color="0.5", lw=0.7, ls="--", zorder=0)
            ax.set_xlim(-0.55, 0.55)
            ax.set_xticks([0])
            ax.set_xticklabels(["CAVI-Q2"] if i == last else [], rotation=60, ha="right", fontsize=6.5)
            if i == 0:
                ax.set_title("CAVI-Q2", fontsize=7.5, color=CAVI_COLOR)
            ax = axes[i, cr]
            ax.axhline(1.0 if ratio else 0.0, color="0.5", lw=0.7, ls="--", zorder=0)
            for k, m in enumerate(ms):
                strip(ax, k, d_L.filter(pl.col("method") == m), rcol, COLOR[m])
            if ratio and b == 0:
                _log_axis(ax)
            ax.set_xlim(-0.6, len(ms) - 0.4)
            ax.set_xticks(range(len(ms)))
            ax.set_xticklabels([LABEL[m] for m in ms] if i == last else [], rotation=60, ha="right",
                               fontsize=6.5)
            if i == 0:
                ax.set_title(f"L = {L}: arm vs CAVI-Q2", fontsize=7.5)
            for a_, lab in ((axes[i, ca], alabel), (axes[i, cr], rlabel)):
                a_.grid(True, axis="y", alpha=0.25)
                a_.tick_params(labelsize=6.5, labelleft=True)
                if b == 0:
                    a_.set_ylabel(lab, fontsize=7.5)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="0.3", marker=DESIGN_MARKER[d], mfc="none", ls="", ms=5, label=DESIGN_LABEL[d])
         for d in DESIGN_ORDER]
    h.append(Line2D([], [], color="0.3", lw=1.6,
                    label=r"median per $\lambda_0$; left to right $\lambda_0$ = " + ", ".join(f"{v:g}" for v in lams)))
    fig.legend(handles=h, loc="outside upper center", ncol=4, fontsize=7, frameon=False)
    return fig
