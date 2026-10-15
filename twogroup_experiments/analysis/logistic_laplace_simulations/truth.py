"""Experiments 022-024 scored against the simulation truth: PIP calibration, FDP / power, and
declared credible sets (coverage, power, size).

Reads the pipeline's per-fit `pip` and `cs` reductions (results/by_batch/<b>/fits/<m>/reductions),
not fits.parquet:

* pip: per replicate, 200 PIP bins ([k/200, (k+1)/200)) with feature and causal counts.
* cs:  per replicate and component l, the component log BF, the 95% CS size, and for each causal
       the alpha mass ranked above it (``mass_above_causal``). A causal is in the beta-CS iff that
       mass is < beta (the 019 convention, resultslib).

``load`` reduces these on read to three frames, cached in analysis/.cache keyed on the reduction
files' paths, sizes and mtimes:

* bins: summed PIP bins per (exp, L, method, cell, batch of replicates).
* reps: one row per (exp, L, method, cell, replicate): causal count, features and causals at or
        above each PIP threshold in THRESHOLDS, declared components, declared components that
        cover a causal, causals captured by a declared 95% CS, summed declared 95% CS size.
* comps: one row per declared component (log BF >= MIN_LOG_BF): 95% CS size, whether it covers a
        causal, and the smallest mass above any causal (``best_mass``, for coverage at any level).
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import polars as pl

_HERE = os.path.dirname(os.path.abspath(__file__))
_TG_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_TG_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import laplacelib as R  # noqa: E402
from experiments import loader  # noqa: E402

RESULTS = os.path.join(_TG_ROOT, "results")
CACHE = os.path.join(os.path.dirname(_HERE), ".cache")
SOURCES = [
    ("022", 5, "022-laplace"), ("022", 1, "022-laplace-ser"),
    ("024", 5, "024-laplace-nested"), ("024", 1, "024-laplace-nested-ser"),
    ("023", 5, "023-laplace-gaussian"), ("023", 1, "023-laplace-gaussian-ser"),
]
DESIGN = {"022": "Binary", "024": "Binary-Block", "023": "Gaussian"}
EXPS = list(DESIGN)
MIN_LOG_BF = R.MIN_LOG_BF
NBIN = 200
THRESHOLDS = (0.5, 0.9, 0.95)
CELL = ["m", "T", "depth", "gap", "null"]
IDX95 = 94   # CS_BETA_GRID index of 0.95


def _files(sources) -> list[tuple]:
    cfg = loader.load_config()
    out, seen = [], set()
    for exp, L, sc in sources:
        for coll in loader.collection_method_pairs(cfg, sc).values():
            for bh, mh, mname, _mc, scoord in coll["pairs"]:
                d = f"{RESULTS}/by_batch/{bh}/fits/{mh}/reductions"
                if (bh, mh) in seen or not os.path.exists(f"{d}/cs.parquet"):
                    continue
                seen.add((bh, mh))
                out.append((exp, L, R._method_key(mname), R.cell_meta(scoord), d))
    return out


def _fingerprint(files) -> str:
    h = hashlib.sha1()
    for *_, d in files:
        for r in ("pip", "cs"):
            st = os.stat(f"{d}/{r}.parquet")
            h.update(f"{d}/{r}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def _reduce(files) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    cuts = [int(round(t * NBIN)) for t in THRESHOLDS]
    bin_rows, rep_rows, comp_rows = [], [], []
    for exp, L, method, meta, d in files:
        key = {"exp": exp, "L": L, "method": method, **{c: meta[c] for c in CELL}}
        pip = pl.read_parquet(f"{d}/pip.parquet", columns=["sample_id", "causal_indices",
                                                            "pip_bin_counts", "pip_bin_causal_counts"])
        cnt = np.array(pip["pip_bin_counts"].to_list(), dtype=np.int64)
        cau = np.array(pip["pip_bin_causal_counts"].to_list(), dtype=np.int64)
        bin_rows.append({**key, "counts": cnt.sum(0).tolist(), "causal": cau.sum(0).tolist()})
        above = np.cumsum(cnt[:, ::-1], 1)[:, ::-1]      # features with PIP >= k/200
        above_c = np.cumsum(cau[:, ::-1], 1)[:, ::-1]
        reps = {"sample_id": pip["sample_id"].to_list(),
                "n_causal": [len(c) for c in pip["causal_indices"].to_list()]}
        for t, k in zip(THRESHOLDS, cuts):
            reps[f"sel_{t:g}"] = above[:, k].tolist()
            reps[f"tp_{t:g}"] = above_c[:, k].tolist()
        cs = (pl.read_parquet(f"{d}/cs.parquet", columns=["sample_id", "ser_log_bf", "causal_indices",
                                                          "mass_above_causal", "cs_sizes"])
              .filter(pl.col("ser_log_bf") >= MIN_LOG_BF))
        per_rep: dict[str, list] = {}
        for sid, lbf, ci, mass, sizes in cs.iter_rows():
            best = min(mass) if mass else None
            covers = best is not None and best < 0.95
            comp_rows.append({**key, "ser_log_bf": lbf, "size": sizes[IDX95], "covers": covers,
                              "best_mass": best})
            per_rep.setdefault(sid, []).append((covers, {j for j, m in zip(ci, mass) if m < 0.95},
                                                sizes[IDX95]))
        n_decl, n_cov, n_cap, size_sum = [], [], [], []
        for sid in reps["sample_id"]:
            comps = per_rep.get(sid, [])
            n_decl.append(len(comps))
            n_cov.append(sum(c for c, _, _ in comps))
            n_cap.append(len(set().union(*(s for _, s, _ in comps))) if comps else 0)
            size_sum.append(sum(z for _, _, z in comps))
        rep_rows.append(pl.DataFrame({**reps, "n_declared": n_decl, "n_covering": n_cov,
                                      "n_captured": n_cap, "size_sum": size_sum}).with_columns(
            *[pl.lit(v).alias(k) for k, v in key.items()]))
    bins = pl.DataFrame(bin_rows, infer_schema_length=None)
    return bins, pl.concat(rep_rows, how="diagonal_relaxed"), pl.DataFrame(comp_rows, infer_schema_length=None)


def load(sources=SOURCES) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(bins, reps, comps); see the module docstring. Every frame carries a `design` column."""
    files = _files(sources)
    stem = os.path.join(CACHE, f"truth_v2_{_fingerprint(files)}")
    if os.path.exists(stem + "_comps.parquet"):
        out = tuple(pl.read_parquet(f"{stem}_{n}.parquet") for n in ("bins", "reps", "comps"))
    else:
        out = _reduce(files)
        os.makedirs(CACHE, exist_ok=True)
        for n, df in zip(("bins", "reps", "comps"), out):
            df.write_parquet(f"{stem}_{n}.parquet")
    return tuple(df.with_columns(pl.col("exp").replace_strict(DESIGN).alias("design")) for df in out)


def arms(df: pl.DataFrame) -> list[str]:
    have = set(df["method"].unique().to_list())
    return [m for m in R.METHODS if m in have]


# ------------------------------------------------------------------------------ PIP calibration
def _coarse(counts: np.ndarray, causal: np.ndarray, k: int = 10):
    """200 bins -> 20: (bin midpoints, causal share, feature count)."""
    n, c = counts.reshape(-1, k).sum(1), causal.reshape(-1, k).sum(1)
    mid = (np.arange(len(n)) + 0.5) / len(n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return mid, np.where(n > 0, c / n, np.nan), n


def calibration_figure(bins: pl.DataFrame, L: int, panel: float = 1.45):
    """Rows = arms, columns = designs: share of features that are causal per PIP bin (20 bins)
    against the bin's PIP, non-null cells pooled. Dot area grows with the bin's feature count
    (square root, relative to the panel's largest bin)."""
    import matplotlib.pyplot as plt
    d = bins.filter((pl.col("L") == L) & ~pl.col("null"))
    ms = arms(d)
    fig, axes = plt.subplots(len(ms), len(EXPS), figsize=(panel * len(EXPS) + 0.9, panel * len(ms) + 0.4),
                             sharex=True, sharey=True, squeeze=False)
    for i, m in enumerate(ms):
        for j, exp in enumerate(EXPS):
            ax = axes[i, j]
            g = d.filter((pl.col("method") == m) & (pl.col("exp") == exp))
            ax.plot([0, 1], [0, 1], ls=":", c="0.5", lw=0.8, zorder=0)
            if g.height:
                cnt = np.array(g["counts"].to_list()).sum(0).astype(float)
                cau = np.array(g["causal"].to_list()).sum(0).astype(float)
                mid, rate, n = _coarse(cnt, cau)
                ok = n > 0
                ax.scatter(mid[ok], rate[ok], s=4 + 36 * np.sqrt(n[ok] / n[ok].max()),
                           color=R.METHOD_COLOR[m], edgecolor="white", linewidth=0.3, zorder=3)
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=6.5)
            if i == 0:
                ax.set_title(DESIGN[exp], fontsize=8)
            if i == len(ms) - 1:
                ax.set_xlabel("PIP", fontsize=7.5)
            if j == 0:
                ax.set_ylabel(R.METHOD_LABEL[m], fontsize=7.5, color=R.METHOD_COLOR[m])
    fig.supylabel("share of features that are causal", fontsize=8)
    fig.tight_layout()
    return fig


def calibration_table(bins: pl.DataFrame, lo: float = 0.9) -> pl.DataFrame:
    """Per (design, L, arm), features with PIP >= lo, non-null cells: mean PIP (from bin
    midpoints) and the share that are causal."""
    k = int(round(lo * NBIN))
    mids = (np.arange(NBIN) + 0.5) / NBIN
    out = []
    for (design, L, m), g in bins.filter(~pl.col("null")).group_by("design", "L", "method"):
        cnt = np.array(g["counts"].to_list()).sum(0)[k:].astype(float)
        cau = np.array(g["causal"].to_list()).sum(0)[k:].astype(float)
        n = cnt.sum()
        out.append({"design": design, "L": L, "method": m, "n": n,
                    "pip": float((cnt * mids[k:]).sum() / n) if n else None,
                    "causal": float(cau.sum() / n) if n else None})
    return pl.DataFrame(out)


# ------------------------------------------------------------------------------ FDP / power
def fdp_power_figure(bins: pl.DataFrame, max_fdp: float = 1.0):
    """Rows = L (5, 1), columns = designs: power (share of causals with PIP >= t) against FDP
    (share of features with PIP >= t that are not causal), sweeping t over the 200 bin edges,
    non-null cells pooled. The dot marks t = 0.95. x runs to `max_fdp`; below 1, y runs to the
    largest power reached within it (shared across panels)."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(EXPS), figsize=(6.5, 4.6), sharex=True, sharey=True, squeeze=False)
    k95 = int(round(0.95 * NBIN))
    top = 0.0
    for i, L in enumerate((5, 1)):
        d = bins.filter((pl.col("L") == L) & ~pl.col("null"))
        for j, exp in enumerate(EXPS):
            ax = axes[i, j]
            for m in arms(d):
                g = d.filter((pl.col("method") == m) & (pl.col("exp") == exp))
                if g.height == 0:
                    continue
                cnt = np.array(g["counts"].to_list()).sum(0).astype(float)
                cau = np.array(g["causal"].to_list()).sum(0).astype(float)
                sel, tp = np.cumsum(cnt[::-1])[::-1], np.cumsum(cau[::-1])[::-1]
                power = tp / max(cau.sum(), 1)
                fdp = np.where(sel > 0, (sel - tp) / np.maximum(sel, 1), np.nan)
                ax.plot(fdp, power, color=R.METHOD_COLOR[m], lw=1.1, label=R.METHOD_LABEL[m])
                vis = power[fdp <= max_fdp]
                top = max(top, float(vis.max()) if vis.size else 0.0)
                ax.scatter([fdp[k95]], [power[k95]], s=14, color=R.METHOD_COLOR[m],
                           marker=R.METHOD_MARKER[m], edgecolor="white", linewidth=0.4, zorder=4)
            ax.axvline(0.05, color="0.6", lw=0.6, ls=":")
            ax.set_xlim(0, max_fdp)
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(DESIGN[exp], fontsize=8)
            if i == 1:
                ax.set_xlabel("FDP", fontsize=7.5)
            if j == 0:
                ax.set_ylabel(f"L = {L}\npower", fontsize=8)
    axes[0, 0].set_ylim(0, 1.02 if max_fdp >= 1 else min(1.02, 1.08 * top))
    handles = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(handles.values(), handles.keys(), loc="upper center", ncol=min(len(handles), 5),
               fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9 if len(handles) > 5 else 0.94))
    return fig


def threshold_table(reps: pl.DataFrame, t: float) -> pl.DataFrame:
    """Per (design, L, arm) at PIP threshold t: FDR = mean over non-null fits of the fit's FDP
    (0 when nothing is selected), power = share of causals selected, and the share of null fits
    that select anything."""
    sel, tp = pl.col(f"sel_{t:g}"), pl.col(f"tp_{t:g}")
    fdp = pl.when(sel > 0).then((sel - tp) / sel).otherwise(0.0)
    sig = (reps.filter(~pl.col("null")).group_by("design", "L", "method")
           .agg(fdp.mean().alias("fdr"), (tp.sum() / pl.col("n_causal").sum()).alias("power")))
    nul = (reps.filter(pl.col("null")).group_by("design", "L", "method")
           .agg((sel > 0).mean().alias("null_any")))
    return sig.join(nul, on=["design", "L", "method"], how="left")


# ------------------------------------------------------------------------------ credible sets
def cs_table(reps: pl.DataFrame, comps: pl.DataFrame) -> pl.DataFrame:
    """Per (design, L, arm), declared 95% CSs (component log BF >= 2) in non-null cells:
    coverage = covering / declared CSs, power = captured causals / causals, size = mean and
    median CS size, cs_per_fit = declared CSs per fit; null_cs_per_fit = declared CSs per null
    fit (every one is a false discovery)."""
    by = ["design", "L", "method"]
    c = (comps.filter(~pl.col("null")).group_by(by)
         .agg(pl.col("covers").mean().alias("coverage"), pl.col("size").mean().alias("mean_size"),
              pl.col("size").median().alias("median_size")))
    r = (reps.filter(~pl.col("null")).group_by(by)
         .agg((pl.col("n_captured").sum() / pl.col("n_causal").sum()).alias("power"),
              pl.col("n_declared").mean().alias("cs_per_fit")))
    n = (reps.filter(pl.col("null")).group_by(by)
         .agg(pl.col("n_declared").mean().alias("null_cs_per_fit")))
    return r.join(c, on=by, how="left").join(n, on=by, how="left")


def cs_reliability_figure(comps: pl.DataFrame, beta_lo: float = 0.5):
    """Rows = L (5, 1), columns = designs: coverage of declared CSs at nominal level beta (share
    whose beta-CS contains a causal, from best_mass < beta) against beta, non-null cells.
    Below the diagonal = coverage under nominal."""
    import matplotlib.pyplot as plt
    grid = np.linspace(beta_lo, 0.99, 50)
    fig, axes = plt.subplots(2, len(EXPS), figsize=(6.5, 4.6), sharex=True, sharey=True, squeeze=False)
    for i, L in enumerate((5, 1)):
        d = comps.filter((pl.col("L") == L) & ~pl.col("null"))
        for j, exp in enumerate(EXPS):
            ax = axes[i, j]
            for m in arms(d):
                bm = d.filter((pl.col("method") == m) & (pl.col("exp") == exp))["best_mass"]
                bm = bm.fill_null(np.inf).to_numpy()   # no causal in the CS's alpha support: never covers
                if bm.size == 0:
                    continue
                ax.plot(grid, [(bm < b).mean() for b in grid], color=R.METHOD_COLOR[m], lw=1.1,
                        label=R.METHOD_LABEL[m])
            ax.plot([beta_lo, 1], [beta_lo, 1], ls=":", c="0.4", lw=0.8)
            ax.set_xlim(beta_lo, 1)
            ax.set_ylim(beta_lo, 1)
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(DESIGN[exp], fontsize=8)
            if i == 1:
                ax.set_xlabel(r"nominal level $\beta$", fontsize=7.5)
            if j == 0:
                ax.set_ylabel(f"L = {L}\ncoverage", fontsize=8)
    handles = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(handles.values(), handles.keys(), loc="upper center", ncol=min(len(handles), 5),
               fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9 if len(handles) > 5 else 0.94))
    return fig


def md(tab: pl.DataFrame, value: str, fmt: str, legend: str = "") -> str:
    """Markdown table: rows = design (Binary, Binary-Block, Gaussian) x L (5, 1), columns = arms
    in R.METHODS order; '-' where an arm has no value."""
    ms = arms(tab)
    lines = ["| design | L | " + " | ".join(R.METHOD_LABEL[m] for m in ms) + " |",
             "|---|---|" + "---|" * len(ms)]
    for exp in EXPS:
        for L in (5, 1):
            g = tab.filter((pl.col("design") == DESIGN[exp]) & (pl.col("L") == L))
            if g.height == 0:
                continue
            cells = []
            for m in ms:
                x = g.filter(pl.col("method") == m)[value]
                cells.append("-" if x.len() == 0 or x[0] is None else fmt.format(x[0]))
            lines.append(f"| {DESIGN[exp]} | {L} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + (f"\n\n{legend}" if legend else "")


# (key, row label, (numerator, denominator) reps columns, null fits?). Estimate = sum(num) / sum(den)
# over replicates, or mean(num) when den is None.
CS_DOTS = [
    ("coverage", "coverage", ("n_covering", "n_declared"), False),
    ("power", "power", ("n_captured", "n_causal"), False),
    ("size", "mean CS size", ("size_sum", "n_declared"), False),
    ("null_cs", "declared CSs\nper null fit", ("n_declared", None), True),
]


def _boot(num: np.ndarray, den: np.ndarray | None, n_boot: int = 1000, seed: int = 0):
    """Estimate sum(num) / sum(den) (or mean(num)) and a 95% interval from resampling replicates."""
    rng = np.random.default_rng(seed)
    den = np.ones_like(num) if den is None else den
    if den.sum() == 0:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    with np.errstate(invalid="ignore", divide="ignore"):
        b = num[idx].sum(1) / den[idx].sum(1)
    return float(num.sum() / den.sum()), float(np.nanquantile(b, 0.025)), float(np.nanquantile(b, 0.975))


# ------------------------------------------------------------------------------ vs CAVI-Q2, per scenario
_SCEN = ["exp", "design", "L"] + [c for c in CELL if c != "null"]


def scenario_frame(reps: pl.DataFrame) -> pl.DataFrame:
    """One row per (signal scenario, arm): coverage, power and mean size of declared 95% CSs, and
    each as a difference from CAVI-Q2 in the same scenario (size as a ratio). A scenario is one
    cell of the grid (design, L, sweep value, T, gap); every arm sees the same simulated
    replicates."""
    df = (reps.filter(~pl.col("null")).group_by(_SCEN + ["method"])
          .agg((pl.col("n_covering").sum() / pl.col("n_declared").sum()).alias("coverage"),
               (pl.col("n_captured").sum() / pl.col("n_causal").sum()).alias("power"),
               (pl.col("size_sum").sum() / pl.col("n_declared").sum()).alias("size"))
          .with_columns(pl.col(c).fill_nan(None) for c in ("coverage", "size")))
    ref = df.filter(pl.col("method") == "cavi").select(
        _SCEN + [pl.col(c).alias(f"{c}_cavi") for c in ("coverage", "power", "size")])
    return (df.filter(pl.col("method") != "cavi").join(ref, on=_SCEN, how="inner", nulls_equal=True)
            .with_columns((pl.col("coverage") - pl.col("coverage_cavi")).alias("d_coverage"),
                          (pl.col("power") - pl.col("power_cavi")).alias("d_power"),
                          (pl.col("size") / pl.col("size_cavi")).alias("r_size")))


VS_CAVI = [
    ("d_coverage", "coverage\n− CAVI-Q2", False),
    ("d_power", "power\n− CAVI-Q2", False),
    ("r_size", "mean CS size\n÷ CAVI-Q2", True),
]


def vs_cavi_table(sf: pl.DataFrame, col: str, ratio: bool = False, digits: int = 3) -> pl.DataFrame:
    """Per (design, L, arm): 'median [min, max]' of `col` over scenarios."""
    f = f"{{:.{digits}f}}"
    out = []
    for (design, L, m), g in sf.group_by("design", "L", "method"):
        y = g[col].drop_nulls().to_numpy()
        y = y[np.isfinite(y)]
        if y.size == 0:
            continue
        med = float(np.exp(np.median(np.log(y)))) if ratio else float(np.median(y))
        out.append({"design": design, "L": L, "method": m,
                    "cell": f"{f.format(med)} [{f.format(y.min())}, {f.format(y.max())}]"})
    return pl.DataFrame(out)


def _log_axis(ax):
    """Log y with plain tick labels at 1, 2, 5 x powers of ten."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _arm_grid(rows: list[tuple[str, bool]], ms_by_L: dict[int, list[str]], title: str):
    """Figure with one row per (y label, log y) and two columns, L = 5 then L = 1, each as wide as
    its arm count; x = arm. Returns (fig, axes)."""
    import matplotlib.pyplot as plt
    Ls = [L for L in (5, 1) if ms_by_L.get(L)]
    fig, axes = plt.subplots(len(rows), len(Ls), figsize=(6.5, 1.75 * len(rows) + 1.2), sharey="row",
                             squeeze=False,
                             gridspec_kw={"width_ratios": [len(ms_by_L[L]) + 0.6 for L in Ls]})
    for j, L in enumerate(Ls):
        ms = ms_by_L[L]
        for i, (label, log) in enumerate(rows):
            ax = axes[i, j]
            if log:
                _log_axis(ax)
            ax.grid(True, axis="y", alpha=0.25)
            ax.tick_params(labelsize=6.5)
            ax.set_xlim(-0.6, len(ms) - 0.4)
            ax.set_xticks(range(len(ms)))
            ax.set_xticklabels([R.METHOD_LABEL[m] for m in ms] if i == len(rows) - 1 else [],
                               rotation=60, ha="right", fontsize=6.5)
            if i == 0:
                ax.set_title(f"L = {L}", fontsize=8)
            if j == 0:
                ax.set_ylabel(label, fontsize=7.5)
    if title:
        fig.suptitle(title, fontsize=8.5)
    return fig, axes, Ls


def cs_dot_figure(reps: pl.DataFrame, exp: str):
    """One design. Rows = coverage, power, mean size of declared 95% CSs (signal fits) and
    declared CSs per null fit; columns = L = 5 and L = 1; x = arm. Bars are 95% intervals from
    resampling replicates."""
    d = reps.filter(pl.col("exp") == exp)
    ms_by_L = {L: arms(d.filter(pl.col("L") == L)) for L in (5, 1)}
    fig, axes, Ls = _arm_grid([(label, key == "size") for key, label, _, _ in CS_DOTS], ms_by_L,
                              DESIGN[exp])
    size_hi = 1.0
    for j, L in enumerate(Ls):
        for i, (key, _label, (num, den), null) in enumerate(CS_DOTS):
            ax = axes[i, j]
            for k, m in enumerate(ms_by_L[L]):
                g = d.filter((pl.col("method") == m) & (pl.col("L") == L) & (pl.col("null") == null))
                if g.height == 0:
                    continue
                est, lo, hi = _boot(g[num].to_numpy().astype(float),
                                    None if den is None else g[den].to_numpy().astype(float))
                if np.isnan(est):
                    continue
                ax.errorbar([k], [est], yerr=[[est - lo], [hi - est]], fmt=R.METHOD_MARKER[m], ms=4.5,
                            color=R.METHOD_COLOR[m], lw=0.9, capsize=0)
                if key == "size":
                    size_hi = max(size_hi, hi)
            if key == "coverage":
                ax.axhline(0.95, color="0.5", lw=0.7, ls="--")
    for i, (key, *_r) in enumerate(CS_DOTS):
        if key == "size":   # the row shares y across L
            axes[i, 0].set_ylim(1, 1.15 * size_hi)
    fig.tight_layout()
    return fig


DESIGN_MARKER = {"022": "o", "024": "s", "023": "^"}


VS_CAVI_ABS = [("coverage_cavi", "coverage", False), ("power_cavi", "power", False),
               ("size_cavi", "mean CS size", True)]   # CAVI-Q2's own value, row-aligned with VS_CAVI


def vs_cavi_figure(sf: pl.DataFrame, seed: int = 0):
    """All designs pooled. For each L (5, then 1) two columns: CAVI-Q2's own coverage, power and
    mean CS size per scenario, then each arm minus CAVI-Q2 (size: over). One point per scenario;
    colour = arm, marker = design (offset, jittered); black bar = median over all scenarios of
    all designs (geometric for sizes). Rows are aligned; CAVI columns share y, as do the arm
    columns."""
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    ms_by_L = {L: arms(sf.filter(pl.col("L") == L)) for L in (5, 1)}
    Ls = [L for L in (5, 1) if ms_by_L[L]]
    widths = []
    for L in Ls:
        widths += [1.25, len(ms_by_L[L]) + 0.6]
    fig, axes = plt.subplots(len(VS_CAVI), 2 * len(Ls), figsize=(6.5, 1.7 * len(VS_CAVI) + 1.4),
                             squeeze=False, layout="constrained", gridspec_kw={"width_ratios": widths})
    for i in range(len(VS_CAVI)):
        for c in range(2, 2 * len(Ls)):
            axes[i, c].sharey(axes[i, c % 2])
    offs = {e: o for e, o in zip(EXPS, (-0.24, 0.0, 0.24))}

    def strip(ax, x, ys_by_exp, color, ratio):
        pooled = []
        for exp, y in ys_by_exp.items():
            y = y[np.isfinite(y)]
            if y.size == 0:
                continue
            pooled.append(y)
            ax.scatter(x + offs[exp] + rng.uniform(-0.06, 0.06, y.size), y, s=9, marker=DESIGN_MARKER[exp],
                       facecolor="none", edgecolor=color, linewidth=0.6, alpha=0.8, zorder=2)
        if pooled:
            y = np.concatenate(pooled)
            med = np.exp(np.median(np.log(y))) if ratio else np.median(y)
            ax.plot([x - 0.36, x + 0.36], [med, med], color="k", lw=1.4, zorder=1)

    ref = sf.unique(subset=_SCEN, keep="first")   # one row per scenario carries CAVI-Q2's values
    last = len(VS_CAVI) - 1
    for b, L in enumerate(Ls):
        ca, cr = 2 * b, 2 * b + 1
        dref, d, ms = ref.filter(pl.col("L") == L), sf.filter(pl.col("L") == L), ms_by_L[L]
        for i, ((acol, alabel, alog), (col, label, ratio)) in enumerate(zip(VS_CAVI_ABS, VS_CAVI)):
            ax = axes[i, ca]
            strip(ax, 0, {e: dref.filter(pl.col("exp") == e)[acol].drop_nulls().to_numpy() for e in EXPS},
                  R.METHOD_COLOR["cavi"], alog)
            if alog and b == 0:
                _log_axis(ax)
            if acol == "coverage_cavi":
                ax.axhline(0.95, color="0.5", lw=0.7, ls="--", zorder=0)
            ax.set_xlim(-0.55, 0.55)
            ax.set_xticks([0])
            ax.set_xticklabels(["CAVI-Q2"] if i == last else [], rotation=60, ha="right", fontsize=6.5)
            if i == 0:
                ax.set_title("CAVI-Q2", fontsize=7.5, color=R.METHOD_COLOR["cavi"])
            ax = axes[i, cr]
            ax.axhline(1.0 if ratio else 0.0, color="0.5", lw=0.7, ls="--", zorder=0)
            for k, m in enumerate(ms):
                g = d.filter(pl.col("method") == m)
                strip(ax, k, {e: g.filter(pl.col("exp") == e)[col].drop_nulls().to_numpy() for e in EXPS},
                      R.METHOD_COLOR[m], ratio)
            if ratio and b == 0:
                _log_axis(ax)
            ax.set_xlim(-0.6, len(ms) - 0.4)
            ax.set_xticks(range(len(ms)))
            ax.set_xticklabels([R.METHOD_LABEL[m] for m in ms] if i == last else [],
                               rotation=60, ha="right", fontsize=6.5)
            if i == 0:
                ax.set_title(f"L = {L}: arm vs CAVI-Q2", fontsize=7.5)
            for a_, lab in ((axes[i, ca], alabel), (axes[i, cr], label.split("\n")[-1])):
                a_.grid(True, axis="y", alpha=0.25)
                a_.tick_params(labelsize=6.5)
                a_.tick_params(labelleft=True)   # every column: its neighbour is on another scale
                if b == 0:
                    a_.set_ylabel(lab, fontsize=7.5)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="0.3", marker=DESIGN_MARKER[e], mfc="none", ls="", ms=5, label=DESIGN[e])
         for e in EXPS] + [Line2D([], [], color="k", lw=1.4, label="median, all scenarios")]
    fig.legend(handles=h, loc="outside upper center", ncol=4, fontsize=7, frameon=False)
    return fig
