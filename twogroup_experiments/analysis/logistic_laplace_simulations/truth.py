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

* bins: summed PIP bins per (exp, L, method, cell).
* reps: one row per (exp, L, method, cell, replicate): causal count, features and causals at or
        above each PIP threshold in THRESHOLDS, declared components, declared components that
        cover a causal, causals captured by a declared 95% CS.
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
            per_rep.setdefault(sid, []).append((covers, {j for j, m in zip(ci, mass) if m < 0.95}))
        n_decl, n_cov, n_cap = [], [], []
        for sid in reps["sample_id"]:
            comps = per_rep.get(sid, [])
            n_decl.append(len(comps))
            n_cov.append(sum(c for c, _ in comps))
            n_cap.append(len(set().union(*(s for _, s in comps))) if comps else 0)
        rep_rows.append(pl.DataFrame({**reps, "n_declared": n_decl, "n_covering": n_cov,
                                      "n_captured": n_cap}).with_columns(
            *[pl.lit(v).alias(k) for k, v in key.items()]))
    bins = pl.DataFrame(bin_rows, infer_schema_length=None)
    return bins, pl.concat(rep_rows, how="diagonal_relaxed"), pl.DataFrame(comp_rows, infer_schema_length=None)


def load(sources=SOURCES) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(bins, reps, comps); see the module docstring. Every frame carries a `design` column."""
    files = _files(sources)
    stem = os.path.join(CACHE, f"truth_{_fingerprint(files)}")
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
def fdp_power_figure(bins: pl.DataFrame):
    """Rows = L (5, 1), columns = designs: power (share of causals with PIP >= t) against FDP
    (share of features with PIP >= t that are not causal), sweeping t over the 200 bin edges,
    non-null cells pooled. The dot marks t = 0.95."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(EXPS), figsize=(6.5, 4.6), sharex=True, sharey=True, squeeze=False)
    k95 = int(round(0.95 * NBIN))
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
                ax.scatter([fdp[k95]], [power[k95]], s=14, color=R.METHOD_COLOR[m],
                           marker=R.METHOD_MARKER[m], edgecolor="white", linewidth=0.4, zorder=4)
            ax.axvline(0.05, color="0.6", lw=0.6, ls=":")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(DESIGN[exp], fontsize=8)
            if i == 1:
                ax.set_xlabel("FDP", fontsize=7.5)
            if j == 0:
                ax.set_ylabel(f"L = {L}\npower", fontsize=8)
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
