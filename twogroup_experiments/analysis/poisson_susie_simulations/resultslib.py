"""Analysis + plotting layer for experiment 019 (logistic-SuSiE approximations).

Design goal: **slice and facet freely**. Every figure function takes the same two
knobs so a layout can be re-cut without touching plot code:

  * ``filt``  -- a dict ``{coord_col: value | [values]}`` (or a ``pl.Expr``) applied
    to the atomic frame BEFORE anything is computed. Subset to single-effect cells,
    one design, one intercept, one ``fit_L``, etc.
  * ``facet`` -- a list of coordinate columns that become panels. Everything NOT in
    ``facet`` (and not ``method``) is *aggregated over*. ``facet=["design"]`` pools
    the T ladder within a design; ``facet=["T","b0"]`` makes a T x b0 grid pooled
    over design. ``facet=[]`` is one fully-aggregated panel.

``method`` is always the series/colour (line plots) or the x-axis category (dotplots),
never a facet -- the whole experiment is a method comparison.

Coordinate columns carried on every atomic row (from ``load_results``):
    design, design_n, b0, T, Lstar, gap, fit_L, corr, density, rho, is_null

Atomic grains:
    pip frame -- one row per (cell, method, replicate); 200-bin PIP arrays + causal PIPs.
    cs  frame -- one row per (cell, method, replicate, component l); per-component
                 ser_log_bf, cs_size95, captured95, best_mass, and the full cs_sizes /
                 mass_above_causal lists over the CS_BETA_GRID.

Coverage convention (matches analysis/op_common.py): a causal is inside the beta-CS
iff its ``mass_above_causal < beta``; ``cs_sizes[94]`` is the 95% CS size.
"""
from __future__ import annotations

import glob
import hashlib
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import polars as pl

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import load_results as lr

_CACHE = os.path.join(_HERE, ".cache")


def _sc_pairs(sc: str, results_root: str):
    """(batch_hash, method_hash) pairs for a supercollection (empty if it doesn't exist)."""
    import experiments.loader as loader
    try:
        colls = loader.collection_method_pairs(loader.load_config(), sc).values()
    except Exception:
        return []
    out, seen = [], set()
    for coll in colls:
        for bh, mh, *_ in coll["pairs"]:
            if (bh, mh) not in seen:
                seen.add((bh, mh))
                out.append((bh, mh))
    return out


def _fingerprint(paths) -> str:
    """Short hash over (path, size) of existing input files -> cache key. Size not mtime:
    content-addressed fits recomputed in place keep the same size (stable cache), while a
    changed set of fits or changed content shifts the size (correct invalidation)."""
    h = hashlib.sha1()
    for p in sorted(paths):
        try:
            h.update(f"{p}:{os.stat(p).st_size}".encode())
        except FileNotFoundError:
            continue
    return h.hexdigest()[:16]


def _cached(name: str, sig_paths, builder):
    """Return builder() but memoized to .cache/<name>_<fingerprint>.parquet. The
    fingerprint of the input files auto-invalidates when reductions/fits change."""
    os.makedirs(_CACHE, exist_ok=True)
    cf = os.path.join(_CACHE, f"{name}_{_fingerprint(sig_paths)}.parquet")
    if os.path.exists(cf):
        return pl.read_parquet(cf)
    df = builder()
    for old in glob.glob(os.path.join(_CACHE, f"{name}_*.parquet")):
        if old != cf:
            os.remove(old)
    if not df.is_empty():
        df.write_parquet(cf)
    return df

# --------------------------------------------------------------------------- #
# Methods: palette + display + order. gIBSS / CAVI-cf / global-JJ / local-JJ.
# --------------------------------------------------------------------------- #
# Poisson arms: update rule = HUE (gIBSS blue, CAVI vermillion/orange, score purple),
# variational family = SHADE (Q2 dark, Q1 light). The fixed unit-prior (`_pv1`) twins reuse
# their EB twin's colour -- the notebook shows EB and pv1 in SEPARATE panels, never overlaid.
_EB_ORDER = ["q2_gibss", "q1_gibss", "q2_cavi", "q1_cavi", "q2_score"]
METHOD_ORDER = _EB_ORDER + [m + "_pv1" for m in _EB_ORDER]
METHOD_LABEL = {
    "q2_gibss": "gIBSS·Q2", "q1_gibss": "gIBSS·Q1",
    "q2_cavi": "CAVI-cf·Q2", "q1_cavi": "CAVI-sn·Q1",
    "q2_score": "score",
    "q2_gibss_pv1": "gIBSS·Q2 (pv1)", "q1_gibss_pv1": "gIBSS·Q1 (pv1)",
    "q2_cavi_pv1": "CAVI-cf·Q2 (pv1)", "q1_cavi_pv1": "CAVI-sn·Q1 (pv1)",
    "q2_score_pv1": "score (pv1)",
}
_BASE_COLOR = {
    "q2_gibss": "#0072B2",  # blue (dark)  -- gIBSS, Gaussian effect
    "q1_gibss": "#56B4E9",  # sky (light)  -- gIBSS, free-form effect
    "q2_cavi": "#D55E00",   # vermillion   -- CAVI exact offset, Gaussian effect
    "q1_cavi": "#E69F00",   # orange       -- CAVI exact offset, free-form effect
    "q2_score": "#CC79A7",  # reddish purple -- score linearization
}
METHOD_COLOR = {**_BASE_COLOR, **{m + "_pv1": c for m, c in _BASE_COLOR.items()}}


def method_color(m: str) -> str:
    return METHOD_COLOR.get(m, "#888888")


def methods_present(df: pl.DataFrame) -> list[str]:
    have = set(df["method"].unique().to_list())
    return [m for m in METHOD_ORDER if m in have]


# --------------------------------------------------------------------------- #
# CS_BETA_GRID (0.01..0.99, 1.0) and the 95% index.
# --------------------------------------------------------------------------- #
from utils import CS_BETA_GRID  # noqa: E402

BETA_GRID = np.asarray(CS_BETA_GRID.tolist(), dtype=float)
IDX95 = int(np.argmin(np.abs(BETA_GRID - 0.95)))  # == 94


# --------------------------------------------------------------------------- #
# Design labels + coordinate metadata.
# --------------------------------------------------------------------------- #
DESIGN_LABEL = {
    500: "AR1-Gaussian (n=500)",
    1000: "Bin-Markov (n=1000, q=.5)",
    10000: "Bin-Markov (n=10000, q=.05)",
}
_DESIGN_ORDER = {lab: n for n, lab in DESIGN_LABEL.items()}

# per-facet-column ordering key + short axis label (for panel titles)
_COORD_TITLE = {
    "design": lambda v: v,
    "design_n": lambda v: f"n={v}",
    "b0": lambda v: f"b0={v:g}",
    "lambda0": lambda v: ("λ0=-" if v is None else f"λ0={v:g}"),
    "T": lambda v: ("null" if v is None else f"T={v:g}"),
    "Lstar": lambda v: ("null" if v == 0 else f"L*={v}"),
    "gap": lambda v: ("gap=-" if v is None else f"gap={v}"),
    "fit_L": lambda v: f"fit L={v}",
}


def _coord_sort_key(col: str, val):
    if col == "design":
        return _DESIGN_ORDER.get(val, 1 << 30)
    if val is None:
        return 1 << 30  # nulls last
    return val


def coord_title(col: str, val) -> str:
    return _COORD_TITLE.get(col, lambda v: f"{col}={v}")(val)


def _add_coords(df: pl.DataFrame) -> pl.DataFrame:
    """Attach the `design` label + `is_null` (Lstar==0) columns; blank T on nulls."""
    if df.is_empty():
        return df
    return df.with_columns(
        pl.col("design_n").replace_strict(DESIGN_LABEL, default="?").alias("design"),
        (pl.col("Lstar") == 0).alias("is_null"),
    ).with_columns(
        pl.when(pl.col("Lstar") == 0).then(None).otherwise(pl.col("T")).alias("T"),
    )


# --------------------------------------------------------------------------- #
# Loaders (thin wrappers over load_results.reduction_frame with derived columns).
# --------------------------------------------------------------------------- #
def load_pip(sc: str = "021-poisson", results_root: str = "results") -> pl.DataFrame:
    paths = [f"{results_root}/by_batch/{bh}/fits/{mh}/reductions/pip.parquet"
             for bh, mh in _sc_pairs(sc, results_root)]
    return _cached(f"pip_{sc}", paths,
                   lambda: _add_coords(lr.reduction_frame(sc, "pip", results_root)))


def load_cs(sc: str = "021-poisson", results_root: str = "results") -> pl.DataFrame:
    paths = [f"{results_root}/by_batch/{bh}/fits/{mh}/reductions/cs.parquet"
             for bh, mh in _sc_pairs(sc, results_root)]
    return _cached(f"cs_{sc}", paths, lambda: _load_cs_build(sc, results_root))


def _load_cs_build(sc: str, results_root: str) -> pl.DataFrame:
    cs = _add_coords(lr.reduction_frame(sc, "cs", results_root))
    if cs.is_empty():
        return cs
    return cs.with_columns(
        (pl.col("causal_indices").list.len() > 0).alias("has_causal"),
        pl.col("cs_sizes").list.get(IDX95).alias("cs_size95"),
        pl.col("mass_above_causal").list.min().alias("best_mass"),  # best-covered causal
    ).with_columns(
        (pl.col("has_causal") & (pl.col("best_mass") < 0.95)).alias("captured95"),
    )


def elbo_frame(sc="021-poisson", results_root="results") -> pl.DataFrame:
    """Per-fit common-Q2 ELBO (`q2_elbo`), one row per (cell, method, fit_L, replicate), tagged
    with batch_hash + rep so methods can be PAIRED (same sim data). Read from fits.parquet (not
    in the reductions). Q1 arms carry None (a free-form state has no Q2 ELBO). Cached."""
    paths = [f"{results_root}/by_batch/{bh}/fits/{mh}/fits.parquet"
             for bh, mh in _sc_pairs(sc, results_root)]
    return _cached("elbo", paths, lambda: _add_coords(lr.elbo_frame(sc, results_root)))


def load_both(sc="021-poisson", results_root="results") -> tuple[pl.DataFrame, pl.DataFrame]:
    """pip, cs frames for the 021-poisson supercollection (all 20 arms in one tier)."""
    return load_pip(sc, results_root), load_cs(sc, results_root)


# --------------------------------------------------------------------------- #
# Filtering + faceting primitives.
# --------------------------------------------------------------------------- #
def apply_filter(df: pl.DataFrame, filt) -> pl.DataFrame:
    """filt is None, a pl.Expr, or a dict {col: scalar | list}. Unknown cols raise."""
    if filt is None:
        return df
    if isinstance(filt, pl.Expr):
        return df.filter(filt)
    exprs = []
    for col, val in filt.items():
        if col not in df.columns:
            raise KeyError(f"filter column {col!r} not in frame ({df.columns})")
        exprs.append(pl.col(col).is_in(val if isinstance(val, (list, tuple, set)) else [val]))
    return df.filter(pl.all_horizontal(exprs)) if exprs else df


def facet_combos(df: pl.DataFrame, facet: list[str]) -> list[tuple]:
    """Ordered list of value-tuples for the facet columns present in df."""
    if not facet:
        return [()]
    combos = df.select(facet).unique().iter_rows()
    return sorted(combos, key=lambda t: tuple(_coord_sort_key(c, v) for c, v in zip(facet, t)))


def facet_label(facet: list[str], combo: tuple) -> str:
    return " | ".join(coord_title(c, v) for c, v in zip(facet, combo))


def _facet_mask(facet: list[str], combo: tuple) -> pl.Expr:
    if not facet:
        return pl.lit(True)
    parts = [
        (pl.col(c).is_null() if v is None else (pl.col(c) == v))
        for c, v in zip(facet, combo)
    ]
    return pl.all_horizontal(parts)


def _grid_shape(facet: list[str], combos: list[tuple], ncol_max: int = 4):
    """(nrow, ncol, index->(i,j)). 2 facets => rows x cols; else wrap into <=ncol_max."""
    n = len(combos)
    if len(facet) == 2:
        rows = sorted({c[0] for c in combos}, key=lambda v: _coord_sort_key(facet[0], v))
        cols = sorted({c[1] for c in combos}, key=lambda v: _coord_sort_key(facet[1], v))
        pos = {(r, c): (i, j) for i, r in enumerate(rows) for j, c in enumerate(cols)}
        return len(rows), len(cols), (lambda combo: pos[combo]), rows, cols
    ncol = min(max(n, 1), ncol_max)
    nrow = (n + ncol - 1) // ncol
    return nrow, ncol, (lambda combo: divmod(combos.index(combo), ncol)), None, None


def _legend(fig, methods, *, loc="center left", bbox=(0.99, 0.5), extra=None):
    handles = [plt.Line2D([], [], color=method_color(m), lw=2.2, label=METHOD_LABEL.get(m, m))
               for m in methods]
    if extra:
        handles += extra
    fig.legend(handles=handles, loc=loc, bbox_to_anchor=bbox, fontsize=8, frameon=False)


# --------------------------------------------------------------------------- #
# Bin helpers (PIP calibration + power/FDP both consume the 200-bin arrays).
# --------------------------------------------------------------------------- #
_NBIN = 200


def _sum_bins(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros(_NBIN)
    causal = np.zeros(_NBIN)
    for c, cc in zip(df["pip_bin_counts"].to_list(), df["pip_bin_causal_counts"].to_list()):
        counts += np.asarray(c, dtype=float)
        causal += np.asarray(cc, dtype=float)
    return counts, causal


def _coarsen(counts: np.ndarray, causal: np.ndarray, k: int = 10):
    """200 fine bins -> 20 coarse bins. Returns (mid, rate, n_total)."""
    nc = counts.reshape(-1, k).sum(1)
    nca = causal.reshape(-1, k).sum(1)
    ncoarse = len(nc)
    mid = (np.arange(ncoarse) + 0.5) / ncoarse
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(nc > 0, nca / nc, np.nan)
    return mid, rate, nc


def _power_fdp(counts: np.ndarray, causal: np.ndarray):
    """Reverse-cumulative power (recall) + FDP over the 200 PIP thresholds.

    Also returns the index of the most stringent threshold that still declares
    at least one feature (highest PIP bin with rc > 0), or None if nothing is
    ever declared. That point is the FDP/power at the tightest usable cutoff.
    """
    rc = np.cumsum(counts[::-1])[::-1]
    rca = np.cumsum(causal[::-1])[::-1]
    tot = max(causal.sum(), 1)
    power = rca / tot
    fdp = (rc - rca) / np.maximum(rc, 1)
    declared = np.flatnonzero(rc > 0)
    strict = int(declared[-1]) if declared.size else None
    return power, fdp, strict


# =========================================================================== #
# PLOT 1 - PIP calibration. Honors the repo rule: one method per panel.
#   rows = facet combos, cols = methods. Dot area ~ #features in the bin.
# =========================================================================== #
def pip_calibration(pip: pl.DataFrame, *, facet=("design",), filt=None,
                    methods=None, panel=1.9, title=None):
    facet = list(facet)
    d = apply_filter(pip, filt)
    if d.is_empty():
        return _placeholder("no PIP data for this slice")
    methods = [m for m in (methods or methods_present(d))
               if not d.filter(pl.col("method") == m).is_empty()]
    combos = facet_combos(d, facet)
    nrow, ncol = len(combos), len(methods)
    fig, axes = plt.subplots(nrow, ncol, figsize=(panel * ncol, panel * nrow),
                             squeeze=False, sharex=True, sharey=True)
    for i, combo in enumerate(combos):
        dd = d.filter(_facet_mask(facet, combo))
        for j, meth in enumerate(methods):
            ax = axes[i][j]
            counts, causal = _sum_bins(dd.filter(pl.col("method") == meth))
            mid, rate, ntot = _coarsen(counts, causal)
            ok = ntot > 0
            ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=0.8, zorder=0)
            if ok.any():
                sizes = 6 + 40 * ntot[ok] / ntot[ok].max()
                ax.scatter(mid[ok], rate[ok], s=sizes, color=method_color(meth),
                           edgecolor="white", linewidth=0.4, zorder=3)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(True, alpha=0.25)
            if i == 0:
                ax.set_title(METHOD_LABEL.get(meth, meth), fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{facet_label(facet, combo)}\nemp. causal freq.", fontsize=8)
            if i == nrow - 1:
                ax.set_xlabel("PIP", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98 if title else 1])
    return fig


# =========================================================================== #
# PLOT 2 - FDP vs power tradeoff (PIP sweep). Methods overlaid per panel.
# =========================================================================== #
def power_fdp(pip: pl.DataFrame, *, facet=("design",), filt=None, methods=None,
              max_fdp=0.5, panel=(3.4, 3.0), ncol_max=4, title=None):
    facet = list(facet)
    d = apply_filter(pip, filt)
    if d.is_empty():
        return _placeholder("no PIP data for this slice")
    methods = [m for m in (methods or methods_present(d))
               if not d.filter(pl.col("method") == m).is_empty()]
    combos = facet_combos(d, facet)
    nrow, ncol, place, *_ = _grid_shape(facet, combos, ncol_max)
    fig, axes = plt.subplots(nrow, ncol, figsize=(panel[0] * ncol, panel[1] * nrow),
                             squeeze=False, sharex=True, sharey=True)
    used = set()
    for combo in combos:
        i, j = place(combo); used.add((i, j))
        ax = axes[i][j]
        dd = d.filter(_facet_mask(facet, combo))
        for meth in methods:
            counts, causal = _sum_bins(dd.filter(pl.col("method") == meth))
            if causal.sum() == 0:
                continue
            power, fdp, strict = _power_fdp(counts, causal)
            c = method_color(meth)
            ax.plot(fdp, power, color=c, lw=1.6)
            if strict is not None:
                ax.scatter(fdp[strict], power[strict], s=42, color=c,
                           edgecolor="white", linewidth=0.8, zorder=5)
        ax.set_xlim(0, max_fdp); ax.set_ylim(0, 1.02); ax.grid(True, alpha=0.3)
        ax.set_title(facet_label(facet, combo), fontsize=9)
        if j == 0:
            ax.set_ylabel("power (recall)", fontsize=9)
        if i == nrow - 1:
            ax.set_xlabel("FDP", fontsize=9)
    for i in range(nrow):
        for j in range(ncol):
            if (i, j) not in used:
                axes[i][j].axis("off")
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.86, 0.98 if title else 1])
    _legend(fig, methods)
    return fig


# =========================================================================== #
# ROC helper (from op_common) - sweep score high->low; returns (fpr, tpr, auc).
# =========================================================================== #
def _roc(scores, labels):
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, float)
    n_pos, n_neg = labels.sum(), (1 - labels).sum()
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(-scores)
    ls = labels[order]
    tpr = np.concatenate([[0.0], np.cumsum(ls) / n_pos])
    fpr = np.concatenate([[0.0], np.cumsum(1 - ls) / n_neg])
    ranks = scores.argsort(kind="mergesort").argsort() + 1
    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return fpr, tpr, float(auc)


def _pr(scores, labels):
    """Precision-recall of ranking `scores` (descending) against binary `labels`.
    Returns (recall, precision, average_precision, prevalence) or None if degenerate.
    AP is the sklearn step estimator (mean precision at the positives)."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, float)
    n_pos, n = labels.sum(), len(labels)
    if n_pos == 0 or n_pos == n:
        return None
    ls = labels[np.argsort(-scores)]
    tp = np.cumsum(ls)
    precision = tp / np.arange(1, n + 1)
    recall = tp / n_pos
    ap = float(np.sum(precision * ls) / n_pos)
    # prepend the (recall=0) edge so the curve starts cleanly at the first declaration
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[precision[0]], precision])
    return recall, precision, ap, float(n_pos / n)


# =========================================================================== #
# PLOT 3 - logBF_SER detection ROC. Rank credible sets by ser_log_bf; a TRUE
#   POSITIVE is a component whose 95% CS covers a causal (captured95). The
#   negative class is genuine false credible sets: the NULL cells (beta=0) plus
#   the rare enriched mis-localizations. Because a 95% CS on p features is large
#   and almost always covers, the enriched-only label saturates (~0.98 positive)
#   and its ROC is degenerate -- the discriminative signal comes from contrasting
#   against the nulls, so we always pool in the matched nulls (same design / b0 /
#   fit_L as the enriched cells in scope), robust to whatever `facet` is chosen.
# =========================================================================== #
# coordinates a null cell shares with an enriched cell (nulls carry no T/gap/Lstar)
_NULL_MATCH = ["design_n", "lambda0", "fit_L"]


# TPR/FPR axis text per label_kind.
_ROC_AXES = {
    "localize": ("TPR (covers a causal)", "FPR (false CS: null / mis-localized)"),
    "detect": ("TPR (enriched flagged)", "FPR (null flagged)"),
    "discovery": ("TPR (localized discovery)", "FPR (null / dispersed / missed)"),
}


def logbf_roc(cs: pl.DataFrame, *, facet=("design",), filt=None, methods=None,
              label_kind="localize", max_size=50, min_class=15, panel=(3.6, 3.6),
              ncol_max=3, title=None):
    """logBF ordering ROC. label_kind:
      "localize"  -- TP = component's 95% CS covers a causal (captured95). Credible-set
                     quality; negatives are false CS (nulls + mis-localizations).
      "detect"    -- TP = the simulation is non-null (~is_null). Pure enrichment
                     detection by logBFSER -- the right lens for single-effect SER fits
                     (one component per dataset).
      "discovery" -- TP = a localized discovery (covers a causal AND 95%-CS size <
                     max_size). The multi-effect lens: with L=10 components most are
                     spurious, so ~is_null would mislabel them; this scores whether
                     logBF ranks real, tight credible sets above null/dispersed/missed.
    In every case the matched null cells (same design / b0 / fit_L) are pooled in as
    part of the negative population, so the ROC never degenerates on an all-positive label.
    """
    facet = list(facet)
    lab_col = {
        "localize": pl.col("captured95"),
        "detect": pl.col("is_null").not_(),
        "discovery": _discovery_label(max_size),
    }[label_kind]
    enr = apply_filter(cs, filt).filter(~pl.col("is_null"))
    nul = cs.filter(pl.col("is_null"))
    if enr.is_empty():
        return _placeholder("no enriched CS data for this slice")
    methods = methods or methods_present(enr)
    combos = facet_combos(enr, facet)
    nrow, ncol, place, *_ = _grid_shape(facet, combos, ncol_max)
    fig, axes = plt.subplots(nrow, ncol, figsize=(panel[0] * ncol, panel[1] * nrow),
                             squeeze=False, sharex=True, sharey=True)
    used = set()
    for combo in combos:
        i, j = place(combo); used.add((i, j))
        ax = axes[i][j]
        ee = enr.filter(_facet_mask(facet, combo))
        # matched nulls: same design / intercept / fit_L as any enriched cell here
        scope = ee.select(_NULL_MATCH).unique()
        nn = nul.join(scope, on=_NULL_MATCH, how="semi")
        dd = pl.concat([ee, nn], how="diagonal_relaxed").with_columns(
            (lab_col if isinstance(lab_col, pl.Expr) else pl.col(lab_col)).alias("_lab"))
        handles = []
        for meth in methods:
            g = dd.filter(pl.col("method") == meth)
            if g.height == 0:
                continue
            labels = g["_lab"].to_numpy().astype(float)
            n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
            res = _roc(g["ser_log_bf"].to_numpy(), labels)
            lab = METHOD_LABEL.get(meth, meth)
            if res is None or min(n_pos, n_neg) < min_class:
                handles.append(plt.Line2D([], [], color=method_color(meth),
                                          label=f"{lab} (n/a)"))
                continue
            fpr, tpr, auc = res
            ax.plot(fpr, tpr, color=method_color(meth), lw=1.7)
            handles.append(plt.Line2D([], [], color=method_color(meth),
                                      label=f"{lab} ({auc:.2f})"))
        ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(True, alpha=0.3)
        ax.set_title(facet_label(facet, combo), fontsize=9)
        ax.legend(handles=handles, fontsize=7, loc="lower right", frameon=False)
        ytxt, xtxt = _ROC_AXES[label_kind]
        if j == 0:
            ax.set_ylabel(ytxt, fontsize=9)
        if i == nrow - 1:
            ax.set_xlabel(xtxt, fontsize=9)
    for i in range(nrow):
        for j in range(ncol):
            if (i, j) not in used:
                axes[i][j].axis("off")
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98 if title else 1])
    return fig


# PLOT 3b - logBF precision-recall. Same populations/labels as logbf_roc, but the
# PR view does not saturate under the heavy negative imbalance of the over-specified
# (L=10) multi-effect fits, where the ROC pins every method at AUC~1. Chance = the
# positive prevalence (grey dashed).
_PR_AXES = {
    "localize": ("precision (covers a causal)", "recall (covered CS found)"),
    "detect": ("precision (enriched)", "recall (enriched flagged)"),
    "discovery": ("precision (localized discovery)", "recall (discoveries found)"),
}


def logbf_pr(cs: pl.DataFrame, *, facet=("design",), filt=None, methods=None,
             label_kind="discovery", max_size=50, min_class=15, panel=(3.6, 3.6),
             ncol_max=3, title=None):
    """Precision-recall of the `ser_log_bf` ranking, mirroring `logbf_roc` (same
    label_kind semantics and matched-null pooling). Legend shows average precision;
    the grey dashed line is the positive prevalence (a no-skill classifier)."""
    facet = list(facet)
    lab_col = {
        "localize": pl.col("captured95"),
        "detect": pl.col("is_null").not_(),
        "discovery": _discovery_label(max_size),
    }[label_kind]
    enr = apply_filter(cs, filt).filter(~pl.col("is_null"))
    nul = cs.filter(pl.col("is_null"))
    if enr.is_empty():
        return _placeholder("no enriched CS data for this slice")
    methods = methods or methods_present(enr)
    combos = facet_combos(enr, facet)
    nrow, ncol, place, *_ = _grid_shape(facet, combos, ncol_max)
    fig, axes = plt.subplots(nrow, ncol, figsize=(panel[0] * ncol, panel[1] * nrow),
                             squeeze=False, sharex=True, sharey=True)
    used = set()
    for combo in combos:
        i, j = place(combo); used.add((i, j))
        ax = axes[i][j]
        ee = enr.filter(_facet_mask(facet, combo))
        scope = ee.select(_NULL_MATCH).unique()
        nn = nul.join(scope, on=_NULL_MATCH, how="semi")
        dd = pl.concat([ee, nn], how="diagonal_relaxed").with_columns(
            (lab_col if isinstance(lab_col, pl.Expr) else pl.col(lab_col)).alias("_lab"))
        handles, prevs = [], []
        for meth in methods:
            g = dd.filter(pl.col("method") == meth)
            if g.height == 0:
                continue
            labels = g["_lab"].to_numpy().astype(float)
            n_pos = int(labels.sum())
            res = _pr(g["ser_log_bf"].to_numpy(), labels)
            lab = METHOD_LABEL.get(meth, meth)
            if res is None or min(n_pos, len(labels) - n_pos) < min_class:
                handles.append(plt.Line2D([], [], color=method_color(meth),
                                          label=f"{lab} (n/a)"))
                continue
            recall, precision, ap, prev = res
            prevs.append(prev)
            ax.plot(recall, precision, color=method_color(meth), lw=1.7)
            handles.append(plt.Line2D([], [], color=method_color(meth),
                                      label=f"{lab} ({ap:.2f})"))
        if prevs:
            ax.axhline(float(np.mean(prevs)), ls=":", c="grey", lw=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(True, alpha=0.3)
        ax.set_title(facet_label(facet, combo), fontsize=9)
        ax.legend(handles=handles, fontsize=7, loc="lower left", frameon=False)
        ytxt, xtxt = _PR_AXES[label_kind]
        if j == 0:
            ax.set_ylabel(ytxt, fontsize=9)
        if i == nrow - 1:
            ax.set_xlabel(xtxt, fontsize=9)
    for i in range(nrow):
        for j in range(ncol):
            if (i, j) not in used:
                axes[i][j].axis("off")
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98 if title else 1])
    return fig


# Coordinate columns that identify one content-addressed simulation cell. The
# per-cell rate is the unit the dotplots spread over (IQR reflects cell-to-cell
# difficulty heterogeneity within a panel; it collapses when a panel is 1 cell).
CELL_COORDS = ["design", "design_n", "lambda0", "T", "Lstar", "gap", "fit_L"]


# =========================================================================== #
# CS per-CELL metrics at a declaration threshold (logBF > min_log_bf), 95%
# level. Used by PLOT 4. One row per (cell, method), pooled over the cell's reps.
#   coverage = fraction of the cell's declared CS that capture a causal
#   power    = fraction of the cell's (rep x causal) captured by a declared CS
#   size     = median 95%-CS size over the cell's declared CS
# =========================================================================== #
def cs_per_cell(cs: pl.DataFrame, *, methods: list[str], min_log_bf=2.0,
                small_size=50) -> pl.DataFrame:
    keys = [c for c in CELL_COORDS if c in cs.columns] + ["method"]
    d = cs.filter(pl.col("method").is_in(methods)).with_columns(
        (pl.col("ser_log_bf") >= min_log_bf).alias("declared"))
    decl = d.filter(pl.col("declared"))
    cov_sz = decl.group_by(keys).agg(
        pl.col("captured95").mean().alias("coverage"),
        pl.col("cs_size95").median().alias("size"),
        # resolution: fraction of declared CS that are small (< small_size), and the
        # median size *among the small ones* -- separates "did it resolve" from "how tight".
        (pl.col("cs_size95") < small_size).mean().alias("frac_small"),
        pl.col("cs_size95").filter(pl.col("cs_size95") < small_size).median().alias("size_small"),
        pl.len().alias("n_declared"),
    )
    # power: fraction of the cell's (rep x causal) captured by ANY declared CS
    ex = (
        d.filter(pl.col("has_causal"))
        .explode(["mass_above_causal", "causal_indices"])
        .with_columns((pl.col("declared") & (pl.col("mass_above_causal") < 0.95)).alias("cap"))
    )
    pwr = (
        ex.group_by(keys + ["sample_id", "batch_hash", "causal_indices"])
        .agg(pl.col("cap").any().alias("cap"))
        .group_by(keys).agg(pl.col("cap").mean().alias("power"))
    )
    return cov_sz.join(pwr, on=keys, how="full", coalesce=True)


METHOD_SHORT = {"globaljj": "gJJ", "localjj": "lJJ", "gibss": "gIB",
                "gibss_profiled": "gIBp", "cavi": "cf"}


def cs_scenario_summary(cs: pl.DataFrame, *, scenario=("design", "b0", "T"), filt=None,
                        methods=None, min_log_bf=2.0) -> pl.DataFrame:
    """Pooled 95%-CS coverage / power / median size per (scenario.., method).

    Declared = ser_log_bf >= min_log_bf. coverage = fraction of declared CS covering a
    causal; power = fraction of (rep x causal) captured by a declared CS; size = median
    95%-CS size over declared CS. All pooled over the scenario's replicates (and any
    coords not in `scenario`, e.g. gap)."""
    scenario = list(scenario)
    d = apply_filter(cs, filt)
    methods = [m for m in (methods or methods_present(d))
               if not d.filter(pl.col("method") == m).is_empty()]
    keys = scenario + ["method"]
    dd = d.filter(pl.col("method").is_in(methods)).with_columns(
        (pl.col("ser_log_bf") >= min_log_bf).alias("declared"))
    decl = dd.filter(pl.col("declared"))
    cov_sz = decl.group_by(keys).agg(
        pl.col("captured95").mean().alias("coverage"),
        pl.col("cs_size95").median().alias("size"),
        pl.len().alias("n_declared"),
    )
    ex = (
        dd.filter(pl.col("has_causal"))
        .explode(["mass_above_causal", "causal_indices"])
        .with_columns((pl.col("declared") & (pl.col("mass_above_causal") < 0.95)).alias("cap"))
    )
    pwr = (
        ex.group_by(keys + ["sample_id", "batch_hash", "causal_indices"])
        .agg(pl.col("cap").any().alias("cap"))
        .group_by(keys).agg(pl.col("cap").mean().alias("power"))
    )
    return cov_sz.join(pwr, on=keys, how="full", coalesce=True)


# (metric key, header label, bold direction). coverage is never bolded (it is shaded
# by severity of under-coverage instead); power bolds the max, size the min.
_CS_TABLE_METRICS = [("coverage", "coverage", "none"),
                     ("power", "power", "max"),
                     ("size", "median size", "min")]

# Under-coverage heat scale (ColorBrewer Reds): (threshold, background, text). A coverage
# below `threshold` gets that shade; deeper red = worse violation. >= 0.95 is unshaded.
_COVERAGE_SHADES = [
    (0.70, "#a50f15", "#ffffff"),   # darkest
    (0.80, "#de2d26", "#ffffff"),
    (0.90, "#fb6a4a", "#ffffff"),
    (0.95, "#fcbba1", "#000000"),   # lightest
]


def _coverage_cell_style(v: float) -> str:
    """Inline td style shading coverage by severity of under-coverage ('' if >= 0.95)."""
    for thr, bg, fg in _COVERAGE_SHADES:
        if v < thr:
            return f' style="background-color:{bg};color:{fg}"'
    return ""


def _best_idx(vals, direction, target=0.95):
    if direction == "none":
        return set()
    present = [(i, v) for i, v in enumerate(vals) if v is not None]
    if not present:
        return set()
    if direction == "max":
        best = max(v for _, v in present)
    elif direction == "min":
        best = min(v for _, v in present)
    else:  # near a target
        best = min((v for _, v in present), key=lambda v: abs(v - target))
    return {i for i, v in present if v == best}


def cs_scenario_tables(cs: pl.DataFrame, *, scenario=("design", "b0", "T"), filt=None,
                       methods=None, min_log_bf=2.0, group="metric") -> str:
    """HTML: one 95%-CS results table per design, with a two-level column header.
    group='metric' -> top row = coverage / power / median size, each spanning the
    methods beneath (best method per metric bolded). group='method' -> top row =
    methods, each spanning the three metrics beneath."""
    scenario = list(scenario)
    summ = cs_scenario_summary(cs, scenario=scenario, filt=filt, methods=methods,
                               min_log_bf=min_log_bf)
    if summ.is_empty():
        return "_(no declared CS for this slice)_"
    methods = methods or methods_present(apply_filter(cs, filt))
    gcol, row_cols = scenario[0], scenario[1:]
    fmt = {"coverage": "{:.2f}", "power": "{:.2f}", "size": "{:g}"}
    mlabels = [METHOD_LABEL.get(m, m) for m in methods]
    metric_titles = [ml for _, ml, _ in _CS_TABLE_METRICS]

    def _td(v, metric, bold):
        if v is None:
            return "<td>-</td>"
        s = fmt[metric].format(v)
        style = _coverage_cell_style(v) if metric == "coverage" else ""
        inner = f"<strong>{s}</strong>" if bold else s
        return f"<td{style}>{inner}</td>"

    # two-level header: (outer spans, inner labels)
    if group == "metric":
        outer, inner, span = metric_titles, mlabels * len(metric_titles), len(methods)
    else:
        outer, inner, span = mlabels, metric_titles * len(methods), len(_CS_TABLE_METRICS)

    blocks = []
    for gval in sorted(summ[gcol].unique().to_list(), key=lambda v: _coord_sort_key(gcol, v)):
        g = summ.filter(pl.col(gcol) == gval)
        h1 = ["<th></th>"] * len(row_cols) + [f'<th colspan="{span}" style="text-align:center">{o}</th>'
                                              for o in outer]
        h2 = [f"<th>{c}</th>" for c in row_cols] + [f"<th>{c}</th>" for c in inner]
        rows_html = []
        for combo in facet_combos(g, row_cols):
            rr = g
            for c, v in zip(row_cols, combo):
                rr = rr.filter(pl.col(c).is_null() if v is None else (pl.col(c) == v))
            vals = {mk: [(rr.filter(pl.col("method") == m)[mk][0]
                          if rr.filter(pl.col("method") == m).height else None)
                         for m in methods] for mk, _, _ in _CS_TABLE_METRICS}
            # rank on the DISPLAYED (rounded) value so display-ties are all bolded, not one.
            rvals = {mk: [None if v is None else float(fmt[mk].format(v)) for v in vals[mk]]
                     for mk, _, _ in _CS_TABLE_METRICS}
            best = {mk: _best_idx(rvals[mk], d) for mk, _, d in _CS_TABLE_METRICS}
            tds = [f"<td>{_short_val(v)}</td>" for v in combo]
            if group == "metric":
                for mk, _, _ in _CS_TABLE_METRICS:
                    for i in range(len(methods)):
                        tds.append(_td(vals[mk][i], mk, i in best[mk]))
            else:
                for i in range(len(methods)):
                    for mk, _, _ in _CS_TABLE_METRICS:
                        tds.append(_td(vals[mk][i], mk, i in best[mk]))
            rows_html.append("<tr>" + "".join(tds) + "</tr>")
        head = gval if gcol == "design" else coord_title(gcol, gval)
        blocks.append(
            f"<p><strong>{head}</strong></p>\n"
            '<div style="overflow-x:auto">\n'
            '<table class="table table-sm">\n<thead>\n'
            f'<tr>{"".join(h1)}</tr>\n<tr>{"".join(h2)}</tr>\n</thead>\n<tbody>\n'
            + "\n".join(rows_html) + "\n</tbody>\n</table>\n</div>")
    return "\n\n".join(blocks)


def elbo_scenario_summary(elbo: pl.DataFrame, *, scenario=("design", "b0", "T"), filt=None,
                          methods=None, ref="cavi") -> pl.DataFrame:
    """Per (scenario.., method): paired ΔELBO vs `ref` (median + IQR, nats) and win rate.
    ELBO is comparable only within a dataset, so we pair on (batch_hash, rep, fit_L)."""
    scenario = list(scenario)
    d = apply_filter(elbo, filt)
    methods = [m for m in (methods or methods_present(d))
               if not d.filter(pl.col("method") == m).is_empty()]
    d = d.filter(pl.col("method").is_in(methods) & pl.col("q2_elbo").is_not_null())
    key = ["batch_hash", "rep", "fit_L"]
    refd = d.filter(pl.col("method") == ref).select(key + ["q2_elbo"]).rename({"q2_elbo": "elbo_ref"})
    mx = d.group_by(key).agg(pl.col("q2_elbo").max().alias("elbo_max"))
    j = (
        d.join(refd, on=key, how="inner").join(mx, on=key, how="inner")
        .with_columns((pl.col("q2_elbo") - pl.col("elbo_ref")).alias("dELBO"),
                      (pl.col("q2_elbo") >= pl.col("elbo_max") - 1e-6).alias("is_win"))
    )
    return j.group_by(scenario + ["method"]).agg(
        pl.col("dELBO").median().alias("med_delbo"),
        pl.col("dELBO").quantile(0.25).alias("q25"),
        pl.col("dELBO").quantile(0.75).alias("q75"),
        pl.col("is_win").mean().alias("win_rate"),
        pl.len().alias("n"),
    )


def elbo_scenario_tables(elbo: pl.DataFrame, *, scenario=("design", "b0", "T"), filt=None,
                         methods=None, ref="cavi", group="metric") -> str:
    """HTML: one common-Q2 ELBO table per design, two-level header. Blocks = median ΔELBO
    vs `ref` (nats; ref is 0 by construction) and win rate (fraction of paired fits where
    the method attains the max ELBO). Best per block bolded (ΔELBO highest, win rate highest)."""
    scenario = list(scenario)
    summ = elbo_scenario_summary(elbo, scenario=scenario, filt=filt, methods=methods, ref=ref)
    if summ.is_empty():
        return "_(no ELBO data for this slice)_"
    methods = methods or methods_present(apply_filter(elbo, filt))
    gcol, row_cols = scenario[0], scenario[1:]
    mlabels = [METHOD_LABEL.get(m, m) for m in methods]
    blocks = [("med_delbo", f"median ΔELBO vs {METHOD_LABEL.get(ref, ref)} (nats)", "{:+.2f}", "max")]
    titles = [t for _, t, _, _ in blocks]

    def _fmt(col, fmtspec, v):
        if v is None:
            return "-"
        return f"{v:.0%}" if fmtspec == "pct" else fmtspec.format(v)

    if group == "metric":
        outer, inner, span = titles, mlabels * len(blocks), len(methods)
    else:
        outer, inner, span = mlabels, titles * len(methods), len(blocks)

    out = []
    for gval in sorted(summ[gcol].unique().to_list(), key=lambda v: _coord_sort_key(gcol, v)):
        g = summ.filter(pl.col(gcol) == gval)
        h1 = ["<th></th>"] * len(row_cols) + [f'<th colspan="{span}" style="text-align:center">{o}</th>' for o in outer]
        h2 = [f"<th>{c}</th>" for c in row_cols] + [f"<th>{c}</th>" for c in inner]
        rows_html = []
        for combo in facet_combos(g, row_cols):
            rr = g
            for c, v in zip(row_cols, combo):
                rr = rr.filter(pl.col(c).is_null() if v is None else (pl.col(c) == v))
            per = {}
            for col, _, fmtspec, direction in blocks:
                vals = [(rr.filter(pl.col("method") == m)[col][0]
                         if rr.filter(pl.col("method") == m).height else None) for m in methods]
                rvals = [None if v is None else float(_fmt(col, fmtspec, v).rstrip("%"))
                         for v in vals]
                per[col] = (vals, _best_idx(rvals, direction))

            def _cells_for(i, col, fmtspec):
                vals, best = per[col]
                s = _fmt(col, fmtspec, vals[i])
                return f"<td>{'<strong>' + s + '</strong>' if i in best else s}</td>"

            tds = [f"<td>{_short_val(v)}</td>" for v in combo]
            if group == "metric":
                for col, _, fmtspec, _ in blocks:
                    tds += [_cells_for(i, col, fmtspec) for i in range(len(methods))]
            else:
                for i in range(len(methods)):
                    tds += [_cells_for(i, col, fmtspec) for col, _, fmtspec, _ in blocks]
            rows_html.append("<tr>" + "".join(tds) + "</tr>")
        head = gval if gcol == "design" else coord_title(gcol, gval)
        out.append(
            f"<p><strong>{head}</strong></p>\n<div style=\"overflow-x:auto\">\n"
            '<table class="table table-sm">\n<thead>\n'
            f'<tr>{"".join(h1)}</tr>\n<tr>{"".join(h2)}</tr>\n</thead>\n<tbody>\n'
            + "\n".join(rows_html) + "\n</tbody>\n</table>\n</div>")
    return "\n\n".join(out)


def _iqr(vals: np.ndarray):
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    return float(np.median(vals)), float(np.percentile(vals, 25)), float(np.percentile(vals, 75))


# =========================================================================== #
# PLOT 4 - coverage / power / size dotplot (median + IQR across the panel's
#   cells) for declared 95% CS (logBF > 2). rows = facet combos, cols = the 3
#   metrics, x = method.
# =========================================================================== #
_METRIC_LABEL = {
    "coverage": "coverage", "power": "power", "size": "median 95%-CS size",
    "frac_small": "frac. CS resolved (<50)", "size_small": "median size",
}
_FRAC_METRICS = {"coverage", "power", "frac_small"}
_SIZE_METRICS = {"size", "size_small"}


def cs_operating_dotplot(cs: pl.DataFrame, *, facet=("design",), filt=None, methods=None,
                         metrics=("coverage", "power", "size_small"),
                         min_log_bf=2.0, small_size=50, nominal=0.95, panel=(2.9, 2.7),
                         title=None, log_size=True):
    facet = list(facet)
    d = apply_filter(cs, filt)
    if d.is_empty():
        return _placeholder("no CS data for this slice")
    methods = [m for m in (methods or methods_present(d))
               if not d.filter(pl.col("method") == m).is_empty()]
    per = cs_per_cell(d, methods=methods, min_log_bf=min_log_bf, small_size=small_size)
    combos = facet_combos(d, facet)
    metrics = list(metrics)
    nrow, ncol = len(combos), len(metrics)
    fig, axes = plt.subplots(nrow, ncol, figsize=(panel[0] * ncol, panel[1] * nrow),
                             squeeze=False, sharex=True)
    xpos = {m: k for k, m in enumerate(methods)}
    for i, combo in enumerate(combos):
        pp = per.filter(_facet_mask(facet, combo)) if facet else per
        for j, metric in enumerate(metrics):
            ax = axes[i][j]
            for meth in methods:
                vals = pp.filter(pl.col("method") == meth)[metric].to_numpy().astype(float)
                med, lo, hi = _iqr(vals)
                if np.isnan(med):
                    continue
                x = xpos[meth]
                ax.errorbar([x], [med], yerr=[[med - lo], [hi - med]], fmt="o",
                            color=method_color(meth), capsize=3, markersize=6, lw=1.6)
            if metric in _FRAC_METRICS:
                ax.set_ylim(-0.03, 1.03)
                if metric == "coverage":
                    ax.axhline(nominal, ls="--", c="grey", lw=0.8)
            elif metric in _SIZE_METRICS and log_size:
                ax.set_yscale("log")
            ax.set_xlim(-0.5, len(methods) - 0.5)
            ax.set_xticks(list(xpos.values()))
            ax.set_xticklabels([METHOD_LABEL.get(m, m) for m in methods], rotation=30,
                               ha="right", fontsize=7)
            ax.grid(True, axis="y", alpha=0.3)
            if i == 0:
                ax.set_title(_METRIC_LABEL.get(metric, metric), fontsize=10)
            if j == 0:
                ax.set_ylabel(facet_label(facet, combo), fontsize=8)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98 if title else 1])
    return fig


# =========================================================================== #
# PLOT 4b - per-method histogram of declared 95%-CS sizes. One square panel per
#   method, filled with the method colour. Pooled over the filter's cells/reps.
# =========================================================================== #
def cs_size_histogram(cs: pl.DataFrame, *, filt=None, methods=None, min_log_bf=2.0,
                      bins=24, log_x=True, panel=2.5, title=None):
    """One square panel per method: histogram of declared (`ser_log_bf` >= min_log_bf)
    95%-CS sizes. Each size bin is split into true positives (the CS covers a causal,
    `captured95`) drawn FILLED in the method colour at the base, with false positives
    (the CS misses -- mislocalized, or a null detection when the slice includes nulls)
    stacked ON TOP as a HOLLOW cap in the same colour (same edge + width, so the two
    segments line up as one bar). The hollow cap is thus the number of false-positive
    credible sets, readable across methods and size. Log-spaced bins by default (CS size
    spans orders of magnitude); dashed grey line = median size; shared y so heights
    compare."""
    d = apply_filter(cs, filt)
    if d.is_empty():
        return _placeholder("no CS data for this slice")
    methods = [m for m in (methods or methods_present(d))
               if not d.filter(pl.col("method") == m).is_empty()]
    d = d.filter(pl.col("method").is_in(methods) & (pl.col("ser_log_bf") >= min_log_bf))
    # per method: (all sizes, TP sizes, FP sizes) -- TP = captured95, FP = not.
    parts = {}
    for m in methods:
        dm = d.filter(pl.col("method") == m)
        allm = dm["cs_size95"].to_numpy().astype(float)
        tp = dm.filter(pl.col("captured95"))["cs_size95"].to_numpy().astype(float)
        fp = dm.filter(~pl.col("captured95"))["cs_size95"].to_numpy().astype(float)
        parts[m] = (allm, tp, fp)
    nonempty = [v for v, _, _ in parts.values() if len(v)]
    allv = np.concatenate(nonempty) if nonempty else np.array([1.0])
    hi = max(2.0, float(np.nanmax(allv)))
    edges = (np.logspace(0, np.log10(hi), bins + 1) if log_x
             else np.linspace(1.0, hi, bins + 1))
    left, width = edges[:-1], np.diff(edges)
    ncol = len(methods)
    fig, axes = plt.subplots(1, ncol, figsize=(panel * ncol, panel), squeeze=False,
                             sharey=True)
    for j, m in enumerate(methods):
        ax = axes[0][j]
        allm, tp, fp = parts[m]
        c = method_color(m)
        if len(allm):
            fp_ct, _ = np.histogram(fp, bins=edges)
            tp_ct, _ = np.histogram(tp, bins=edges)
            # TP: filled base (coloured face + matching coloured edge); FP: hollow, stacked
            # on TOP with the SAME coloured edge + linewidth, so the two segments line up as
            # one bar -- the hollow cap reads the # of false positives.
            ax.bar(left, tp_ct, width=width, align="edge",
                   facecolor=c, edgecolor=c, linewidth=0.7)
            ax.bar(left, fp_ct, width=width, align="edge", bottom=tp_ct,
                   facecolor="none", edgecolor=c, linewidth=0.7)
            ax.axvline(float(np.median(allm)), ls="--", c="grey", lw=0.9)
        if log_x:
            ax.set_xscale("log")
        ax.set_box_aspect(1)                       # square panel regardless of data range
        ax.set_title(METHOD_LABEL.get(m, m), fontsize=10, color=c)
        ax.set_xlabel("95%-CS size", fontsize=8)
        if j == 0:
            ax.set_ylabel("count", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    # one shared legend: filled = covers a causal (TP), hollow = misses (FP)
    tp_key = Patch(facecolor="0.4", edgecolor="0.4", label="covers causal (TP)")
    fp_key = Patch(facecolor="none", edgecolor="0.4", label="misses (FP)")
    axes[0][-1].legend(handles=[tp_key, fp_key], fontsize=7, loc="upper right",
                       frameon=False)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98 if title else 1])
    return fig


def fp_size_table(cs: pl.DataFrame, *, methods=None, filt=None, min_log_bf=2.0,
                  thresholds=(1, 5, 10, 50)) -> str:
    """Markdown table: among each method's FALSE-POSITIVE credible sets (declared at
    `ser_log_bf` >= min_log_bf, and NOT `captured95` -- mislocalized, or a null
    detection when the slice includes nulls), the cumulative fraction whose 95%-CS size
    is <= t, for each t in `thresholds`. A method whose FPs pile up at size<=1 makes
    confident-but-wrong singleton calls; one whose FPs are large is at least honestly
    diffuse. n_FP is the denominator (declared CS that miss)."""
    d = apply_filter(cs, filt)
    methods = [m for m in (methods or methods_present(d))
               if not d.filter(pl.col("method") == m).is_empty()]
    d = d.filter(pl.col("method").is_in(methods) & (pl.col("ser_log_bf") >= min_log_bf))
    fp = d.filter(~pl.col("captured95"))
    hdr = ["Method", "n_FP"] + [f"size<={t}" for t in thresholds]
    lines = ["| " + " | ".join(hdr) + " |",
             "|" + "|".join(["---"] * len(hdr)) + "|"]
    for m in methods:
        sz = fp.filter(pl.col("method") == m)["cs_size95"].to_numpy().astype(float)
        n = len(sz)
        cells = [METHOD_LABEL.get(m, m), str(n)]
        cells += [(f"{float((sz <= t).mean()):.2f}" if n else "-") for t in thresholds]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# =========================================================================== #
# Calibrated-alpha machinery (PLOT 5). Per (facet.., method): find the CS level
# alpha* whose empirical coverage == nominal over declared CS, then read off
# power + size at alpha*. Coverage is monotone in beta so we invert on the grid.
# =========================================================================== #
def _coverage_curve(g: pl.DataFrame):
    """coverage(beta) over declared components that target a causal."""
    mass = g["best_mass"].to_numpy()
    mass = mass[~np.isnan(mass)]
    if len(mass) == 0:
        return None
    return np.array([np.mean(mass < b) for b in BETA_GRID])


def _calibrated_beta(cov: np.ndarray, nominal: float):
    """Smallest grid beta with coverage >= nominal (None if never reached)."""
    hit = np.where(cov >= nominal)[0]
    if len(hit) == 0:
        return None, None
    k = int(hit[0])
    return k, float(BETA_GRID[k])


def cs_calibrated_table(cs: pl.DataFrame, *, facet: list[str], methods: list[str],
                        min_log_bf=2.0, nominal=0.95) -> pl.DataFrame:
    facet = list(facet)
    d = cs.filter(pl.col("method").is_in(methods)).with_columns(
        (pl.col("ser_log_bf") >= min_log_bf).alias("declared"))
    rows = []
    combos = facet_combos(d, facet)
    for combo in combos:
        dd = d.filter(_facet_mask(facet, combo)) if facet else d
        for meth in methods:
            decl = dd.filter((pl.col("method") == meth) & pl.col("declared"))
            enr = decl.filter(pl.col("has_causal"))
            cov = _coverage_curve(enr)
            rec = {c: v for c, v in zip(facet, combo)}
            rec["method"] = meth
            if cov is None:
                rows.append({**rec, "beta_star": None, "coverage95": None,
                             "power": None, "size": None, "n_declared": decl.height})
                continue
            k, beta = _calibrated_beta(cov, nominal)
            cov95 = float(cov[IDX95])
            if k is None:
                rows.append({**rec, "beta_star": None, "coverage95": cov95,
                             "power": None, "size": None, "n_declared": decl.height})
                continue
            # size at beta*: median cs_size at that grid index over declared CS
            sizes = np.asarray(decl["cs_sizes"].to_list(), dtype=float)
            size_star = float(np.median(sizes[:, k])) if sizes.ndim == 2 and sizes.shape[0] else np.nan
            # power at beta*: fraction of causals captured (mass < beta*)
            ex = (
                dd.filter((pl.col("method") == meth) & pl.col("has_causal"))
                .explode(["mass_above_causal", "causal_indices"])
                .with_columns((pl.col("declared") & (pl.col("mass_above_causal") < beta)).alias("cap"))
            )
            keys = ["sample_id", "batch_hash", "causal_indices"]
            pc = ex.group_by(keys).agg(pl.col("cap").any().alias("cap"))
            power = float(pc["cap"].mean()) if pc.height else np.nan
            rows.append({**rec, "beta_star": beta, "coverage95": cov95,
                         "power": power, "size": size_star, "n_declared": decl.height})
    return pl.DataFrame(rows)


def cs_calibrated_dotplot(cs: pl.DataFrame, *, facet=("design",), filt=None, methods=None,
                          metrics=("power", "size"), min_log_bf=2.0, nominal=0.95,
                          panel=(2.9, 2.7), title=None, log_size=True, annotate_beta=True):
    facet = list(facet)
    d = apply_filter(cs, filt)
    if d.is_empty():
        return _placeholder("no CS data for this slice")
    methods = [m for m in (methods or methods_present(d))
               if not d.filter(pl.col("method") == m).is_empty()]
    tab = cs_calibrated_table(d, facet=facet, methods=methods,
                              min_log_bf=min_log_bf, nominal=nominal)
    combos = facet_combos(d, facet)
    metrics = list(metrics)
    nrow, ncol = len(combos), len(metrics)
    fig, axes = plt.subplots(nrow, ncol, figsize=(panel[0] * ncol, panel[1] * nrow),
                             squeeze=False, sharex=True)
    xpos = {m: k for k, m in enumerate(methods)}
    for i, combo in enumerate(combos):
        tt = tab
        for c, v in zip(facet, combo):
            tt = tt.filter(pl.col(c).is_null() if v is None else (pl.col(c) == v))
        for j, metric in enumerate(metrics):
            ax = axes[i][j]
            for meth in methods:
                r = tt.filter(pl.col("method") == meth)
                if r.height == 0 or r[metric][0] is None:
                    continue
                x = xpos[meth]
                ax.plot([x], [r[metric][0]], "o", color=method_color(meth), markersize=7)
                if annotate_beta and metric == metrics[0] and r["beta_star"][0] is not None:
                    ax.annotate(f"{r['beta_star'][0]:.2f}", (x, r[metric][0]),
                                textcoords="offset points", xytext=(0, 6),
                                ha="center", fontsize=6, color=method_color(meth))
            if metric == "power":
                ax.set_ylim(-0.03, 1.03)
            elif metric == "size" and log_size:
                ax.set_yscale("log")
            ax.set_xlim(-0.5, len(methods) - 0.5)
            ax.set_xticks(list(xpos.values()))
            ax.set_xticklabels([METHOD_LABEL.get(m, m) for m in methods], rotation=30,
                               ha="right", fontsize=7)
            ax.grid(True, axis="y", alpha=0.3)
            if i == 0:
                ax.set_title(f"{_METRIC_LABEL.get(metric, metric)} @ calibrated alpha", fontsize=9)
            if j == 0:
                ax.set_ylabel(facet_label(facet, combo), fontsize=8)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98 if title else 1])
    return fig, tab


# =========================================================================== #
# AUROC birds-eye. A component is a DISCOVERY when its 95% CS covers a causal
# AND is not dispersed (cs_size95 < max_size -- a generous localization gate).
# Per (setting.., method) we compute the AUROC of logBFSER separating discoveries
# from non-discoveries (missed or dispersed enriched components), then tile it as
# a heatmap: rows = method, columns = simulation setting.
# =========================================================================== #
def _discovery_label(max_size: int) -> pl.Expr:
    return pl.col("captured95") & (pl.col("cs_size95") < max_size)


def auroc_by_setting(cs: pl.DataFrame, *, setting=("design", "b0", "T"), filt=None,
                     methods=None, max_size=50, min_class=5) -> pl.DataFrame:
    """One row per (setting.., method): AUROC of ser_log_bf separating discoveries
    (covered AND cs_size95 < max_size) from non-discoveries, within that setting.
    AUROC is null when either class has < min_class members."""
    setting = list(setting)
    d = apply_filter(cs, filt).with_columns(_discovery_label(max_size).alias("_disc"))
    methods = [m for m in (methods or methods_present(d))
               if not d.filter(pl.col("method") == m).is_empty()]
    rows = []
    for combo in facet_combos(d, setting):
        dd = d.filter(_facet_mask(setting, combo))
        for meth in methods:
            g = dd.filter(pl.col("method") == meth)
            rec = {c: v for c, v in zip(setting, combo)}
            rec["method"] = meth
            if g.height == 0:
                rows.append({**rec, "auroc": None, "n_pos": 0, "n_neg": 0})
                continue
            lab = g["_disc"].to_numpy().astype(float)
            n_pos, n_neg = int(lab.sum()), int((1 - lab).sum())
            res = _roc(g["ser_log_bf"].to_numpy(), lab)
            auroc = res[2] if (res is not None and min(n_pos, n_neg) >= min_class) else None
            rows.append({**rec, "auroc": auroc, "n_pos": n_pos, "n_neg": n_neg})
    return pl.DataFrame(rows)


def _setting_Z(tab: pl.DataFrame, value_col: str, setting: list[str],
               methods: list[str], combos: list[tuple]) -> np.ndarray:
    """method x setting matrix of value_col from a tidy per-(setting,method) table."""
    Z = np.full((len(methods), len(combos)), np.nan)
    for j, combo in enumerate(combos):
        for i, meth in enumerate(methods):
            r = tab
            for c, v in zip(setting, combo):
                r = r.filter(pl.col(c).is_null() if v is None else (pl.col(c) == v))
            r = r.filter(pl.col("method") == meth)
            if r.height and r[value_col][0] is not None:
                Z[i, j] = r[value_col][0]
    return Z


def _draw_setting_heatmap(Z, methods, combos, setting, *, vmin, vmax, cbar_label, title,
                          annot=True, annot_thresh=None, col_w=0.34, panel_h=0.55,
                          cmap="viridis"):
    """Shared renderer: rows = method, cols = setting combos; first setting coord is a
    white-separated band, the rest form the x tick; NaN cells drawn grey."""
    def _short(v):
        if v is None:
            return "-"
        return f"{int(v)}" if isinstance(v, float) and float(v).is_integer() else str(v)
    thr = annot_thresh if annot_thresh is not None else (vmin + vmax) / 2
    fig, ax = plt.subplots(figsize=(max(col_w * len(combos) + 2.0, 6), panel_h * len(methods) + 1.6))
    im = ax.imshow(Z, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    ax.set_facecolor("0.85"); im.cmap.set_bad("0.85")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([METHOD_LABEL.get(m, m) for m in methods], fontsize=8)
    lead = [c[0] for c in combos]
    ax.set_xticks(range(len(combos)))
    ax.set_xticklabels(["/".join(_short(v) for v in combo[1:]) or "" for combo in combos],
                       fontsize=6, rotation=90)
    if len(setting) >= 2:
        ax.set_xlabel(f"setting  ({' / '.join(setting[1:])});  block = {setting[0]}", fontsize=8)
        second = [c[1] for c in combos]
        for j in range(1, len(combos)):
            if lead[j] != lead[j - 1]:
                ax.axvline(j - 0.5, color="white", lw=2.5)
            elif second[j] != second[j - 1]:
                ax.axvline(j - 0.5, color="0.6", lw=0.6)
        prev, start = lead[0], 0
        for j in range(1, len(combos) + 1):
            if j == len(combos) or lead[j] != prev:
                ax.text((start + j - 1) / 2, -0.9, coord_title(setting[0], prev),
                        ha="center", va="bottom", fontsize=7, fontweight="bold")
                if j < len(combos):
                    prev, start = lead[j], j
    else:
        ax.set_xlabel(f"setting ({setting[0]})", fontsize=8)
    if annot:
        for i in range(len(methods)):
            for j in range(len(combos)):
                if not np.isnan(Z[i, j]):
                    ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center",
                            fontsize=5, color="w" if Z[i, j] < thr else "k")
    fig.colorbar(im, ax=ax, shrink=0.7, pad=0.01, label=cbar_label)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


def auroc_setting_heatmap(cs: pl.DataFrame, *, setting=("design", "b0", "T"), filt=None,
                          methods=None, max_size=50, min_class=5, title=None, **kw):
    """Heatmap of per-setting logBF discovery-AUROC. Grey = too few of one class to score."""
    setting = list(setting)
    tab = auroc_by_setting(cs, setting=setting, filt=filt, methods=methods,
                           max_size=max_size, min_class=min_class)
    if tab.is_empty():
        return _placeholder("no data for this slice")
    methods = methods or methods_present(apply_filter(cs, filt))
    combos = facet_combos(tab, setting)
    Z = _setting_Z(tab, "auroc", setting, methods, combos)
    return _draw_setting_heatmap(
        Z, methods, combos, setting, vmin=0.5, vmax=1.0, annot_thresh=0.8,
        cbar_label="discovery AUROC (logBF)",
        title=title or f"logBFSER discovery-AUROC  (discovery = covers causal & 95%-CS size < {max_size})", **kw)


# =========================================================================== #
# AUPRC (average precision) birds-eye. Summarizes the PIP power-vs-FDP curve:
# how well the PIP vector ranks the causal variable(s) above the p-1 decoys. AP
# is the base-rate-aware summary (the PIP AUROC saturates because the causal is
# rare); it is computed from the pooled 200-bin PIP histograms per (setting, method).
# =========================================================================== #
def _ap_from_bins(counts: np.ndarray, causal: np.ndarray):
    """Average precision from pooled PIP histograms (200 bins, high PIP = better)."""
    cd, ccd = counts[::-1], causal[::-1]         # index 0 = highest-PIP bin
    sel, tp = np.cumsum(cd), np.cumsum(ccd)      # include more as threshold lowers
    totc = causal.sum()
    if totc == 0 or sel[-1] == 0:
        return None
    prec = tp / np.maximum(sel, 1)
    rec = tp / totc
    rec_prev = np.concatenate([[0.0], rec[:-1]])
    return float(np.sum((rec - rec_prev) * prec))


def ap_by_setting(pip: pl.DataFrame, *, setting=("design", "b0", "T"), filt=None,
                  methods=None, min_causal=5) -> pl.DataFrame:
    """One row per (setting.., method): average precision of PIP ranking causal vs
    non-causal variables, pooled over the setting's replicates. Null when the pooled
    causal count < min_causal."""
    setting = list(setting)
    d = apply_filter(pip, filt)
    methods = [m for m in (methods or methods_present(d))
               if not d.filter(pl.col("method") == m).is_empty()]
    rows = []
    for combo in facet_combos(d, setting):
        dd = d.filter(_facet_mask(setting, combo))
        for meth in methods:
            g = dd.filter(pl.col("method") == meth)
            rec = {c: v for c, v in zip(setting, combo)}
            rec["method"] = meth
            if g.height == 0:
                rows.append({**rec, "ap": None, "n_causal": 0})
                continue
            counts, causal = _sum_bins(g)
            ncaus = int(causal.sum())
            ap = _ap_from_bins(counts, causal) if ncaus >= min_causal else None
            rows.append({**rec, "ap": ap, "n_causal": ncaus})
    return pl.DataFrame(rows)


def _short_val(v) -> str:
    if v is None:
        return "-"
    return f"{int(v)}" if isinstance(v, float) and float(v).is_integer() else str(v)


def setting_tables_markdown(tab: pl.DataFrame, *, value: str, group_col: str,
                            row_cols: list[str], methods: list[str], fmt="{:.2f}",
                            higher_better=True, na="-") -> str:
    """One markdown table per `group_col` value (e.g. design). Rows = combos of
    `row_cols`, columns = methods; the best method in each row is bolded. `tab` is a
    tidy per-(setting, method) table carrying `value`."""
    row_cols = list(row_cols)
    blocks = []
    for gval in sorted(tab[group_col].unique().to_list(),
                       key=lambda v: _coord_sort_key(group_col, v)):
        g = tab.filter(pl.col(group_col) == gval)
        header = list(row_cols) + [METHOD_LABEL.get(m, m) for m in methods]
        lines = ["| " + " | ".join(header) + " |",
                 "| " + " | ".join("---" for _ in header) + " |"]
        for combo in facet_combos(g, row_cols):
            rr = g
            for c, v in zip(row_cols, combo):
                rr = rr.filter(pl.col(c).is_null() if v is None else (pl.col(c) == v))
            vals = []
            for m in methods:
                x = rr.filter(pl.col("method") == m)
                vals.append(x[value][0] if (x.height and x[value][0] is not None) else None)
            present = [v for v in vals if v is not None]
            best = (max if higher_better else min)(present) if present else None
            cells = []
            for v in vals:
                if v is None:
                    cells.append(na)
                else:
                    s = fmt.format(v)
                    cells.append(f"**{s}**" if (best is not None and v == best) else s)
            rowlab = [_short_val(v) for v in combo]
            lines.append("| " + " | ".join(rowlab + cells) + " |")
        head = gval if group_col == "design" else coord_title(group_col, gval)
        blocks.append(f"**{head}**\n\n" + "\n".join(lines))
    return "\n\n".join(blocks)


def ap_setting_tables(pip: pl.DataFrame, *, setting=("design", "b0", "T"), filt=None,
                      methods=None, min_causal=5, fmt="{:.2f}") -> str:
    """Markdown: one AP table per design (first coord), rows = the remaining setting
    coords, columns = methods, best-per-row bolded."""
    setting = list(setting)
    tab = ap_by_setting(pip, setting=setting, filt=filt, methods=methods, min_causal=min_causal)
    if tab.is_empty():
        return "_(no PIP data for this slice)_"
    methods = methods or methods_present(apply_filter(pip, filt))
    return setting_tables_markdown(tab, value="ap", group_col=setting[0],
                                   row_cols=setting[1:], methods=methods, fmt=fmt)


def ap_setting_heatmap(pip: pl.DataFrame, *, setting=("design", "b0", "T"), filt=None,
                       methods=None, min_causal=5, title=None, **kw):
    """Heatmap of per-setting PIP average precision. Grey = too few causals pooled."""
    setting = list(setting)
    tab = ap_by_setting(pip, setting=setting, filt=filt, methods=methods, min_causal=min_causal)
    if tab.is_empty():
        return _placeholder("no PIP data for this slice")
    methods = methods or methods_present(apply_filter(pip, filt))
    combos = facet_combos(tab, setting)
    Z = _setting_Z(tab, "ap", setting, methods, combos)
    return _draw_setting_heatmap(
        Z, methods, combos, setting, vmin=0.0, vmax=1.0, annot_thresh=0.5, cmap="magma",
        cbar_label="PIP average precision", title=title or "PIP average precision (AUPRC)", **kw)


# =========================================================================== #
# Credible-set CALIBRATION / overconfidence. Coverage of the beta-CS is P(best_mass
# < beta): its ECDF is the calibration curve, and comparing it to the diagonal (or to
# the nominal level) exposes overconfidence -- anti-conservative sets that cover the
# truth less often than they claim. Both views below are pure functions of best_mass
# (already stored); no size/alpha needed.
# =========================================================================== #
def cs_calibration_curve(cs: pl.DataFrame, *, facet=("design",), filt=None, methods=None,
                         min_log_bf=2.0, beta_lo=0.5, panel=(3.4, 3.6), title=None):
    """Reliability curve: nominal CS level beta (x) vs empirical coverage P(best_mass <
    beta) (y), among credible sets declared at ser_log_bf >= min_log_bf. One panel per
    facet combo, methods overlaid, diagonal = calibrated; below diagonal = overconfident."""
    facet = list(facet)
    d = apply_filter(cs, filt).filter(pl.col("has_causal") & (pl.col("ser_log_bf") >= min_log_bf))
    if d.is_empty():
        return _placeholder("no declared CS for this slice")
    methods = methods or methods_present(d)
    combos = facet_combos(d, facet)
    grid = np.linspace(max(beta_lo, 0.0), 0.99, 50)
    ncol = len(combos)
    fig, axes = plt.subplots(1, ncol, figsize=(panel[0] * ncol, panel[1]), squeeze=False, sharey=True)
    for j, combo in enumerate(combos):
        ax = axes[0][j]
        dd = d.filter(_facet_mask(facet, combo))
        for m in methods:
            bm = dd.filter(pl.col("method") == m)["best_mass"].to_numpy()
            bm = bm[~np.isnan(bm)]
            if len(bm):
                ax.plot(grid, [np.mean(bm < b) for b in grid], color=method_color(m),
                        lw=1.9, label=METHOD_LABEL.get(m, m))
        ax.plot([beta_lo, 1], [beta_lo, 1], ls=":", c="k", lw=1)
        ax.set_xlim(beta_lo, 1); ax.set_ylim(beta_lo, 1); ax.grid(True, alpha=0.3)
        ax.set_title(facet_label(facet, combo), fontsize=9)
        ax.set_xlabel(r"nominal CS level $\beta$", fontsize=9)
    axes[0][0].set_ylabel("empirical coverage", fontsize=9)
    axes[0][-1].legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle(title or rf"CS calibration at logBFSER $\geq$ {min_log_bf:g} "
                 r"(below the diagonal = overconfident)", fontsize=11)
    fig.tight_layout()
    return fig


def _seg_below(ax, x, y, level, color, lw=1.8):
    """Plot y vs x, solid where y>=level, dotted where y<level (anti-conservative)."""
    y = np.asarray(y, float)
    above = np.where(y >= level, y, np.nan)
    below = np.where(y < level, y, np.nan)
    for k in range(1, len(y)):  # bridge crossings so segments meet
        if np.isfinite(y[k]) and np.isfinite(y[k - 1]) and (y[k] >= level) != (y[k - 1] >= level):
            for arr in (above, below):
                if not np.isfinite(arr[k]):
                    arr[k] = y[k]
                if not np.isfinite(arr[k - 1]):
                    arr[k - 1] = y[k - 1]
    ax.plot(x, above, color=color, lw=lw, ls="-")
    ax.plot(x, below, color=color, lw=lw, ls=":")


def coverage_vs_logbf(cs: pl.DataFrame, *, betas=(0.99, 0.95, 0.80, 0.50), filt=None,
                      methods=None, xmax=10.0, min_n=30, panel=(3.4, 2.5), title=None):
    """Coverage of the beta-CS as the logBFSER declaration threshold is swept, faceted
    rows = nominal beta, cols = design. Each curve goes DOTTED where it falls below its
    nominal level (red line) -- i.e. where the method is anti-conservative / overconfident.
    x capped at xmax (and at each design's logBF support)."""
    d = apply_filter(cs, filt).filter(pl.col("has_causal"))
    if d.is_empty():
        return _placeholder("no CS data for this slice")
    # drop methods with no rows in this slice (e.g. an SER-only method in a MULTI filter),
    # else their empty quantiles poison the shared x-range min().
    present = set(d["method"].unique().to_list())
    methods = [m for m in (methods or methods_present(d)) if m in present]
    betas = list(betas)
    designs = [dn for dn in DESIGN_LABEL if dn in d["design_n"].unique().to_list()]
    fig, axes = plt.subplots(len(betas), len(designs),
                             figsize=(panel[0] * len(designs), panel[1] * len(betas)),
                             sharex="col", sharey="row", squeeze=False)
    for jc, dn in enumerate(designs):
        dd = d.filter(pl.col("design_n") == dn)
        xmin = min(dd.filter(pl.col("method") == m)["ser_log_bf"].quantile(0.02) for m in methods)
        xhi = min(xmax, min(dd.filter(pl.col("method") == m)["ser_log_bf"].quantile(0.99) for m in methods))
        tg = np.linspace(xmin, xhi, 40)
        for ir, beta in enumerate(betas):
            ax = axes[ir][jc]
            ax.axhline(beta, ls="-", c="#d62728", lw=1.3, zorder=0)
            ax.grid(True, alpha=0.3)
            for m in methods:
                g = dd.filter(pl.col("method") == m)
                lbf, bm = g["ser_log_bf"].to_numpy(), g["best_mass"].to_numpy()
                cov = [np.mean(bm[lbf >= t] < beta) if (lbf >= t).sum() > min_n else np.nan for t in tg]
                _seg_below(ax, tg, cov, beta, method_color(m))
            ax.set_xlim(xmin, xhi)
            if ir == 0:
                ax.set_title(DESIGN_LABEL[dn], fontsize=9)
            if jc == 0:
                ax.set_ylabel(rf"$\beta$ = {beta:g} CS" + "\ncoverage", fontsize=8)
            if ir == len(betas) - 1:
                ax.set_xlabel("logBFSER threshold", fontsize=8)
    handles = [plt.Line2D([], [], color=method_color(m), lw=2, label=METHOD_LABEL.get(m, m)) for m in methods]
    axes[0][-1].legend(handles=handles, fontsize=6.5, frameon=False, loc="lower left")
    fig.suptitle(title or r"Coverage of the $\beta$-CS vs logBFSER threshold "
                 r"(red = nominal $\beta$; dotted = below nominal / overconfident)", fontsize=11)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
def _placeholder(msg: str):
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=11)
    ax.set_axis_off()
    return fig
