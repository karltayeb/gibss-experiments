"""CS-family hooks (12 analyses) + legacy Snakemake entry-point.

Hook protocol: each analysis registers aggregate(rows)->stats and
draw(ax, stats, *, color, linestyle, label) via add_hook.  The generic
driver (generate_plots.render_plot) owns filtering and faceting; these
functions handle per-series geometry only.

Causal guard: analyses with simulation_filter:has_causal in library.yaml
skip rows whose causal_indices list is empty inside aggregate().
Per-analysis guards documented inline.
"""
import sys
from pathlib import Path

import numpy as np

_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from analyses.hooks import add_hook

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_CS_POWER_FDP_BETAS = [1.0, 0.95, 0.50]
# index into CS_BETA_GRID (100-element array: 0.01..0.99 step 0.01, then 1.0)
_CS_BETA_SIZE_INDICES = {0.95: 94, 0.50: 49}
_CS_LBF_MARKERS = [0.0, 2.0]
_CS_LBF_MARKER_STYLES = ["o", "s"]
_CS_BETA_LABELS = {1.0: "100% CS", 0.95: "95% CS", 0.50: "50% CS"}
_MAX_FDP = 0.5

_DEFAULT_BETA = 0.95
_DEFAULT_MAX_CS_SIZE = 10000
_DEFAULT_MIN_LOG_BF = 2.0


# ===========================================================================
# 1. causal_rank  (no has_causal filter in library.yaml)
#    But only rows with non-empty rank_of_causal contribute (null sims have
#    no rank to report, mirroring make_causal_rank_summary's filter).
# ===========================================================================

def _causal_rank_aggregate(rows):
    """Mean causal rank per threshold.

    Per (batch_hash, sample_id, threshold) take min rank across L effects,
    then average across samples.  Grouping-invariant: pooling more batches
    adds more (sample, rank) observations to the mean.
    """
    from collections import defaultdict
    # (batch_hash, sample_id, threshold) -> list[min_rank_for_this_l]
    per_sample: dict = defaultdict(list)
    for r in rows:
        ranks = r.get("rank_of_causal") or []
        if not ranks:
            continue
        thresh = r.get("threshold")
        sid = r.get("sample_id", "")
        bhash = r.get("batch_hash")
        per_sample[(bhash, sid, thresh)].append(min(ranks) + 1)

    if not per_sample:
        return {"mean_causal_rank": float("nan"), "by_threshold": None}

    per_thresh: dict = defaultdict(list)
    for (bhash, sid, thresh), rank_list in per_sample.items():
        per_thresh[thresh].append(min(rank_list))

    if len(per_thresh) == 1:
        thresh, sample_ranks = next(iter(per_thresh.items()))
        return {"mean_causal_rank": float(np.mean(sample_ranks)), "by_threshold": None}

    by_thresh = sorted(
        [(t, float(np.mean(rs))) for t, rs in per_thresh.items()
         if t is not None and rs],
        key=lambda x: x[0],
    )
    return {"mean_causal_rank": float("nan"), "by_threshold": by_thresh}


def _causal_rank_draw(ax, stats, *, color, linestyle, label):
    if stats.get("by_threshold") is not None:
        thresholds = [t for t, _ in stats["by_threshold"]]
        means = [m for _, m in stats["by_threshold"]]
        ax.plot(thresholds, means, color=color, linestyle=linestyle,
                linewidth=1.5, marker="o", label=label)
    else:
        y = stats["mean_causal_rank"]
        if np.isnan(y):
            return
        ax.axhline(y=y, color=color, linestyle=linestyle, linewidth=1.5, label=label)
    ax.set_xlabel("Threshold")
    ax.set_ylim(bottom=1)
    ax.set_ylabel("Mean causal rank")


add_hook("causal_rank", "cs", _causal_rank_aggregate, _causal_rank_draw)


# ===========================================================================
# 2. preceding_posterior_mass_ecdf  (no has_causal in library.yaml)
#    Internally filters to causal rows (rank_of_causal non-empty), matching
#    make_preceding_mass_ecdf_summary's filter.
# ===========================================================================

def _preceding_posterior_mass_ecdf_aggregate(rows):
    """Collect mass_above_causal values from rows with non-empty rank_of_causal."""
    masses = []
    for r in rows:
        if not (r.get("rank_of_causal") or []):
            continue
        macs = r.get("mass_above_causal") or []
        masses.extend(float(m) for m in macs)

    if not masses:
        return {"masses": np.array([])}
    return {"masses": np.sort(np.asarray(masses, dtype=float))}


def _preceding_posterior_mass_ecdf_draw(ax, stats, *, color, linestyle, label):
    vals = stats["masses"]
    if len(vals) == 0:
        return
    y = np.arange(1, len(vals) + 1) / len(vals)
    ax.plot(vals, y, color=color, linestyle=linestyle, linewidth=1.2, label=label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mass above causal")
    ax.set_ylabel("coverage")


add_hook("preceding_posterior_mass_ecdf", "cs",
         _preceding_posterior_mass_ecdf_aggregate,
         _preceding_posterior_mass_ecdf_draw)


# ===========================================================================
# 3. cs_dot_summary  (has_causal)
#
# NOTE — grouping caveat: each series is one method in one collection
# (driver uses facet_col="design", color="family").  aggregate collapses all
# rows to a single (power, coverage, cs_size) scalar dot at beta=0.95.
# draw renders a point at (cs_size, power); not all 3 original panels fit
# in a single ax so the most informative pair is chosen.
# ===========================================================================

def _cs_dot_summary_aggregate(rows):
    """Compute (power, cs_size) at beta=_DEFAULT_BETA for the series.
    Causal guard: skip rows with empty causal_indices.
    """
    from utils import CS_BETA_GRID as _CBG
    beta_arr = np.asarray(_CBG.tolist())
    beta_idx = int(np.argmin(np.abs(beta_arr - _DEFAULT_BETA)))

    cs_sizes = []
    causal_disc: dict = {}  # (bhash, sid, causal_idx) -> discovered bool

    for r in rows:
        cis = r.get("causal_indices") or []
        if not cis:
            continue
        ser_valid = r.get("ser_log_bf", 0.0) >= _DEFAULT_MIN_LOG_BF
        sizes = r.get("cs_sizes") or []
        ranks = r.get("rank_of_causal") or []
        sid = r.get("sample_id", "")
        bhash = r.get("batch_hash")

        if sizes and ser_valid:
            cs_size = sizes[beta_idx]
            if cs_size <= _DEFAULT_MAX_CS_SIZE:
                cs_sizes.append(cs_size)

        for causal_rank, causal_idx in zip(ranks, cis):
            key = (bhash, sid, int(causal_idx))
            if ser_valid and sizes:
                cs_size = sizes[beta_idx]
                disc = cs_size <= _DEFAULT_MAX_CS_SIZE and causal_rank < cs_size
            else:
                disc = False
            if key not in causal_disc:
                causal_disc[key] = False
            causal_disc[key] = causal_disc[key] or disc

    power = float(np.mean(list(causal_disc.values()))) if causal_disc else float("nan")
    cs_size_mean = float(np.mean(cs_sizes)) if cs_sizes else float("nan")
    return {"power": power, "cs_size": cs_size_mean, "beta": _DEFAULT_BETA}


def _cs_dot_summary_draw(ax, stats, *, color, linestyle, label):
    cs_size = stats.get("cs_size", float("nan"))
    power = stats.get("power", float("nan"))
    if np.isnan(cs_size) or np.isnan(power):
        return
    ax.scatter([cs_size], [power], color=color, s=60, zorder=3, label=label)
    ax.set_xlabel(f"Mean CS size (β={stats['beta']:.2f})")
    ax.set_ylabel("Power")


add_hook("cs_dot_summary", "cs", _cs_dot_summary_aggregate, _cs_dot_summary_draw)


# ===========================================================================
# 4. cs_calibrated_dot  (has_causal)
#
# NOTE — grouping caveat: same as cs_dot_summary — one dot per series.
# aggregate calibrates beta using the mass_above_causal quantile at
# target_coverage, then reports power/cs_size at that calibrated beta.
# ===========================================================================

def _cs_calibrated_dot_aggregate(rows):
    """Calibrated dot: find beta' = quantile(mass_above_causal, target_coverage).
    Causal guard: skip rows with empty causal_indices.
    """
    from utils import CS_BETA_GRID as _CBG
    beta_arr = np.asarray(_CBG.tolist())
    TARGET = _DEFAULT_BETA

    masses = []
    for r in rows:
        cis = r.get("causal_indices") or []
        if not cis:
            continue
        if r.get("ser_log_bf", 0.0) < _DEFAULT_MIN_LOG_BF:
            continue
        macs = r.get("mass_above_causal") or []
        masses.extend(float(m) for m in macs)

    if not masses:
        return {"power": float("nan"), "cs_size": float("nan"),
                "calibrated_beta": float("nan"), "beta": TARGET}

    cal_beta = float(np.clip(round(float(np.quantile(masses, TARGET, method="higher")), 2), 0.01, 0.99))
    cal_idx = int(np.argmin(np.abs(beta_arr - cal_beta)))

    cs_sizes = []
    causal_disc: dict = {}

    for r in rows:
        cis = r.get("causal_indices") or []
        if not cis:
            continue
        ser_valid = r.get("ser_log_bf", 0.0) >= _DEFAULT_MIN_LOG_BF
        sizes = r.get("cs_sizes") or []
        ranks = r.get("rank_of_causal") or []
        sid = r.get("sample_id", "")
        bhash = r.get("batch_hash")

        if sizes and ser_valid:
            cs_size = sizes[cal_idx]
            if cs_size <= _DEFAULT_MAX_CS_SIZE:
                cs_sizes.append(cs_size)

        for causal_rank, causal_idx in zip(ranks, cis):
            key = (bhash, sid, int(causal_idx))
            if ser_valid and sizes:
                cs_size = sizes[cal_idx]
                disc = cs_size <= _DEFAULT_MAX_CS_SIZE and causal_rank < cs_size
            else:
                disc = False
            if key not in causal_disc:
                causal_disc[key] = False
            causal_disc[key] = causal_disc[key] or disc

    power = float(np.mean(list(causal_disc.values()))) if causal_disc else float("nan")
    cs_size_mean = float(np.mean(cs_sizes)) if cs_sizes else float("nan")
    return {"power": power, "cs_size": cs_size_mean,
            "calibrated_beta": cal_beta, "beta": TARGET}


def _cs_calibrated_dot_draw(ax, stats, *, color, linestyle, label):
    cs_size = stats.get("cs_size", float("nan"))
    power = stats.get("power", float("nan"))
    if np.isnan(cs_size) or np.isnan(power):
        return
    cal_beta = stats.get("calibrated_beta", float("nan"))
    lbl = f"{label} (β'={cal_beta:.2f})" if not np.isnan(cal_beta) else label
    ax.scatter([cs_size], [power], color=color, s=60, zorder=3, label=lbl)
    ax.set_xlabel("Mean CS size (calibrated β)")
    ax.set_ylabel("Power")


add_hook("cs_calibrated_dot", "cs", _cs_calibrated_dot_aggregate, _cs_calibrated_dot_draw)


# ===========================================================================
# 5. cs_size_power  (has_causal)
#    Scatter: nominal dot (filled) and calibrated dot (open), connected.
# ===========================================================================

def _cs_size_power_aggregate(rows):
    """Compute nominal and calibrated (cs_size, power) for the series.
    Causal guard: skip rows with empty causal_indices.
    """
    from utils import CS_BETA_GRID as _CBG
    beta_arr = np.asarray(_CBG.tolist())
    TARGET = _DEFAULT_BETA
    nom_idx = int(np.argmin(np.abs(beta_arr - TARGET)))

    masses = []
    for r in rows:
        cis = r.get("causal_indices") or []
        if not cis:
            continue
        if r.get("ser_log_bf", 0.0) < _DEFAULT_MIN_LOG_BF:
            continue
        macs = r.get("mass_above_causal") or []
        masses.extend(float(m) for m in macs)

    cal_beta = TARGET
    if masses:
        cal_beta = float(np.clip(round(float(np.quantile(masses, TARGET, method="higher")), 2), 0.01, 0.99))
    cal_idx = int(np.argmin(np.abs(beta_arr - cal_beta)))

    def _at_idx(idx):
        cs_sizes = []
        causal_disc: dict = {}
        for r in rows:
            cis = r.get("causal_indices") or []
            if not cis:
                continue
            ser_valid = r.get("ser_log_bf", 0.0) >= _DEFAULT_MIN_LOG_BF
            sizes = r.get("cs_sizes") or []
            ranks = r.get("rank_of_causal") or []
            sid = r.get("sample_id", "")
            bhash = r.get("batch_hash")
            if sizes and ser_valid:
                cs_size = sizes[idx]
                if cs_size <= _DEFAULT_MAX_CS_SIZE:
                    cs_sizes.append(cs_size)
            for causal_rank, causal_idx in zip(ranks, cis):
                key = (bhash, sid, int(causal_idx))
                if ser_valid and sizes:
                    cs_size = sizes[idx]
                    disc = cs_size <= _DEFAULT_MAX_CS_SIZE and causal_rank < cs_size
                else:
                    disc = False
                if key not in causal_disc:
                    causal_disc[key] = False
                causal_disc[key] = causal_disc[key] or disc
        power = float(np.mean(list(causal_disc.values()))) if causal_disc else float("nan")
        cs_size_mean = float(np.mean(cs_sizes)) if cs_sizes else float("nan")
        return power, cs_size_mean, cs_sizes

    nom_power, nom_cs, nom_dist = _at_idx(nom_idx)
    cal_power, cal_cs, cal_dist = _at_idx(cal_idx)
    return {"nom_power": nom_power, "nom_cs_size": nom_cs, "nom_size_dist": nom_dist,
            "cal_power": cal_power, "cal_cs_size": cal_cs, "cal_size_dist": cal_dist,
            "nom_beta": TARGET, "cal_beta": cal_beta}


def _cs_size_power_draw_which(which):
    """Tradeoff plane at ONE operating point (which='nom' or 'cal'): y = power,
    dot = MEAN CS size (median ~1 collapses; mean reflects the skewed tail and
    spreads methods), horizontal IQR whisker for spread. One point per method, no
    connectors — compare nominal vs calibrated across the two panels."""
    def draw(ax, stats, *, color, linestyle, label):
        def _ok(v): return v is not None and not np.isnan(v)
        x = stats.get(f"{which}_cs_size")
        y = stats.get(f"{which}_power")
        dist = stats.get(f"{which}_size_dist") or []
        if not (_ok(x) and _ok(y)):
            return
        if dist:
            q25, q75 = np.percentile(np.asarray(dist, dtype=float), [25, 75])
            ax.plot([q25, q75], [y, y], color=color, lw=1.2, alpha=0.5, zorder=2,
                    solid_capstyle="butt")
        ax.scatter([x], [y], s=48, color=color, zorder=4, label=label)
        ax.set_xlabel("Mean CS size (dot) · IQR (whisker)")
        ax.set_ylabel("Power")
        ax.margins(0.08)
    return draw


def _cs_size_power_draw(ax, stats, *, color, linestyle, label):
    """Both operating points overlaid: nominal (filled) -> calibrated (open) dots
    at MEAN CS size, IQR whiskers, connected. For standalone faceted plots."""
    def _ok(v): return v is not None and not np.isnan(v)

    def hpt(which, filled):
        x, y = stats.get(f"{which}_cs_size"), stats.get(f"{which}_power")
        dist = stats.get(f"{which}_size_dist") or []
        if not (_ok(x) and _ok(y)):
            return None
        if dist:
            q25, q75 = np.percentile(np.asarray(dist, dtype=float), [25, 75])
            ax.plot([q25, q75], [y, y], color=color, lw=1.1, alpha=0.45, zorder=2,
                    solid_capstyle="butt")
        ax.scatter([x], [y], s=46, zorder=4, edgecolors=color, linewidths=1.3,
                   facecolors=(color if filled else "none"))
        return x

    nx = hpt("nom", True)
    cx = hpt("cal", False)
    if nx is not None and cx is not None:
        ax.plot([nx, cx], [stats["nom_power"], stats["cal_power"]], color=color, lw=0.8, zorder=2)
    ax.plot([], [], color=color, marker="s", linestyle="none", label=label)
    ax.set_xlabel("Mean CS size · IQR")
    ax.set_ylabel("Power")
    ax.margins(0.08)


add_hook("cs_size_power", "cs", _cs_size_power_aggregate, _cs_size_power_draw)
add_hook("cs_size_power_nom", "cs", _cs_size_power_aggregate, _cs_size_power_draw_which("nom"))
add_hook("cs_size_power_cal", "cs", _cs_size_power_aggregate, _cs_size_power_draw_which("cal"))


# ===========================================================================
# 6. cs_radius_power  (has_causal)
#    Like cs_size_power but x = mean causal radius.
# ===========================================================================

def _cs_radius_power_aggregate(rows):
    """Compute nominal and calibrated (causal_radius, power).
    Causal guard: skip rows with empty causal_indices.
    """
    from utils import CS_BETA_GRID as _CBG
    beta_arr = np.asarray(_CBG.tolist())
    TARGET = _DEFAULT_BETA
    nom_idx = int(np.argmin(np.abs(beta_arr - TARGET)))

    masses = []
    for r in rows:
        cis = r.get("causal_indices") or []
        if not cis:
            continue
        if r.get("ser_log_bf", 0.0) < _DEFAULT_MIN_LOG_BF:
            continue
        macs = r.get("mass_above_causal") or []
        masses.extend(float(m) for m in macs)

    cal_beta = TARGET
    if masses:
        cal_beta = float(np.clip(round(float(np.quantile(masses, TARGET, method="higher")), 2), 0.01, 0.99))
    cal_idx = int(np.argmin(np.abs(beta_arr - cal_beta)))

    def _at_idx(idx):
        # (bhash, sid, causal_idx) -> (discovered, max_radius)
        causal_disc: dict = {}
        for r in rows:
            cis = r.get("causal_indices") or []
            if not cis:
                continue
            ser_valid = r.get("ser_log_bf", 0.0) >= _DEFAULT_MIN_LOG_BF
            sizes = r.get("cs_sizes") or []
            ranks = r.get("rank_of_causal") or []
            radii_raw = r.get("cs_causal_radius") or []
            sid = r.get("sample_id", "")
            bhash = r.get("batch_hash")
            for ci_pos, (causal_rank, causal_idx) in enumerate(zip(ranks, cis)):
                key = (bhash, sid, int(causal_idx))
                if ser_valid and sizes:
                    cs_size = sizes[idx]
                    disc = cs_size <= _DEFAULT_MAX_CS_SIZE and causal_rank < cs_size
                    radius = (radii_raw[ci_pos][idx]
                              if radii_raw and ci_pos < len(radii_raw)
                              and idx < len(radii_raw[ci_pos]) else None)
                else:
                    disc = False
                    radius = None
                if key not in causal_disc:
                    causal_disc[key] = (False, None)
                prev_disc, prev_rad = causal_disc[key]
                new_disc = prev_disc or disc
                # Take max radius across L effects for the same key (mirrors .max() in viz_utils)
                if disc and radius is not None:
                    new_rad = radius if prev_rad is None else max(radius, prev_rad)
                else:
                    new_rad = prev_rad
                causal_disc[key] = (new_disc, new_rad)

        if not causal_disc:
            return float("nan"), float("nan")
        discoveries = [d for d, _ in causal_disc.values()]
        radii_vals = [rad for d, rad in causal_disc.values() if d and rad is not None]
        power = float(np.mean(discoveries))
        mean_radius = float(np.mean(radii_vals)) if radii_vals else float("nan")
        return power, mean_radius

    nom_power, nom_radius = _at_idx(nom_idx)
    cal_power, cal_radius = _at_idx(cal_idx)
    return {"nom_power": nom_power, "nom_radius": nom_radius,
            "cal_power": cal_power, "cal_radius": cal_radius,
            "nom_beta": TARGET, "cal_beta": cal_beta}


def _cs_radius_power_draw(ax, stats, *, color, linestyle, label):
    def _ok(v): return v is not None and not np.isnan(v)
    nom_x, nom_y = stats.get("nom_radius"), stats.get("nom_power")
    cal_x, cal_y = stats.get("cal_radius"), stats.get("cal_power")
    if _ok(nom_x) and _ok(nom_y) and _ok(cal_x) and _ok(cal_y):
        ax.plot([nom_x, cal_x], [nom_y, cal_y], color=color, linewidth=0.8, zorder=2)
    if _ok(nom_x) and _ok(nom_y):
        ax.scatter([nom_x], [nom_y], color=color, s=60, zorder=3, marker="o", label=label)
    if _ok(cal_x) and _ok(cal_y):
        ax.scatter([cal_x], [cal_y], color=color, s=60, zorder=3,
                   marker="o", facecolors="none", edgecolors=color, linewidths=1.5)
    ax.set_xlabel("Mean causal radius")
    ax.set_ylabel("Power")
    # autoscale to the data (radii cluster near the design's max), with margins,
    # instead of forcing the full 0..1 box where points pile in one corner.
    ax.margins(0.18)
    ax.autoscale_view()


add_hook("cs_radius_power", "cs", _cs_radius_power_aggregate, _cs_radius_power_draw)


# ===========================================================================
# 7. cs_power_fdp  (no has_causal in library.yaml)
#    Power/FDP curves sweeping over ser_log_bf.  Null-simulation rows
#    contribute to the FDP denominator (same as old _cs_power_fdp_curves).
# ===========================================================================

def _detection_labels(rows):
    """Pure SER detection: TP = non-null sim, FP = null sim — the SER 'fires'
    purely by ser_log_bf, no CS-coverage requirement (localization is measured by
    coverage/radius/size, not here). Returns (sorted_lbf, sorted_is_nonnull,
    n_non_null, n_null) sorted by descending ser_log_bf."""
    is_nonnull = np.array([len(r.get("rank_of_causal") or []) > 0 for r in rows])
    lbf = np.asarray([r.get("ser_log_bf", -99.0) for r in rows], dtype=float)
    order = np.argsort(-lbf)
    return lbf[order], is_nonnull[order], int(is_nonnull.sum()), int((~is_nonnull).sum())


def _cs_power_fdp_aggregate(rows):
    """CS-level detection precision-recall, swept by ser_log_bf. TP=non-null fires,
    FP=null fires (no localization requirement). power=TP/non-null, FDP=FP/reported."""
    sorted_lbf, nonnull, n_nn, _ = _detection_labels(rows)
    cum_tp = np.cumsum(nonnull)
    cum_fp = np.cumsum(~nonnull)
    return {
        "lbf": sorted_lbf,
        "power": cum_tp / max(n_nn, 1),
        "fdp": cum_fp / np.maximum(cum_tp + cum_fp, 1),
    }


def _cs_power_fdp_draw(ax, stats, *, color, linestyle, label):
    lbf, power, fdp = stats["lbf"], stats["power"], stats["fdp"]
    ax.plot(fdp, power, color=color, linestyle=linestyle, linewidth=1.5, label=label)
    for thresh, marker in zip(_CS_LBF_MARKERS, _CS_LBF_MARKER_STYLES):
        mask = lbf >= thresh
        if mask.any():
            i = int(np.where(mask)[0][-1])
            ax.plot(fdp[i], power[i], marker=marker, color=color,
                    markersize=6, zorder=5, linestyle="none")
    ax.set_xlabel("FDP")
    ax.set_ylabel("Power")
    ax.set_xlim(0.0, _MAX_FDP)
    ax.set_ylim(0.0, 1.05)


add_hook("cs_power_fdp", "cs", _cs_power_fdp_aggregate, _cs_power_fdp_draw)


# --- ROC for SER detection (paired null/non-null sims -> balanced classes, so
# FPR is honest; rank-based -> immune to per-method ser_log_bf scale). ----------
def _cs_roc_aggregate(rows):
    """SER-detection ROC swept by ser_log_bf: TPR over non-null sims vs FPR over
    null sims. Same TP/FP labels as cs_power_fdp; AUC summarizes."""
    sorted_lbf, nonnull, n_nn, n_null = _detection_labels(rows)
    tpr = np.concatenate([[0.0], np.cumsum(nonnull) / max(n_nn, 1)])
    fpr = np.concatenate([[0.0], np.cumsum(~nonnull) / max(n_null, 1)])
    _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = float(_trap(tpr, fpr)) if (n_null and n_nn) else float("nan")
    return {"tpr": tpr, "fpr": fpr, "auc": auc}


def _cs_roc_draw(ax, stats, *, color, linestyle, label):
    ax.plot([0, 1], [0, 1], color="#cccccc", lw=1, zorder=0)  # chance diagonal
    ax.plot(stats["fpr"], stats["tpr"], color=color, linestyle=linestyle,
            linewidth=1.5, label=label)
    k = getattr(ax, "_auc_count", 0)
    ax.text(0.97, 0.05 + 0.07 * k, f"AUC={stats['auc']:.3f}", transform=ax.transAxes,
            fontsize=6.5, color=color, va="bottom", ha="right", zorder=4)
    ax._auc_count = k + 1
    ax.set_xlabel("FPR (null sims)")
    ax.set_ylabel("TPR (non-null sims)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)


add_hook("cs_roc", "cs", _cs_roc_aggregate, _cs_roc_draw)


# ===========================================================================
# Operating points: power / coverage / size at standard CS levels (50/80/90/95),
# filled = at the nominal beta, open = at the CALIBRATED beta (the beta whose
# empirical coverage equals that nominal level). The filled->open gap is the
# cost of honest calibration. Three hooks share one compute, one metric each.
# ===========================================================================
_OP_TARGETS = [0.8, 0.9, 0.95, 0.99]


def _op_aggregate(rows):
    from utils import CS_BETA_GRID as _CBG
    beta = np.asarray(_CBG.tolist())
    # Calibration is over the SAME (ungated) population as the coverage metric, so
    # the calibrated beta reproduces the target empirical coverage. (Gating by
    # ser_log_bf here would calibrate the detected subset, not coverage.)
    masses = []
    for r in rows:
        if not (r.get("causal_indices") or []):
            continue
        masses.extend(float(m) for m in (r.get("mass_above_causal") or []))
    masses = np.asarray(masses)

    def metrics_at(idx):
        cov, powr, sizes = {}, {}, []
        for r in rows:
            cis = r.get("causal_indices") or []
            if not cis:
                continue
            sizes_l = r.get("cs_sizes") or []
            ranks = r.get("rank_of_causal") or []
            sid, bh = r.get("sample_id", ""), r.get("batch_hash")
            ser_valid = r.get("ser_log_bf", 0.0) >= _DEFAULT_MIN_LOG_BF
            cs_size = sizes_l[idx] if sizes_l and idx < len(sizes_l) else None
            valid_size = cs_size is not None and cs_size <= _DEFAULT_MAX_CS_SIZE
            if valid_size and ser_valid:
                sizes.append(cs_size)
            for rank, ci in zip(ranks, cis):
                key = (bh, sid, int(ci))
                covered = cs_size is not None and rank < cs_size          # pure coverage
                detected = covered and ser_valid and valid_size           # in a usable CS
                cov[key] = cov.get(key, False) or covered
                powr[key] = powr.get(key, False) or detected
        nan = float("nan")
        return (
            float(np.mean(list(powr.values()))) if powr else nan,
            float(np.mean(list(cov.values()))) if cov else nan,
            float(np.mean(sizes)) if sizes else nan,
            sizes,  # full size distribution (for box plots)
        )

    nom = {"power": [], "coverage": [], "size": [], "size_dist": []}
    cal = {"power": [], "coverage": [], "size": [], "size_dist": []}
    for T in _OP_TARGETS:
        nidx = int(np.argmin(np.abs(beta - T)))
        cb = (float(np.clip(round(float(np.quantile(masses, T, method="higher")), 2), 0.01, 1.0))
              if masses.size else T)
        cidx = int(np.argmin(np.abs(beta - cb)))
        for store, idx in ((nom, nidx), (cal, cidx)):
            p, c, s, sd = metrics_at(idx)
            store["power"].append(p); store["coverage"].append(c)
            store["size"].append(s); store["size_dist"].append(sd)
    return {"targets": _OP_TARGETS, "nom": nom, "cal": cal}


def _op_draw(metric, ylabel, ref):
    def draw(ax, stats, *, color, linestyle, label):
        n = len(stats["targets"])
        x0 = np.arange(n)
        M = max(int(getattr(ax, "_n_series", 1)), 1)
        i = int(getattr(ax, "_series_index", 0))
        off = (i - (M - 1) / 2.0) * (0.8 / M) if M > 1 else 0.0   # dodge methods within level
        x = x0 + off
        nom = np.asarray(stats["nom"][metric], dtype=float)
        cal = np.asarray(stats["cal"][metric], dtype=float)
        if i == 0:  # background + reference drawn once per panel
            for c in range(n):
                if c % 2 == 1:
                    ax.axvspan(c - 0.5, c + 0.5, color="0.94", zorder=0)
            if ref:  # nominal-level target as a short horizontal bar per group
                for c, t in enumerate(stats["targets"]):
                    ax.plot([c - 0.42, c + 0.42], [t, t], color="0.6", lw=1.2, zorder=1)
        # connect nominal<->calibrated within each level (no across-level line)
        for xi, a, b in zip(x, nom, cal):
            if np.isfinite(a) and np.isfinite(b):
                ax.plot([xi, xi], [a, b], color=color, lw=0.8, zorder=2)
        ax.scatter(x, nom, s=26, color=color, zorder=3, label=label)            # filled = nominal
        ax.scatter(x, cal, s=32, facecolors="none", edgecolors=color,
                   linewidths=1.2, zorder=4)                                     # open = calibrated
        ax.set_xticks(x0)
        ax.set_xticklabels([f"{int(t * 100)}" for t in stats["targets"]])
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_xlabel("nominal CS level (%)")
        ax.set_ylabel(ylabel)
        if metric == "power":
            ax.margins(y=0.10)                # dynamic (zoom to data)
        elif metric == "size":
            ax.set_yscale("log")              # spans orders of magnitude
            ax.margins(y=0.15)
        else:                                 # coverage: dynamic (zoom to data + targets)
            ax.margins(y=0.10)
    return draw


def _op_size_box_draw(which, other, ylabel):
    """CS-size DISTRIBUTION per level as boxes for `which` (nom/cal), with an open
    circle at the `other` median for reference. Methods dodged, log y."""
    def draw(ax, stats, *, color, linestyle, label):
        n = len(stats["targets"])
        x0 = np.arange(n)
        M = max(int(getattr(ax, "_n_series", 1)), 1)
        i = int(getattr(ax, "_series_index", 0))
        w = 0.8 / M
        x = x0 + (i - (M - 1) / 2.0) * w if M > 1 else x0
        if i == 0:
            for c in range(n):
                if c % 2 == 1:
                    ax.axvspan(c - 0.5, c + 0.5, color="0.94", zorder=0)
        dist = [np.asarray(d, dtype=float) if d else np.array([np.nan]) for d in stats[which]["size_dist"]]
        bp = ax.boxplot(dist, positions=x, widths=w * 0.85, patch_artist=True,
                        showfliers=False, manage_ticks=False, zorder=3)
        for box in bp["boxes"]:
            box.set(facecolor=color, alpha=0.30, edgecolor=color, linewidth=1.0)
        for part in ("whiskers", "caps", "medians"):
            for ln in bp[part]:
                ln.set_color(color)
        other_med = [float(np.median(d)) if d else np.nan for d in stats[other]["size_dist"]]
        ax.scatter(x, other_med, s=30, facecolors="none", edgecolors=color, linewidths=1.2,
                   zorder=5, label=label)  # open = other-beta median
        ax.set_xticks(x0)
        ax.set_xticklabels([f"{int(t * 100)}" for t in stats["targets"]])
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_xlabel("nominal CS level (%)")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
    return draw


add_hook("cs_op_power", "cs", _op_aggregate, _op_draw("power", "Power", ref=False))
add_hook("cs_op_coverage", "cs", _op_aggregate, _op_draw("coverage", "Empirical coverage", ref=True))
add_hook("cs_op_size", "cs", _op_aggregate, _op_size_box_draw("nom", "cal", "CS size (nominal)"))
add_hook("cs_op_size_cal", "cs", _op_aggregate, _op_size_box_draw("cal", "nom", "CS size (calibrated)"))


# ===========================================================================
# Shared beta-trace aggregate (used by cs_power_size_coverage_trace and
# cs_coverage_trace — same statistic, different draw).
#
# Computes (beta, power, coverage, cs_size) across CS_BETA_GRID.
# Mirrors make_cs_power_size_coverage_summary:
#   - coverage/cs_size: CS-level (per l effect), filtered by ser_log_bf and max_cs_size
#   - power: per (batch_hash, sample_id, causal_idx), OR across l effects
#
# has_causal guard: skip rows with empty causal_indices.
# ===========================================================================

def _beta_trace_aggregate(rows):
    """Compute (beta, power, coverage, cs_size) trace across all CS_BETA_GRID values.
    Causal guard: skip rows with empty causal_indices.
    """
    from utils import CS_BETA_GRID as _CBG
    n_betas = len(_CBG)

    # CS-level accumulators (coverage, cs_size): filtered by ser_log_bf and max_cs_size
    valid_cs_counts = np.zeros(n_betas, dtype=float)
    covered_sums = np.zeros(n_betas, dtype=float)
    cs_size_sums = np.zeros(n_betas, dtype=float)

    # Power: per (bhash, sid, causal_idx) -> bool array per beta
    causal_disc: dict = {}

    for r in rows:
        cis = r.get("causal_indices") or []
        if not cis:
            continue
        ser_valid = r.get("ser_log_bf", 0.0) >= _DEFAULT_MIN_LOG_BF
        sizes = r.get("cs_sizes") or []
        ranks = r.get("rank_of_causal") or []
        sid = r.get("sample_id", "")
        bhash = r.get("batch_hash")

        if not sizes:
            continue

        sizes_arr = np.asarray(sizes, dtype=float)
        valid_size_m = sizes_arr <= _DEFAULT_MAX_CS_SIZE  # per beta

        if ser_valid:
            valid_cs_counts += valid_size_m.astype(float)
            if ranks:
                min_rank = min(ranks)
                covered_arr = (min_rank < sizes_arr).astype(float)
                covered_sums += np.where(valid_size_m, covered_arr, 0.0)
            cs_size_sums += np.where(valid_size_m, sizes_arr, 0.0)

            # Power accumulation
            for causal_rank, causal_idx in zip(ranks, cis):
                key = (bhash, sid, int(causal_idx))
                valid_covered = valid_size_m & (causal_rank < sizes_arr)
                if key not in causal_disc:
                    causal_disc[key] = np.zeros(n_betas, dtype=bool)
                causal_disc[key] |= valid_covered

    betas = list(_CBG.tolist())

    if causal_disc:
        disc_stack = np.stack(list(causal_disc.values()))  # (n_causals, n_betas)
        power_arr = disc_stack.mean(axis=0).astype(float)
    else:
        power_arr = np.full(n_betas, float("nan"))

    with np.errstate(invalid="ignore", divide="ignore"):
        coverage_arr = np.where(valid_cs_counts > 0,
                                covered_sums / valid_cs_counts, float("nan"))
        cs_size_arr = np.where(valid_cs_counts > 0,
                               cs_size_sums / valid_cs_counts, float("nan"))

    return {
        "betas": betas,
        "power": power_arr.tolist(),
        "coverage": coverage_arr.tolist(),
        "cs_size": cs_size_arr.tolist(),
    }


# ===========================================================================
# 8. cs_power_size_coverage_trace  (has_causal)
#    draw: coverage vs nominal beta (calibration trace) with y=x reference.
# ===========================================================================

def _cs_power_size_coverage_trace_draw(ax, stats, *, color, linestyle, label):
    betas = np.asarray(stats["betas"])
    coverage = np.asarray(stats["coverage"], dtype=float)
    ax.plot([0, 1], [0, 1], color="#cccccc", lw=1, zorder=0)
    valid = ~np.isnan(coverage)
    if valid.any():
        ax.plot(betas[valid], coverage[valid],
                color=color, linestyle=linestyle, linewidth=2.0, label=label)
    ax.set_xlabel("Nominal coverage (β)")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)


add_hook("cs_power_size_coverage_trace", "cs",
         _beta_trace_aggregate, _cs_power_size_coverage_trace_draw)


# ===========================================================================
# 9. cs_coverage_trace  (has_causal)
#    draw: power vs empirical coverage.
# ===========================================================================

def _cs_coverage_trace_draw(ax, stats, *, color, linestyle, label):
    betas = np.asarray(stats["betas"])
    coverage = np.asarray(stats["coverage"], dtype=float)
    power = np.asarray(stats["power"], dtype=float)
    valid = ~(np.isnan(coverage) | np.isnan(power))
    if not valid.any():
        return
    idx = np.argsort(coverage[valid])
    ax.plot(coverage[valid][idx], power[valid][idx],
            color=color, linestyle=linestyle, linewidth=2.0, label=label)
    # mark beta=0.95 point
    close_idx = int(np.argmin(np.abs(betas - 0.95)))
    if valid[close_idx]:
        ax.plot(coverage[close_idx], power[close_idx],
                marker="o", markersize=6, color=color, linestyle="none")
    ax.axvline(0.95, color="0.6", lw=1, ls=":", zorder=1)  # honest 95% coverage
    ax.set_xlabel("Empirical coverage")
    ax.set_ylabel("Power")


add_hook("cs_coverage_trace", "cs", _beta_trace_aggregate, _cs_coverage_trace_draw)


# ===========================================================================
# Shared coverage-size aggregate (used by cs_coverage_size and cs_calibration).
#
# Mirrors make_cs_coverage_size_curves: no ser_log_bf or max_cs_size filter.
# has_causal guard: skip rows with empty causal_indices.
# ===========================================================================

def _cs_coverage_size_aggregate(rows):
    """Compute (beta, coverage, cs_size, cs_size_frac) trace.  No BF/size filter.
    Causal guard: skip rows with empty causal_indices.
    """
    from utils import CS_BETA_GRID as _CBG
    n_betas = len(_CBG)

    covered_sums = np.zeros(n_betas, dtype=float)
    cs_size_sums = np.zeros(n_betas, dtype=float)
    cs_size_frac_sums = np.zeros(n_betas, dtype=float)
    n_cs_counts = np.zeros(n_betas, dtype=float)

    for r in rows:
        cis = r.get("causal_indices") or []
        if not cis:
            continue
        sizes = r.get("cs_sizes") or []
        ranks = r.get("rank_of_causal") or []
        n_features = r.get("n_features", 1) or 1

        if not sizes:
            continue

        sizes_arr = np.asarray(sizes, dtype=float)
        n_cs_counts += 1.0
        cs_size_sums += sizes_arr
        cs_size_frac_sums += sizes_arr / n_features

        if ranks:
            min_rank = min(ranks)
            covered_sums += (min_rank < sizes_arr).astype(float)

    betas = list(_CBG.tolist())
    with np.errstate(invalid="ignore", divide="ignore"):
        coverage_arr = np.where(n_cs_counts > 0, covered_sums / n_cs_counts, float("nan"))
        cs_size_arr = np.where(n_cs_counts > 0, cs_size_sums / n_cs_counts, float("nan"))
        cs_size_frac_arr = np.where(n_cs_counts > 0,
                                    cs_size_frac_sums / n_cs_counts, float("nan"))

    return {
        "betas": betas,
        "coverage": coverage_arr.tolist(),
        "cs_size": cs_size_arr.tolist(),
        "cs_size_frac": cs_size_frac_arr.tolist(),
    }


# ===========================================================================
# 10. cs_coverage_size  (has_causal)
#     draw: empirical coverage vs cs_size_frac.
# ===========================================================================

def _cs_coverage_size_draw(ax, stats, *, color, linestyle, label):
    betas = np.asarray(stats["betas"])
    coverage = np.asarray(stats["coverage"], dtype=float)
    cs_size_frac = np.asarray(stats["cs_size_frac"], dtype=float)
    valid = ~(np.isnan(coverage) | np.isnan(cs_size_frac))
    if not valid.any():
        return
    ax.plot(coverage[valid], cs_size_frac[valid],
            color=color, linestyle=linestyle, linewidth=2.0, label=label)
    close_idx = int(np.argmin(np.abs(betas - 0.95)))
    if valid[close_idx]:
        ax.plot(coverage[close_idx], cs_size_frac[close_idx],
                marker="o", markersize=6, color=color, linestyle="none")
    ax.axvline(0.95, color="0.6", lw=1, ls=":", zorder=1)  # honest 95% coverage
    ax.set_xlabel("Empirical coverage")
    ax.set_ylabel("Mean CS size fraction")
    # autoscale: coverage clusters near 1 in well-powered settings, so a forced
    # 0..1 box crushes the curves into a vertical line.
    ax.margins(0.12)
    ax.autoscale_view()


add_hook("cs_coverage_size", "cs", _cs_coverage_size_aggregate, _cs_coverage_size_draw)


# --- per-method size-vs-coverage with median + IQR band (facet_wrap by method) -
def _cs_size_band_aggregate(rows):
    from utils import CS_BETA_GRID as _CBG
    n_betas = len(_CBG)
    covered = np.zeros(n_betas); n_cs = np.zeros(n_betas)
    size_rows = []
    for r in rows:
        if not (r.get("causal_indices") or []):
            continue
        sizes = r.get("cs_sizes") or []
        if not sizes:
            continue
        arr = np.asarray(sizes, dtype=float)
        size_rows.append(arr)
        n_cs += 1.0
        ranks = r.get("rank_of_causal") or []
        if ranks:
            covered += (min(ranks) < arr).astype(float)
    betas = list(_CBG.tolist())
    if not size_rows:
        nans = [float("nan")] * n_betas
        return {"coverage": nans, "med": nans, "q25": nans, "q75": nans}
    mat = np.vstack(size_rows)
    with np.errstate(invalid="ignore"):
        coverage = np.where(n_cs > 0, covered / n_cs, np.nan)
    return {
        "coverage": coverage.tolist(),
        "med": np.median(mat, axis=0).tolist(),
        "q25": np.percentile(mat, 25, axis=0).tolist(),
        "q75": np.percentile(mat, 75, axis=0).tolist(),
    }


def _cs_size_band_draw(ax, stats, *, color, linestyle, label):
    cov = np.asarray(stats["coverage"], dtype=float)
    med = np.asarray(stats["med"], dtype=float)
    q25 = np.asarray(stats["q25"], dtype=float)
    q75 = np.asarray(stats["q75"], dtype=float)
    m = ~(np.isnan(cov) | np.isnan(med))
    if not m.any():
        return
    ax.fill_between(cov[m], q25[m], q75[m], color=color, alpha=0.25, zorder=2)
    ax.plot(cov[m], med[m], color=color, linewidth=1.6, label=label, zorder=3)
    ax.axvline(0.95, color="0.6", lw=1, ls=":", zorder=1)
    ax.set_xlabel("Empirical coverage")
    ax.set_ylabel("CS size")
    ax.set_yscale("log")
    ax.margins(0.10)


add_hook("cs_size_band", "cs", _cs_size_band_aggregate, _cs_size_band_draw)


# ===========================================================================
# 11. cs_coverage_radius  (has_causal)
#     Coverage uses no BF/size filter (mirrors make_cs_coverage_size_curves).
#     Radius uses BF/size filter (mirrors make_cs_radius_power_summary).
#     draw: empirical coverage vs mean causal radius.
# ===========================================================================

def _cs_coverage_radius_aggregate(rows):
    """Compute (beta, coverage, cs_causal_radius) trace.
    Coverage: no BF/size filter.  Radius: with BF/size filter.
    Causal guard: skip rows with empty causal_indices.
    """
    from utils import CS_BETA_GRID as _CBG
    n_betas = len(_CBG)

    covered_sums = np.zeros(n_betas, dtype=float)
    n_cs_counts = np.zeros(n_betas, dtype=float)
    # (bhash, sid, causal_idx) -> per-beta radius (keep max across l effects)
    causal_radius: dict = {}

    for r in rows:
        cis = r.get("causal_indices") or []
        if not cis:
            continue
        ser_valid = r.get("ser_log_bf", 0.0) >= _DEFAULT_MIN_LOG_BF
        sizes = r.get("cs_sizes") or []
        ranks = r.get("rank_of_causal") or []
        radii_raw = r.get("cs_causal_radius") or []
        sid = r.get("sample_id", "")
        bhash = r.get("batch_hash")

        if not sizes:
            continue

        sizes_arr = np.asarray(sizes, dtype=float)

        # Coverage — no filter
        n_cs_counts += 1.0
        if ranks:
            min_rank = min(ranks)
            covered_sums += (min_rank < sizes_arr).astype(float)

        # Radius — with filter
        if ser_valid and radii_raw:
            for ci_pos, (causal_rank, causal_idx) in enumerate(zip(ranks, cis)):
                if ci_pos >= len(radii_raw):
                    continue
                key = (bhash, sid, int(causal_idx))
                if key not in causal_radius:
                    causal_radius[key] = np.full(n_betas, np.nan)
                causal_radii_per_beta = radii_raw[ci_pos]
                for beta_idx, (cs_size, radius) in enumerate(
                        zip(sizes, causal_radii_per_beta)):
                    if (cs_size <= _DEFAULT_MAX_CS_SIZE and causal_rank < cs_size
                            and radius is not None):
                        prev = causal_radius[key][beta_idx]
                        # take max across l effects (mirrors .max() in make_cs_radius_power_summary)
                        if np.isnan(prev) or float(radius) > prev:
                            causal_radius[key][beta_idx] = float(radius)

    betas = list(_CBG.tolist())
    with np.errstate(invalid="ignore", divide="ignore"):
        coverage_arr = np.where(n_cs_counts > 0, covered_sums / n_cs_counts, float("nan"))

    if causal_radius:
        rad_stack = np.stack(list(causal_radius.values()))  # (n_causals, n_betas)
        with np.errstate(invalid="ignore"):
            radius_mean = np.nanmean(rad_stack, axis=0)
        all_nan = np.all(np.isnan(rad_stack), axis=0)
        radius_mean[all_nan] = float("nan")
    else:
        radius_mean = np.full(n_betas, float("nan"))

    return {
        "betas": betas,
        "coverage": coverage_arr.tolist(),
        "cs_causal_radius": radius_mean.tolist(),
    }


def _cs_coverage_radius_draw(ax, stats, *, color, linestyle, label):
    betas = np.asarray(stats["betas"])
    coverage = np.asarray(stats["coverage"], dtype=float)
    radius = np.asarray(stats["cs_causal_radius"], dtype=float)
    valid = ~(np.isnan(coverage) | np.isnan(radius))
    if not valid.any():
        return
    ax.plot(coverage[valid], radius[valid],
            color=color, linestyle=linestyle, linewidth=2.0, label=label)
    close_idx = int(np.argmin(np.abs(betas - 0.95)))
    if valid[close_idx]:
        ax.plot(coverage[close_idx], radius[close_idx],
                marker="o", markersize=6, color=color, linestyle="none")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Empirical coverage")
    ax.set_ylabel("Mean causal radius")


add_hook("cs_coverage_radius", "cs", _cs_coverage_radius_aggregate, _cs_coverage_radius_draw)


# ===========================================================================
# 12. cs_calibration  (has_causal)
#     Same data as cs_coverage_size (reuses _cs_coverage_size_aggregate).
#     draw: nominal beta vs empirical coverage with y=x diagonal.
# ===========================================================================

def _cs_calibration_draw(ax, stats, *, color, linestyle, label):
    betas = np.asarray(stats["betas"])
    coverage = np.asarray(stats["coverage"], dtype=float)
    ax.plot([0, 1], [0, 1], color="#cccccc", lw=1, zorder=0)
    valid = ~np.isnan(coverage)
    if not valid.any():
        return
    ax.plot(betas[valid], coverage[valid],
            color=color, linestyle=linestyle, linewidth=2.0, label=label)
    close_idx = int(np.argmin(np.abs(betas - 0.95)))
    if valid[close_idx]:
        ax.plot(betas[close_idx], coverage[close_idx],
                marker="o", markersize=6, color=color, linestyle="none")
    ax.set_xlabel("Nominal coverage (β)")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)


add_hook("cs_calibration", "cs", _cs_coverage_size_aggregate, _cs_calibration_draw)


if "snakemake" in globals():
    import sys as _sys
    from pathlib import Path as _Path
    _parent = str(_Path(__file__).parent.parent)
    if _parent not in _sys.path:
        _sys.path.insert(0, _parent)
    from analyses import pip as _pip_mod  # register pip-family hooks alongside cs hooks
    import generate_plots
    from experiments import loader as _loader
    _wc = snakemake.wildcards
    _cfg = _loader.load_config()
    _spec = _loader.resolve_plot_spec(_cfg, _wc.supercollection, _wc.plot_name)
    from analyses.hooks import HOOKS
    _bundle = _loader.load_sc_bundle(_cfg, _wc.supercollection, [HOOKS[_spec["analysis"]].requires])
    generate_plots.render_plot(_bundle, _spec, snakemake.output[0])
