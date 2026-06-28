"""PIP-family hook implementations (pip_calibration, power_fdp, causal_pip, mass_above_causal)."""
import sys
from pathlib import Path

import numpy as np

_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import viz_utils
from analyses.hooks import add_hook

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------
_N_PIP_BINS = 200

MAX_FDP = 0.5  # used by power_fdp draw


# ---------------------------------------------------------------------------
# pip_calibration
# ---------------------------------------------------------------------------

_N_COARSE_BINS = 20  # display bins (width 0.05), centers 0.025..0.975


def _coarsen(fine: np.ndarray) -> np.ndarray:
    """Sum the reduction's fine bins (e.g. 200, width 0.005) down to
    _N_COARSE_BINS display bins (width 0.05). Falls back to the raw array if it
    doesn't divide evenly."""
    n = len(fine)
    if n % _N_COARSE_BINS == 0:
        return fine.reshape(_N_COARSE_BINS, n // _N_COARSE_BINS).sum(axis=1)
    return fine


def _pip_calibration_aggregate(rows):
    # pool bin counts across all rows in the series (grouping-invariant: sum)
    total = None
    causal = None
    for r in rows:
        c = np.asarray(r["pip_bin_counts"], dtype=float)
        k = np.asarray(r["pip_bin_causal_counts"], dtype=float)
        total = c if total is None else total + c
        causal = k if causal is None else causal + k
    # class-stratified Brier from the FINE bins (pip ~ bin center): causal feats
    # target 1, null feats target 0. Each class normalized within itself, so the
    # 99.8% null mass can't swamp the causal term.
    fine_centers = (np.arange(len(total)) + 0.5) / len(total)
    nulls = total - causal
    n_caus, n_null = causal.sum(), nulls.sum()
    brier_causal = float(np.sum(causal * (1.0 - fine_centers) ** 2) / n_caus) if n_caus > 0 else float("nan")
    brier_null = float(np.sum(nulls * fine_centers ** 2) / n_null) if n_null > 0 else float("nan")

    total = _coarsen(total)
    causal = _coarsen(causal)
    n_bins = len(total)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    with np.errstate(invalid="ignore", divide="ignore"):
        emp = np.where(total > 0, causal / total, np.nan)
    lo, hi = _wilson(causal, total)
    return {"centers": centers, "empirical": emp, "counts": total,
            "ci_lo": lo, "ci_hi": hi,
            "brier_causal": brier_causal, "brier_null": brier_null}


def _wilson(k, n, z: float = 1.96):
    """Wilson score interval for a binomial proportion, vectorized. Returns
    (lo, hi) arrays; NaN where n==0."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    out_lo = np.full_like(n, np.nan)
    out_hi = np.full_like(n, np.nan)
    m = n > 0
    if m.any():
        p = k[m] / n[m]
        nn = n[m]
        denom = 1.0 + z * z / nn
        center = (p + z * z / (2 * nn)) / denom
        half = z * np.sqrt(p * (1 - p) / nn + z * z / (4 * nn * nn)) / denom
        out_lo[m] = center - half
        out_hi[m] = center + half
    return out_lo, out_hi


def _pip_calibration_draw(ax, stats, *, color, linestyle, label):
    # Markers only (no connecting line): per-method bins are sparse, so a line
    # produces noisy vertical streaks. Size each point by its bin count so
    # well-supported bins read as larger dots. linestyle is unused here.
    del linestyle
    ax.plot([0, 1], [0, 1], color="#cccccc", lw=1, zorder=0)  # y=x reference
    counts = stats["counts"]
    m = counts > 0
    if m.any():
        x = stats["centers"][m]
        c = counts[m]
        sizes = 8.0 + 60.0 * (c / c.max())  # scale dot area by bin support
        # vertical Wilson CI for each bin's empirical frequency
        ax.vlines(x, stats["ci_lo"][m], stats["ci_hi"][m],
                  color=color, alpha=0.35, linewidth=1.0, zorder=2)
        ax.scatter(x, stats["empirical"][m], s=sizes,
                   color=color, edgecolors="none", alpha=0.85, label=label, zorder=3)
        # class-stratified Brier: causal feats (target 1) vs null feats (target 0),
        # each normalized within its class so the ~99.8% null mass can't swamp it.
        # B1 = sharpness on causals (low = causals get high PIP); B0 = null
        # over-confidence (low = nulls correctly ~0).
        k = getattr(ax, "_ece_count", 0)
        ax.text(0.05, 0.95 - 0.10 * k,
                f"Brier  causal={stats['brier_causal']:.3f}\n        null={stats['brier_null']:.4f}",
                transform=ax.transAxes, fontsize=6.5, color=color, va="top", ha="left", zorder=4)
        ax._ece_count = k + 1
    ax.set_xlabel("Predicted PIP"); ax.set_ylabel("Empirical frequency")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


add_hook("pip_calibration", "pip", _pip_calibration_aggregate, _pip_calibration_draw)


# ---------------------------------------------------------------------------
# power_fdp (requires "pip")
# ---------------------------------------------------------------------------

def _power_fdp_aggregate(rows):
    """Pool pip_bin_counts/pip_bin_causal_counts and derive power/FDP curves."""
    counts = np.zeros(_N_PIP_BINS, dtype=float)
    causal = np.zeros(_N_PIP_BINS, dtype=float)
    for r in rows:
        counts += np.asarray(r["pip_bin_counts"], dtype=float)
        causal += np.asarray(r["pip_bin_causal_counts"], dtype=float)
    power, fdp = viz_utils._bins_to_power_fdp(counts, causal)
    return {"fdp": fdp, "power": power}


def _power_fdp_draw(ax, stats, *, color, linestyle, label):
    fdp = stats["fdp"]
    power = stats["power"]
    mask = fdp <= MAX_FDP
    ax.plot(fdp[mask], power[mask], color=color, linestyle=linestyle,
            linewidth=1.5, label=label)
    ax.set_xlabel("FDP")
    ax.set_ylabel("Power")
    ax.set_xlim(0.0, MAX_FDP)
    ax.set_ylim(0.0, 1.05)
    ax.set_box_aspect(1)


add_hook("power_fdp", "pip", _power_fdp_aggregate, _power_fdp_draw)


# ---------------------------------------------------------------------------
# causal_pip (requires "pip"; skip null-simulation rows in aggregate)
# ---------------------------------------------------------------------------

def _causal_pip_aggregate(rows):
    """Collect causal PIPs from rows with non-empty causal_indices.

    Groups by threshold so thresholded methods yield a mean_pip vs threshold
    curve; non-thresholded methods yield a single mean value.
    """
    from collections import defaultdict
    per_threshold: dict = defaultdict(list)
    for r in rows:
        cis = r.get("causal_indices") or []
        if not cis:  # null simulation — skip
            continue
        cpips = r.get("causal_pips") or []
        thresh = r.get("threshold")  # None for non-thresholded methods
        per_threshold[thresh].extend(float(p) for p in cpips)

    if not per_threshold:
        return {"mean_causal_pip": float("nan"), "by_threshold": None}

    # Single threshold (or all None): return a scalar mean
    if len(per_threshold) == 1:
        thresh, pips = next(iter(per_threshold.items()))
        return {
            "mean_causal_pip": float(np.mean(pips)) if pips else float("nan"),
            "by_threshold": None,
        }

    # Multiple thresholds: return sorted (threshold, mean_pip) pairs (drop None key)
    by_thresh = sorted(
        [(t, float(np.mean(pips))) for t, pips in per_threshold.items()
         if t is not None and pips],
        key=lambda x: x[0],
    )
    return {"mean_causal_pip": float("nan"), "by_threshold": by_thresh}


def _causal_pip_draw(ax, stats, *, color, linestyle, label):
    if stats["by_threshold"] is not None:
        thresholds = [t for t, _ in stats["by_threshold"]]
        means = [m for _, m in stats["by_threshold"]]
        ax.plot(thresholds, means, color=color, linestyle=linestyle,
                linewidth=1.5, marker="o", label=label)
    else:
        y = stats["mean_causal_pip"]
        if np.isnan(y):
            return
        ax.axhline(y=y, color=color, linestyle=linestyle, linewidth=1.5, label=label)
    ax.set_xlabel("Threshold")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Mean causal PIP")


add_hook("causal_pip", "pip", _causal_pip_aggregate, _causal_pip_draw)


# ---------------------------------------------------------------------------
# mass_above_causal (requires "cs")
# ---------------------------------------------------------------------------

def _mass_above_causal_aggregate(rows):
    """Pick best SER per (batch_hash, sample_id, threshold, causal_idx) by highest causal_alpha.

    Mirrors expand_mass_above_causal_from_compact: for each causal in each sample,
    uses the SER effect l that had the highest alpha (credibility) for that causal.
    batch_hash + sample_id uniquely identifies the sample across batches/collections.
    """
    from collections import defaultdict
    best: dict = {}  # (batch_hash, sample_id, threshold, causal_idx) -> (alpha, mass)
    for r in rows:
        cis = r.get("causal_indices") or []
        if not cis:  # null simulation — skip
            continue
        cas = r.get("causal_alpha") or []
        mas = r.get("mass_above_causal") or []
        sid = r.get("sample_id", "")
        thresh = r.get("threshold")
        bhash = r.get("batch_hash")
        for ci, alpha, mass in zip(cis, cas, mas):
            key = (bhash, sid, thresh, int(ci))
            if key not in best or float(alpha) > best[key][0]:
                best[key] = (float(alpha), float(mass))

    masses = [v[1] for v in best.values()]
    if not masses:
        return {"mean_mass_above_causal": float("nan"), "by_threshold": None}
    return {"mean_mass_above_causal": float(np.mean(masses)), "by_threshold": None}


def _mass_above_causal_draw(ax, stats, *, color, linestyle, label):
    if stats.get("by_threshold") is not None:
        thresholds = [t for t, _ in stats["by_threshold"]]
        means = [m for _, m in stats["by_threshold"]]
        ax.plot(thresholds, means, color=color, linestyle=linestyle,
                linewidth=1.5, marker="o", label=label)
    else:
        y = stats["mean_mass_above_causal"]
        if np.isnan(y):
            return
        ax.axhline(y=y, color=color, linestyle=linestyle, linewidth=1.5, label=label)
    ax.set_xlabel("Threshold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Mean mass above causal")


add_hook("mass_above_causal", "cs", _mass_above_causal_aggregate, _mass_above_causal_draw)


if "snakemake" in globals():
    import sys as _sys
    from pathlib import Path as _Path
    _parent = str(_Path(__file__).parent.parent)
    if _parent not in _sys.path:
        _sys.path.insert(0, _parent)
    from analyses import cs as _cs_mod  # register cs-family hooks alongside pip hooks
    import generate_plots
    from experiments import loader as _loader
    _wc = snakemake.wildcards
    _cfg = _loader.load_config()
    _spec = _loader.resolve_plot_spec(_cfg, _wc.supercollection, _wc.plot_name)
    from analyses.hooks import HOOKS
    _bundle = _loader.load_sc_bundle(_cfg, _wc.supercollection, [HOOKS[_spec["analysis"]].requires])
    generate_plots.render_plot(_bundle, _spec, snakemake.output[0])
