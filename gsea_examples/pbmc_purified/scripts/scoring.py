"""Per-gene ranking score and hit-list selection for the Lifan GSEA pipeline.

The ranking score is a two-sided significance: ``-log10(LFSR)`` clamped so that
LFSR == 0 (numerically underflowed, i.e. maximally significant) does not become
``inf``. A small deterministic tie-break by ``|z| = |posterior_mean| / posterior_sd``
gives a strict total order so the Cox partial likelihood has no tied event times.
"""
from __future__ import annotations

import numpy as np

# Tie-break magnitude. Sub-dominant to any real gap in -log10(LFSR), so it only
# orders genes that are otherwise tied (e.g. all clamped at the LFSR floor).
_TIEBREAK_SCALE = 1e-6


def significance_score(
    lfsr: np.ndarray,
    posterior_mean: np.ndarray,
    posterior_sd: np.ndarray,
    *,
    lfsr_floor: float,
) -> np.ndarray:
    """Two-sided significance score, higher = more significant.

    ``-log10(clip(LFSR, lfsr_floor, 1))`` plus a tie-break proportional to
    ``|posterior_mean| / posterior_sd``.
    """
    lfsr = np.asarray(lfsr, dtype=float)
    posterior_mean = np.asarray(posterior_mean, dtype=float)
    posterior_sd = np.asarray(posterior_sd, dtype=float)

    neglog = -np.log10(np.clip(lfsr, lfsr_floor, 1.0))
    abs_z = np.abs(posterior_mean / posterior_sd)
    max_abs_z = np.max(abs_z)
    tiebreak = (abs_z / max_abs_z) * _TIEBREAK_SCALE if max_abs_z > 0 else 0.0
    return neglog + tiebreak


def ranking_score(
    posterior_mean: np.ndarray,
    posterior_sd: np.ndarray,
    lfsr: np.ndarray,
    *,
    kind: str,
    lfsr_floor: float = 1e-20,
    sd_floor: float = 1e-4,
) -> np.ndarray:
    """Two-sided (or signed) per-gene ranking score, higher = more interesting.

    kind:
      - abs_loading    : |posterior_mean|          (recommended; robust, no saturation)
      - signed_loading : posterior_mean            (directional analysis)
      - abs_z          : |posterior_mean / max(posterior_sd, sd_floor)|
      - neglog_lfsr    : two-sided significance, -log10(LFSR) with |z| tie-break
    """
    posterior_mean = np.asarray(posterior_mean, dtype=float)
    posterior_sd = np.asarray(posterior_sd, dtype=float)
    if kind == "abs_loading":
        return np.abs(posterior_mean)
    if kind == "signed_loading":
        return posterior_mean
    if kind == "abs_z":
        return np.abs(posterior_mean / np.maximum(posterior_sd, sd_floor))
    if kind == "neglog_lfsr":
        return significance_score(lfsr, posterior_mean, posterior_sd, lfsr_floor=lfsr_floor)
    raise ValueError(f"unknown score kind: {kind}")


def select_hits(
    score: np.ndarray,
    *,
    threshold: float | None = None,
    quantile: float | None = None,
    topk: int | None = None,
) -> np.ndarray:
    """Boolean hit-list mask over ``score`` (higher = more significant).

    Exactly one of ``threshold``, ``quantile``, ``topk`` must be given:
      - ``threshold``: ``score >= threshold``.
      - ``quantile``: ``score >= quantile-th quantile`` (i.e. top ``1 - quantile``).
      - ``topk``: the ``topk`` genes with the largest score.
    """
    given = {
        "threshold": threshold,
        "quantile": quantile,
        "topk": topk,
    }
    supplied = [name for name, value in given.items() if value is not None]
    if len(supplied) != 1:
        raise ValueError(
            "select_hits requires exactly one of threshold, quantile, topk; "
            f"got {supplied or 'none'}."
        )

    score = np.asarray(score, dtype=float)
    if threshold is not None:
        return score >= float(threshold)
    if quantile is not None:
        if not 0.0 < quantile < 1.0:
            raise ValueError("quantile must be in (0, 1).")
        return score >= float(np.quantile(score, quantile))
    # topk
    topk = int(topk)
    if topk <= 0:
        raise ValueError("topk must be positive.")
    mask = np.zeros(len(score), dtype=bool)
    if topk >= len(score):
        mask[:] = True
        return mask
    order = np.argsort(-score, kind="mergesort")
    mask[order[:topk]] = True
    return mask
