"""Shared helpers for slide figures.

Every figure script writes its PDF through `save()`, which also drops a
`<name>.prov.json` sidecar recording the script, git commit, timestamp, and
params that produced it. That sidecar is the provenance record - no generated
figure should enter the deck without one.

Run scripts with uv, e.g.:
    uv run python scripts/fig06_resolution_spectrum.py --out figures/fig06_resolution_spectrum.pdf
or, preferably, via the Snakemake pipeline:
    snakemake -s figures.smk all -c4
"""
from __future__ import annotations

import datetime as _dt
import json
import pathlib
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# House style: plain, legible, no chrome. The content is the message.
plt.rcParams.update(
    {
        "figure.figsize": (5.0, 3.1),
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "savefig.bbox": "tight",
        "savefig.dpi": 200,
    }
)

# Two-group palette, matched to the review surface. Used for neutral schematic
# elements (there is no repo convention for causal status).
TEAL = "#0B6E75"
AMBER = "#CC7A34"
INK = "#16202E"
MUT = "#93A2B0"

# Null / non-null scheme - colour-menu option 3 (cool grey / crimson), chosen in review.
NULL_C = "#7D8CA3"
NONNULL_C = "#B0322B"

# Method colors - mirror twogroup_experiments/viz_utils.py (Okabe-Ito palette),
# so slide figures match the repo's result plots.
M_ORACLE = "#CC79A7"        # twogroup_oracle (rose)
M_ESTIMATED = "#D55E00"     # twogroup estimated (vermillion)
M_TWOGROUP_LOC = "#C0392B"  # twogroup_loc_fam (crimson)
M_TWOGROUP_SCALE = "#FF6347"  # twogroup_scale_fam (tomato)
M_COX = "#009E73"           # cox / forward ordering (green)
M_COX_REVERSED = "#E69F00"  # cox_reversed / reverse ordering (amber)
M_LOGISTIC = "#0072B2"      # logistic_threshold (blue)
M_LINEAR = "#8B4513"        # linear

_HERE = pathlib.Path(__file__).resolve().parent


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_HERE, text=True
        ).strip()
    except Exception:
        return "unknown"


def save(fig, out, params: dict, script: str) -> None:
    """Save `fig` to `out` and write a provenance sidecar next to it."""
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    prov = {
        "output": out.name,
        "script": pathlib.Path(script).name,
        "git_commit": _git_hash(),
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "params": params,
    }
    out.with_suffix(".prov.json").write_text(json.dumps(prov, indent=2) + "\n")
    plt.close(fig)


def placeholder(out, slide: str, title: str, note: str, params: dict, script: str) -> None:
    """Emit a labelled placeholder figure so the deck compiles before the real
    figure exists. Sub-agents replace the body of the calling script; the
    provenance plumbing stays identical."""
    fig, ax = plt.subplots(figsize=(5.6, 3.0))    # aspect ~1.87, within the slide band
    ax.axis("off")
    ax.add_patch(
        plt.Rectangle(
            (0.015, 0.015), 0.97, 0.97, fill=False, ls="--", ec=MUT, lw=1.2,
            transform=ax.transAxes,
        )
    )
    ax.text(0.5, 0.66, f"{slide}", ha="center", va="center", fontsize=10,
            color=AMBER, weight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.53, title, ha="center", va="center", fontsize=13,
            color=INK, weight="bold", transform=ax.transAxes, wrap=True)
    ax.text(0.5, 0.34, note, ha="center", va="center", fontsize=8,
            color="0.35", transform=ax.transAxes, wrap=True)
    ax.text(0.5, 0.09, "PLACEHOLDER - provenance stub", ha="center", va="center",
            fontsize=7, color=MUT, style="italic", transform=ax.transAxes)
    save(fig, out, params, script)
