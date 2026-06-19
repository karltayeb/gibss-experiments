from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
_parent = str(Path(__file__).parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import plot_ready
import viz_utils
from analyses import ANALYSIS_RENDERERS
from analyses._common import foreground_methods as _foreground_methods_impl


def _foreground_methods(method_metadata: pl.DataFrame, settings: dict) -> set[str]:
    """Filter method_filter to methods present in method_metadata."""
    return _foreground_methods_impl(method_metadata, settings)


def _method_order(method_metadata: pl.DataFrame, foreground: set[str]) -> list[str]:
    from analyses._common import method_order
    return method_order(method_metadata, foreground)


def render_analysis(bundle: dict, args: dict, analysis: str, output_path: str) -> None:
    """Render a single analysis type and save to output_path."""
    fig = ANALYSIS_RENDERERS[analysis](bundle, args)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
