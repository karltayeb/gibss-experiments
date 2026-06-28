"""analyses package: renderer family modules.

pip and cs analyses are registered in HOOKS (analyses/hooks.py) and driven
by the plot-spec system via the Snakemake pipeline. The legacy
ANALYSIS_RENDERERS registry was removed after the Snakemake migration.
"""
from __future__ import annotations

from analyses import pip, cs
from analyses._common import foreground_methods, method_order, set_agg_facecolor

__all__ = [
    "foreground_methods",
    "method_order",
    "set_agg_facecolor",
    "pip",
    "cs",
]
