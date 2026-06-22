"""analyses package: renderer family modules + assembled ANALYSIS_RENDERERS registry."""
from __future__ import annotations

from analyses import pip, cs, logbf, f1
from analyses import paired
from analyses._common import foreground_methods, method_order, set_agg_facecolor

ANALYSIS_RENDERERS: dict = {
    **pip.RENDERERS,
    **cs.RENDERERS,
    **logbf.RENDERERS,
    **f1.RENDERERS,
    **paired.RENDERERS,
}

__all__ = [
    "ANALYSIS_RENDERERS",
    "foreground_methods",
    "method_order",
    "set_agg_facecolor",
    "pip",
    "cs",
    "logbf",
    "f1",
    "paired",
]
