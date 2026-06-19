from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

import core
from gibss import distributions as _distributions

EXPERIMENTS_DIR = Path(__file__).resolve().parent


def format_float(value: float) -> str:
    return f"{float(value):.2f}"


def resolve_callable(name: str) -> Any:
    if not hasattr(core, name):
        raise KeyError(f"Unknown callable in core: {name!r}")
    return getattr(core, name)


def resolve_distribution(node: dict[str, Any]) -> Any:
    if not isinstance(node, dict) or len(node) != 1:
        raise ValueError(f"Distribution node must be a single-key map, got {node!r}")
    (type_name, ctor_kwargs), = node.items()
    if not hasattr(_distributions, type_name):
        raise KeyError(f"Unknown distribution type: {type_name!r}")
    return getattr(_distributions, type_name)(**(ctor_kwargs or {}))


def load_library(experiments_dir: Path | None = None) -> dict[str, Any]:
    base = Path(experiments_dir) if experiments_dir is not None else EXPERIMENTS_DIR
    data = yaml.safe_load((base / "library.yaml").read_text(encoding="utf-8")) or {}
    for section in ("defaults", "designs", "enrichments", "signals", "errors",
                    "methods", "reductions", "analyses", "analysis_groups"):
        data.setdefault(section, {})
    return data
