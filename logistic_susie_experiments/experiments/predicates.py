"""Boolean predicates for filtering method coordinates and simulation descriptors.

Referenced by name in library.yaml (method_filter / simulation_filter) and
resolved at DAG-build time via loader.resolve_predicate().
"""
from __future__ import annotations

from typing import Any


def has_causal(sim_descriptor: dict[str, Any]) -> bool:
    """Return True if the simulation has a non-zero causal effect."""
    return sim_descriptor["enrichment"]["arguments"].get("causal_effect", 0) != 0
