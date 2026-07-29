"""Response transforms: ``simulation -> Response`` as a composable pipeline.

A method is handed the data it observes and fits it; it does not reach into the
simulation to pick a field. The mapping ``simulation -> observation`` is a first-
class, composable component - the response transform.

See docs/superpowers/specs/2026-07-27-response-transform-design.md.

Phase 1 covers the vector subset (glm/linear family). ``reverse``/``rank``/
``survival`` and the ``two_group`` extractor arrive with the cox/twogroup migration.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class Response:
    """What a method observes. ``values`` is the observation; ``f0``/``f1`` are the
    two-group nuisance *initialization* (default carried from the simulation, dropped
    by ``forget``). ``sim`` is never stored here - the transform has simulation access
    but the object handed to the fit stays sim-free, so it cannot reach the estimand.
    """

    values: np.ndarray | None = None
    kind: str = "vector"  # "vector" | "survival" | "two_group"
    se: np.ndarray | None = None
    event_type: np.ndarray | None = None
    f0: Any = None
    f1: Any = None


# ---------------------------------------------------------------------------
# Steps: uniform type ``(simulation, Response) -> Response``. There is no
# privileged extractor/combinator split - extractors ignore the incoming (empty)
# response, combinators ignore the simulation, ``oracle``/``forget`` use both.
# ---------------------------------------------------------------------------

def _require_values(r: Response, step: str) -> None:
    if r.values is None:
        raise ValueError(
            f"response step {step!r} needs a populated response; "
            "start the pipeline with an extractor (z/stat)."
        )


def step_z(simulation, r: Response) -> Response:
    return replace(r, values=np.asarray(simulation.z, dtype=float))


def step_stat(simulation, r: Response) -> Response:
    return replace(
        r,
        values=np.asarray(simulation.thetahat, dtype=float),
        se=np.asarray(simulation.se, dtype=float),
    )


def step_standardize(simulation, r: Response) -> Response:
    _require_values(r, "standardize")
    if r.se is None:
        raise ValueError("standardize needs `se`; start from `stat`.")
    return replace(r, values=r.values / r.se)


def step_abs(simulation, r: Response) -> Response:
    _require_values(r, "abs")
    return replace(r, values=np.abs(r.values))


def step_threshold(simulation, r: Response, *, t: float) -> Response:
    _require_values(r, "threshold")
    return replace(r, values=(r.values > float(t)).astype(float))


def step_forget(simulation, r: Response) -> Response:
    return replace(r, f0=None, f1=None)


_STEPS: dict[str, Callable[..., Response]] = {
    "z": step_z,
    "stat": step_stat,
    "standardize": step_standardize,
    "abs": step_abs,
    "threshold": step_threshold,
    "forget": step_forget,
}

# Steps whose scalar shorthand (``{threshold: 2.0}``) binds to a named argument.
_SCALAR_ARG = {"threshold": "t"}


def _bind_step(name: str, args: dict[str, Any]) -> Callable[..., Response]:
    if name not in _STEPS:
        raise KeyError(f"unknown response step {name!r}; options: {sorted(_STEPS)}")
    fn = _STEPS[name]
    return partial(fn, **args) if args else fn


def resolve_step(step: Any) -> Callable[..., Response]:
    """A step is a bare name (``"abs"``), or a single-key map with args
    (``{threshold: 2.0}`` or ``{threshold: {t: 2.0}}``)."""
    if isinstance(step, str):
        return _bind_step(step, {})
    if isinstance(step, dict) and len(step) == 1:
        (name, raw), = step.items()
        if isinstance(raw, dict):
            args = raw
        elif name in _SCALAR_ARG:
            args = {_SCALAR_ARG[name]: raw}
        else:
            raise ValueError(
                f"response step {name!r} takes no scalar shorthand; pass a map of args."
            )
        return _bind_step(name, args)
    raise ValueError(
        f"response step must be a name or single-key map, got {step!r}"
    )


def build_response_transform(spec: Any) -> Callable[[Any], Response]:
    """Compile a response spec (a step, or a list of steps) into ``simulation -> Response``.

    Examples: ``"z"``; ``[stat, standardize, abs]``; ``[stat, standardize, abs, {threshold: 2.0}]``.
    """
    steps = spec if isinstance(spec, list) else [spec]
    resolved = [resolve_step(s) for s in steps]

    def transform(simulation) -> Response:
        r = Response()
        for step in resolved:
            r = step(simulation, r)
        if r.values is None:
            raise ValueError(
                f"response pipeline {spec!r} did not populate values; "
                "start with an extractor (z/stat)."
            )
        return r

    return transform
