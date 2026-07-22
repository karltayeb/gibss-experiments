from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
import hashlib
import json
from typing import Any

import numpy as np

from gibss.distributions import Normal, NormalMixture, PointMass
from simulations.distributions import Exponential
@dataclass
class SimulationSpec:
    design_sampler: Any
    effect_sampler: Any
    intercept: float
    f0: Any
    f1: Any
    error_sampler: Any
    base_seed: int
    hash: str
    name: str = ""
    membership: str = "stochastic"


@dataclass(frozen=True)
class TwoGroupSimulation:
    X: np.ndarray
    intercept: float
    causal_indices: list[int]
    causal_effects: list[float]
    b: np.ndarray
    z: np.ndarray
    theta: np.ndarray
    thetahat: np.ndarray
    se: np.ndarray
    f0: Any
    f1: Any


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _to_python(value: Any) -> Any:
    if is_dataclass(value):
        return _to_python(asdict(value))
    if isinstance(value, dict):
        return {key: _to_python(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(val) for val in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _distribution_struct(distribution: Any) -> dict[str, Any]:
    base = {
        "type": type(distribution).__name__,
        "value": None,
        "loc": None,
        "scale": None,
        "rate": None,
        "weights": None,
        "locs": None,
        "scales": None,
    }
    if isinstance(distribution, PointMass):
        base["value"] = float(distribution.value)
    elif isinstance(distribution, Normal):
        base["loc"] = float(distribution.loc)
        base["scale"] = float(distribution.scale)
    elif isinstance(distribution, Exponential):
        base["rate"] = float(distribution.rate)
    elif isinstance(distribution, NormalMixture):
        base["weights"] = _to_python(distribution.weights)
        base["locs"] = _to_python(distribution.locs)
        base["scales"] = _to_python(distribution.scales)
    return base


def replicate_seed(base_seed: int, simulation_hash: str, replicate: int) -> int:
    digest = hashlib.sha256(
        f"{int(base_seed)}:{simulation_hash}:{int(replicate)}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def simulate(simulation_spec: SimulationSpec, replicate: int):
    """
    Planned refactor contract:

    - `SimulationSpec` is pure data.
    - `simulate(simulation_spec, replicate)` materializes one `TwoGroupSimulation`.
    - the realized simulation may contain `X`.
    - realized simulations are regenerated on demand rather than cached.
    - fit results, not simulations, are the intended cache boundary.
    """
    rng = np.random.default_rng(
        replicate_seed(simulation_spec.base_seed, simulation_spec.hash, int(replicate))
    )
    X = simulation_spec.design_sampler(rng)
    causal_indices, causal_effects = simulation_spec.effect_sampler(X, rng)
    causal_indices = np.asarray(causal_indices, dtype=int)
    causal_effects = np.asarray(causal_effects, dtype=float)

    b = np.zeros(X.shape[1], dtype=float)
    b[causal_indices] = causal_effects
    logits = float(simulation_spec.intercept) + np.asarray(X @ b, dtype=float)
    if simulation_spec.membership == "deterministic":
        z = (logits > 0).astype(int)
    else:
        z = rng.binomial(1, _sigmoid(logits)).astype(int)

    theta = np.empty(X.shape[0], dtype=float)
    n_null = int(np.sum(z == 0))
    n_signal = int(np.sum(z == 1))
    theta[z == 0] = np.asarray(simulation_spec.f0.sample(rng, size=n_null), dtype=float)
    theta[z == 1] = np.asarray(
        simulation_spec.f1.sample(rng, size=n_signal), dtype=float
    )

    se = np.ones(X.shape[0], dtype=float)
    if simulation_spec.error_sampler is None:
        noise = rng.normal(scale=se, size=X.shape[0])
    else:
        noise = simulation_spec.error_sampler(rng, se)
    thetahat = theta + noise

    return TwoGroupSimulation(
        X=X,
        intercept=float(simulation_spec.intercept),
        causal_indices=causal_indices.tolist(),
        causal_effects=causal_effects.tolist(),
        b=b,
        z=z,
        theta=theta,
        thetahat=thetahat,
        se=se,
        f0=simulation_spec.f0,
        f1=simulation_spec.f1,
    )


def _score(simulation: TwoGroupSimulation) -> np.ndarray:
    return np.abs(
        np.asarray(simulation.thetahat, dtype=float)
        / np.asarray(simulation.se, dtype=float)
    )


def _extract_ser_struct(state: Any, l: int) -> dict[str, Any]:
    effect = state.single_effects[l]
    return {
        "mu": _to_python(effect.mu),
        "var": _to_python(effect.var),
        "alpha": _to_python(effect.alpha),
        "prior_variance": float(effect.prior_variance),
        "marginal_log_likelihood": float(effect.marginal_log_likelihood),
        "null_log_likelihood": float(effect.null_log_marginal),
        "ser_log_bf": float(np.asarray(state.ser_log_bayes_factor[l])),
        "kl": float(effect.kl),
    }


# Family-state fields that hold response/distribution/array OBJECTS, not scalars:
# skipped when serializing the flat family_state struct (distributions are captured
# separately by `_extract_twogroup_state_struct`; the response and per-row arrays are
# not needed downstream and would not JSON-serialize cleanly).
_FAMILY_STATE_SKIP = frozenset({"response", "f0", "f1", "llr", "row_param"})


def _extract_family_state_struct(state: Any) -> dict[str, Any]:
    """Flat, JSON-serializable snapshot of the (daf5a24) engine family state.

    The glm-based families (logistic/twogroup) carry the *string* enrichment axis
    on ``intercept`` ("shared"/"profiled"/"null") and the fitted *value* on
    ``intercept_value``; the linear family carries the numeric intercept directly on
    ``intercept``. Normalize so downstream always finds a numeric ``intercept`` (the
    string axis is preserved as ``intercept_kind``).
    """
    fs = state.family_state
    out: dict[str, Any] = {}
    for field in fs.__dataclass_fields__:  # type: ignore[attr-defined]
        if field in _FAMILY_STATE_SKIP:
            continue
        out[field] = _to_python(getattr(fs, field))
    if hasattr(fs, "intercept_value"):
        out["intercept_kind"] = out.pop("intercept", None)
        out["intercept"] = float(fs.intercept_value)
    return out


def _extract_twogroup_state_struct(state: Any) -> dict[str, Any] | None:
    """Two-group f0/f1 + posterior enrichment, or None for non-twogroup families.

    The daf5a24 twogroup family marginalizes z analytically: f0/f1 live directly on
    the family state (no inner/outer nesting) and the posterior enrichment is
    ``twogroup.compute_Ez(state)``.
    """
    from gibss import twogroup as _twogroup

    fs = state.family_state
    if getattr(fs, "llr", None) is None or not hasattr(fs, "f0"):
        return None
    return {
        "f0": _distribution_struct(fs.f0),
        "f1": _distribution_struct(fs.f1),
        "Ez": _to_python(np.asarray(_twogroup.compute_Ez(state))),
        "update_f0": bool(
            getattr(fs.f0, "estimate_loc", False)
            or getattr(fs.f0, "estimate_scale", False)
        ),
        "update_f1": bool(
            getattr(fs.f1, "estimate_loc", False)
            or getattr(fs.f1, "estimate_scale", False)
        ),
        "n_null_iter": int(getattr(fs, "init_em_iters", 0)),
    }


def _make_cs_struct(
    state: Any, simulation: TwoGroupSimulation, l: int, coverage: float = 0.95
) -> dict[str, Any]:
    alpha = np.asarray(state.single_effects[l].alpha, dtype=float)
    cs = tuple(int(idx) for idx in state.get_credible_sets(coverage=coverage)[l])
    top_feature = int(np.argmax(alpha))
    causal_indices = [int(idx) for idx in simulation.causal_indices]
    return {
        "cs": list(cs),
        "cs_size": len(cs),
        "causal_indices": causal_indices,
        "causal_in_cs": any(idx in cs for idx in causal_indices),
        "top_feature": top_feature,
        "top_feature_is_causal": top_feature in causal_indices,
    }


def _make_fit_summary_struct(
    state: Any, simulation: TwoGroupSimulation, n_selected: int | None
) -> dict[str, Any]:
    return {
        "n_selected": None if n_selected is None else int(n_selected),
        "n_iter": int(state.n_iter),
        "converged": bool(state.converged),
    }


def _callable_path(func: Any) -> str:
    module_name = getattr(func, "__module__", None)
    qualname = getattr(func, "__qualname__", None)
    if (
        not module_name
        or not qualname
        or "<locals>" in qualname
        or "<lambda>" in qualname
    ):
        raise TypeError(
            "Only importable top-level functions are supported in simulation specs."
        )
    return f"{module_name}:{qualname}"


def canonical_json_bytes(node: Any) -> bytes:
    """Stable JSON serialization for plain-Python coordinate dicts."""
    return json.dumps(
        node,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def spec_hash(spec_node: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(spec_node)).hexdigest()


# ---------------------------------------------------------------------------
# Re-exports from simulations/ sub-package
# (keeps getattr(core, name) working; inspect.getfile follows the real definition)
# ---------------------------------------------------------------------------
from simulations.design.markov import gaussian_markov_X, uniform_markov_X
from simulations.design.genesets import hallmark_gene_sets_X, c4_gene_sets_X, msigdb_gene_sets_X, gobp_gene_sets_X
from simulations.design.degenerate import null_enrich_X
from simulations.effect.effects import uniform_single_effect, sized_single_effect
from simulations.error.errors import noiseless_error_sampler, t_error_sampler

# ---------------------------------------------------------------------------
# Re-exports from fits/ sub-package
# (keeps core.<name> working; inspect.getfile follows the real definition)
# ---------------------------------------------------------------------------
from fits.cox import fit_cox_method, summarize_cox_method, run_cox_method
from fits.logistic import fit_logistic_method, summarize_logistic_method, run_logistic_method
from fits.twogroup import fit_twogroup_method, summarize_twogroup_method, run_twogroup_method
from fits.linear import fit_linear_method, summarize_linear_method, run_linear_method
