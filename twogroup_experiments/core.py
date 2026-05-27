from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from functools import partial
import hashlib
import importlib
import inspect
import json
import math
from typing import Any, Iterable, Mapping

import numpy as np

from gibss import cox, engine, localjj, twogroup
from gibss.distributions import Normal, NormalMixture, PointMass
from gseasusie.genesets import load_gene_sets

HASH_KEY = "__spec_hash__"


@dataclass(frozen=True)
class SimulationSpec:
    name: str
    design_sampler: Any
    effect_sampler: Any
    intercept: float
    f0: Any
    f1: Any
    base_seed: int
    error_sampler: Any = None


@dataclass(frozen=True)
class MethodSpec:
    name: str
    fit_function: Any
    summarize_function: Any
    kwargs: dict[str, Any]


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
        "weights": None,
        "locs": None,
        "scales": None,
    }
    if isinstance(distribution, PointMass):
        base["value"] = float(distribution.value)
    elif isinstance(distribution, Normal):
        base["loc"] = float(distribution.loc)
        base["scale"] = float(distribution.scale)
    elif isinstance(distribution, NormalMixture):
        base["weights"] = _to_python(distribution.weights)
        base["locs"] = _to_python(distribution.locs)
        base["scales"] = _to_python(distribution.scales)
    return base


def simulation_hash(simulation_spec: SimulationSpec) -> str:
    return spec_hash(dehydrate_spec(simulation_spec))


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
    sim_hash = simulation_hash(simulation_spec)
    rng = np.random.default_rng(
        replicate_seed(simulation_spec.base_seed, sim_hash, int(replicate))
    )
    X = simulation_spec.design_sampler(rng)
    causal_indices, causal_effects = simulation_spec.effect_sampler(X, rng)
    causal_indices = np.asarray(causal_indices, dtype=int)
    causal_effects = np.asarray(causal_effects, dtype=float)

    b = np.zeros(X.shape[1], dtype=float)
    b[causal_indices] = causal_effects
    logits = float(simulation_spec.intercept) + np.asarray(X @ b, dtype=float)
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
        "null_log_likelihood": float(effect.null_log_likelihood),
        "ser_log_bf": float(np.asarray(state.ser_log_bayes_factor[l])),
        "kl": float(effect.kl),
    }


def _extract_family_state_struct(state: Any) -> dict[str, Any]:
    family_state = (
        state.family_state.inner_family_state
        if hasattr(state.family_state, "inner_family_state")
        else state.family_state
    )
    return _to_python(family_state)


def _extract_twogroup_state_struct(state: Any) -> dict[str, Any] | None:
    family_state = state.family_state
    if not hasattr(family_state, "inner_family_state"):
        return None
    return {
        "f0": _distribution_struct(family_state.f0),
        "f1": _distribution_struct(family_state.f1),
        "Ez": _to_python(family_state.Ez),
        "update_f0": bool(family_state.update_f0),
        "update_f1": bool(family_state.update_f1),
        "n_null_iter": int(family_state.n_null_iter),
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


def fit_cox_method(simulation: TwoGroupSimulation, *, threshold, time_sign, L=1):
    score = _score(simulation)
    if threshold is None:
        event_type = np.ones_like(score, dtype=int)
    else:
        event_type = (score > float(threshold)).astype(int)
    data = cox.prep_data(
        simulation.X,
        event_time=time_sign * score,
        event_type=event_type,
    )
    state = cox.initialize_state(
        data,
        L=L,
        family_state_kwargs={"estimate_prior_variance": False},
    )
    fitted = engine.fit_ibss(data, state, cox.default_schedule())
    return {
        "state": fitted,
        "threshold": threshold,
        "n_selected": int(event_type.sum()),
    }


def summarize_cox_method(
    fit_obj,
    simulation: TwoGroupSimulation,
    *,
    threshold,
    time_sign,
    L=1,
):
    del time_sign, threshold, L
    state = fit_obj["state"]
    n_effects = len(state.single_effects)
    return {
        "threshold": fit_obj["threshold"],
        "single_effects": [_extract_ser_struct(state, l) for l in range(n_effects)],
        "credible_sets": [_make_cs_struct(state, simulation, l) for l in range(n_effects)],
        "family_state": _extract_family_state_struct(state),
        "two_group_state": _extract_twogroup_state_struct(state),
        "fit_summary": _make_fit_summary_struct(state, simulation, fit_obj["n_selected"]),
    }


def fit_logistic_method(
    simulation: TwoGroupSimulation, *, response_source, threshold=None, L=1
):
    if response_source == "z":
        y = np.asarray(simulation.z, dtype=float)
    elif response_source == "score_threshold":
        if threshold is None:
            raise ValueError("score_threshold logistic method requires a threshold.")
        y = (_score(simulation) > float(threshold)).astype(float)
    else:
        raise ValueError(f"Unsupported logistic response_source: {response_source}")

    data = localjj.prep_data(simulation.X, y)
    state = localjj.initialize_state(
        data,
        L=L,
        family_state_kwargs={"estimate_prior_variance": False},
    )
    fitted = engine.fit_ibss(data, state, localjj.default_schedule())
    return {
        "state": fitted,
        "threshold": threshold,
        "n_selected": int(np.asarray(y).sum()),
    }


def summarize_logistic_method(
    fit_obj,
    simulation: TwoGroupSimulation,
    *,
    response_source,
    threshold=None,
    L=1,
):
    del response_source, threshold, L
    state = fit_obj["state"]
    n_effects = len(state.single_effects)
    return {
        "threshold": fit_obj["threshold"],
        "single_effects": [_extract_ser_struct(state, l) for l in range(n_effects)],
        "credible_sets": [_make_cs_struct(state, simulation, l) for l in range(n_effects)],
        "family_state": _extract_family_state_struct(state),
        "two_group_state": _extract_twogroup_state_struct(state),
        "fit_summary": _make_fit_summary_struct(state, simulation, fit_obj["n_selected"]),
    }


def fit_twogroup_method(
    simulation: TwoGroupSimulation,
    *,
    f1,
    L=1,
    oracle_init=False,
    n_null_iter=20,
    n_intercept_iter=20,
):
    if oracle_init:
        resolved_f1 = Normal(
            loc=simulation.f1.loc,
            scale=simulation.f1.scale,
            estimate_loc=f1.estimate_loc,
            estimate_scale=f1.estimate_scale,
        )
    else:
        resolved_f1 = simulation.f1 if f1 is None else f1
    y0 = np.full(simulation.X.shape[0], 0.5, dtype=float)
    inner_data = localjj.prep_data(simulation.X, y0)
    inner_state = localjj.initialize_state(
        inner_data,
        L=L,
        family_state_kwargs={"estimate_prior_variance": False},
    )
    data = twogroup.prep_data(simulation.X, bhat=simulation.thetahat, se=simulation.se)
    state = twogroup.initialize_state(
        data,
        inner_state=inner_state,
        f0=simulation.f0,
        f1=resolved_f1,
        n_null_iter=n_null_iter,
        n_intercept_iter=n_intercept_iter,
    )
    fitted = engine.fit_ibss(
        data,
        state,
        twogroup.default_schedule(localjj.default_schedule()),
    )
    return {
        "state": fitted,
        "threshold": None,
        "n_selected": None,
    }


def summarize_twogroup_method(
    fit_obj,
    simulation: TwoGroupSimulation,
    *,
    f1,
    L=1,
    oracle_init=False,
    n_null_iter=20,
    n_intercept_iter=20,
):
    del f1, L, oracle_init, n_null_iter, n_intercept_iter
    state = fit_obj["state"]
    n_effects = len(state.single_effects)
    return {
        "threshold": fit_obj["threshold"],
        "single_effects": [_extract_ser_struct(state, l) for l in range(n_effects)],
        "credible_sets": [_make_cs_struct(state, simulation, l) for l in range(n_effects)],
        "family_state": _extract_family_state_struct(state),
        "two_group_state": _extract_twogroup_state_struct(state),
        "fit_summary": _make_fit_summary_struct(state, simulation, fit_obj["n_selected"]),
    }


TWOGROUP_DEFAULT_F1INIT = Normal(
    loc=0.0,
    scale=1.0,
    estimate_loc=True,
    estimate_scale=True,
)

TWOGROUP_SCALE_FAM_F1INIT = Normal(
    loc=0.0,
    scale=1.0,
    estimate_loc=False,
    estimate_scale=True,
)

TWOGROUP_LOC_FAM_F1INIT = Normal(
    loc=0.0,
    scale=0.1,
    estimate_loc=True,
    estimate_scale=False,
)


COX_HEAVY = MethodSpec(
    name="cox_heavy_L1",
    fit_function=fit_cox_method,
    summarize_function=summarize_cox_method,
    kwargs={
        "threshold": None,
        "time_sign": 1.0,
        "L": 1,
    },
)

LOGISTIC_ORACLE = MethodSpec(
    name="logistic_oracle_L1",
    fit_function=fit_logistic_method,
    summarize_function=summarize_logistic_method,
    kwargs={
        "response_source": "z",
        "threshold": None,
        "L": 1,
    },
)

_TG_ITER_KWARGS = {"n_null_iter": 20, "n_intercept_iter": 20}

TWOGROUP_ORACLE = MethodSpec(
    name="twogroup_oracle_L1",
    fit_function=fit_twogroup_method,
    summarize_function=summarize_twogroup_method,
    kwargs={"f1": None, "L": 1, **_TG_ITER_KWARGS},
)

TWOGROUP = MethodSpec(
    name="twogroup_L1",
    fit_function=fit_twogroup_method,
    summarize_function=summarize_twogroup_method,
    kwargs={"f1": TWOGROUP_DEFAULT_F1INIT, "L": 1, **_TG_ITER_KWARGS},
)

TWOGROUP_ORACLE_INIT = MethodSpec(
    name="twogroup_oracle_init_L1",
    fit_function=fit_twogroup_method,
    summarize_function=summarize_twogroup_method,
    kwargs={"f1": TWOGROUP_DEFAULT_F1INIT, "oracle_init": True, "L": 1, **_TG_ITER_KWARGS},
)

TWOGROUP_SCALE_FAM = MethodSpec(
    name="twogroup_scale_fam_L1",
    fit_function=fit_twogroup_method,
    summarize_function=summarize_twogroup_method,
    kwargs={"f1": TWOGROUP_SCALE_FAM_F1INIT, "L": 1, **_TG_ITER_KWARGS},
)

TWOGROUP_LOC_FAM = MethodSpec(
    name="twogroup_loc_fam_L1",
    fit_function=fit_twogroup_method,
    summarize_function=summarize_twogroup_method,
    kwargs={"f1": TWOGROUP_LOC_FAM_F1INIT, "L": 1, **_TG_ITER_KWARGS},
)

LOGISTIC_THRESHOLD_2_0 = MethodSpec(
    name="logistic_threshold_L1",
    fit_function=fit_logistic_method,
    summarize_function=summarize_logistic_method,
    kwargs={
        "response_source": "score_threshold",
        "threshold": 2.0,
        "L": 1,
    },
)

COX_LIGHT_THRESHOLD_2_0 = MethodSpec(
    name="cox_light_threshold_L1",
    fit_function=fit_cox_method,
    summarize_function=summarize_cox_method,
    kwargs={
        "threshold": 2.0,
        "time_sign": -1.0,
        "L": 1,
    },
)


def run_method_spec(method_spec: MethodSpec, simulation: TwoGroupSimulation):
    return method_spec.fit_function(simulation, **method_spec.kwargs)


def summarize_method_spec(
    method_spec: MethodSpec,
    fit_obj,
    simulation: TwoGroupSimulation,
) -> dict[str, Any]:
    row = method_spec.summarize_function(
        fit_obj,
        simulation,
        **method_spec.kwargs,
    )
    return {"method": method_spec.name, **row}


def identity_design_sampler(rng: np.random.Generator) -> np.ndarray:
    del rng
    return np.eye(3, dtype=float)


def hallmark_gene_sets_X(rng: np.random.Generator) -> np.ndarray:
    del rng
    gene_sets = load_gene_sets(source="msigdb", collection="h.all")
    return gene_sets.to_sparse()


def c4_gene_sets_X(rng: np.random.Generator) -> np.ndarray:
    del rng
    gene_sets = load_gene_sets(source="msigdb", collection="c4.all")
    return gene_sets.to_sparse()


def gaussian_markov_X(
    rng: np.random.Generator, *, n: int, p: int, rho: float
) -> np.ndarray:
    """
    Generate ``n`` independent Gaussian Markov chains of length ``p``.

    Each row is a stationary AR(1) process across columns with
    ``X[i, j + 1] | X[i, j] ~ N(rho * X[i, j], 1 - rho**2)``.
    """
    if n < 0 or p < 0:
        raise ValueError("n and p must be non-negative.")
    if abs(rho) > 1:
        raise ValueError("gaussian_markov_X requires |rho| <= 1.")
    X = np.empty((n, p), dtype=float)
    if n == 0 or p == 0:
        return X
    X[:, 0] = rng.normal(size=n)
    innovation_scale = float(np.sqrt(max(0.0, 1.0 - rho**2)))
    for j in range(1, p):
        X[:, j] = rho * X[:, j - 1] + innovation_scale * rng.normal(size=n)
    return X


def bernoulli_markov_X(
    n: int, p: int, m: int, prob: float, rho: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Generate ``n`` independent binomial Markov chains of length ``p``.

    The returned matrix has entries in ``{0, ..., m}``. Each row is built as
    the sum of ``m`` independent Bernoulli Markov chains with stationary
    success probability ``prob``. Adjacent-column correlation is ``rho``.
    """
    if n < 0 or p < 0:
        raise ValueError("n and p must be non-negative.")
    if m < 0:
        raise ValueError("m must be non-negative.")
    if not (0.0 <= prob <= 1.0):
        raise ValueError("prob must be in [0, 1].")
    if not (0.0 <= rho <= 1.0):
        raise ValueError("bernoulli_markov_X requires rho in [0, 1].")
    X = np.empty((n, p), dtype=int)
    if n == 0 or p == 0:
        return X
    if m == 0:
        X.fill(0)
        return X

    states = rng.binomial(1, prob, size=(n, m)).astype(int)
    X[:, 0] = states.sum(axis=1)
    for j in range(1, p):
        copy_mask = rng.random(size=(n, m)) < rho
        redraw = rng.binomial(1, prob, size=(n, m)).astype(int)
        states = np.where(copy_mask, states, redraw)
        X[:, j] = states.sum(axis=1)
    return X


def uniform_markov_X(
    rng: np.random.Generator, *, n: int, p: int, rho: float
) -> np.ndarray:
    """
    Generate ``n`` independent uniform Markov chains of length ``p``.

    Each row is formed by applying the Gaussian CDF coordinatewise to a
    stationary Gaussian AR(1) chain. Marginals are Uniform(0, 1), and the
    within-row dependence is induced by a Gaussian copula with latent
    adjacent-column correlation ``rho``.
    """
    gaussian_X = gaussian_markov_X(rng, n=n, p=p, rho=rho)
    gaussian_cdf = np.vectorize(
        lambda x: 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0))),
        otypes=[float],
    )
    return gaussian_cdf(gaussian_X)


def null_enrich_X(rng: np.random.Generator, *, n: int = 4384) -> np.ndarray:
    del rng
    return np.zeros((n, 1), dtype=float)


def uniform_single_effect(
    X: np.ndarray,
    rng: np.random.Generator,
    causal_effect: float,
) -> tuple[list[int], list[float]]:
    index = int(rng.integers(0, X.shape[1]))
    return [index], [float(causal_effect)]


def t_error_sampler(
    rng: np.random.Generator,
    se: np.ndarray,
    *,
    df: float,
) -> np.ndarray:
    """Standardized t-distributed error: unit variance regardless of df."""
    scale = se * np.sqrt((df - 2.0) / df)
    return rng.standard_t(df, size=len(se)) * scale


def canonicalize_node(node: Any) -> Any:
    if isinstance(node, dict) and HASH_KEY in node:
        node = {key: value for key, value in node.items() if key != HASH_KEY}
    if is_dataclass(node):
        return {
            "type": "dataclass",
            "path": _callable_path(type(node)),
            "fields": {
                field.name: canonicalize_node(getattr(node, field.name))
                for field in fields(node)
            },
        }
    if isinstance(node, tuple) and hasattr(node, "_fields"):
        return {
            "type": "namedtuple",
            "path": _callable_path(type(node)),
            "fields": {
                field: canonicalize_node(getattr(node, field)) for field in node._fields
            },
        }
    if isinstance(node, tuple):
        return {
            "type": "tuple",
            "items": [canonicalize_node(value) for value in node],
        }
    if isinstance(node, dict):
        return {str(key): canonicalize_node(value) for key, value in node.items()}
    if isinstance(node, list):
        return [canonicalize_node(value) for value in node]
    if isinstance(node, np.ndarray):
        return [canonicalize_node(value) for value in node.tolist()]
    if isinstance(node, np.generic):
        return canonicalize_node(node.item())
    if hasattr(node, "tolist") and not isinstance(node, (str, bytes)):
        try:
            return canonicalize_node(node.tolist())
        except Exception:
            pass
    if isinstance(node, (str, int, float, bool)) or node is None:
        return node
    raise TypeError(f"Unsupported node for canonicalization: {type(node)!r}")


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


def _load_callable(path: str) -> Any:
    module_name, qualname = path.split(":", maxsplit=1)
    value = importlib.import_module(module_name)
    for attr in qualname.split("."):
        value = getattr(value, attr)
    return value


def _dehydrate_constructed_instance(node: Any) -> dict[str, Any] | None:
    if isinstance(node, type):
        return None
    if callable(node):
        return None
    node_type = type(node)
    if node_type.__module__ == "builtins":
        return None

    try:
        signature = inspect.signature(node_type)
    except (TypeError, ValueError):
        return None

    kwargs: dict[str, Any] = {}
    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            return None
        if hasattr(node, parameter.name):
            kwargs[parameter.name] = dehydrate_node(getattr(node, parameter.name))
            continue
        if parameter.default is inspect._empty:
            return None

    return {
        "type": "partial",
        "call": True,
        "func": dehydrate_node(node_type),
        "args": dehydrate_node(()),
        "kwargs": kwargs,
    }


def dehydrate_node(node: Any) -> Any:
    if is_dataclass(node):
        return {
            "type": "dataclass",
            "path": _callable_path(type(node)),
            "fields": {
                field.name: dehydrate_node(getattr(node, field.name))
                for field in fields(node)
            },
        }
    if isinstance(node, tuple) and hasattr(node, "_fields"):
        return {
            "type": "namedtuple",
            "path": _callable_path(type(node)),
            "fields": {
                field: dehydrate_node(getattr(node, field)) for field in node._fields
            },
        }
    if isinstance(node, tuple):
        return {
            "type": "tuple",
            "items": [dehydrate_node(value) for value in node],
        }
    if isinstance(node, partial):
        return {
            "type": "partial",
            "func": dehydrate_node(node.func),
            "args": dehydrate_node(node.args),
            "kwargs": dehydrate_node(node.keywords or {}),
        }
    constructed = _dehydrate_constructed_instance(node)
    if constructed is not None:
        return constructed
    if callable(node):
        return {"type": "callable", "path": _callable_path(node)}
    if isinstance(node, dict):
        return {str(key): dehydrate_node(value) for key, value in node.items()}
    if isinstance(node, (list, np.ndarray, np.generic)):
        return canonicalize_node(node)
    if hasattr(node, "tolist") and not isinstance(node, (str, bytes)):
        try:
            return canonicalize_node(node.tolist())
        except Exception:
            pass
    if isinstance(node, (str, int, float, bool)) or node is None:
        return node
    raise TypeError(f"Unsupported node for dehydration: {type(node)!r}")


def dehydrate_hashed(node: Any) -> dict[str, Any]:
    dehydrated = dehydrate_node(node)
    if not isinstance(dehydrated, dict):
        raise TypeError(
            "dehydrate_hashed requires an object that dehydrates to a dict."
        )
    return {**dehydrated, HASH_KEY: spec_hash(dehydrated)}


def dehydrate_spec(spec: SimulationSpec) -> dict[str, Any]:
    result = {}
    for field in fields(spec):
        value = getattr(spec, field.name)
        if value is None and field.default is None:
            continue
        result[field.name] = dehydrate_node(value)
    return result


def dehydrate_simulation_semantics(spec: SimulationSpec) -> dict[str, Any]:
    return {
        "design_sampler": dehydrate_node(spec.design_sampler),
        "effect_sampler": dehydrate_node(spec.effect_sampler),
        "intercept": dehydrate_node(spec.intercept),
        "f0": dehydrate_node(spec.f0),
        "f1": dehydrate_node(spec.f1),
        "base_seed": dehydrate_node(spec.base_seed),
    }


def rehydrate_node(node: Any) -> Any:
    if isinstance(node, list):
        return [rehydrate_node(value) for value in node]
    if not isinstance(node, dict):
        return node
    if HASH_KEY in node:
        node = {key: value for key, value in node.items() if key != HASH_KEY}

    node_type = node.get("type")
    if node_type == "callable":
        return _load_callable(node["path"])
    if node_type == "partial":
        func = rehydrate_node(node["func"])
        args = rehydrate_node(node["args"])
        kwargs = rehydrate_node(node["kwargs"])
        if node.get("call", False):
            return func(*args, **kwargs)
        return partial(func, *args, **kwargs)
    if node_type == "tuple":
        return tuple(rehydrate_node(value) for value in node["items"])
    if node_type == "dataclass":
        dataclass_type = _load_callable(node["path"])
        values = {
            key: rehydrate_node(value) for key, value in node.get("fields", {}).items()
        }
        return dataclass_type(**values)
    if node_type == "namedtuple":
        namedtuple_type = _load_callable(node["path"])
        fields = {
            key: rehydrate_node(value) for key, value in node.get("fields", {}).items()
        }
        return namedtuple_type(**fields)
    return {key: rehydrate_node(value) for key, value in node.items()}


def rehydrate_spec(node: dict[str, Any]) -> SimulationSpec:
    canonical_node = canonicalize_node(node)
    return SimulationSpec(
        name=str(canonical_node["name"]),
        design_sampler=rehydrate_node(canonical_node["design_sampler"]),
        effect_sampler=rehydrate_node(canonical_node["effect_sampler"]),
        intercept=float(canonical_node["intercept"]),
        f0=rehydrate_node(canonical_node["f0"]),
        f1=rehydrate_node(canonical_node["f1"]),
        base_seed=int(canonical_node["base_seed"]),
        error_sampler=rehydrate_node(canonical_node["error_sampler"])
            if "error_sampler" in canonical_node else None,
    )


def canonical_json_bytes(node: Any) -> bytes:
    return json.dumps(
        canonicalize_node(node),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def spec_hash(spec_node: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(spec_node)).hexdigest()


def _iter_specs(
    specs: Iterable[SimulationSpec] | Mapping[str, SimulationSpec],
) -> Iterable[SimulationSpec]:
    if isinstance(specs, Mapping):
        return specs.values()
    return specs


def build_hash_registry(
    specs: Iterable[SimulationSpec] | Mapping[str, SimulationSpec],
) -> dict[str, SimulationSpec]:
    registry: dict[str, SimulationSpec] = {}
    semantics_by_hash: dict[str, dict[str, Any]] = {}
    for spec in _iter_specs(specs):
        digest = spec_hash(dehydrate_spec(spec))
        semantics = canonicalize_node(dehydrate_spec(spec))
        if digest in semantics_by_hash and semantics_by_hash[digest] != semantics:
            raise ValueError(f"Inconsistent semantic spec for hash {digest}")
        semantics_by_hash.setdefault(digest, semantics)
        registry.setdefault(digest, spec)
    return registry


def build_alias_registry(
    specs: Iterable[SimulationSpec] | Mapping[str, SimulationSpec],
) -> dict[str, str]:
    if isinstance(specs, Mapping):
        registry: dict[str, str] = {}
        for alias, spec in specs.items():
            if alias in registry:
                raise ValueError(f"Duplicate alias: {alias}")
            registry[alias] = spec_hash(dehydrate_spec(spec))
        return registry

    registry: dict[str, str] = {}
    raise TypeError(
        "build_alias_registry now requires an alias-to-spec mapping, not bare specs."
    )


def alias_to_hash_mapping(
    simulation_dispatch: Mapping[str, SimulationSpec],
) -> dict[str, str]:
    return build_alias_registry(simulation_dispatch)
