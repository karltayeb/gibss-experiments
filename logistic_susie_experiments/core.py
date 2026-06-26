from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from typing import Any

import numpy as np


@dataclass
class SimulationSpec:
    """Pure-data description of a logistic simulation scenario.

    Two axes only: a ``design_sampler`` (generates X) and an ``effect_sampler``
    (the enrichment: which features are causal and with what effect). The binary
    response is drawn directly from the logistic model — there is no z-score /
    f0/f1 / error layer (that belongs to twogroup_experiments).
    """

    design_sampler: Any
    effect_sampler: Any
    intercept: float
    base_seed: int
    hash: str
    name: str = ""


@dataclass(frozen=True)
class LogisticSimulation:
    X: Any
    intercept: float
    causal_indices: list[int]
    causal_effects: list[float]
    b: np.ndarray
    y: np.ndarray


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


def replicate_seed(base_seed: int, simulation_hash: str, replicate: int) -> int:
    digest = hashlib.sha256(
        f"{int(base_seed)}:{simulation_hash}:{int(replicate)}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def simulate(simulation_spec: SimulationSpec, replicate: int) -> LogisticSimulation:
    """Materialize one logistic simulation for the given replicate.

    ``y ~ Bernoulli(sigmoid(intercept + X @ b))`` — well-specified for logistic
    SuSiE. Realized simulations are regenerated on demand rather than cached;
    fit results are the intended cache boundary.
    """
    rng = np.random.default_rng(
        replicate_seed(simulation_spec.base_seed, simulation_spec.hash, int(replicate))
    )
    X = simulation_spec.design_sampler(rng)
    causal_indices, causal_effects = simulation_spec.effect_sampler(X, rng)
    causal_indices = np.asarray(causal_indices, dtype=int)
    causal_effects = np.asarray(causal_effects, dtype=float)

    p = int(X.shape[1])
    b = np.zeros(p, dtype=float)
    b[causal_indices] = causal_effects
    logits = float(simulation_spec.intercept) + np.asarray(X @ b, dtype=float)
    y = rng.binomial(1, _sigmoid(logits)).astype(int)

    return LogisticSimulation(
        X=X,
        intercept=float(simulation_spec.intercept),
        causal_indices=causal_indices.tolist(),
        causal_effects=causal_effects.tolist(),
        b=b,
        y=y,
    )


# ---------------------------------------------------------------------------
# Fit-state extraction (shared across logistic implementations)
# ---------------------------------------------------------------------------

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


def _make_cs_struct(
    state: Any, simulation: LogisticSimulation, l: int, coverage: float = 0.95
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
    state: Any, simulation: LogisticSimulation, n_selected: int | None
) -> dict[str, Any]:
    return {
        "n_selected": None if n_selected is None else int(n_selected),
        "n_iter": int(state.n_iter),
        "converged": bool(state.converged),
    }


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

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
# Re-exports from sub-packages
# (keeps getattr(core, name) working; inspect.getfile follows the real
# definition so Snakemake tracks the right source file)
# ---------------------------------------------------------------------------
from simulations.design.markov import gaussian_markov_X, uniform_markov_X
from simulations.design.genesets import (
    hallmark_gene_sets_X,
    c4_gene_sets_X,
    msigdb_gene_sets_X,
)
from simulations.effect.effects import uniform_single_effect, uniform_k_effects
from fits.logistic import (
    fit_logistic_method,
    summarize_logistic_method,
    run_logistic_method,
)
from fits.score import (
    fit_score_method,
    summarize_score_method,
    run_score_method,
)
from fits.block_irls import (
    fit_block_irls_method,
    summarize_block_irls_method,
    run_block_irls_method,
)
from fits.irls_steps import (
    fit_irls_method,
    summarize_irls_method,
    run_irls_method,
)
from fits.globaljj_steps import (
    fit_globaljj_method,
    summarize_globaljj_method,
    run_globaljj_method,
)
