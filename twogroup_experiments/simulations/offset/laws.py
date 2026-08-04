"""Random-offset laws for the linear predictor.

The offset is a mean-zero-*by-default* (but general) per-observation term added to
the logit before ``z`` is drawn:  ``eta_i = intercept + (X b)_i + psi_i``.  It stands
in for the leave-one-out random offset a SER sees inside a SuSiE fit; the experiment
asks when a point estimate of it suffices vs. integrating over it.

An ``OffsetLaw`` is a *per-observation* distribution.  The simulation draws ``psi``
from it (via ``sample``) but stores only the LAW, never the realized draw - a fit
legitimately knows the offset distribution, never its realization.  Each fit reduces
the law to what its quadrature needs:

    fixed-offset (point)  -> law.mean            (n,)
    Gaussian quadrature   -> law.mean, law.variance   (n,), (n,)
    mixture quadrature    -> law.mixture()        (weights, means, vars), each (n, K)

Every array is per-row ``(n,)`` or ``(n, K)``; i.i.d. / homoskedastic / mean-zero are
just degenerate shapes of the same objects.  See
docs/superpowers/specs/2026-07-27-response-transform-design.md for the sibling
response-transform seam this mirrors.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2


class OffsetLaw(ABC):
    """A per-observation offset distribution. Implementations expose the three
    reductions the fits consume (``mean``, ``variance``, ``mixture``) plus ``sample``
    (used only inside ``simulate``) and ``summary`` (compact record for serialization).
    """

    @property
    @abstractmethod
    def mean(self) -> np.ndarray:  # (n,)  E[psi_i]
        ...

    @property
    @abstractmethod
    def variance(self) -> np.ndarray:  # (n,)  Var[psi_i]
        ...

    @abstractmethod
    def mixture(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(weights, means, vars), each (n, K): a Gaussian-mixture representation.
        Exact for the Gaussian family; a finite scale-mixture approximation for t."""

    @abstractmethod
    def sample(self, rng: np.random.Generator) -> np.ndarray:  # (n,)
        ...

    @abstractmethod
    def summary(self) -> dict:
        """JSON-able record (kind + per-row moments) for the simulation struct."""


@dataclass(frozen=True)
class GaussianMixtureOffsetLaw(OffsetLaw):
    """Per-row Gaussian mixture. ``weights``/``means``/``vars`` are all ``(n, K)``.

    Covers every Gaussian case as a shape: K=1 -> (heteroskedastic) Gaussian; K>1 with
    a shared ``means`` column -> zero-mean scale mixture; K>1 with distinct ``means`` ->
    multimodal.  ``mixture()`` is exact.
    """

    weights: np.ndarray  # (n, K), rows sum to 1
    means: np.ndarray    # (n, K)
    vars: np.ndarray     # (n, K)

    @property
    def mean(self) -> np.ndarray:
        return np.sum(self.weights * self.means, axis=1)

    @property
    def variance(self) -> np.ndarray:
        second = np.sum(self.weights * (self.vars + self.means**2), axis=1)
        return second - self.mean**2

    def mixture(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.weights, self.means, self.vars

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        n, _ = self.weights.shape
        # per-row categorical over components, then a Gaussian draw from the pick
        u = rng.random(n)
        comp = (u[:, None] < np.cumsum(self.weights, axis=1)).argmax(axis=1)
        rows = np.arange(n)
        return rng.normal(self.means[rows, comp], np.sqrt(self.vars[rows, comp]))

    def summary(self) -> dict:
        return {
            "kind": "gaussian_mixture",
            "n_components": int(self.weights.shape[1]),
            "mean": self.mean.tolist(),
            "variance": self.variance.tolist(),
        }


@dataclass(frozen=True)
class StudentTOffsetLaw(OffsetLaw):
    """Per-row location-scale Student-t: ``psi_i = means_i + scales_i * t_df``.

    ``mean``/``variance`` are exact (df>1 / df>2). ``mixture()`` returns a finite
    Gaussian scale-mixture approximation of the t (a t IS a continuous scale mixture of
    normals, ``t_nu = N(0, s^2 * nu / W)`` with ``W ~ chi^2_nu``): ``n_components``
    equal-weight quantile nodes of ``chi^2_df``.
    """

    means: np.ndarray   # (n,)
    scales: np.ndarray  # (n,)
    df: float
    n_components: int = 10

    def __post_init__(self) -> None:
        if float(self.df) <= 2.0:
            raise ValueError("StudentTOffsetLaw needs df > 2 for a finite variance.")

    @property
    def mean(self) -> np.ndarray:
        return np.asarray(self.means, dtype=float)

    @property
    def variance(self) -> np.ndarray:
        return np.asarray(self.scales, dtype=float) ** 2 * (self.df / (self.df - 2.0))

    def mixture(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # t = N(0, scale^2 * df / W), W ~ chi^2_df: equal-weight quantile nodes of chi^2_df
        # give component var = scale^2 * df / w_k. A coarse quantile grid underweights the
        # heavy-variance (small-W) tail, so rescale the per-scale variances to match the
        # exact t variance df/(df-2) - keeps the tail SHAPE, fixes the first two moments.
        k = np.arange(self.n_components)
        w_nodes = chi2.ppf((k + 0.5) / self.n_components, self.df)  # (K,)
        var_per_scale = self.df / w_nodes  # (K,) multiplied by scale_i^2 below
        var_per_scale *= (self.df / (self.df - 2.0)) / var_per_scale.mean()  # moment-match
        n = self.means.shape[0]
        weights = np.full((n, self.n_components), 1.0 / self.n_components)
        means = np.repeat(np.asarray(self.means, float)[:, None], self.n_components, axis=1)
        vars_ = (np.asarray(self.scales, float)[:, None] ** 2) * var_per_scale[None, :]
        return weights, means, vars_

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        n = self.means.shape[0]
        return np.asarray(self.means, float) + np.asarray(self.scales, float) * rng.standard_t(self.df, size=n)

    def summary(self) -> dict:
        return {
            "kind": "student_t",
            "df": float(self.df),
            "mean": self.mean.tolist(),
            "variance": self.variance.tolist(),
        }


# ---------------------------------------------------------------------------
# Offset samplers: ``f(rng, n) -> (psi, OffsetLaw)``. The per-row means are drawn
# (when ``mean_scale > 0``) and then FROZEN into the returned law (known to the fit);
# only ``psi`` (the realized draw) is hidden by ``simulate``.
# ---------------------------------------------------------------------------

def _row_means(rng: np.random.Generator, n: int, mean_scale: float) -> np.ndarray:
    """Per-row offset means mu_i ~ N(0, mean_scale^2), frozen as a known law param.
    Always draws n values (so the RNG stream is stable across mean_scale); scale 0 -> 0."""
    return float(mean_scale) * rng.standard_normal(n)


def gaussian_offset(
    rng: np.random.Generator,
    n: int,
    *,
    scale: float,
    mean_scale: float = 0.0,
) -> tuple[np.ndarray, GaussianMixtureOffsetLaw]:
    """Unimodal Gaussian offset: per-row mean ``mu_i ~ N(0, mean_scale^2)``, shared
    variance ``scale^2``.  ``mean_scale=0`` is the mean-zero scale-sweep baseline."""
    means = _row_means(rng, n, mean_scale)[:, None]                 # (n, 1)
    vars_ = np.full((n, 1), float(scale) ** 2)                       # (n, 1)
    weights = np.ones((n, 1))
    law = GaussianMixtureOffsetLaw(weights=weights, means=means, vars=vars_)
    return law.sample(rng), law


def heteroskedastic_gaussian_offset(
    rng: np.random.Generator,
    n: int,
    *,
    scale: float,
    var_cv: float,
    mean_scale: float = 0.0,
) -> tuple[np.ndarray, GaussianMixtureOffsetLaw]:
    """Heteroskedastic Gaussian: per-row mean and per-row variance.  Row variances are
    ``scale^2 * g_i`` with ``g_i`` a mean-1 Gamma of coefficient of variation ``var_cv``
    (frozen as a known law param)."""
    means = _row_means(rng, n, mean_scale)[:, None]
    shape = 1.0 / float(var_cv) ** 2
    g = rng.gamma(shape=shape, scale=1.0 / shape, size=n)            # mean 1, CV=var_cv
    vars_ = (float(scale) ** 2 * g)[:, None]
    weights = np.ones((n, 1))
    law = GaussianMixtureOffsetLaw(weights=weights, means=means, vars=vars_)
    return law.sample(rng), law


def multimodal_offset(
    rng: np.random.Generator,
    n: int,
    *,
    weights: list[float],
    locs: list[float],
    scales: list[float],
    mean_scale: float = 0.0,
) -> tuple[np.ndarray, GaussianMixtureOffsetLaw]:
    """Multimodal Gaussian mixture with distinct component means, shared across rows
    (broadcast to (n, K)); an optional per-row mean shift ``mu_i ~ N(0, mean_scale^2)``
    is added to every component's location."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    locs = np.asarray(locs, dtype=float)
    scales = np.asarray(scales, dtype=float)
    K = w.shape[0]
    shift = _row_means(rng, n, mean_scale)[:, None]                 # (n, 1)
    means = locs[None, :] + shift                                   # (n, K)
    vars_ = np.repeat((scales**2)[None, :], n, axis=0)              # (n, K)
    weights_nk = np.repeat(w[None, :], n, axis=0)                   # (n, K)
    law = GaussianMixtureOffsetLaw(weights=weights_nk, means=means, vars=vars_)
    return law.sample(rng), law


def t_offset(
    rng: np.random.Generator,
    n: int,
    *,
    scale: float,
    df: float,
    mean_scale: float = 0.0,
    n_components: int = 10,
) -> tuple[np.ndarray, StudentTOffsetLaw]:
    """Student-t offset: per-row mean, ``t_df`` errors of scale ``scale``.  The fit's
    mixture reduction approximates the t with ``n_components`` scale-mixture nodes."""
    means = _row_means(rng, n, mean_scale)
    scales = np.full(n, float(scale))
    law = StudentTOffsetLaw(means=means, scales=scales, df=float(df), n_components=int(n_components))
    return law.sample(rng), law
