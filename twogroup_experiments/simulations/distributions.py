from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True, slots=True)
class Exponential:
    rate: float = 1.0

    def __post_init__(self) -> None:
        if float(self.rate) <= 0.0:
            raise ValueError("Exponential rate must be positive.")

    def sample(
        self,
        rng: np.random.Generator,
        size: int | tuple[int, ...],
    ) -> np.ndarray:
        return rng.exponential(scale=1.0 / float(self.rate), size=size)

    def log_likelihood(self, x):
        x = jnp.asarray(x)
        rate = float(self.rate)
        return jnp.where(x >= 0.0, jnp.log(rate) - rate * x, -jnp.inf)

    def log_likelihood_nm(self, bhat, se):
        del bhat, se
        raise NotImplementedError(
            "Exponential normal-means likelihood is not implemented; use it for exact-event simulations."
        )

    def update_nm(self, bhat, se, weights):
        del bhat, se, weights
        return self
