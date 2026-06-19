from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass
class ReductionContext:
    fits: pl.DataFrame
    sims: pl.DataFrame
    sample_metadata: pl.DataFrame
    sim_coordinate: dict
