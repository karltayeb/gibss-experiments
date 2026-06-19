from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import polars as pl
import plot_ready


def test_build_sample_metadata_atomic():
    sims = pl.from_dicts([{"replicate": 0, "simulation": {}}, {"replicate": 1, "simulation": {}}])
    md = plot_ready.build_sample_metadata("BH", sims)
    assert md["sample_id"].to_list() == ["BH::0", "BH::1"]
    assert md["batch_hash"].to_list() == ["BH", "BH"]
