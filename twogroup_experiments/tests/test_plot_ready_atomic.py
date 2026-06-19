from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import polars as pl
import plot_ready
import core
import utils
from experiments import loader


def test_build_sample_metadata_atomic():
    sims = pl.from_dicts([{"replicate": 0, "simulation": {}}, {"replicate": 1, "simulation": {}}])
    md = plot_ready.build_sample_metadata("BH", sims)
    assert md["sample_id"].to_list() == ["BH::0", "BH::1"]
    assert md["batch_hash"].to_list() == ["BH", "BH"]


def _make_atomic_fixture(tmp_path):
    """Run one tiny batch+method end-to-end, return (fits_df, sims_df, sample_md)."""
    lib = loader.load_library(Path(__file__).resolve().parent / "fixtures" / "experiments")
    spec = loader.resolve_simulation(lib, "gaussian_p8", "ser_b2", "loc_2.0", "gaussian")
    method = loader.expand_method("cox_heavy", lib["methods"]["cox_heavy"])[0]
    reps = (0, 1)
    sims_df = utils.simulate_batch(spec, replicates=reps)
    bh = spec.hash
    fits_df = utils.fit_batch_method(spec, method_coord=method, replicates=reps).with_columns(
        pl.lit(bh).alias("batch_hash"))
    sample_md = plot_ready.build_sample_metadata(bh, sims_df)
    return fits_df, sims_df, sample_md


# pip and cs builders now live in reductions/pip.py and reductions/cs.py.
# These tests verify backward-compatible calling via the reductions package.

def test_build_pip_plot_data_atomic(tmp_path):
    from reductions import ReductionContext
    from reductions.pip import build
    fits_df, sims_df, sample_md = _make_atomic_fixture(tmp_path)
    bh = sample_md["batch_hash"][0]
    # build via the new ctx API
    ctx = ReductionContext(fits=fits_df, sims=sims_df, sample_metadata=sample_md, sim_coordinate={})
    out = build(ctx)
    assert out["sample_id"].to_list() == [f"{bh}::0", f"{bh}::1"]
    assert "pip_bin_counts" in out.columns
    assert "batch_hash" in out.columns
    assert all(v == bh for v in out["batch_hash"].to_list())


def test_build_cs_plot_data_atomic(tmp_path):
    from reductions import ReductionContext
    from reductions.cs import build
    fits_df, sims_df, sample_md = _make_atomic_fixture(tmp_path)
    bh = sample_md["batch_hash"][0]
    ctx = ReductionContext(fits=fits_df, sims=sims_df, sample_metadata=sample_md, sim_coordinate={})
    out = build(ctx)
    expected_sample_ids = {f"{bh}::0", f"{bh}::1"}
    assert set(out["sample_id"].to_list()) == expected_sample_ids
    for col in ("sample_id", "batch_hash", "method", "threshold", "l", "causal_indices"):
        assert col in out.columns
    assert all(v == bh for v in out["batch_hash"].to_list())
