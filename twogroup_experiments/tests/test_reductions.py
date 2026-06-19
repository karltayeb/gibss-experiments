from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl
import core
import utils
import plot_ready
from experiments import loader
from reductions import ReductionContext
import reductions.pip as red_pip
import reductions.cs as red_cs
import reductions.f1 as red_f1
import reductions.enrich as red_enrich


# ---------------------------------------------------------------------------
# Shared fixture builders
# ---------------------------------------------------------------------------

def _make_cox_ctx(tmp_path):
    """Run one tiny batch+cox_heavy method end-to-end, return (ctx, sample_md, bh)."""
    lib = loader.load_library(Path(__file__).resolve().parent / "fixtures" / "experiments")
    spec = loader.resolve_simulation(lib, "gaussian_p8", "ser_b2", "loc_2.0", "gaussian")
    coord = loader.simulation_coordinate(lib, "gaussian_p8", "ser_b2", "loc_2.0", "gaussian")
    method = loader.expand_method("cox_heavy", lib["methods"]["cox_heavy"])[0]
    reps = (0, 1)
    sims_df = utils.simulate_batch(spec, replicates=reps)
    bh = spec.hash
    fits_df = utils.fit_batch_method(spec, method_coord=method, replicates=reps).with_columns(
        pl.lit(bh).alias("batch_hash")
    )
    sample_md = plot_ready.build_sample_metadata(bh, sims_df)
    ctx = ReductionContext(fits=fits_df, sims=sims_df, sample_metadata=sample_md, sim_coordinate=coord)
    return ctx, bh


def _make_twogroup_ctx(tmp_path):
    """Run one tiny batch+twogroup method end-to-end, return (ctx, sample_md, bh)."""
    lib = loader.load_library(Path(__file__).resolve().parent / "fixtures" / "experiments")
    spec = loader.resolve_simulation(lib, "gaussian_p8", "ser_b2", "loc_2.0", "gaussian")
    coord = loader.simulation_coordinate(lib, "gaussian_p8", "ser_b2", "loc_2.0", "gaussian")
    method = loader.expand_method("twogroup", lib["methods"]["twogroup"])[0]
    reps = (0, 1)
    sims_df = utils.simulate_batch(spec, replicates=reps)
    bh = spec.hash
    fits_df = utils.fit_batch_method(spec, method_coord=method, replicates=reps).with_columns(
        pl.lit(bh).alias("batch_hash")
    )
    sample_md = plot_ready.build_sample_metadata(bh, sims_df)
    ctx = ReductionContext(fits=fits_df, sims=sims_df, sample_metadata=sample_md, sim_coordinate=coord)
    return ctx, bh


# ---------------------------------------------------------------------------
# pip
# ---------------------------------------------------------------------------

def test_reduction_build_pip_atomic(tmp_path):
    ctx, bh = _make_cox_ctx(tmp_path)
    out = red_pip.build(ctx)
    assert "pip_bin_counts" in out.columns
    assert "batch_hash" in out.columns
    assert set(out["sample_id"].to_list()) == {f"{bh}::0", f"{bh}::1"}
    assert all(v == bh for v in out["batch_hash"].to_list())


# ---------------------------------------------------------------------------
# cs
# ---------------------------------------------------------------------------

def test_reduction_build_cs_atomic(tmp_path):
    ctx, bh = _make_cox_ctx(tmp_path)
    out = red_cs.build(ctx)
    expected_sample_ids = {f"{bh}::0", f"{bh}::1"}
    assert set(out["sample_id"].to_list()) == expected_sample_ids
    for col in ("sample_id", "batch_hash", "method", "threshold", "l", "causal_indices"):
        assert col in out.columns
    assert all(v == bh for v in out["batch_hash"].to_list())


# ---------------------------------------------------------------------------
# f1
# ---------------------------------------------------------------------------

def test_reduction_build_f1_atomic(tmp_path):
    ctx, bh = _make_twogroup_ctx(tmp_path)
    out = red_f1.build(ctx)
    expected_sample_ids = {f"{bh}::0", f"{bh}::1"}
    assert set(out["sample_id"].to_list()) == expected_sample_ids
    for col in ("sample_id", "batch_hash", "method", "f1_loc", "f1_scale", "true_f1_loc", "true_f1_scale"):
        assert col in out.columns
    assert all(v == bh for v in out["batch_hash"].to_list())
    # Check coordinate is read correctly: loc=2.0, scale=0.1 from fixture library.yaml
    assert out["true_f1_loc"][0] == 2.0
    assert out["true_f1_scale"][0] == 0.1


# ---------------------------------------------------------------------------
# enrich
# ---------------------------------------------------------------------------

def test_reduction_build_enrich_atomic(tmp_path):
    ctx, bh = _make_twogroup_ctx(tmp_path)
    out = red_enrich.build(ctx)
    expected_sample_ids = {f"{bh}::0", f"{bh}::1"}
    assert set(out["sample_id"].to_list()) == expected_sample_ids
    for col in ("sample_id", "batch_hash", "method", "est_intercept", "mu_at_causal", "true_intercept", "true_effect"):
        assert col in out.columns
    assert all(v == bh for v in out["batch_hash"].to_list())
