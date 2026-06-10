from __future__ import annotations

from pathlib import Path

import yaml


CORRELATION_SUPERCOLLECTIONS = {
    "005-gaussian-correlation-loc-non-null",
    "005-gaussian-correlation-loc",
    "005-gaussian-correlation-scale-non-null",
    "005-gaussian-correlation-scale",
    "005-uniform-correlation-loc-non-null",
    "005-uniform-correlation-loc",
    "005-uniform-correlation-scale-non-null",
    "005-uniform-correlation-scale",
}


def _load_plot_config(name: str) -> dict:
    return yaml.safe_load((Path("plot_configs") / name).read_text()) or {}


def test_005_correlation_supercollections_are_non_null_and_with_null_pairs():
    cfg = _load_plot_config("005_correlation.yaml")

    assert set(cfg["supercollections"]) == CORRELATION_SUPERCOLLECTIONS

    collections = cfg["collections"]
    for sc_name, sc in cfg["supercollections"].items():
        coll_names = [entry["name"] for entry in sc["collections"]]
        assert len(coll_names) == 6
        assert [entry["alias"] for entry in sc["collections"]] == [
            "rho=0.00",
            "rho=0.50",
            "rho=0.80",
            "rho=0.90",
            "rho=0.95",
            "rho=0.99",
        ]

        if sc_name.endswith("-non-null"):
            assert all("__enrichment=b0_-2.00_b_2.00__" in name for name in coll_names)
            assert all("__signal=loc_2.00" in name or "__signal=scale_2.00" in name for name in coll_names)
            continue

        assert all(name in collections for name in coll_names)
        for name in coll_names:
            simulations = collections[name]["simulations"]
            assert len(simulations) == 2
            non_null, null = simulations
            assert "__enrichment=b0_-2.00_b_2.00__" in non_null
            assert "__enrichment=b0_-2.00_b_0.00__" in null
            assert non_null.replace("__enrichment=b0_-2.00_b_2.00__", "__enrichment=b0_-2.00_b_0.00__") == null
            assert "__signal=loc_2.00" in non_null or "__signal=scale_2.00" in non_null


def test_correlation_matched_positive_and_null_simulations_are_registered():
    from config import SIMULATION_BY_NAME

    for family in ("gaussian", "uniform"):
        for rho in ("0.00", "0.50", "0.80", "0.90", "0.95", "0.99"):
            for signal in ("loc_2.00", "scale_2.00"):
                for enrichment in ("b0_-2.00_b_2.00", "b0_-2.00_b_0.00"):
                    name = (
                        f"design={family}_rho_{rho}_n_500_p_100"
                        f"__enrichment={enrichment}"
                        f"__signal={signal}"
                    )
                    assert name in SIMULATION_BY_NAME


def test_scale_plot_blocks_include_minimal_no_linear_setting():
    for path in sorted(Path("plot_configs").glob("*.yaml")):
        cfg = _load_plot_config(path.name)
        for sc_name, sc in (cfg.get("supercollections") or {}).items():
            for entry in sc.get("plots", []):
                settings = entry.get("settings", [])
                if "minimal-scale" in settings:
                    assert "minimal_no_linear" in settings, (
                        f"{path}:{sc_name} has minimal-scale without minimal_no_linear"
                    )


def test_radius_plots_are_in_cs_plot_groups():
    cfg = _load_plot_config("main.yaml")

    cs_group = cfg["plot_type_groups"]["cs"]
    cs_non_null_group = cfg["plot_type_groups"]["cs_non_null"]

    assert "cs_radius_power" in cs_group
    assert "agg_cs_radius_power" in cs_group
    assert "cs_coverage_radius" in cs_non_null_group
    assert "agg_cs_coverage_radius" in cs_non_null_group
