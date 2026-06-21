from __future__ import annotations

import os
from pathlib import Path

import yaml

from scripts import symlink_plots


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=True))


def test_link_plot_config_links_only_declared_supercollections(tmp_path: Path) -> None:
    plot_configs = tmp_path / "plot_configs"
    plot_configs.mkdir()
    _write_yaml(
        plot_configs / "main.yaml",
        {
            "plot_type_groups": {"standard": ["power_fdp", "pip_calibration"]},
            "settings": {"minimal": None},
        },
    )
    _write_yaml(
        plot_configs / "003_loc_snr.yaml",
        {
            "supercollections": {
                "loc-sc": {
                    "plots": [
                        {
                            "settings": ["minimal"],
                            "plot_type_groups": ["standard"],
                        }
                    ]
                }
            }
        },
    )
    _write_yaml(
        plot_configs / "004_scale_snr.yaml",
        {
            "supercollections": {
                "scale-sc": {
                    "plots": [
                        {
                            "settings": ["minimal"],
                            "plot_types": ["power_fdp"],
                        }
                    ]
                }
            }
        },
    )

    results = tmp_path / "results"
    loc_pdf = results / "supercollections" / "loc-sc" / "power_fdp" / "minimal.pdf"
    loc_pdf.parent.mkdir(parents=True)
    loc_pdf.write_text("loc")
    missing_pdf = results / "supercollections" / "loc-sc" / "pip_calibration" / "minimal.pdf"
    scale_pdf = results / "supercollections" / "scale-sc" / "power_fdp" / "minimal.pdf"
    scale_pdf.parent.mkdir(parents=True)
    scale_pdf.write_text("scale")

    result = symlink_plots.link_plot_config(
        "003_loc_snr",
        results=results,
        plot_configs_dir=plot_configs,
    )

    by_type = results / "plots" / "by_type" / "power_fdp" / "minimal" / "loc-sc.pdf"
    by_sc = results / "plots" / "by_sc" / "loc-sc" / "minimal" / "power_fdp.pdf"
    scale_link = results / "plots" / "by_type" / "power_fdp" / "minimal" / "scale-sc.pdf"

    assert result.linked == 2
    assert result.missing == 1
    assert result.removed == 0
    assert by_type.is_symlink()
    assert Path(os.readlink(by_type)) == Path(os.path.relpath(loc_pdf, by_type.parent))
    assert by_sc.is_symlink()
    assert not scale_link.exists()
    assert not missing_pdf.exists()


def test_purge_plot_config_removes_only_declared_symlinks(tmp_path: Path) -> None:
    plot_configs = tmp_path / "plot_configs"
    plot_configs.mkdir()
    _write_yaml(
        plot_configs / "main.yaml",
        {"plot_type_groups": {"standard": ["power_fdp"]}},
    )
    _write_yaml(
        plot_configs / "004_scale_snr.yaml",
        {
            "supercollections": {
                "scale-sc": {
                    "plots": [
                        {
                            "settings": ["minimal"],
                            "plot_type_groups": ["standard"],
                        }
                    ]
                }
            }
        },
    )

    results = tmp_path / "results"
    target = results / "supercollections" / "scale-sc" / "power_fdp" / "minimal.pdf"
    target.parent.mkdir(parents=True)
    target.write_text("scale")
    keep_target = results / "supercollections" / "other-sc" / "power_fdp" / "minimal.pdf"
    keep_target.parent.mkdir(parents=True)
    keep_target.write_text("other")

    remove_link = results / "plots" / "by_type" / "power_fdp" / "minimal" / "scale-sc.pdf"
    keep_link = results / "plots" / "by_type" / "power_fdp" / "minimal" / "other-sc.pdf"
    remove_link.parent.mkdir(parents=True)
    remove_link.symlink_to(os.path.relpath(target, remove_link.parent))
    keep_link.symlink_to(os.path.relpath(keep_target, keep_link.parent))

    result = symlink_plots.purge_plot_config(
        "004_scale_snr",
        results=results,
        plot_configs_dir=plot_configs,
    )

    assert result.removed == 1
    assert result.linked == 0
    assert result.missing == 0
    assert not remove_link.exists()
    assert keep_link.is_symlink()
