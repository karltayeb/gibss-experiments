#!/usr/bin/env python3
"""Create symlink views of results/supercollections/{sc}/{plot_type}/{settings}.pdf.

Two views under results/plots/:
  by_type/{plot_type}/{settings}/{supercollection}.pdf   -- compare across supercollections
  by_sc/{supercollection}/{settings}/{plot_type}.pdf     -- browse one supercollection

Usage:
    uv run python scripts/symlink_plots.py
    uv run python scripts/symlink_plots.py --results /path/to/results
    uv run python scripts/symlink_plots.py link 003_loc_snr
    uv run python scripts/symlink_plots.py purge 004_scale_snr
"""
import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import yaml


DEFAULT_PLOT_CONFIGS_DIR = Path("plot_configs")


@dataclass(frozen=True)
class LinkResult:
    linked: int = 0
    removed: int = 0
    missing: int = 0


def _link(src: Path, dst: Path) -> bool:
    rel_target = os.path.relpath(src, dst.parent)
    if dst.is_symlink():
        if os.readlink(dst) == rel_target:
            return False
        dst.unlink()
    elif dst.exists():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(rel_target)
    return True


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def _merge_plot_configs(plot_configs_dir: Path) -> dict:
    merged: dict = {}
    for path in sorted(plot_configs_dir.glob("*.yaml")):
        cfg = _load_yaml(path)
        for key, val in cfg.items():
            if key not in merged:
                merged[key] = val
            elif isinstance(merged[key], dict) and isinstance(val, dict):
                merged[key].update(val)
    return merged


def _resolve_plot_config_path(plot_config: str, plot_configs_dir: Path) -> Path:
    path = Path(plot_config)
    candidates = [path]
    if path.suffix != ".yaml":
        candidates.append(path.with_suffix(".yaml"))
        candidates.append(plot_configs_dir / f"{plot_config}.yaml")
    candidates.append(plot_configs_dir / path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No plot config found for {plot_config!r}")


def _resolve_sc_plot_pairs(sc_cfg: dict, plot_type_groups: dict) -> list[tuple[str, str]]:
    seen: dict[tuple[str, str], None] = {}
    for entry in sc_cfg.get("plots", []):
        settings_list = entry.get("settings", [])
        plot_types = list(entry.get("plot_types", []))
        for group in entry.get("plot_type_groups", []):
            plot_types += plot_type_groups.get(group, [])
        for settings in settings_list:
            for plot_type in plot_types:
                seen[(settings, plot_type)] = None
    return list(seen.keys())


def _plot_link_paths(
    results: Path,
    supercollection: str,
    settings: str,
    plot_type: str,
) -> tuple[Path, Path, Path]:
    src = results / "supercollections" / supercollection / plot_type / f"{settings}.pdf"
    by_type = results / "plots" / "by_type" / plot_type / settings / f"{supercollection}.pdf"
    by_sc = results / "plots" / "by_sc" / supercollection / settings / f"{plot_type}.pdf"
    return src, by_type, by_sc


def _iter_config_plot_paths(
    plot_config: str,
    *,
    results: Path,
    plot_configs_dir: Path,
):
    path = _resolve_plot_config_path(plot_config, plot_configs_dir)
    cfg = _load_yaml(path)
    merged = _merge_plot_configs(plot_configs_dir)
    plot_type_groups = merged.get("plot_type_groups", {})

    for sc_name, sc_cfg in sorted(cfg.get("supercollections", {}).items()):
        for settings, plot_type in _resolve_sc_plot_pairs(sc_cfg, plot_type_groups):
            yield _plot_link_paths(results, sc_name, settings, plot_type)


def symlink_plots(results: Path) -> None:
    sc_root = results / "supercollections"
    by_type = results / "plots" / "by_type"
    by_sc = results / "plots" / "by_sc"

    count = 0
    for pdf in sorted(sc_root.glob("*/*/*.pdf")):
        sc = pdf.parts[-3]
        plot_type = pdf.parts[-2]
        settings = pdf.stem

        count += _link(pdf, by_type / plot_type / settings / f"{sc}.pdf")
        count += _link(pdf, by_sc / sc / settings / f"{plot_type}.pdf")

    print(f"Created/updated {count} symlinks under {results / 'plots'}")


def link_plot_config(
    plot_config: str,
    *,
    results: Path = Path("results"),
    plot_configs_dir: Path = DEFAULT_PLOT_CONFIGS_DIR,
) -> LinkResult:
    linked = 0
    missing = 0
    for src, by_type, by_sc in _iter_config_plot_paths(
        plot_config,
        results=results,
        plot_configs_dir=plot_configs_dir,
    ):
        if not src.exists():
            missing += 1
            continue
        linked += _link(src, by_type)
        linked += _link(src, by_sc)
    return LinkResult(linked=linked, missing=missing)


def purge_plot_config(
    plot_config: str,
    *,
    results: Path = Path("results"),
    plot_configs_dir: Path = DEFAULT_PLOT_CONFIGS_DIR,
) -> LinkResult:
    removed = 0
    for _src, by_type, by_sc in _iter_config_plot_paths(
        plot_config,
        results=results,
        plot_configs_dir=plot_configs_dir,
    ):
        for link in (by_type, by_sc):
            if link.is_symlink():
                link.unlink()
                removed += 1
    return LinkResult(removed=removed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=Path("results"), type=Path)
    parser.add_argument("--plot-configs-dir", default=DEFAULT_PLOT_CONFIGS_DIR, type=Path)
    subparsers = parser.add_subparsers(dest="command")

    link_parser = subparsers.add_parser("link")
    link_parser.add_argument("plot_config")

    purge_parser = subparsers.add_parser("purge")
    purge_parser.add_argument("plot_config")

    args = parser.parse_args()

    if args.command == "link":
        result = link_plot_config(
            args.plot_config,
            results=args.results,
            plot_configs_dir=args.plot_configs_dir,
        )
        print(
            f"Created/updated {result.linked} symlinks under {args.results / 'plots'} "
            f"({result.missing} missing source PDFs)"
        )
    elif args.command == "purge":
        result = purge_plot_config(
            args.plot_config,
            results=args.results,
            plot_configs_dir=args.plot_configs_dir,
        )
        print(f"Removed {result.removed} symlinks under {args.results / 'plots'}")
    else:
        symlink_plots(args.results)
