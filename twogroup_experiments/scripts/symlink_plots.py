#!/usr/bin/env python3
"""Create results/plots/{plot_type}/{settings}/{supercollection}.pdf symlinks.

Scans results/supercollections/{sc}/{plot_type}/{settings}.pdf and deposits
relative symlinks under results/plots/ for easy cross-supercollection browsing.

Usage:
    uv run python scripts/symlink_plots.py
    uv run python scripts/symlink_plots.py --results /path/to/results
"""
import argparse
import os
from pathlib import Path


def symlink_plots(results: Path) -> None:
    sc_root = results / "supercollections"
    plots_root = results / "plots"

    count = 0
    for pdf in sorted(sc_root.glob("*/*/*.pdf")):
        sc = pdf.parts[-3]
        plot_type = pdf.parts[-2]
        settings = pdf.stem

        dst = plots_root / plot_type / settings / f"{sc}.pdf"
        dst.parent.mkdir(parents=True, exist_ok=True)

        rel_target = os.path.relpath(pdf, dst.parent)
        if dst.is_symlink():
            if os.readlink(dst) == rel_target:
                continue
            dst.unlink()
        elif dst.exists():
            dst.unlink()

        dst.symlink_to(rel_target)
        count += 1

    print(f"Created/updated {count} symlinks under {plots_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results", type=Path)
    args = parser.parse_args()
    symlink_plots(args.results)
