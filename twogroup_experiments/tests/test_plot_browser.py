from __future__ import annotations

import os
from pathlib import Path

from scripts.plot_browser import (
    PlotImageCache,
    PlotIndex,
    PlotSelection,
    shortcut_action,
    shortcut_help_items,
)


def _write_pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4\n% test\n")


def test_plot_index_ignores_non_pdf_files(tmp_path: Path) -> None:
    root = tmp_path / "results" / "plots" / "by_type"
    _write_pdf(root / "power_fdp" / "minimal" / "001-loc.pdf")
    (root / "power_fdp" / "minimal" / "notes.txt").write_text("ignore me")

    index = PlotIndex.from_root(root)

    assert index.plot_types() == ["power_fdp"]
    assert index.supercollections(False, "power_fdp", "minimal") == ["001-loc"]


def test_plot_index_handles_symlinked_pdfs(tmp_path: Path) -> None:
    root = tmp_path / "results" / "plots" / "by_type"
    target = tmp_path / "results" / "supercollections" / "001-loc" / "power_fdp" / "minimal.pdf"
    _write_pdf(target)
    link = root / "power_fdp" / "minimal" / "001-loc.pdf"
    link.parent.mkdir(parents=True)
    link.symlink_to(os.path.relpath(target, link.parent))

    index = PlotIndex.from_root(root)

    selection = PlotSelection(False, "power_fdp", "minimal", "001-loc")
    assert index.path_for(selection) == link


def test_plot_index_groups_aggregate_and_nonaggregate_plot_types(tmp_path: Path) -> None:
    root = tmp_path / "results" / "plots" / "by_type"
    _write_pdf(root / "power_fdp" / "minimal" / "001-loc.pdf")
    _write_pdf(root / "agg_power_fdp" / "minimal" / "001-loc.pdf")

    index = PlotIndex.from_root(root)

    assert index.plot_types() == ["power_fdp"]
    assert index.aggregate_options("power_fdp") == [False, True]


def test_normalize_selection_updates_dependents_when_toggling_agg(tmp_path: Path) -> None:
    root = tmp_path / "results" / "plots" / "by_type"
    _write_pdf(root / "power_fdp" / "minimal" / "002-loc.pdf")
    _write_pdf(root / "agg_power_fdp" / "summary" / "001-scale.pdf")

    index = PlotIndex.from_root(root)

    selection = index.normalize(
        PlotSelection(True, "power_fdp", "minimal", "002-loc"),
        previous=PlotSelection(False, "power_fdp", "minimal", "002-loc"),
    )

    assert selection == PlotSelection(True, "power_fdp", "summary", "001-scale")
    assert index.path_for(selection) == root / "agg_power_fdp" / "summary" / "001-scale.pdf"


def test_normalize_selection_keeps_nearest_valid_method_and_supercollection(tmp_path: Path) -> None:
    root = tmp_path / "results" / "plots" / "by_type"
    _write_pdf(root / "pip_calibration" / "minimal" / "001-loc.pdf")
    _write_pdf(root / "power_fdp" / "full" / "003-scale.pdf")
    _write_pdf(root / "power_fdp" / "minimal" / "002-loc.pdf")

    index = PlotIndex.from_root(root)

    selection = index.normalize(
        PlotSelection(False, "power_fdp", "minimal", "001-loc"),
        previous=PlotSelection(False, "pip_calibration", "minimal", "001-loc"),
    )

    assert selection == PlotSelection(False, "power_fdp", "minimal", "002-loc")


def test_plot_image_cache_renders_pdf_to_stable_png_path(tmp_path: Path) -> None:
    pdf = tmp_path / "plots" / "power_fdp" / "minimal" / "001-loc.pdf"
    _write_pdf(pdf)
    cache = PlotImageCache(tmp_path / "cache", render_pdf=lambda src, dst, scale: dst.write_bytes(b"png"))

    image = cache.image_for(pdf)

    assert image.suffix == ".png"
    assert image.parent == tmp_path / "cache"
    assert image.read_bytes() == b"png"


def test_plot_image_cache_reuses_existing_image_for_unchanged_pdf(tmp_path: Path) -> None:
    pdf = tmp_path / "plots" / "power_fdp" / "minimal" / "001-loc.pdf"
    _write_pdf(pdf)
    calls = []

    def render_pdf(src: Path, dst: Path, scale: float) -> None:
        calls.append(src)
        dst.write_bytes(b"png")

    cache = PlotImageCache(tmp_path / "cache", render_pdf=render_pdf)

    first = cache.image_for(pdf)
    second = cache.image_for(pdf)

    assert first == second
    assert calls == [pdf]


def test_plot_image_cache_url_quotes_cached_image_name(tmp_path: Path) -> None:
    pdf = tmp_path / "plots" / "power fdp" / "minimal" / "001 loc.pdf"
    _write_pdf(pdf)
    cache = PlotImageCache(
        tmp_path / "cache dir",
        route="/plot-images",
        render_pdf=lambda src, dst, scale: dst.write_bytes(b"png"),
    )

    url = cache.url_for(cache.image_for(pdf))

    assert url.startswith("/plot-images/")
    assert " " not in url
    assert url.endswith(".png")


def test_shortcut_action_uses_option_a_for_aggregate_toggle() -> None:
    assert shortcut_action("a", meta=False, alt=True) == "toggle_aggregate"
    assert shortcut_action("a", meta=True, alt=False) is None


def test_shortcut_action_handles_macos_option_a_key_event() -> None:
    assert shortcut_action("å", code="KeyA", meta=False, alt=True) == "toggle_aggregate"


def test_shortcut_action_maps_requested_arrow_navigation() -> None:
    assert shortcut_action("arrowleft", meta=True, alt=False, ctrl=False) == "previous_supercollection"
    assert shortcut_action("arrowright", meta=True, alt=False, ctrl=False) == "next_supercollection"
    assert shortcut_action("arrowleft", meta=False, alt=True, ctrl=False) == "previous_method"
    assert shortcut_action("arrowright", meta=False, alt=True, ctrl=False) == "next_method"
    assert shortcut_action("arrowup", meta=True, alt=False, ctrl=False) == "previous_plot_type"
    assert shortcut_action("arrowdown", meta=True, alt=False, ctrl=False) == "next_plot_type"
    assert shortcut_action("arrowleft", meta=False, alt=False, ctrl=True) is None
    assert shortcut_action("arrowright", meta=False, alt=False, ctrl=True) is None


def test_shortcut_help_items_match_navigation_shortcuts() -> None:
    assert ("Cmd Left/Right", "Supercollection") in shortcut_help_items()
    assert ("Option Left/Right", "Method collection") in shortcut_help_items()
    assert ("Cmd Up/Down", "Plot type") in shortcut_help_items()
    assert ("Option A", "Aggregate") in shortcut_help_items()


def test_plot_browser_script_declares_uv_runtime_dependencies() -> None:
    script = Path("scripts/plot_browser.py").read_text()

    assert "# /// script" in script
    assert '"nicegui"' in script
    assert '"pypdfium2"' in script
