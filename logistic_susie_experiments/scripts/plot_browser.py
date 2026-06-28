#!/usr/bin/env python3
# /// script
# dependencies = [
#   "nicegui",
#   "pypdfium2",
#   "pillow",
# ]
# ///
"""Browse plot PDFs with a small NiceGUI app.

Experiment-first: pick an experiment (supercollection), then the available
analyses and plots are populated conditional on that choice. Reads the canonical
tree results/supercollections/{experiment}/{analysis}/{plot}.pdf directly (no
symlink view needed).

Run:
    uv run scripts/plot_browser.py
"""
from __future__ import annotations

import argparse
import hashlib
from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote


DEFAULT_PLOT_ROOT = Path("results/supercollections")
DEFAULT_CACHE_DIR = Path("results/plots/.plot_browser_cache")
DEFAULT_RENDER_SCALE = 3.0

RenderPdf = Callable[[Path, Path, float], None]


@dataclass(frozen=True, order=True)
class PlotSelection:
    experiment: str
    analysis: str
    plot: str


class PlotIndex:
    """Index of plot PDFs, queried experiment-first.

    Source layout: results/supercollections/{experiment}/{analysis}/{plot}.pdf
    """

    def __init__(self, files: dict[PlotSelection, Path]) -> None:
        self.files = dict(sorted(files.items()))

    @classmethod
    def from_root(cls, root: Path = DEFAULT_PLOT_ROOT) -> "PlotIndex":
        files: dict[PlotSelection, Path] = {}
        if not root.exists():
            return cls(files)
        for pdf in sorted(root.glob("*/*/*.pdf")):
            if not pdf.is_file() and not pdf.is_symlink():
                continue
            selection = PlotSelection(
                experiment=pdf.parts[-3],
                analysis=pdf.parts[-2],
                plot=pdf.stem,
            )
            files[selection] = pdf
        return cls(files)

    def is_empty(self) -> bool:
        return not self.files

    def experiments(self) -> list[str]:
        return sorted({s.experiment for s in self.files})

    def analyses(self, experiment: str) -> list[str]:
        return sorted(
            {s.analysis for s in self.files if s.experiment == experiment}
        )

    def plots(self, experiment: str, analysis: str) -> list[str]:
        return sorted(
            {
                s.plot
                for s in self.files
                if s.experiment == experiment and s.analysis == analysis
            }
        )

    def path_for(self, selection: PlotSelection) -> Path | None:
        return self.files.get(selection)

    def first_selection(self) -> PlotSelection | None:
        if not self.files:
            return None
        return next(iter(self.files))

    def normalize(
        self,
        selection: PlotSelection,
        *,
        previous: PlotSelection | None = None,
    ) -> PlotSelection:
        """Snap a (possibly stale) selection to a valid one, cascading
        experiment -> analysis -> plot. When an upstream field changes, prefer
        carrying the previous downstream choice if it is still available."""
        if not self.files:
            raise ValueError("No PDFs found")

        experiment = _nearest(self.experiments(), selection.experiment)

        analyses = self.analyses(experiment)
        analysis_pref = selection.analysis
        if previous and previous.experiment != experiment:
            analysis_pref = previous.analysis
        analysis = _nearest(analyses, analysis_pref)

        plots = self.plots(experiment, analysis)
        plot_pref = selection.plot
        if previous and (
            previous.experiment != experiment or previous.analysis != analysis
        ):
            plot_pref = previous.plot
        plot = _nearest(plots, plot_pref)

        return PlotSelection(experiment, analysis, plot)

    def cycle_experiment(self, selection: PlotSelection, delta: int) -> PlotSelection:
        experiment = _cycle(self.experiments(), selection.experiment, delta)
        return self.normalize(
            PlotSelection(experiment, selection.analysis, selection.plot),
            previous=selection,
        )

    def cycle_analysis(self, selection: PlotSelection, delta: int) -> PlotSelection:
        analysis = _cycle(
            self.analyses(selection.experiment), selection.analysis, delta
        )
        return self.normalize(
            PlotSelection(selection.experiment, analysis, selection.plot),
            previous=selection,
        )

    def cycle_plot(self, selection: PlotSelection, delta: int) -> PlotSelection:
        plot = _cycle(
            self.plots(selection.experiment, selection.analysis),
            selection.plot,
            delta,
        )
        return self.normalize(
            PlotSelection(selection.experiment, selection.analysis, plot),
            previous=selection,
        )


def _nearest(items: list[str], preferred: str) -> str:
    if not items:
        raise ValueError("Cannot choose from an empty sequence")
    if preferred in items:
        return preferred
    index = bisect_left(items, preferred)
    if index >= len(items):
        return items[-1]
    return items[index]


def _cycle(items: list[str], current: str, delta: int) -> str:
    if not items:
        raise ValueError("Cannot cycle an empty sequence")
    if current not in items:
        return _nearest(items, current)
    return items[(items.index(current) + delta) % len(items)]


def shortcut_action(
    key: str,
    *,
    meta: bool,
    alt: bool,
    ctrl: bool = False,
) -> str | None:
    key = key.lower()
    if key in {"arrowleft", "left"}:
        if meta and not alt and not ctrl:
            return "previous_experiment"
        if alt and not meta and not ctrl:
            return "previous_analysis"
        return None
    if key in {"arrowright", "right"}:
        if meta and not alt and not ctrl:
            return "next_experiment"
        if alt and not meta and not ctrl:
            return "next_analysis"
        return None
    if key in {"arrowup", "up"}:
        if meta and not alt and not ctrl:
            return "previous_plot"
        return None
    if key in {"arrowdown", "down"}:
        if meta and not alt and not ctrl:
            return "next_plot"
        return None
    return None


def shortcut_help_items() -> list[tuple[str, str]]:
    return [
        ("Cmd Left/Right", "Experiment"),
        ("Option Left/Right", "Analysis"),
        ("Cmd Up/Down", "Plot"),
    ]


class PlotImageCache:
    def __init__(
        self,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        *,
        route: str = "/plot-images",
        scale: float = DEFAULT_RENDER_SCALE,
        render_pdf: RenderPdf | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.route = route.rstrip("/")
        self.scale = scale
        self.render_pdf = render_pdf or render_pdf_first_page

    def image_for(self, pdf: Path) -> Path:
        image = self.cache_dir / f"{self._cache_key(pdf)}.png"
        if image.exists():
            return image

        image.parent.mkdir(parents=True, exist_ok=True)
        tmp = image.with_suffix(".tmp.png")
        self.render_pdf(pdf, tmp, self.scale)
        tmp.replace(image)
        return image

    def url_for(self, image: Path) -> str:
        quoted = quote(image.name)
        return f"{self.route}/{quoted}"

    def _cache_key(self, pdf: Path) -> str:
        stat = pdf.stat()
        resolved = pdf.resolve()
        payload = "|".join(
            [
                str(resolved),
                str(stat.st_size),
                str(stat.st_mtime_ns),
                f"{self.scale:.3f}",
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def render_pdf_first_page(pdf: Path, image: Path, scale: float) -> None:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError(
            "Rendering PDF previews requires pypdfium2. Run with: "
            "uv run scripts/plot_browser.py"
        ) from exc

    document = pdfium.PdfDocument(pdf)
    try:
        if len(document) == 0:
            raise ValueError(f"PDF has no pages: {pdf}")
        page = document[0]
        try:
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()
            pil_image.save(image)
        finally:
            page.close()
    finally:
        document.close()


def run_app(
    root: Path = DEFAULT_PLOT_ROOT,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    from nicegui import app, ui

    root = root.resolve()
    cache = PlotImageCache(cache_dir.resolve())
    index = PlotIndex.from_root(root)

    cache.cache_dir.mkdir(parents=True, exist_ok=True)
    app.add_static_files(cache.route, cache.cache_dir)
    ui.add_head_html(
        """
        <style>
        body { margin: 0; background: #f7f7f4; color: #24292f; }
        .plot-shell { height: 100vh; display: grid; grid-template-columns: 280px 1fr; }
        .plot-sidebar {
            border-right: 1px solid #d8d8d0; background: #fcfcfa; padding: 14px;
            gap: 12px; align-content: start;
        }
        .plot-viewer {
            height: 100vh; background: #e9e9e3; display: flex; align-items: center;
            justify-content: center; overflow: hidden; padding: 10px;
        }
        .plot-image {
            display: block; max-width: 100%; max-height: 100%; width: auto; height: auto;
            object-fit: contain; box-shadow: 0 1px 6px rgba(0, 0, 0, 0.16);
            background: white;
        }
        .shortcut-help {
            margin-top: 4px; padding-top: 12px; border-top: 1px solid #d8d8d0;
            color: #545b64; font-size: 12px; line-height: 1.35;
        }
        .shortcut-row {
            display: flex; justify-content: space-between; gap: 10px; align-items: baseline;
            padding: 3px 0;
        }
        .shortcut-key {
            color: #24292f; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            white-space: nowrap;
        }
        @media (max-width: 760px) {
            .plot-shell { grid-template-columns: 1fr; grid-template-rows: auto 1fr; }
            .plot-sidebar { border-right: 0; border-bottom: 1px solid #d8d8d0; }
            .plot-viewer { height: calc(100vh - 238px); }
        }
        </style>
        """
    )

    if index.is_empty():
        with ui.column().classes("w-full h-screen items-center justify-center gap-2"):
            ui.label("No PDFs found").classes("text-xl font-medium")
            ui.label(str(root)).classes("text-sm text-gray-600")
        ui.run(host=host, port=port, title="Plot Browser", reload=False)
        return

    state = {"selection": index.first_selection(), "syncing": False}
    assert state["selection"] is not None

    experiment_select = None
    analysis_select = None
    plot_select = None
    image = None

    def sync_controls() -> None:
        nonlocal experiment_select, analysis_select, plot_select, image
        selection = state["selection"]
        assert selection is not None

        state["syncing"] = True
        try:
            experiment_select.options = index.experiments()
            experiment_select.value = selection.experiment

            analysis_select.options = index.analyses(selection.experiment)
            analysis_select.value = selection.analysis

            plot_select.options = index.plots(selection.experiment, selection.analysis)
            plot_select.value = selection.plot

            path = index.path_for(selection)
            if path is not None:
                preview = cache.image_for(path)
                image.props(f'src="{cache.url_for(preview)}"')

            for control in (experiment_select, analysis_select, plot_select, image):
                control.update()
        finally:
            state["syncing"] = False

    def set_selection(selection: PlotSelection) -> None:
        previous = state["selection"]
        state["selection"] = index.normalize(selection, previous=previous)
        sync_controls()

    def on_experiment_change(event) -> None:
        if state["syncing"]:
            return
        selection = state["selection"]
        set_selection(
            PlotSelection(event.value, selection.analysis, selection.plot)
        )

    def on_analysis_change(event) -> None:
        if state["syncing"]:
            return
        selection = state["selection"]
        set_selection(
            PlotSelection(selection.experiment, event.value, selection.plot)
        )

    def on_plot_change(event) -> None:
        if state["syncing"]:
            return
        selection = state["selection"]
        set_selection(
            PlotSelection(selection.experiment, selection.analysis, event.value)
        )

    def on_key(event) -> None:
        action = getattr(event, "action", None)
        if action is not None and not getattr(action, "keydown", False):
            return
        key_arg = getattr(event, "key", "")
        key = str(getattr(key_arg, "name", key_arg)).lower()
        modifiers = getattr(event, "modifiers", None)
        meta = bool(getattr(modifiers, "meta", False))
        alt = bool(getattr(modifiers, "alt", False))
        ctrl = bool(getattr(modifiers, "ctrl", False))
        selection = state["selection"]

        action_name = shortcut_action(key, meta=meta, alt=alt, ctrl=ctrl)

        if action_name == "previous_experiment":
            set_selection(index.cycle_experiment(selection, -1))
        elif action_name == "next_experiment":
            set_selection(index.cycle_experiment(selection, 1))
        elif action_name == "previous_analysis":
            set_selection(index.cycle_analysis(selection, -1))
        elif action_name == "next_analysis":
            set_selection(index.cycle_analysis(selection, 1))
        elif action_name == "previous_plot":
            set_selection(index.cycle_plot(selection, -1))
        elif action_name == "next_plot":
            set_selection(index.cycle_plot(selection, 1))

    with ui.element("div").classes("plot-shell"):
        with ui.column().classes("plot-sidebar"):
            experiment_select = ui.select(
                [], label="Experiment", on_change=on_experiment_change
            ).classes("w-full")
            analysis_select = ui.select(
                [], label="Analysis", on_change=on_analysis_change
            ).classes("w-full")
            plot_select = ui.select(
                [], label="Plot", on_change=on_plot_change
            ).classes("w-full")
            with ui.element("div").classes("shortcut-help"):
                ui.label("Shortcuts").classes("text-xs font-medium text-gray-700")
                for keys, description in shortcut_help_items():
                    with ui.element("div").classes("shortcut-row"):
                        ui.label(keys).classes("shortcut-key")
                        ui.label(description)
        with ui.element("div").classes("plot-viewer"):
            image = ui.element("img").classes("plot-image")

    ui.keyboard(on_key=on_key)
    sync_controls()
    ui.run(host=host, port=port, title="Plot Browser", reload=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_PLOT_ROOT, type=Path)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8080, type=int)
    args = parser.parse_args()

    run_app(args.root, cache_dir=args.cache_dir, host=args.host, port=args.port)


if __name__ in {"__main__", "__mp_main__"}:
    main()
