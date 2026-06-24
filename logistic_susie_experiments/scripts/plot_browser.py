#!/usr/bin/env python3
# /// script
# dependencies = [
#   "nicegui",
#   "pypdfium2",
#   "pillow",
# ]
# ///
"""Browse PDFs under results/plots/by_type with a small NiceGUI app.

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


DEFAULT_PLOT_ROOT = Path("results/plots/by_type")
DEFAULT_CACHE_DIR = Path("results/plots/.plot_browser_cache")
DEFAULT_RENDER_SCALE = 3.0

RenderPdf = Callable[[Path, Path, float], None]


@dataclass(frozen=True, order=True)
class PlotSelection:
    agg: bool
    plot_type: str
    method_collection: str
    supercollection: str


class PlotIndex:
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
            raw_plot_type = pdf.parts[-3]
            agg = raw_plot_type.startswith("agg_")
            plot_type = raw_plot_type.removeprefix("agg_")
            selection = PlotSelection(
                agg=agg,
                plot_type=plot_type,
                method_collection=pdf.parts[-2],
                supercollection=pdf.stem,
            )
            files[selection] = pdf
        return cls(files)

    def is_empty(self) -> bool:
        return not self.files

    def plot_types(self, agg: bool | None = None) -> list[str]:
        return sorted(
            {
                selection.plot_type
                for selection in self.files
                if agg is None or selection.agg == agg
            }
        )

    def aggregate_options(self, plot_type: str) -> list[bool]:
        return sorted(
            {
                selection.agg
                for selection in self.files
                if selection.plot_type == plot_type
            }
        )

    def method_collections(self, agg: bool, plot_type: str) -> list[str]:
        return sorted(
            {
                selection.method_collection
                for selection in self.files
                if selection.agg == agg and selection.plot_type == plot_type
            }
        )

    def supercollections(
        self,
        agg: bool,
        plot_type: str,
        method_collection: str,
    ) -> list[str]:
        return sorted(
            {
                selection.supercollection
                for selection in self.files
                if selection.agg == agg
                and selection.plot_type == plot_type
                and selection.method_collection == method_collection
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
        if not self.files:
            raise ValueError("No PDFs found")

        plot_types = self.plot_types()
        plot_type = _nearest(plot_types, selection.plot_type)

        agg_options = self.aggregate_options(plot_type)
        agg = selection.agg if selection.agg in agg_options else agg_options[0]

        methods = self.method_collections(agg, plot_type)
        method_preference = selection.method_collection
        if previous and previous.plot_type != plot_type:
            method_preference = previous.method_collection
        method = _nearest(methods, method_preference)

        supers = self.supercollections(agg, plot_type, method)
        super_preference = selection.supercollection
        if previous and (
            previous.plot_type != plot_type
            or previous.agg != agg
            or previous.method_collection != method
        ):
            super_preference = previous.supercollection
        supercollection = _nearest(supers, super_preference)

        return PlotSelection(agg, plot_type, method, supercollection)

    def cycle_plot_type(self, selection: PlotSelection, delta: int) -> PlotSelection:
        plot_type = _cycle(self.plot_types(), selection.plot_type, delta)
        return self.normalize(
            PlotSelection(
                selection.agg,
                plot_type,
                selection.method_collection,
                selection.supercollection,
            ),
            previous=selection,
        )

    def cycle_method(self, selection: PlotSelection, delta: int) -> PlotSelection:
        method = _cycle(
            self.method_collections(selection.agg, selection.plot_type),
            selection.method_collection,
            delta,
        )
        return self.normalize(
            PlotSelection(
                selection.agg,
                selection.plot_type,
                method,
                selection.supercollection,
            ),
            previous=selection,
        )

    def cycle_supercollection(
        self,
        selection: PlotSelection,
        delta: int,
    ) -> PlotSelection:
        supercollection = _cycle(
            self.supercollections(
                selection.agg,
                selection.plot_type,
                selection.method_collection,
            ),
            selection.supercollection,
            delta,
        )
        return self.normalize(
            PlotSelection(
                selection.agg,
                selection.plot_type,
                selection.method_collection,
                supercollection,
            ),
            previous=selection,
        )

    def toggle_aggregate(self, selection: PlotSelection) -> PlotSelection:
        return self.normalize(
            PlotSelection(
                not selection.agg,
                selection.plot_type,
                selection.method_collection,
                selection.supercollection,
            ),
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
    code: str | None = None,
) -> str | None:
    key = key.lower()
    if alt and not meta and not ctrl and (key in {"a", "å"} or code == "KeyA"):
        return "toggle_aggregate"
    if key in {"arrowleft", "left"}:
        if meta and not alt and not ctrl:
            return "previous_supercollection"
        if alt and not meta and not ctrl:
            return "previous_method"
        return None
    if key in {"arrowright", "right"}:
        if meta and not alt and not ctrl:
            return "next_supercollection"
        if alt and not meta and not ctrl:
            return "next_method"
        return None
    if key in {"arrowup", "up"}:
        if meta and not alt and not ctrl:
            return "previous_plot_type"
        return None
    if key in {"arrowdown", "down"}:
        if meta and not alt and not ctrl:
            return "next_plot_type"
        return None
    return None


def shortcut_help_items() -> list[tuple[str, str]]:
    return [
        ("Cmd Left/Right", "Supercollection"),
        ("Option Left/Right", "Output"),
        ("Cmd Up/Down", "Plot type"),
        ("Option A", "Aggregate"),
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

    plot_select = None
    method_select = None
    super_select = None
    agg_switch = None
    image = None

    def sync_controls() -> None:
        nonlocal plot_select, method_select, super_select, agg_switch, image
        selection = state["selection"]
        assert selection is not None

        state["syncing"] = True
        try:
            plot_select.options = index.plot_types()
            plot_select.value = selection.plot_type

            agg_switch.value = selection.agg
            if len(index.aggregate_options(selection.plot_type)) > 1:
                agg_switch.enable()
            else:
                agg_switch.disable()

            method_select.options = index.method_collections(
                selection.agg,
                selection.plot_type,
            )
            method_select.value = selection.method_collection

            super_select.options = index.supercollections(
                selection.agg,
                selection.plot_type,
                selection.method_collection,
            )
            super_select.value = selection.supercollection

            path = index.path_for(selection)
            if path is not None:
                preview = cache.image_for(path)
                image.props(f'src="{cache.url_for(preview)}"')

            for control in (plot_select, method_select, super_select, agg_switch, image):
                control.update()
        finally:
            state["syncing"] = False

    def set_selection(selection: PlotSelection) -> None:
        previous = state["selection"]
        state["selection"] = index.normalize(selection, previous=previous)
        sync_controls()

    def on_plot_change(event) -> None:
        if state["syncing"]:
            return
        selection = state["selection"]
        set_selection(
            PlotSelection(
                selection.agg,
                event.value,
                selection.method_collection,
                selection.supercollection,
            )
        )

    def on_method_change(event) -> None:
        if state["syncing"]:
            return
        selection = state["selection"]
        set_selection(
            PlotSelection(
                selection.agg,
                selection.plot_type,
                event.value,
                selection.supercollection,
            )
        )

    def on_super_change(event) -> None:
        if state["syncing"]:
            return
        selection = state["selection"]
        set_selection(
            PlotSelection(
                selection.agg,
                selection.plot_type,
                selection.method_collection,
                event.value,
            )
        )

    def on_agg_change(event) -> None:
        if state["syncing"]:
            return
        selection = state["selection"]
        set_selection(
            PlotSelection(
                bool(event.value),
                selection.plot_type,
                selection.method_collection,
                selection.supercollection,
            )
        )

    def on_key(event) -> None:
        action = getattr(event, "action", None)
        if action is not None and not getattr(action, "keydown", False):
            return
        key_arg = getattr(event, "key", "")
        key = str(getattr(key_arg, "name", key_arg)).lower()
        code = getattr(key_arg, "code", None)
        modifiers = getattr(event, "modifiers", None)
        meta = bool(getattr(modifiers, "meta", False))
        alt = bool(getattr(modifiers, "alt", False))
        ctrl = bool(getattr(modifiers, "ctrl", False))
        selection = state["selection"]

        action_name = shortcut_action(key, meta=meta, alt=alt, ctrl=ctrl, code=code)

        if action_name == "toggle_aggregate":
            set_selection(index.toggle_aggregate(selection))
        elif action_name == "previous_method":
            set_selection(index.cycle_method(selection, -1))
        elif action_name == "previous_plot_type":
            set_selection(index.cycle_plot_type(selection, -1))
        elif action_name == "previous_supercollection":
            set_selection(index.cycle_supercollection(selection, -1))
        elif action_name == "next_method":
            set_selection(index.cycle_method(selection, 1))
        elif action_name == "next_plot_type":
            set_selection(index.cycle_plot_type(selection, 1))
        elif action_name == "next_supercollection":
            set_selection(index.cycle_supercollection(selection, 1))

    with ui.element("div").classes("plot-shell"):
        with ui.column().classes("plot-sidebar"):
            agg_switch = ui.switch("Aggregate", on_change=on_agg_change)
            plot_select = ui.select([], label="Plot type", on_change=on_plot_change).classes(
                "w-full"
            )
            method_select = ui.select(
                [],
                label="Output",
                on_change=on_method_change,
            ).classes("w-full")
            super_select = ui.select(
                [],
                label="Supercollection",
                on_change=on_super_change,
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
