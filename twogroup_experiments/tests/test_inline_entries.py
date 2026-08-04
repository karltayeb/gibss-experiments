"""Inline (anchor) axis entries: name-or-dict resolution, content-addressed hashing, naming."""
from __future__ import annotations

from experiments.loader import (
    load_library,
    resolve_simulation,
    sim_hash,
    simulation_coordinate,
    _axis_display,
    _hashable_entry,
)


_INLINE_ENRICH = {
    "label": "ser_b1.8_m",
    "function": "sized_single_effect",
    "arguments": {"causal_effect": 1.8, "size_lo": 80, "size_hi": 120},
    "intercept": -2.0,
}


def test_inline_entry_hashes_like_named_equivalent():
    # An inline dict must hash IDENTICALLY to the named library entry with the same
    # functional content (content-addressed -> dedup + stable hashes).
    lib = load_library()
    named = sim_hash(simulation_coordinate(lib, "gobp_10_500", "op_b1.80_s80_120", "binary", "noiseless"))
    inline = sim_hash(simulation_coordinate(lib, "gobp_10_500", _INLINE_ENRICH, "binary", "noiseless"))
    assert named == inline


def test_label_is_stripped_from_hash():
    # Adding/removing the display-only label must not change the coordinate hash.
    lib = load_library()
    with_label = simulation_coordinate(lib, "gobp_10_500", _INLINE_ENRICH, "binary", "noiseless")
    without = dict(_INLINE_ENRICH)
    without.pop("label")
    no_label = simulation_coordinate(lib, "gobp_10_500", without, "binary", "noiseless")
    assert sim_hash(with_label) == sim_hash(no_label)
    assert "label" not in with_label["enrichment"]


def test_inline_entry_uses_label_for_sim_name():
    lib = load_library()
    spec = resolve_simulation(lib, "gobp_10_500", _INLINE_ENRICH, "binary", "noiseless", "gaussian_s1.0")
    assert "ser_b1.8_m" in spec.name


def test_axis_display_synthesizes_when_no_label():
    entry = {"function": "sized_single_effect", "arguments": {"causal_effect": 1.8}}
    disp = _axis_display(entry)
    assert "sized_single_effect" in disp and "1.8" in disp
    assert _axis_display("op_b1.80_s80_120") == "op_b1.80_s80_120"  # string passthrough


def test_named_string_entries_unchanged():
    # Regression guard: named-string resolution still produces the same (label-free) entry.
    lib = load_library()
    coord = simulation_coordinate(lib, "gobp_10_500", "op_b1.80_s80_120", "binary", "noiseless")
    assert coord["enrichment"] == _hashable_entry(lib["enrichments"]["op_b1.80_s80_120"])
