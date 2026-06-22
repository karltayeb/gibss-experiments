from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import core
from experiments import loader


def test_ser_bneg2_is_depletion():
    lib = loader.load_library()
    ent = lib["enrichments"]["ser_bneg2"]
    assert ent["intercept"] == -2.0
    assert ent["arguments"]["causal_effect"] == -2.0
    spec = loader.resolve_simulation(lib, "hallmark", "ser_bneg2", "loc_2.0", "gaussian")
    assert spec.intercept == -2.0
    sim = core.simulate(spec, 0)
    b = np.asarray(sim.b)
    assert b.min() == -2.0  # one causal column with a negative (depleting) effect


def test_003_has_depletion_twin():
    cfg = loader.load_config()
    scs = cfg["supercollections"]
    assert "003-hallmark-loc-snr-depletion" in scs
    # depletion collections use ser_bneg2 + null_b0
    dep = scs["003-hallmark-loc-snr-depletion"]
    enrich_list = dep["collections"]["template"]["enrichment"]
    assert enrich_list == ["ser_bneg2", "null_b0"]
    # twin mirrors the enrichment SC's analyses and methods
    base = "003-hallmark-loc-snr"
    twin = "003-hallmark-loc-snr-depletion"
    assert set(loader.resolve_sc_analyses(cfg, twin)) == set(loader.resolve_sc_analyses(cfg, base))
    assert (set(loader.resolve_methods_for_sc(cfg["library"], scs[twin]))
            == set(loader.resolve_methods_for_sc(cfg["library"], scs[base])))


def test_003_enrichment_sc_unchanged():
    cfg = loader.load_config()
    base = cfg["supercollections"]["003-hallmark-loc-snr"]
    # enrichment SC still uses ser_b2 + null_b0
    assert base["collections"]["template"]["enrichment"] == ["ser_b2", "null_b0"]
    # manifest still builds with both SCs present
    m = loader.manifest_dict(cfg["library"], cfg)
    assert m["batches"]
