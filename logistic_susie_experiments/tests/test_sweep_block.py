from experiments import loader


def _sweep_block():
    return {
        "name": "gaussian",
        "sweep": {
            "design": {"function": "gaussian_markov_X", "n": 1000, "p": 500},
            "rho": [0, 0.9],
            "b0": [0, -2],
            "target_logbf": [0, 4, 16],   # 0 = null (explicit, no auto-append)
        },
    }


def test_sweep_expands_to_per_rho_collections():
    lib = loader.load_library()
    cols = loader._expand_block(lib, "t", _sweep_block())
    assert len(cols) == 2  # one per rho
    # plain cartesian: 2 b0 x 3 target_logbf (incl 0=null) = 6 sims
    assert len(cols[0]["simulations"]) == 6


def test_sweep_injects_design_and_b0_into_enrichment():
    lib = loader.load_library()
    cols = loader._expand_block(lib, "t", _sweep_block())
    coords = cols[0]["coordinates"]
    enr = [c["coordinate"]["enrichment"] for c in coords]
    signal = [e for e in enr if e["arguments"]["target_logbf"] != 0][0]
    assert signal["function"] == "logbf_single_effect"
    assert signal["arguments"]["design"] == "gaussian"
    assert "b0" in signal["arguments"] and signal["intercept"] == signal["arguments"]["b0"]
    null = [e for e in enr if e["arguments"]["target_logbf"] == 0][0]
    assert null["arguments"]["target_logbf"] == 0


def test_sweep_binary_design_key():
    lib = loader.load_library()
    blk = _sweep_block()
    blk["sweep"]["design"] = {"function": "binary_markov_X", "n": 1000, "p": 500, "freq": 0.1}
    cols = loader._expand_block(lib, "t", blk)
    enr = cols[0]["coordinates"][0]["coordinate"]["enrichment"]
    assert enr["arguments"]["design"] == "binary q=0.1"
