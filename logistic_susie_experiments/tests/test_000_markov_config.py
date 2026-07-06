from experiments import loader


def test_000_markov_supercollection_expands():
    cfg = loader.load_config()
    assert "000_markov" in cfg["supercollections"]
    sc = cfg["supercollections"]["000_markov"]
    cols = loader.expand_collections(cfg["library"], "000_markov", sc["collections"])
    # 5 design types (incl binary_q0.9) x 6 rho = 30 collections
    assert len(cols) == 30
    # each collection unions 15 signal (3 b0 x 5 logBF) + 3 null = 18 sims
    assert len(cols[0]["simulations"]) == 18
    # methods: 12 names x 2 priors (fixed/eb) = 24
    methods = loader.resolve_methods_for_sc(cfg["library"], sc)
    assert len(methods) == 24


def test_000_global_local_removed():
    import pathlib
    assert not (pathlib.Path("experiments") / "000_global_local.yaml").exists()


def test_000_markov_expands_via_sweep():
    cfg = loader.load_config()
    sc = cfg["supercollections"]["000_markov"]
    cols = loader.expand_collections(cfg["library"], "000_markov", sc["collections"])
    assert len(cols) == 30                       # 5 designs (incl q0.9) x 6 rho
    assert len(cols[0]["simulations"]) == 18     # 3 b0 x 5 logbf + 3 null
    methods = loader.resolve_methods_for_sc(cfg["library"], sc)
    assert len(methods) == 24                     # 12 names x 2 priors


def test_000_markov_plot_keys_are_path_safe():
    """Plot keys become Snakemake `plot_name` wildcards (regex [A-Za-z0-9_\\-]+),
    so they must not contain dots — e.g. use `binary_q05__*`, not
    `binary_q0.5__*` (the filter value keeps the dot; the key must not)."""
    import re
    import yaml
    from pathlib import Path
    spec = yaml.safe_load(Path("experiments/000_markov.yaml").read_text())
    plots = spec["supercollections"]["000_markov"]["plots"]
    bad = [k for k in plots if not re.fullmatch(r"[A-Za-z0-9_\-]+", k)]
    assert not bad, f"plot keys not path-safe (contain dots): {bad}"
