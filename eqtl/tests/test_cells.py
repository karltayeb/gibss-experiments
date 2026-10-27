from eqtl.cells import batches, collection_members, expand, load_config


def test_batches_cover_the_replicates():
    assert batches(5) == ((0, 5),)
    assert batches(100, 100) == ((0, 100),)
    assert batches(250, 100) == ((0, 100), (100, 200), (200, 250))


def test_config_expands_and_merges():
    cfg = load_config("config.yaml")
    cells = expand(cfg)
    assert all(len(h) == 16 for h in cells)
    spec = cfg["collections"]["pilot"]
    grid = spec["grid"]
    n_cells = 1
    for v in grid.values():
        n_cells *= len(v) if isinstance(v, list) else 1
    pilot = [c for c in cells.values() if "pilot" in c.collections]
    assert len(pilot) == n_cells
    # pilot cells are also main cells where the grid overlaps: reps merge to the max and the
    # cell carries both batch layouts
    main = cfg["collections"]["001-main"]
    shared = [c for c in pilot if "001-main" in c.collections]
    assert shared
    assert all(c.reps == max(spec["reps"], main["reps"]) for c in shared)
    assert all(set(c.batches) == set(batches(spec["reps"]) + batches(main["reps"], main.get("batch_size"))) for c in shared)
    members = collection_members(cfg, cells, "pilot")
    assert len(members) == len(pilot) * len(spec["arms"]) * len(spec["L"]) * len(batches(spec["reps"], spec.get("batch_size")))
    assert set(L for _, _, L, _, _ in members) == {"1", "5"}
    assert set((s, e) for _, _, _, s, e in members) == {(0, spec["reps"])}


def test_population_override_and_grid_union(tmp_path):
    import yaml
    cfg = load_config("config.yaml")
    cfg["collections"] = {
        "a": {"reps": 2, "L": [1], "arms": ["linear_int"], "population": {"right": 31_000_000},
              "grid": [{"n": 50, "mu0": 10, "k_causal": [1, 2], "beta": 0.3}, {"n": 50, "mu0": 10, "k_causal": 0, "beta": 0.0}]},
    }
    cells = expand(cfg)
    assert len(cells) == 3
    assert all(c.population.right == 31_000_000 for c in cells.values())
    assert sorted(c.expression.k_causal for c in cells.values()) == [0, 1, 2]
    assert all(c.batches == ((0, 2),) for c in cells.values())
