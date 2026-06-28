from viz_facet import apply_filter, assign_groups

ROWS = [
    {"family": "irls", "step": "one_step", "design": "gaussian", "v": 1},
    {"family": "irls", "step": "converged", "design": "gaussian", "v": 2},
    {"family": "irls", "step": "converged", "design": "c4", "v": 3},
    {"family": "globaljj", "step": "converged", "design": "c4", "v": 4},
]


def test_apply_filter_scalar_equality():
    out = apply_filter(ROWS, {"family": "irls"})
    assert {r["v"] for r in out} == {1, 2, 3}


def test_apply_filter_list_membership():
    out = apply_filter(ROWS, {"design": ["c4"], "step": ["converged"]})
    assert {r["v"] for r in out} == {3, 4}


def test_apply_filter_empty_filter_passes_all():
    assert len(apply_filter(ROWS, {})) == 4


def test_assign_groups_facet_col_and_color():
    grid = assign_groups(ROWS, facet_col="design", color="family")
    assert grid.row_keys == [None]
    assert set(grid.col_keys) == {"gaussian", "c4"}
    c4 = grid.cell(None, "c4")
    colors = {s.color_key for s in c4}
    assert colors == {"irls", "globaljj"}
    irls_series = next(s for s in c4 if s.color_key == "irls")
    assert {r["v"] for r in irls_series.rows} == {3}


def test_assign_groups_series_pools_remaining_rows():
    # color only -> one facet cell, series by family pools across step/design
    grid = assign_groups(ROWS, color="family")
    cell = grid.cell(None, None)
    irls = next(s for s in cell if s.color_key == "irls")
    assert {r["v"] for r in irls.rows} == {1, 2, 3}  # pooled
