# Viz2 Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify `notebooks/viz2.py` by moving data loading and remaining logic to `viz2_logic.py`.

**Architecture:** Refactor the notebook to be a pure UI layer, delegating all data processing and plotting logic to the `viz2_logic` module.

**Tech Stack:** Python, Marimo, Polars, Matplotlib, YAML.

---

### Task 1: Add Data Loading Functions to `viz2_logic.py`

**Files:**
- Modify: `viz2_logic.py`
- Test: `tests/test_viz2_logic.py`

- [ ] **Step 1: Write tests for data loading functions**

```python
def test_load_pip_threshold_plot_data(tmp_path):
    import polars as pl
    from viz2_logic import load_pip_threshold_plot_data
    
    # Create mock parquet file
    df = pl.DataFrame({"replicate": [1]})
    batch_dir = tmp_path / "batches" / "b1" / "fits" / "f1"
    batch_dir.mkdir(parents=True)
    df.write_parquet(batch_dir / "pip_threshold_plot_data.parquet")
    
    loaded = load_pip_threshold_plot_data(tmp_path)
    assert len(loaded) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_viz2_logic.py`
Expected: FAIL (ImportError or AttributeError)

- [ ] **Step 3: Implement data loading functions in `viz2_logic.py`**

```python
def load_collection_manifest(collection_root: Path) -> dict:
    manifest_path = collection_root / "collection_spec.yaml"
    return yaml.safe_load(manifest_path.read_text())

def load_pip_threshold_plot_data(collection_root: Path) -> pl.DataFrame:
    paths = sorted(collection_root.glob("batches/*/fits/*/pip_threshold_plot_data.parquet"))
    if not paths: return empty_pip_threshold_plot_data()
    return viz2_metadata.add_plot_metadata_columns(
        pl.concat([pl.read_parquet(p) for p in paths], how="diagonal_relaxed")
    )

# ... and similar for other data types
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_viz2_logic.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add viz2_logic.py tests/test_viz2_logic.py
git commit -m "feat: add data loading functions to viz2_logic"
```

### Task 2: Migrate Remaining Helper Functions

**Files:**
- Modify: `viz2_logic.py`

- [ ] **Step 1: Move `add_plot_metadata_columns` and `method_metadata_from_method_spec_json`**
- [ ] **Step 2: Wrap metadata calls for cleaner notebook usage**

```python
def method_display_order(): return viz2_metadata.method_display_order()
def method_label_map(): return viz2_metadata.method_label_map()
```

- [ ] **Step 3: Commit**

```bash
git add viz2_logic.py
git commit -m "refactor: migrate remaining helpers to viz2_logic"
```

### Task 3: Refactor `notebooks/viz2.py`

**Files:**
- Modify: `notebooks/viz2.py`

- [ ] **Step 1: Update imports and add `viz2_logic`**
- [ ] **Step 2: Remove migrated functions (those now in `viz2_logic`)**
- [ ] **Step 3: Update Marimo cells to call `viz2_logic` functions**
- [ ] **Step 4: Verify notebook runs**

Run: `marimo export notebooks/viz2.py` (or manual check)
Expected: No errors.

- [ ] **Step 5: Commit**

```bash
git add notebooks/viz2.py
git commit -m "refactor: clean up viz2.py notebook"
```
