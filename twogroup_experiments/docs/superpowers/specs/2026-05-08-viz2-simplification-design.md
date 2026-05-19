# 2026-05-08 Viz2 Simplification Design

**Goal:** Simplify `notebooks/viz2.py` by moving data loading, preparation, and rendering logic to `viz2_logic.py`.

**Architecture:** 
- `viz2_logic.py`: Single source of truth for analysis logic and plotting.
- `notebooks/viz2.py`: Pure UI layer. Imports logic, handles Marimo UI controls, and wires inputs to logic functions.

**Components:**
1. **Data Loading:** New functions in `viz2_logic.py` for reading Parquet/YAML files from a collection root.
2. **Logic Migration:** Move all `@app.function` definitions (filters, summaries, chart renders) to `viz2_logic.py`.
3. **Notebook Refactor:** Update `viz2.py` to import `viz2_logic` and call its functions.
