"""Bridge for the R-backed fits (fit_slpr / fit_mgsa). SLPR (glmnet) and MGSA
(Bioconductor `mgsa`) have no faithful Python implementation, so the numeric core
runs in R. This module keeps all Snakemake-facing orchestration in Python (loading
the cached design via `_design`, writing the final parquet), and confines R to the
fit itself, exchanging data through a per-job temp directory:

  membership.mtx   sparse genes x sets membership (MatrixMarket, R reads via Matrix::readMM)
  *.tsv            per-gene response / hit lists / set ids (in), fit results (out)

The exchange is language-neutral and typed enough for this purpose; no R Arrow /
data.table dependency is required (base R + Matrix + the method package only).
"""
import os
import subprocess
import sys

import numpy as np


def write_mm(path, rows, cols, n_rows, n_cols):
    """Write a genes x sets 0/1 membership as a MatrixMarket coordinate file.

    `rows`/`cols` are 0-based COO indices of the 1-entries (as cached by prep).
    MatrixMarket is 1-based, so indices are shifted on write; R reads the result
    with `Matrix::readMM` straight into a sparse matrix.
    """
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    triplets = np.column_stack([rows + 1, cols + 1])
    with open(path, "wb") as fh:
        fh.write(b"%%MatrixMarket matrix coordinate real general\n")
        fh.write(f"{n_rows} {n_cols} {triplets.shape[0]}\n".encode())
        # all stored values are 1.0 (membership indicator)
        np.savetxt(fh, triplets, fmt="%d %d 1")


def run_r(rscript, workdir, threads=1):
    """Run an R helper as `Rscript <rscript> <workdir>`; raise with captured R
    output on failure. BLAS/OpenMP thread pools are pinned to the allocated
    cores so a multi-job SLURM node is not oversubscribed."""
    env = dict(os.environ)
    nt = str(max(1, int(threads)))
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(v, nt)
    r = subprocess.run(
        ["Rscript", "--vanilla", rscript, workdir],
        env=env, capture_output=True, text=True,
    )
    # R helpers echo progress to stderr; surface it so a fit's log is not silent.
    sys.stderr.write(r.stderr)
    if r.returncode != 0:
        raise RuntimeError(
            f"R helper {os.path.basename(rscript)} failed (exit {r.returncode}).\n"
            f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}"
        )
    return r.stdout
