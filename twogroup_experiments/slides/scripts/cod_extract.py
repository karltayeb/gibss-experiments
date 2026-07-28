"""Cost-of-discretizing figure data - extracted from the PIPELINE outputs.

The fits are produced by the results pipeline (NOT re-simulated here):

    uv run snakemake -s twogroup_experiments.snk -c11 \\
        results/supercollections/010-c2-cost-of-discretizing/.done

This script reads those committed pipeline outputs
(results/by_batch/<bh>/fits/<mh>/{fits,reductions/pip}.parquet) and caches the
mean-causal-PIP curves and the logBF detection-ROC curves to
figures/cod_data.json (with provenance). The plot scripts
fig10_pip_vs_threshold.py and fig11_cod_roc.py read that cache, so the slow part
(the pipeline) runs once and the deck rebuilds from the cache.

    uv run python scripts/cod_extract.py --out figures/cod_data.json

Positives = replicates of the enrichment sim; negatives = replicates of the paired
cod_null sim. Score for the ROC = single_effects[0].ser_log_bf (L=1). PIP curves
pool causal PIPs across all n_batches of a coordinate (n_batches=4 -> 200 reps).
"""
import argparse
import datetime as dt
import json
import pathlib
import re
import subprocess
import sys

import numpy as np
import polars as pl

ROOT = pathlib.Path(__file__).resolve().parents[2]  # twogroup_experiments/
sys.path.insert(0, str(ROOT))
from experiments import loader  # noqa: E402

SC = "010-c2-cost-of-discretizing"
RESULTS = ROOT / "results"
CELL_ORDER = ["small-loc", "large-loc", "small-scale", "large-scale"]


def parse(mname):
    """(kind, key, tau) for a method name. refs are tau-free."""
    if mname.startswith("twogroup_oracle"): return ("ref", "twogroup", None)
    if mname.startswith("linear_abs"): return ("ref", "linear_abs", None)
    if mname.startswith("linear_z") or mname.startswith("linear_fixed"): return ("ref", "linear", None)
    if mname.startswith("cox_reversed_binned"): return ("ref", "coxrev", None)
    if mname.startswith("cox_full_binned"): return ("ref", "coxfull", None)
    m = re.search(r"threshold=([0-9.]+)", mname)
    tau = float(m.group(1)) if m else None
    if mname.startswith("cox_binned__"): return ("cox", "cox", tau)
    if mname.startswith("logistic_threshold"): return ("logistic", "logistic", tau)
    return (None, None, None)


def causal_pip_vals(bh, mh):
    p = RESULTS / "by_batch" / bh / "fits" / mh / "reductions" / "pip.parquet"
    if not p.exists(): return []
    df = pl.read_parquet(p)
    return [r["causal_pips"][0] for r in df.iter_rows(named=True) if r["causal_pips"]]


def logbf_vals(bh, mh):
    p = RESULTS / "by_batch" / bh / "fits" / mh / "fits.parquet"
    if not p.exists(): return None
    df = pl.read_parquet(p)
    out = [float(se[0]["ser_log_bf"]) for se in df["single_effects"] if se is not None and len(se) > 0]
    return np.array(out) if out else None


def roc(pos, neg):
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(-scores, kind="mergesort")
    ls = labels[order]
    tpr = np.concatenate([[0.0], np.cumsum(ls) / len(pos)])
    fpr = np.concatenate([[0.0], np.cumsum(1 - ls) / len(neg)])
    ranks = np.empty(len(scores)); ranks[np.argsort(scores, kind="mergesort")] = np.arange(1, len(scores) + 1)
    auc = (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return fpr.tolist(), tpr.tolist(), float(auc)


def is_null(sim):
    enr = sim.get("enrichment", {}) if isinstance(sim, dict) else {}
    return float(enr.get("arguments", {}).get("causal_effect", 0.0)) == 0.0


def main(out):
    if not (RESULTS / "supercollections" / SC / ".done").exists():
        raise SystemExit(
            f"pipeline outputs for {SC} not found under {RESULTS}. Run first:\n"
            f"  uv run snakemake -s twogroup_experiments.snk -c11 "
            f"results/supercollections/{SC}/.done")
    cfg = loader.load_config()
    cmp = loader.collection_method_pairs(cfg, SC)

    pip = {}   # cell -> {"refs":{k:mean}, "cox":{tau:mean}, "logistic":{tau:mean}, "n":int}
    rocd = {}  # cell -> [ {key,tau,fpr,tpr,auc,npos,nneg} ]
    diff = {}  # cell -> realized two-group causal-set logBF (mean over enrichment reps)
    for cell in CELL_ORDER:
        info = cmp[cell]
        pip_acc = {"refs": {}, "cox": {}, "logistic": {}}
        lbf_acc = {}  # (key,tau) -> {"pos":[arr], "neg":[arr]}
        for bh, mh, mname, mcoord, sim in info["pairs"]:
            kind, key, tau = parse(mname)
            if kind is None: continue
            null = is_null(sim)
            # PIP: pool causal PIPs across enrichment batches only
            if not null:
                vals = causal_pip_vals(bh, mh)
                if vals:
                    bucket = pip_acc["refs"] if kind == "ref" else pip_acc[kind]
                    bucket.setdefault(key if kind == "ref" else tau, []).extend(vals)
            # ROC: logBF, pos=enrichment / neg=null
            lb = logbf_vals(bh, mh)
            if lb is not None:
                slot = lbf_acc.setdefault((key, tau), {"pos": [], "neg": []})
                slot["neg" if null else "pos"].append(lb)
        pip[cell] = {
            "refs": {k: float(np.mean(v)) for k, v in pip_acc["refs"].items()},
            "cox": {f"{t:.1f}": float(np.mean(v)) for t, v in pip_acc["cox"].items()},
            "logistic": {f"{t:.1f}": float(np.mean(v)) for t, v in pip_acc["logistic"].items()},
        }
        curves = []
        npos = nneg = 0
        for (key, tau), s in lbf_acc.items():
            if not s["pos"] or not s["neg"]: continue
            pos = np.concatenate(s["pos"]); neg = np.concatenate(s["neg"])
            fpr, tpr, auc = roc(pos, neg)
            curves.append({"key": key, "tau": tau, "fpr": fpr, "tpr": tpr,
                           "auc": auc, "npos": int(len(pos)), "nneg": int(len(neg))})
            npos, nneg = int(len(pos)), int(len(neg))
            if key == "twogroup":
                diff[cell] = float(np.mean(pos))  # realized 2g causal-set logBF
        pip[cell]["n"] = npos
        rocd[cell] = curves

    try:
        git = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        git = "unknown"
    result = {
        "sc": SC, "cells": CELL_ORDER, "thresholds": [1.0, 2.0, 3.0, 4.0],
        "pip": pip, "roc": rocd, "twogroup_logbf": diff,
        "git_commit": git,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    pathlib.Path(out).write_text(json.dumps(result) + "\n")
    print("wrote", out)
    for cell in CELL_ORDER:
        aucs = {c["key"] + (f"@{c['tau']:g}" if c["tau"] else ""): round(c["auc"], 2) for c in rocd[cell]}
        print(f"[{cell}] n={pip[cell]['n']} 2g-logBF={diff.get(cell, float('nan')):.1f}  AUC={aucs}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    main(ap.parse_args().out)
