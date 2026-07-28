"""Slide 10 experiment: causal-PIP-vs-threshold pilot (real SER fits).

Runs the actual two-group / cox / logistic SER fits on ~200 simulated datasets in
a scale-driven regime where f1 is poorly identified, and caches the mean causal
PIP per method to figures/fig10_data.json (with provenance). The figure script
fig10_pip_vs_threshold.py reads that cache and plots - so the slow simulation
runs once, not on every figure build.

    uv run python scripts/fig10_experiment.py --out figures/fig10_data.json

Chosen regime (see slides/notes on the pilot sweep): scale-driven signal, where
estimating f1 is hard, so rank methods (cox-reversed) can beat twogroup-estimated
while the twogroup oracle stays on top.
"""
import argparse
import datetime as dt
import json
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]   # twogroup_experiments/
sys.path.insert(0, str(ROOT))

from gibss.distributions import Normal          # noqa: E402
import core                                      # noqa: E402
from experiments import loader                   # noqa: E402

CONFIG = {
    "design": "gaussian_rho0.50_n500_p100",
    "enrichment": "ser_b2",
    "signal": "scale_1.5",
    "error": "gaussian",
    "n_rep": 200,
    "thresholds": [1.0, 2.0, 3.0, 4.0],
}


def flat_methods():
    return {
        "oracle": {"name": "twogroup_oracle__L=1", "function": "run_twogroup_method",
                   "kwargs": {"f1": None, "L": 1}},
        "estimated": {"name": "twogroup__L=1", "function": "run_twogroup_method",
                      "kwargs": {"f1": Normal(loc=0.0, scale=1.0, estimate_loc=True,
                                              estimate_scale=True), "L": 1}},
        "cox_reversed": {"name": "cox_reversed__L=1", "function": "run_cox_method",
                         "kwargs": {"threshold": None, "time_sign": 1.0, "L": 1}},
    }


def cox_at(t):
    return {"name": f"cox__threshold={t:.2f}__L=1", "function": "run_cox_method",
            "kwargs": {"threshold": t, "time_sign": -1.0, "L": 1}}


def logistic_at(t):
    return {"name": f"logistic_threshold__threshold={t:.2f}__L=1",
            "function": "run_logistic_method",
            "kwargs": {"response_source": "score_threshold", "threshold": t, "L": 1}}


def causal_pip(coord, sim, c):
    fit = loader.run_method(coord, sim)
    return float(np.asarray(fit["single_effects"][0]["alpha"])[c])


def summ(vals):
    v = np.asarray(vals, float)
    return {"mean": float(np.nanmean(v)), "se": float(np.nanstd(v) / np.sqrt(len(v))),
            "n": int(np.sum(~np.isnan(v)))}


def main(out):
    cfg = CONFIG
    lib = loader.load_library()
    spec = loader.resolve_simulation(lib, design=cfg["design"], enrichment=cfg["enrichment"],
                                     signal=cfg["signal"], error=cfg["error"])
    flat = flat_methods()
    taus = cfg["thresholds"]
    acc_flat = {k: [] for k in flat}
    acc_cox = {t: [] for t in taus}
    acc_logi = {t: [] for t in taus}

    for rep in range(cfg["n_rep"]):
        sim = core.simulate(spec, replicate=rep)
        c = sim.causal_indices[0]
        for k, coord in flat.items():
            acc_flat[k].append(causal_pip(coord, sim, c))
        for t in taus:
            acc_cox[t].append(causal_pip(cox_at(t), sim, c))
            acc_logi[t].append(causal_pip(logistic_at(t), sim, c))
        if (rep + 1) % 25 == 0:
            print(f"  {rep + 1}/{cfg['n_rep']}", flush=True)

    try:
        git = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      cwd=ROOT, text=True).strip()
    except Exception:
        git = "unknown"

    result = {
        "config": cfg, "spec": spec.name,
        "flat": {k: summ(v) for k, v in acc_flat.items()},
        "cox_threshold": {f"{t:.1f}": summ(acc_cox[t]) for t in taus},
        "logistic_threshold": {f"{t:.1f}": summ(acc_logi[t]) for t in taus},
        "git_commit": git,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    pathlib.Path(out).write_text(json.dumps(result, indent=2) + "\n")
    print("wrote", out)
    print("flat:", {k: round(v["mean"], 3) for k, v in result["flat"].items()})
    print("cox: ", {k: round(v["mean"], 3) for k, v in result["cox_threshold"].items()})
    print("logi:", {k: round(v["mean"], 3) for k, v in result["logistic_threshold"].items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    main(ap.parse_args().out)
