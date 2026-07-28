"""Slide 8 (misspecification) experiment.

On SCALE-driven data, fit four methods and report mean causal PIP over ~150 sims:
  * twogroup_oracle     - correct f1 (upper bound)
  * twogroup_scale_fam  - correct FAMILY (estimates scale, loc=0)
  * twogroup_loc_fam    - MISSPECIFIED family (estimates loc, scale fixed 0.1)
  * cox_reversed        - rank method, models no f1 at all (should be robust)

Message: assuming the wrong f1 family tanks the two-group; the rank method, which
never models f1, is unaffected. Caches to figures/fig08_data.json.
"""
import argparse
import datetime as dt
import json
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gibss.distributions import Normal          # noqa: E402
import core                                      # noqa: E402
from experiments import loader                   # noqa: E402

CONFIG = {"design": "gaussian_rho0.50_n500_p100", "enrichment": "ser_b2",
          "signal": "scale_1.5", "error": "gaussian", "n_rep": 150}

METHODS = {
    "oracle": {"name": "twogroup_oracle__L=1", "function": "run_twogroup_method",
               "kwargs": {"f1": None, "L": 1}},
    "scale_fam": {"name": "twogroup_scale_fam__L=1", "function": "run_twogroup_method",
                  "kwargs": {"f1": Normal(loc=0.0, scale=1.0, estimate_loc=False,
                                          estimate_scale=True), "L": 1}},
    "loc_scale_fam": {"name": "twogroup__L=1", "function": "run_twogroup_method",
                      "kwargs": {"f1": Normal(loc=0.0, scale=1.0, estimate_loc=True,
                                              estimate_scale=True), "L": 1}},
    "loc_fam": {"name": "twogroup_loc_fam__L=1", "function": "run_twogroup_method",
                "kwargs": {"f1": Normal(loc=0.0, scale=0.1, estimate_loc=True,
                                        estimate_scale=False), "L": 1}},
    "cox_reversed": {"name": "cox_reversed__L=1", "function": "run_cox_method",
                     "kwargs": {"threshold": None, "time_sign": 1.0, "L": 1}},
}


def summ(v):
    v = np.asarray(v, float)
    return {"mean": float(np.nanmean(v)), "se": float(np.nanstd(v) / np.sqrt(len(v)))}


def main(out):
    cfg = CONFIG
    lib = loader.load_library()
    spec = loader.resolve_simulation(lib, design=cfg["design"], enrichment=cfg["enrichment"],
                                     signal=cfg["signal"], error=cfg["error"])
    acc = {k: [] for k in METHODS}
    for rep in range(cfg["n_rep"]):
        sim = core.simulate(spec, replicate=rep)
        c = sim.causal_indices[0]
        for k, coord in METHODS.items():
            fit = loader.run_method(coord, sim)
            acc[k].append(float(np.asarray(fit["single_effects"][0]["alpha"])[c]))
        if (rep + 1) % 25 == 0:
            print(f"  {rep + 1}/{cfg['n_rep']}", flush=True)
    try:
        git = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        git = "unknown"
    res = {"config": cfg, "spec": spec.name, "pip": {k: summ(v) for k, v in acc.items()},
           "git_commit": git, "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    pathlib.Path(out).write_text(json.dumps(res, indent=2) + "\n")
    print("wrote", out, {k: round(v["mean"], 3) for k, v in res["pip"].items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    main(ap.parse_args().out)
