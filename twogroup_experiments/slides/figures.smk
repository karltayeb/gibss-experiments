# Figure provenance pipeline for the GSEA / two-group committee slides.
#
# Every figure in the deck is built here so it traces back to a script + params.
# Each build also writes a figures/<name>.prov.json sidecar (git commit,
# timestamp, params) via scripts/_common.save().
#
#   snakemake -s figures.smk all -c4          # build everything
#   snakemake -s figures.smk figures/fig06_resolution_spectrum.pdf
#
# Two kinds of figure:
#   SCRIPT_FIGS  - produced by a standalone script (local, no cluster).
#   STUB_FIGS    - real content comes from the results pipeline post-migration;
#                  a labelled placeholder stands in until then.

# Standalone analytic / small-sim figures (fast).
SCRIPT_FIGS = {
    "fig_two_regimes":           "scripts/fig_two_regimes.py",
    "fig06_resolution_spectrum": "scripts/fig06_resolution_spectrum.py",
    "fig08_f1_identifiability":  "scripts/fig08_f1_identifiability.py",
    "fig09_logistic_estimand":   "scripts/fig09_logistic_estimand.py",
    "fig14_hazard_loc_scale":    "scripts/fig14_hazard_loc_scale.py",
    "fig15_late_arrival":        "scripts/fig15_late_arrival.py",
    "figB2_cox_poisson":         "scripts/figB2_cox_poisson.py",
}

# fig08 parameterization variants (loc / moment) share one script via --param.
FIG08_VARIANTS = ["loc", "moment"]

# fig10 (PIP) and fig11 (ROC) are the cost-of-discretizing figures. Both read
# figures/cod_data.json, extracted from the RESULTS PIPELINE by cod_extract.py (the
# 010-c2-cost-of-discretizing supercollection, 200 reps/cell) - the fits come from
# the pipeline, not re-simulated here. The cached JSON is committed, so `all` only
# replots. (scripts/fig10_experiment.py is the retired stand-alone pilot.)

# name -> (slide label, title, note). Rendered via scripts/_pipeline_stub.py.
STUB_FIGS = {
    "fig17_wellspec_calibration": ("Slide 17", "When PH holds, it's calibrated",
        "exponential-ranking sim; both directions on the diagonal. seed: 009-cox-well-specified"),
    "fig18_power_vs_resolution":  ("Slide 18", "Power traded for resolution",
        "logistic bigger CS / more power, cox smaller CS. seed: 003-hallmark-loc-snr"),
    "fig19_threshold_sensitivity":("Slide 19", "Threshold sensitivity",
        "cox graceful, logistic dilutes as threshold loosens. NEW sweep"),
    "fig20_punchline_locsweep":   ("Slide 20", "cox-reverse wins when it's hard",
        "derive from a location-signal (SNR) sweep. seed: 003 loc"),
    "figB3_f1_bias":              ("Backup B3", "Estimating f1 biases enrichment",
        "f1 estimate drifts off truth, worst estimating loc+scale. seed: 006 / 008-oracle-em"),
    "figB4_misspecification":     ("Backup B4", "What a misspecified model does",
        "degradation under t-errors / heteroskedasticity; twogroup suffers most. NEW run"),
}

ALL = (list(SCRIPT_FIGS) + list(STUB_FIGS)
       + ["fig10_pip_vs_threshold", "fig11_cod_roc", "fig08_misspecification",
          "fig_ora_redundancy_marginal", "fig_ora_redundancy_causal"])


# Slide 8 (misspecification): slow SER experiment -> cache -> fast plot, like fig10.
rule fig08_data:
    input:
        "scripts/fig08_experiment.py",
    output:
        "figures/fig08_data.json",
    shell:
        "uv run python {input} --out {output}"


rule fig08_figure:
    input:
        script="scripts/fig08_misspecification.py",
        data="figures/fig08_data.json",
        common="scripts/_common.py",
    output:
        "figures/fig08_misspecification.pdf",
    shell:
        "uv run python {input.script} --out {output}"


# fig08 identifiability-ridge variants (loc/moment) remain buildable on demand
# (the scale ridge, fig08_f1_identifiability, stays in SCRIPT_FIGS as a backup).
rule fig08_variant:
    input:
        script="scripts/fig08_f1_identifiability.py",
        common="scripts/_common.py",
    output:
        "figures/fig08_f1_identifiability_{pm}.pdf",
    wildcard_constraints:
        pm="|".join(FIG08_VARIANTS),
    shell:
        "uv run python {input.script} --param {wildcards.pm} --out {output}"


rule all:
    input:
        expand("figures/{name}.pdf", name=ALL),


# Redundancy live example: simulate z-scores under the two-group model over a
# committed GO:BP collection (~4,900 sets, covid-scale), run ORA, and plot the
# volcano as two reveal stages (marginal cloud vs the 10 causal sets underneath).
# The collection (resources/gobp_collection.gmt) is committed and rebuilt only on
# demand via scripts/gobp_prep.py, so this needs no cluster/venv.
rule fig_ora_redundancy:
    input:
        script="scripts/fig_ora_redundancy.py",
        common="scripts/_common.py",
        gmt="resources/gobp_collection.gmt",
    output:
        marginal="figures/fig_ora_redundancy_marginal.pdf",
        causal="figures/fig_ora_redundancy_causal.pdf",
    shell:
        "uv run python {input.script} "
        "--out-marginal {output.marginal} --out-causal {output.causal}"


# Cost-of-discretizing data: extracted from the RESULTS PIPELINE (not re-simulated).
# Run the pipeline supercollection first, then this caches the PIP + ROC curves:
#   uv run snakemake -s twogroup_experiments.snk -c11 \
#       results/supercollections/010-c2-cost-of-discretizing/.done
# cod_extract.py reads results/by_batch/.../{fits,reductions/pip}.parquet. The cache
# (figures/cod_data.json) is committed, so the deck rebuilds without rerunning fits;
# delete it (or edit cod_extract.py) to regenerate from fresh pipeline outputs.
rule cod_data:
    input:
        "scripts/cod_extract.py",
    output:
        "figures/cod_data.json",
    shell:
        "uv run python {input} --out {output}"


rule fig10_figure:
    input:
        script="scripts/fig10_pip_vs_threshold.py",
        data="figures/cod_data.json",
        common="scripts/_common.py",
    output:
        "figures/fig10_pip_vs_threshold.pdf",
    shell:
        "uv run python {input.script} --out {output}"


rule fig11_figure:
    input:
        script="scripts/fig11_cod_roc.py",
        data="figures/cod_data.json",
        common="scripts/_common.py",
    output:
        "figures/fig11_cod_roc.pdf",
    shell:
        "uv run python {input.script} --out {output}"


rule script_figure:
    input:
        script=lambda wc: SCRIPT_FIGS[wc.name],
        common="scripts/_common.py",
    output:
        "figures/{name}.pdf",
    wildcard_constraints:
        name="|".join(SCRIPT_FIGS),
    shell:
        "uv run python {input.script} --out {output}"


rule stub_figure:
    input:
        stub="scripts/_pipeline_stub.py",
        common="scripts/_common.py",
    output:
        "figures/{name}.pdf",
    wildcard_constraints:
        name="|".join(STUB_FIGS),
    params:
        slide=lambda wc: STUB_FIGS[wc.name][0],
        title=lambda wc: STUB_FIGS[wc.name][1],
        note=lambda wc: STUB_FIGS[wc.name][2],
    shell:
        "uv run python {input.stub} --out {output} "
        "--slide {params.slide:q} --title {params.title:q} --note {params.note:q}"
