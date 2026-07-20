"""covid `deseq2` step (bespoke): derive the differential-expression table from the
raw GSE147507 human counts *in-pipeline* (pydeseq2), so `de/` is a generated result
rather than a provided input. Minimal input in, everything else out of the pipeline.

Each configured target is one contrast defined by {series, treatment, control}: select
that series' treatment+control samples from `sample_metadata`, subset the raw counts,
and run pydeseq2 (design ~condition, control = reference level). The output schema is
exactly what `convert` consumes downstream: contrast_id, gene_id (HGNC symbol),
log2FoldChange, stat, padj (plus baseMean, lfcSE, pvalue, and annotation columns).
"""
import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

_COLS = ["contrast_id", "sample_type", "treatment_condition", "control_condition",
         "species", "gene_id", "gene_symbol", "baseMean", "log2FoldChange", "lfcSE",
         "stat", "pvalue", "padj"]


def run_deseq2(counts_path, meta_path, targets, out_path, species="human"):
    counts = pd.read_csv(counts_path, sep="\t", index_col=0)          # genes x samples
    meta = pd.read_csv(meta_path, sep="\t", index_col="sample_id")
    frames = []
    for cid, spec in targets.items():
        series, trt, ctl = spec["series"], spec["treatment"], spec["control"]
        m = meta[(meta["series_id"] == series) & (meta["condition"].isin([trt, ctl]))]
        samples = [s for s in m.index if s in counts.columns]
        if not samples:
            raise ValueError(f"{cid}: no samples for series={series} conditions=({trt},{ctl})")
        X = counts[samples].T                                        # samples x genes
        X = X.loc[:, X.sum(axis=0) > 0]                              # drop all-zero genes
        md = pd.DataFrame({"condition": m.loc[samples, "condition"].astype(str)})
        dds = DeseqDataSet(counts=X, metadata=md, design="~condition",
                           ref_level=["condition", ctl])
        dds.deseq2()
        ds = DeseqStats(dds, contrast=["condition", trt, ctl])
        ds.summary()
        r = ds.results_df.rename_axis("gene_id").reset_index()
        r.insert(0, "contrast_id", cid)
        r["sample_type"] = str(m.loc[samples, "cell_type"].iloc[0])
        r["treatment_condition"] = trt
        r["control_condition"] = ctl
        r["species"] = species
        r["gene_symbol"] = np.nan
        frames.append(r)
        print(f"  {cid}: {len(samples)} samples ({trt} vs {ctl}, {series}), {r.shape[0]} genes tested")
    pd.concat(frames, ignore_index=True)[_COLS].to_csv(out_path, sep="\t", index=False)


if "snakemake" in globals():
    smk = snakemake  # noqa: F821  (injected by Snakemake)
    run_deseq2(smk.input.counts, smk.input.meta, dict(smk.params.targets), smk.output.de)
