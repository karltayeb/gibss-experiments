"""covid `convert` step (bespoke): turn the GSE147507 DESeq2 tables into tidy
per-contrast TSVs that the shared `prep` step consumes.

The DE source (`de/de_results.tsv`) is keyed by HGNC symbol with the usual DESeq2
columns (log2FoldChange, lfcSE, stat, pvalue, padj). This step:

  1. builds a symbol/synonym/Ensembl -> Entrez lookup from NCBI Homo_sapiens.gene_info
     (MSigDB collections are keyed by Entrez),
  2. for each contrast, computes a per-gene ranking `score` (see score_kind) and a
     `signed_effect` (log2FoldChange, for set-level direction annotation),
  3. writes results/units/{contrast}.tsv with columns [original_id, entrez, score,
     signed_effect] and a results/genes/gene_mapping.tsv for provenance.

Genes DESeq2 could not test (NA stat/padj) are dropped: they are not part of the
tested universe, so they should not enter the hit list or the background.
"""
import gzip
import re

import numpy as np
import polars as pl

_ENS = re.compile(r"Ensembl:(ENSG[0-9]+)")


def build_id_maps(gene_info_gz):
    """symbol->entrez, ensembl->entrez, synonym->entrez (unambiguous fallback)."""
    symbol2e, ens2e = {}, {}
    syn_gene = {}      # synonym -> set(entrez)
    primary = set()
    with gzip.open(gene_info_gz, "rt") as fh:
        next(fh)  # header
        for line in fh:
            f = line.rstrip("\n").split("\t")
            geneid, symbol, synonyms, dbxrefs = f[1], f[2], f[4], f[5]
            primary.add(symbol)
            symbol2e.setdefault(symbol, geneid)
            m = _ENS.search(dbxrefs)
            if m:
                ens2e.setdefault(m.group(1), geneid)
            if synonyms != "-":
                for s in synonyms.split("|"):
                    syn_gene.setdefault(s, set()).add(geneid)
    # keep only synonyms that map to exactly one gene and are not a primary symbol
    synonym2e = {s: next(iter(g)) for s, g in syn_gene.items()
                 if len(g) == 1 and s not in primary}
    return symbol2e, ens2e, synonym2e


def map_ids(ids, symbol2e, ens2e, synonym2e):
    entrez, match_type = [], []
    for gid in ids:
        base = gid.split(".")[0]
        if base.startswith("ENSG") and base in ens2e:
            entrez.append(ens2e[base]); match_type.append("ensembl")
        elif gid in symbol2e:
            entrez.append(symbol2e[gid]); match_type.append("symbol")
        elif gid in synonym2e:
            entrez.append(synonym2e[gid]); match_type.append("synonym")
        else:
            entrez.append(None); match_type.append("unmapped")
    return entrez, match_type


def de_score(df, score_kind):
    lfc = df["log2FoldChange"].to_numpy()
    if score_kind == "abs_stat":
        return np.abs(df["stat"].to_numpy())          # |Wald z|, two-sided significance
    if score_kind == "signed_stat":
        return df["stat"].to_numpy()                  # directional (up-regulated first)
    if score_kind == "neglog_padj":
        padj = np.clip(df["padj"].to_numpy(), 1e-300, 1.0)
        return -np.log10(padj)
    raise ValueError(f"unknown score_kind: {score_kind!r}")


def _stat_col(score_kind):
    return "padj" if score_kind == "neglog_padj" else "stat"


def convert(de_path, gene_info_gz, units, score_kind, mapping_out, unit_paths):
    symbol2e, ens2e, synonym2e = build_id_maps(gene_info_gz)
    de = pl.read_csv(de_path, separator="\t", null_values=["NA"])

    all_ids = de["gene_id"].unique().to_list()
    entrez, match_type = map_ids(all_ids, symbol2e, ens2e, synonym2e)
    mapping = pl.DataFrame({"original_id": all_ids, "entrez": entrez, "match_type": match_type})
    n_mapped = mapping.filter(pl.col("match_type") != "unmapped").height
    print(f"mapped {n_mapped}/{len(all_ids)} symbols to Entrez "
          f"({100 * n_mapped / max(len(all_ids), 1):.1f}%)")
    mapping.write_csv(mapping_out, separator="\t")

    need = _stat_col(score_kind)
    for unit, out_path in zip(units, unit_paths):
        sub = (
            de.filter(pl.col("contrast_id") == unit)
            .join(mapping, left_on="gene_id", right_on="original_id", how="left")
            .filter(pl.col("match_type") != "unmapped")
            .drop_nulls([need, "log2FoldChange"])
        )
        if sub.height == 0:
            raise ValueError(f"contrast {unit!r} produced no usable rows "
                             f"(missing in {de_path} or all unmapped/NA)")
        out = pl.DataFrame({
            "original_id": sub["gene_id"],
            "entrez": sub["entrez"],
            "score": de_score(sub, score_kind),
            "signed_effect": sub["log2FoldChange"],
        })
        out.write_csv(out_path, separator="\t")
        print(f"  {unit}: {out.height} genes -> {out_path}")


if "snakemake" in globals():
    smk = snakemake  # noqa: F821  (injected by Snakemake)
    convert(
        smk.input.de, smk.input.gene_info,
        list(smk.params.units), smk.params.score_kind,
        smk.output.mapping, list(smk.output.units),
    )
