# GSEA examples for logistic SuSiE

A curated collection of real-data examples for gene-set enrichment analysis (GSEA)
with logistic SuSiE (and, where the input is a ranking, the Cox variant). Each
subdirectory holds the **input data** for one example - the upstream statistical
object (a differential-expression table or a factor-analysis fit) that gets turned
into gene lists / gene rankings and handed to the enrichment model. Downstream fits
and figures are *not* stored here; they are regenerated from these inputs.

**Runnable pipelines.** `covid/` and `pbmc_purified/` are self-contained Snakemake
+ uv projects that run the full logistic / Cox / ORA SuSiE suite against MSigDB
(`convert -> prep -> fit -> gather`; see each example's `README.md` / `RUNNING.md`).
They mirror the `lifan` example's pipeline: the `prep` / `fit` / `scoring` scripts
are shared verbatim, and only the `convert` step (which reads that example's DE /
factor source and defines the per-gene ranking score) is bespoke. The `lifan`
pipeline itself lives in the `gibss-experiments` repo at `gsea_examples/lifan/` on
`origin/main`; here `lifan/` holds only its input RDS. `deng/` is data-only so far.

The four examples were chosen to cover the two dominant input modalities and to be
interpretable and (mostly) publicly reproducible:

| Example | Data source | Input modality | # analyses | Enrichment model |
|---|---|---|---|---|
| [`covid/`](#covid--gse147507-nhbe) | GSE147507 (public) | Differential expression | 2 contrasts | logistic |
| [`pbmc_purified/`](#pbmc_purified--sorted-immune-cells) | Zheng et al. 2017 (public) | Differential expression | 5-6 cell types | logistic |
| [`deng/`](#deng--developmental-scrna-seq-factors) | Deng et al. developmental scRNA-seq | Matrix-factorization loadings | 38 topics | logistic |
| [`lifan/`](#lifan--ebmf-factor-analysis) | Lifan's EBMF factor analysis | Matrix-factorization loadings | 140 factors | logistic + Cox |

**Input-modality summary.** `covid` and `pbmc_purified` are *differential-expression*
examples (gene lists thresholded from DE statistics). `deng` and `lifan` are
*factor-analysis* examples (per-gene loadings / LFSR from a low-rank decomposition,
converted to gene lists or rankings). Note these factorizations are NMF / SNMF
(deng) and EBMF (lifan) - not a literal cEBMF fit.

---

## `covid/` - GSE147507 NHBE

**Source.** Blanco-Melo et al., GEO accession **GSE147507**: NHBE (normal human
bronchial epithelial) cells and other respiratory models challenged with SARS-CoV-2,
IAV, RSV, and other viruses. Public and fully reproducible.

**Modality.** Differential expression (DESeq2 on pseudobulk counts), then gene lists
thresholded at `abs(log2FoldChange) > 0` and `padj < 0.05`.

**Analyses.** Two NHBE contrasts drive the enrichment comparison:
1. `nhbe_sars_cov_2_vs_mock`
2. `nhbe_iav_vs_mock`

The two-virus contrast is the narrative: a shared inflammatory/interferon core plus
virus-specific response, both well-characterized so credible-set correctness is easy
to judge. `raw/contrast_spec.tsv` enumerates the full set of 20+ contrasts available
in the study (human + ferret, many series) should the example be extended.

**Contents.**
- `raw/` - the public source data: `GSE147507_RawReadCounts_Human.tsv.gz`,
  `GSE147507_RawReadCounts_Ferret.tsv.gz`, `GSE147507_family.soft.gz`,
  `sample_metadata.tsv`, `contrast_spec.tsv`, plus `checksums.sha256` and a raw README.
- `de/` - DESeq2 output: `de_results.tsv` (all contrasts), per-contrast tables under
  `results/`, and the `contrast_manifest.tsv` / `contrast_annotations.tsv` / `run_metadata.tsv`.
- `gene_lists/` - thresholded gene lists fed to the enrichment model, per contrast
  (`gene_lists.tsv`, `results/`, `summary.tsv`).

**Provenance.** `interpretable-gsea/data/gse147507/` (raw) and
`interpretable-gsea/analysis/covid-de/derived/{de,gene_lists}/` (derived). The full
pipeline (DE -> gene lists -> gene sets -> ORA + SuSiE + L=1 agreement diagnostics)
lives in `interpretable-gsea/analysis/covid-de/`.

---

## `pbmc_purified/` - sorted immune cells

**Source.** FACS-purified PBMC populations from **Zheng et al. 2017** (10x Genomics
reference dataset). Public.

**Modality.** Differential expression (DESeq2, no-shrink), one cell type vs the rest;
gene lists taken from the top of the DE ranking.

**Analyses.** 5-6 immune cell types (CD19+ B, CD14+ monocyte, CD34+, CD56+ NK, and
the CD4/CD8 T subsets - sometimes collapsed to a single T-cell group, hence 5 vs 6).
Each cell type has a known signature, so this is the "recovers the right biology,
repeatedly" example rather than a one-off.

**Contents.**
- `de-pbmc-purified-noshrink.RData` - the DESeq2 DE results object (per-cell-type
  statistics) used directly by the enrichment pipeline.

**Provenance.** `interpretable-gsea/data/pbmc-purified/`. The `logistic-susie-gsea`
repo also carries `deseq2-pbmc-purified.RData` and several `analysis/single_cell_pbmc*.Rmd`
notebooks for the same data.

---

## `deng/` - developmental scRNA-seq factors

**Source.** **Deng et al.** mouse pre-implantation embryo scRNA-seq, decomposed with
NMF and semi-NMF (SNMF) topic models.

**Modality.** Matrix-factorization loadings. Each topic's per-gene loading (thresholded
by LFSR) becomes a gene list for enrichment - the factor-analysis analog of a DE list.

**Analyses.** 38 topics total: 19 NMF + 19 SNMF (topics 2-20 for each), with positive
and negative loadings analyzed where applicable.

**Contents.**
- `nmf.rds` - NMF decomposition (loadings + LFSR).
- `snmf.rds` - SNMF decomposition (loadings + LFSR).

**Provenance.** `interpretable-gsea/data/deng/` (also `logistic-susie-gsea/data/deng/`).
Analysis notebook: `logistic-susie-gsea/analysis/deng_example.Rmd`.

---

## `lifan/` - EBMF factor analysis

**Source.** Lifan's EBMF (empirical-Bayes matrix factorization) fit. Human data.

**Modality.** Matrix-factorization loadings. `factor_sumstats.RDS` is an unnamed list
of **140 factors**, each a `20107 gene x 3` matrix with columns `posterior_mean`
(signed loading), `posterior_sd` (~constant), and `LFSR`. Gene ids are a mix of HGNC
symbols (~84%) and Ensembl (~16%); the pipeline maps them to Entrez via NCBI
`Homo_sapiens.gene_info`.

**Analyses.** 140 factors. Each factor is run against MSigDB C2 through logistic SuSiE,
Cox (forward and reversed), and ORA - a two-sided-ranking comparison. The signed
loading + LFSR gives both a hit list (logistic / ORA) and a ranking (Cox), which is
why this example exercises the Cox variant, unlike the DE examples.

**Contents.**
- `factor_sumstats.RDS` (74 MB) - the 140-factor EBMF summary-statistics list.

**Provenance.** The runnable Snakemake pipeline is tracked separately at
`gibss-experiments/gsea_examples/lifan/` on `origin/main` (id->Entrez conversion,
sparse-design prep, per-factor logistic/Cox/ORA fits). The 74 MB RDS is gitignored
there and transferred out of band; this is that file.

---

## Notes

- **Sizes.** `lifan/` 74M, `deng/` 83M, `covid/` 24M, `pbmc_purified/` 7M (~188M total).
  These are large binary/data files; if this collection is version-controlled, keep the
  `.RDS`/`.rds`/`.RData` and the `.tsv.gz` raw counts out of git and transfer them
  out of band.
- **What's here vs not.** Only *inputs* are stored. Gene-set collections (MSigDB / GO)
  are shared reference data pulled at fit time, and downstream fits/figures are
  regenerated - neither is duplicated here.
- **A gap.** None of these is a true cEBMF fit; the factor examples are NMF/SNMF (deng)
  and EBMF (lifan). Add a genuine cEBMF example here if that modality needs its own
  demonstration.
