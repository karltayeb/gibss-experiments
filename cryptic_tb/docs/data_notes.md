# Data notes

What was downloaded, what each file holds, how the two sources were joined, and which
filters produced the matrices in `data/processed/`. Written during discovery on
2026-10-04; counts are from the files actually on disk.

## Sources

Everything is public. `data/` is gitignored.

### CRyPTIC, EBI FTP

Root: `https://ftp.ebi.ac.uk/pub/databases/cryptic/release_june2022/`. The tree has three
useful branches: `reuse/` (recommended entry point), `reproducibility/` (the per-paper
data, including the large variant tables) and `pubs/gwas2022/` (the PLoS Biology GWAS).

| Local path | Source | Size | Contents |
|---|---|---|---|
| `raw/reuse/CRyPTIC_reuse_table_20240917.csv` | `reuse/` | 5.4 MB | 12,287 isolates, one row each |
| `raw/reuse/CRyPTIC_excluded_samples_20220607.tsv` | `reuse/` | 692 KB | 2,922 excluded samples |
| `raw/data_tables/MUTATIONS.csv.gz` | `reproducibility/data_tables/cryptic-analysis-group/` | 1.49 GB | 88.2M rows, gene-annotated calls |
| `raw/data_tables/VARIANTS.csv.gz` | same | 2.8 GB | nucleotide-level calls, genome-wide, with FRS |
| `raw/data_tables/GENOMES.csv.gz` | same | 5.2 MB | per-genome QC, Mykrobe lineage |
| `raw/data_tables/UKMYC_PHENOTYPES.csv.gz` | same | 4.0 MB | per-isolate per-drug MIC readings, plate design |
| `raw/data_tables/SAMPLES.csv.gz` | same | 250 KB | country, collection date, clinical fields |
| `raw/data_tables/MYKROBE_LINEAGE.csv.gz` | same | 518 KB | lineage calls |
| `raw/cryptic-index_20231027.json` | `reproducibility/` | 23 MB | 30,971 sequencing records, VCF paths |
| `raw/gwas2022/site_batches_plus_plate_covariates.txt` | `pubs/gwas2022/data/` | 204 KB | the GWAS paper's own covariate matrix |
| `raw/ref/Mtub_H37Rv_NC000962.3.fasta`, `.gb` | `pubs/gwas2022/ref/` | 4.3 / 9.8 MB | H37Rv reference and annotation |

### WHO mutation catalogue, 2023 second edition

`git clone --depth 1 https://github.com/GTB-tbsequencing/mutation-catalogue-2023` into
`raw/who_catalogue_2023/`. Two files matter, both under `Final Result Files/`:

- `WHO-UCN-TB-2023.6-eng_catalogue_master_file.txt`, 48,152 rows, one per (drug, variant).
  Carries the gene, the tier, the effect, the SOLO counts and `FINAL CONFIDENCE GRADING`
  (`1) Assoc w R` through `5) Not assoc w R`).
- `WHO-UCN-TB-2023.7-eng_genomic_coordinates.txt`, 144,964 rows mapping each variant name
  to `(position, ref, alt)` on NC_000962.3. One name has many spellings, because a protein
  change is reachable by several nucleotide changes.

## Decisions taken during discovery

### The per-isolate VCFs are not worth downloading

One isolate's regenotyped VCF is 19.7 MB with 1,265,713 records, nearly all `0/0`: it
carries a call at every site any isolate varies at. Across 12,287 isolates that is roughly
240 GB. The per-sample (`masked`) VCF is only 25 KB but lists just the non-reference calls,
so a site absent from the file is either reference or uncalled and the two cannot be
distinguished.

The `MUTATIONS` and `VARIANTS` tables hold the same regenotyped calls with depth, FRS and
filter flags already attached, which is why the pipeline reads those instead. Total 4.3 GB
rather than 240 GB.

### Phenotype counts per drug

From the reuse table, isolates with a binary call of the given quality. `HIGH` means at
least two of the three reading methods agreed on the MIC.

| Drug | HIGH n | HIGH R | HIGH+MEDIUM n | HIGH+MEDIUM R |
|---|---|---|---|---|
| RIF | 8,955 | 3,448 | 10,310 | 4,247 |
| INH | 9,518 | 4,467 | 10,868 | 5,328 |
| RFB | 10,041 | 3,195 | 11,392 | 3,966 |
| EMB | 6,614 | 1,460 | 7,697 | 1,791 |
| LEV | 7,773 | 1,178 | 9,126 | 1,602 |
| MXF | 6,784 | 991 | 8,136 | 1,324 |
| ETH | 8,251 | 1,189 | 9,427 | 1,511 |
| KAN | 9,332 | 759 | 10,686 | 985 |
| AMI | 8,972 | 597 | 10,321 | 762 |
| CFZ | 7,762 | 106 | 9,113 | 139 |
| DLM | 8,094 | 85 | 9,442 | 98 |
| LZD | 7,140 | 76 | 8,494 | 90 |
| BDQ | 8,535 | 71 | 9,889 | 94 |

The pipeline keeps `HIGH` and `MEDIUM`, which buys roughly 15% more isolates and 20% more
resistant ones. The three new-drug arms stay thin either way: under 150 resistant isolates
each. That ceiling, not the method, is what limits the discovery claims.

**Pyrazinamide is not in this cohort.** The config originally listed PZA as the
allelic-heterogeneity failure-mode drug, but the UKMYC plates do not include it and the
reuse table has no `PZA_BINARY_PHENOTYPE` column. The pipeline skips the drug and records
the reason. The same failure mode is still reachable through Rv0678 for bedaquiline and
clofazimine, which is where it is now studied.

### Covariates available without further computation

- Site: the two-digit field inside `UNIQUEID`, 11 sites in the cohort, led by Peru (2,643),
  Milan (1,828), South Africa NICD (1,631) and Mumbai (1,509).
- Plate design: `UKMYC5` for 6,269 isolates and `UKMYC6` for 6,018, from `UKMYC_PHENOTYPES`.
- Lineage: `GENOMES.MYKROBE_LINEAGE_NAME_1` covers all 12,287. Lineage 4 (6,034),
  lineage 2 (4,295), lineage 3 (1,053), lineage 1 (671), mixed (202).
- Country: `SAMPLES` covers 9,919 of 12,287, so it is weaker than site and is not used.

Genome-wide principal components are **not yet computed**; they need `VARIANTS`. Lineage
indicators stand in for them for now, and the summary shows where that is not enough.

## Joining CRyPTIC calls to WHO grades

The two sources name the same variant differently, so neither name is a join key:

| Variant | CRyPTIC | WHO |
|---|---|---|
| rpoB codon 450 Ser to Leu | `rpoB` + `S450L` | `rpoB_p.Ser450Leu` |
| the isoniazid promoter variant | `fabG1` + `c-15t` | `inhA_c.-777C>T` |
| an Rv0678 frameshift | `Rv0678` + `139_indel` | `Rv0678_p.Asp47fs` |
| a premature stop | `Rv2752c` + `Q247!` | `Rv2752c_p.Gln247*` |

`src/who.py` bridges them two ways, tried in order.

**Genomic coordinate first.** The catalogue's coordinate file gives
`(position, ref, alt)` for each variant name, and the CRyPTIC `MUTATIONS` row carries
`GENOME_INDEX`, `REF` and `ALT` for any nucleotide-level call. This join is
nomenclature-free, and it is the only one that reaches a variant the two sources file under
different genes. The headline case is the isoniazid promoter: CRyPTIC books it under
`fabG1` at `c-15`, WHO books the same base under `inhA` at `c.-777`, because WHO measures
the shared fabG1-inhA operon promoter from the inhA start codon. WHO lists no `fabG1` tier
for isoniazid at all. A name-only join silently drops this variant, and it is the second
strongest isoniazid signal in the cohort, so dropping it would have been a quiet, serious
error. 30,647 of the 30,699 graded variant names are reachable by coordinate.

**Canonical key second.** Both names are parsed into one key, for example
`("aa", "rpoB", 450, "L")` or `("fs", "Rv0678", 47)`. The frameshift key deliberately omits
the reference amino acid, because CRyPTIC does not record it. Three-letter amino-acid codes
become one letter and CRyPTIC's `!` stop becomes `*`.

**Gene-level loss of function last.** WHO grades many individually-too-rare knockouts only
as a collapsed `{gene}_LoF` feature. A CRyPTIC frameshift with no variant-level grade
inherits that gene-level grade, and the `who_match` column records which route was used, so
an inherited grade is never mistaken for a variant-specific one.

The result: 98% of design columns carry a grade (RIF 381 of 390, INH 311 of 319).

## Filters that produced `data/processed/`

1. `scripts/01_filter_mutations.py` streams `MUTATIONS.csv.gz` once, keeping the ~45 genes
   in any studied drug's WHO tier 1 or 2 plus `fabG1`, the compensatory RIF genes and a few
   others. 88,172,303 rows in, 3,040,431 kept, 193 s, 18.7 MB parquet.
2. `scripts/02_build_design.py` builds one design per drug:
   - isolates with a binary call of kept quality;
   - variants in that drug's gene set, synonymous calls dropped (a flag keeps them as a
     negative control);
   - a call with `IS_NULL` or a failed filter is treated as **missing, not reference**. The
     upstream VCF already applies a minimum FRS of 0.9, so a mixed call arrives as a null
     rather than as a fractional genotype, and `MUTATIONS` records no heterozygous calls at
     all. Missing counts are reported per feature and per drug, and they are large: 121,409
     missing calls for RIF.
   - minor allele count at least 3;
   - variants with an **identical carrier set collapsed into one column**, with the folded
     names kept in `collapsed.csv`. No method can separate them, so presenting them
     separately would be misleading.
   - one gene-level LoF burden column per gene, set when the isolate carries any
     frameshift, premature stop or lost start in that gene.

Genotypes are haploid 0/1 and the matrix is stored sparse (`scipy` CSR, converted to a jax
BCOO for the fit so the design is never densified).

## Still outstanding

- `VARIANTS.csv.gz` (2.8 GB) is still downloading. It is needed for the genome-wide
  secondary analysis and for genome-wide principal components. Until then the covariate
  offset uses lineage indicators, and `results/summary.md` shows three credible sets landing
  on common WHO grade-5 variants, which is the cost of that shortcut.
- Subsampling stability, the semi-synthetic calibration check, the linear-SuSiE comparator
  and the published SOLO grades as a side-by-side are all unstarted.
