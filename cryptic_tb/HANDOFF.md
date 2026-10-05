# CRyPTIC TB drug resistance — logistic SuSiE real-data example

Brief for Claude Code. Goal: set up a reproducible analysis applying logistic SuSiE
(gibss) to *M. tuberculosis* drug resistance in the CRyPTIC data, and report whether it
gives a result worth showing in a stats paper on logistic SuSiE / credible sets.

Follow the repo conventions in `../AGENTS.md`: use `uv`; treat `cryptic_tb/` as the root
of this analysis. Reuse the gibss fitting code in `../logistic_susie_experiments/fits/`
where possible instead of re-implementing.

## Why this example (the pitch)

- Binary phenotype (resistant / susceptible), ~12k isolates, public data.
- Ground truth: the WHO mutation catalogue grades mutations (1 = associated with
  resistance ... 5 = not associated). First-line drugs (RIF, INH) are well explained by
  known mutations -> **validation**. New drugs (BDQ, LZD, CFZ) have many "uncertain
  significance" mutations -> **discovery with uncertainty**.
- Methods gap: the WHO catalogue handles co-occurring mutations with the "SOLO"
  heuristic (grade a mutation only from isolates where it is the sole candidate variant;
  two-tier gene lists; masking). The CRyPTIC GWAS (PLoS Biol 2022, 10.1371/journal.pbio.3001755)
  used marginal LMMs on k-mers vs MIC. Neither jointly attributes resistance among
  co-occurring variants or gives credible sets. Logistic SuSiE does.
- Expected stats talking points: clonal (genome-wide) LD; homoplasy gives resolution;
  compensatory mutations (rpoC/rpoA for RIF) and co-resistance (MDR) create correlated
  non-causal signals; near-separation (resistance mutations ~deterministic);
  allelic heterogeneity (pncA, Rv0678) where single-variant SuSiE should struggle.

## Data

Network note: the Cowork session that wrote this could not reach EBI/GitHub; you
(Claude Code on Karl's machine) should be able to.

1. **CRyPTIC release** (EBI FTP): `https://ftp.ebi.ac.uk/pub/databases/cryptic/`
   (look under `release_june2022/`). Data compendium paper: PLoS Biol 2022,
   10.1371/journal.pbio.3001721. Expected contents (verify — names from memory):
   - a reuse table `CRyPTIC_reuse_table_*.csv` with one row per isolate: `UNIQUEID`,
     ENA accessions, site/country, plate type, and per drug `{DRUG}_MIC`,
     `{DRUG}_BINARY_PHENOTYPE` (R/S), `{DRUG}_PHENOTYPE_QUALITY` (HIGH/MEDIUM/LOW),
     plus a path to the per-isolate VCF.
   - `reproducibility/` data tables (e.g. GENOMES, MUTATIONS, VARIANTS, EFFECTS,
     UKMYC_PHENOTYPES) — MUTATIONS/EFFECTS likely cover catalogue genes only;
     VARIANTS may be genome-wide.
   - per-isolate VCFs (regenotyped against H37Rv, NC_000962.3).
   Start by listing the FTP tree and reading any README; write what you find to
   `docs/data_notes.md` before building anything.
2. **WHO catalogue** (truth set): 2023 2nd edition, GitHub
   `GTB-tbsequencing/mutation-catalogue-2023` (also the 2021 1st edition xlsx).
   Need per drug: gene tiers, mutation names, grading (groups 1-5).
3. Lineage: if not in the reuse table, compute PCs from a genome-wide variant matrix,
   or call lineages with the standard SNP barcode (e.g. TB-Profiler) — PCs are enough.

Use `scripts/00_download.sh` as a starting point; fix paths once you've seen the tree.

## Analysis plan

### Phenotypes
- Drugs: validation **RIF, INH**; discovery **BDQ, LZD, CFZ**; optional EMB, LEV/MXF.
- y = 1 if `BINARY_PHENOTYPE == R`. Keep `PHENOTYPE_QUALITY` in {HIGH, MEDIUM}
  (report n and case counts per drug for both HIGH-only and HIGH+MEDIUM).

### Genotypes (two predictor sets per drug)
- **A. Candidate genes** (main analysis): all variants in WHO tier-1 + tier-2 genes for
  the drug, including promoter/upstream regions as the catalogue defines them
  (e.g. fabG1 c.-15C>T, inhA promoter, ahpC promoter, eis promoter). Include SNPs,
  indels and MNVs as separate binary features (presence in isolate). Keep
  non-synonymous + promoter + frameshift/stop; drop synonymous (keep a flag to include
  them as a negative control). Minor allele count >= 3 (try 5, 10).
- **B. Genome-wide** (secondary): all variants with MAC >= 10 (sparse matrix). Purpose:
  see whether credible sets land on compensatory, co-resistance or lineage-marker
  variants when the model is allowed to look everywhere.
- Haploid: genotype is 0/1. Treat heterozygous/mixed calls (low FRS, e.g. < 0.9) as
  missing; drop isolates with excessive mixed calls; set remaining missing to 0 and
  record how many.
- Collapse exactly-identical columns (identical carrier sets) into one feature but keep
  a map — report them; they are perfectly confounded and SuSiE cannot separate them.

### Covariates
- Sampling site / country (one-hot), plate type (UKMYC5 vs UKMYC6), top ~10 genome-wide
  PCs (lineage). Optionally lineage indicators instead of PCs (sensitivity).
- Check how gibss handles covariates. Preferred: fixed covariates in the model. If
  only offsets are supported: fit `glm(y ~ covariates)` and pass its linear predictor
  as a fixed offset (state the approximation in notes). Report results with and
  without covariates.
- Do NOT condition on resistance to other drugs (it's downstream / collider-like);
  instead report co-resistance as a diagnostic.

### Fits
- gibss logistic SuSiE: `irls` (Laplace) and `logistic_profile` / `logistic_quadrature`
  impls; L in {5, 10, 20}; estimated prior variance (also try fixed, e.g. 1, 4, to
  probe separation sensitivity). Standardize or not — record choice.
- Comparators: linear SuSiE on 0/1 (susieR or the linear impl in
  `../logistic_susie_experiments/fits/linear_susie.py`); marginal logistic Wald/LRT per
  variant; WHO SOLO grades as published.
- 95% credible sets, default purity filter (min |r| 0.5); also report unfiltered.

### Sensitivity / stability (Karl's idea)
- Across impls, L, prior-variance setting, covariate sets, MAC threshold.
- Subsampling: n in {500, 1000, 2000, 5000, all} x 20 reps (stratified by R/S). Track
  CS size, Jaccard stability of CSs across reps, and whether WHO group-1 mutations are
  covered. Stability is not calibration — keep that distinction in the write-up.
- Calibration via semi-synthetic: real genotype matrix (candidate genes), simulate y
  from a few chosen variants with realistic effects + covariate effects; check CS
  coverage. Low priority unless real-data results look good.

## What counts as "interesting/promising" (check explicitly)

1. **Recovery (RIF/INH)**: high-PIP singleton CSs at rpoB S450L, H445X, D435X; katG
   S315T, fabG1 c.-15C>T, inhA promoter / inhA S94A. Fraction of R isolates carrying a
   variant in some CS.
2. **Joint attribution**: CSs that contain multiple co-occurring variants (e.g.
   fabG1 c.-15C>T with inhA I21T/S94A; katG 315 with inhA promoter) — show SuSiE
   expressing ambiguity SOLO hides; and WHO group-1/2 variants that get low PIP because a
   co-occurring variant explains the signal.
3. **Uncertain -> confident**: WHO group-3 ("uncertain significance") variants that land in
   pure, high-PIP CSs (esp. BDQ/LZD/CFZ). List them with carrier counts and R fraction.
4. **Genome-wide run**: do CSs pick up compensatory rpoC/rpoA (RIF) or other-drug
   resistance mutations (co-resistance)? Do covariates remove lineage markers?
5. **Failure modes** (also useful for the paper): allelic heterogeneity in Rv0678
   (BDQ/CFZ) or pncA (PZA) — many rare LoF variants, weak per-variant evidence. Compare
   against a gene-level burden feature (any LoF in gene) added to X.
6. **Separation**: report any features with perfect/near-perfect separation and how prior
   variance estimates behave; compare impls.
7. Linear vs logistic SuSiE: where do PIPs/CSs differ? (Large effects -> expect
   differences.)

## Deliverables

- `results/phenotype_counts.csv` (via `scripts/01_phenotype_table.py`) — per-drug n tested / n resistant by quality; first thing to report back.
- `docs/data_notes.md` — what was downloaded, schemas, filters, sample counts.
- `scripts/` — numbered, re-runnable steps (download -> build matrices -> fit -> summarize).
  A small Snakefile is welcome but not required.
- `data/processed/{drug}/` — X (sparse npz) + feature table (pos, ref, alt, gene,
  mutation name in WHO notation, MAC), y, covariates, sample ids.
- `results/{drug}/` — fits (pickled/npz), CS tables annotated with gene, mutation, WHO
  grade, carriers, %R among carriers, marginal p-value.
- `results/summary.md` — per drug: n, cases, #CSs, CS sizes, recovery vs WHO, the
  most interesting examples for items 1-7, and an honest verdict on whether this is a
  good paper example. Plus a few figures (PIP along gene coordinates coloured by WHO
  grade; CS composition; subsampling stability).

Keep raw data out of git (`data/` is gitignored).

## Verified details (from the compendium paper, PLoS Biol 2022, 10.1371/journal.pbio.3001721)

- Release root: `ftp.ebi.ac.uk/pub/databases/cryptic/release_june2022/` with `reuse/` and
  `reproducibility/` directories.
- `reuse/CRyPTIC_reuse_table_20221019.csv`: 12,288 isolates — binary phenotypes, MICs,
  phenotype quality, ENA accessions, VCF paths. `CRyPTIC_excluded_samples_20220607.tsv`:
  2,922 sequence-only isolates (no usable phenotype).
- Supporting tables: GENOTYPES.csv, VARIANTS.csv, MUTATIONS.csv, SAMPLES.csv, PHENOTYPES.csv.
- 13 drugs: RIF, INH, EMB, AMI, KAN, RFB, LEV, MXF, ETH, BDQ, CFZ, DLM, LZD.
- Binary phenotype: R if MIC > ECOFF. Quality HIGH (>=2 reading methods agree),
  MEDIUM, LOW. Plates UKMYC5 / UKMYC6 (covariate).
- Variant calls: Clockwork v0.8.3 -> Minos v0.11.0, filters depth >= 5, FRS >= 0.9.
- Lineages assigned with Mykrobe (L1-L4 = 99.7%) — check whether lineage is in SAMPLES.csv;
  use as covariate alongside PCs.
- Sampling is enriched for resistance at most sites (do not interpret prevalences).
- Resistance prevalence: INH 49.0%, RIF 38.7%, LEV 17.6%, CFZ 4.4%, DLM 1.6%, LZD 1.3%,
  BDQ 0.9% -> new drugs have only ~100-550 resistant isolates. Consider adding DLM.
- Original analysis code: github.com/kerrimalone/Brankin_Malone_2022.
