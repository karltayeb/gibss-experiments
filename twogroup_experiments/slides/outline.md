# Two-group experiments - presentation outline

Status: planning. Audience: committee/advisor. Section 4 ("Revisiting GSEA") of the
2026-07-30 committee deck; leans on Logistic SuSiE (§1) and GLM SuSiE/GIBSS (§3).
Existing results are **seeds, not final** - most need a rerun after the gibss-mono
migration (twogroup fits route through the new front door / marginal, which *changes*
results; see `project_twogroup_frontdoor_parity_map`). Editorialize every backup title.

## Ground rules (locked)

1. **Beamer**, not Marp. LaTeX source, compiles to PDF.
2. **Simple slide construction** - the content is the message; do not over-style.
   Plain frames, minimal theme, no decorative chrome.
3. **Bibliography** included from the start (`.bib` + `\cite`); actual references get
   filled in a later edit round.
4. **Provenance for every generated figure.** Programmatic artifacts are built by a
   dedicated, self-contained `Snakemake` + scripts under `slides/` so each PDF/PNG traces
   back to the script + params that made it. No hand-run one-offs dropped into the deck.

Naming locked: **cox / cox-reverse** (= Plackett-Luce partial likelihood).
Current structure: 22 main slides (Acts A-D) + backup/derivation bank (B1-B4). See the
review surface artifact for the slide-by-slide with cartoons.

## Framing point (added): enrichment's two jobs

Enrichment analysis serves two purposes: (1) **characterize a strong result** - a few
large-effect genes in a small set; (2) **detect a small coordinated shift** - many genes
each nudged slightly. The threshold **2x2 / hit-list does (1) but is blind to (2)**: no
gene crosses the threshold, so the hit count looks null even when the set is really
shifted. This is the motivation for modeling richer resolutions and it sets up the richer
"cost of discretizing" example (slide 10, owned by the other agent). Figure:
`fig_two_regimes.py`; deck slide sits just before "Data arrives at different resolutions".

---

## 1. Through-line

**One question:** For covariate-moderated gene-set enrichment, how much resolution in
the gene-level data do we actually need? There is a spectrum -

  full test statistics  →  ranks  →  binary membership

Moving right loses information but buys **robustness**: you stop having to model (and
risk misspecifying) the distribution of gene-level effects. The talk walks this spectrum
with one fixed enrichment model (SER/SuSiE over which covariate drives enrichment) and
asks what each resolution costs and buys.

**Three claims the talk defends:**
1. The *correct* model (two-group, full data) is the hardest to fit - estimating `f1`
   is unreliable even with oracle knowledge, and that instability shows up in the
   enrichment inference.
2. Rank-based (Cox) and binary (logistic) models are misspecified *by construction*, but
   the misspecification is analyzable - and the **ordering direction matters**
   (cox-heavy vs cox-light / reversed). We can say *why* one ordering is safer.
3. The practical tradeoff is **power vs resolution**: logistic tends to report larger,
   higher-power credible sets; Cox reports smaller, lower-power sets. Cox-light is less
   sensitive to the binarization threshold than logistic - an attractive middle ground.

**Take-home:** you don't pay much for throwing information away *if* you order/threshold
in the direction the data support; and you can pay a lot for insisting on the "correct"
model when `f1` is not identifiable.

---

## 2. Spotlight track (the actual talk, ~12-15 slides)

Each spotlight is a **small, locally-runnable** figure or a single-dataset illustration -
no cluster needed. Heavy method-comparison sweeps live in the backup bank (§4) and are
pulled up on demand.

| # | Slide | Visual (spotlight) | Runnable now? |
|---|-------|--------------------|---------------|
| 1 | Title / motivation: GSEA from gene-level stats | none | - |
| 2 | The covariate-moderated two-group model | generative-model schematic (draw) | build |
| 3 | SER / SuSiE enrichment prior (one covariate drives π) | small diagram | build |
| 4 | The resolution spectrum: same data → z, ranks, binary | 1 dataset, 3 panels | **local** |
| 5 | Claim 1: estimating `f1` is hard | `f1` identifiability cartoon + bias boxplot | local + backup |
| 6 | Cox as a model of ranks; partial likelihood recap | equations | - |
| 7 | Misspecification: hazard ratio vs arrival time | `hazard_scale.png` (analytic) | **local** |
| 8 | Why direction matters: early vs late arrival sensitivity | perturbation illustration O(1/n) vs O(log n) | **local** |
| 9 | Well-specified check: exponential ranking | small paired sim | local (small) |
| 10 | Logistic = interval-censored binarization | schematic + threshold picture | build |
| 11 | Spotlight result: power vs resolution tradeoff | `power_fdp.png` + `coverage_size.png` | backup (rerun) |
| 12 | Cox-light vs logistic: threshold sensitivity | threshold-sweep line | backup (rerun) |
| 13 | Summary table: resolution × {power, CS size, robustness} | table | - |
| 14 | Recommendations / when to use what | text | - |

Slides 4, 7, 8, 9 are the **illustrative core** - small, exact, and don't depend on the
migration. Build these first; they carry the argument even before the big sims rerun.

---

## 3. Spotlight figures to build locally (priority order)

These are cheap scripts, independent of the cluster and mostly of the migration.

1. **Resolution spectrum panel (slide 4).** One simulated dataset. Panel A: z-scores with
   null/non-null colored. Panel B: same genes as ranks (arrival times), both orderings.
   Panel C: binary membership at a threshold. Shows exactly what information is discarded
   at each step. *New script.*
2. **Hazard-ratio misspecification (slide 7).** Already have `hazard_scale.png` /
   `hazard_loc.png` (InvChi2 blows up for the most informative obs; Chi2-reversed is
   stable). Regenerate from the analytic formulas in
   `Misspecification of the Gaussian two group model.md` so it's reproducible + restyled.
3. **Early vs late arrival sensitivity (slide 8).** Reproduce the O(1/n) vs O(-log n)
   perturbation from `Sensitivity of the cox model to early and late arrivals.md`:
   take one ranking, swap the first vs the last item into the enriched group, plot the
   change in the score. Motivates right-censoring / cox-light directly. *New script.*
4. **`f1` identifiability cartoon (slide 5).** Show the marginal `f̃` and how multiple
   (π, f1) pairs produce nearly the same marginal - why estimating f1 is ill-posed even
   before noise. *New small script.*
5. **Well-specified exponential ranking (slide 9).** Small paired sim: background Exp(1),
   enriched Exp(λ), fit Cox both directions; confirm calibration when PH actually holds.
   Seed exists at `009-*-cox-well-specified` (rerun small, local).

---

## 4. Backup-slide bank (heavy sims - placeholders, editorialized titles)

Pull up on demand. Every title states the question/take-home, not the plot type. All of
these need a **rerun post-migration** (see §5) - current PNGs under `slides/figures/` are
seeds to show the *format*, not final numbers.

| Editorialized title | Backing figure (seed) | Source collection | Rerun status |
|---------------------|-----------------------|-------------------|--------------|
| "Estimating f1 biases the enrichment estimate - even with oracle init" | `f1_boxplot.png` | `006-null-enrich-signal-loc` | rerun |
| "Logistic buys power by reporting larger credible sets" | `power_fdp.png` | `003-hallmark-loc-snr` | rerun |
| "At calibration, Cox resolves signals in smaller sets than logistic" | `coverage_size.png` | `003-hallmark-loc-snr` | rerun |
| "All methods' Bayes factors separate null from non-null" | `bf_roc.png` | `003-hallmark-loc-snr` | rerun |
| "Credible sets are calibrated once β is tuned" | `calibration.png` | `003-hallmark-loc-snr` | rerun |
| "Power/size/coverage trade off along the BF threshold" | `cs_trace.png` | `003-hallmark-loc-snr` | rerun |
| "When PH holds, Cox is well-calibrated (both directions)" | `cox_power.png`, `cox_coverage.png` | `009-hallmark-cox-well-specified` | rerun |
| "Error misspecification (t-errors) hurts the two-group model" | *placeholder* | `abcD` (not yet run) | **new run** |
| "Sample size amplifies partial-likelihood asymmetry" | *placeholder* | `001/003 n-sweep` | **new run** |
| "Location vs scale signal: when the linear model fails" | *placeholder* | loc vs scale SNR | **new run** |
| "Threshold sensitivity: cox-light degrades gracefully, logistic doesn't" | *placeholder* | threshold sweep | **new run** |

---

## 5. Migration + rerun checklist (blocking the backup bank)

- [ ] Migrate twogroup fits to current gibss-mono via the front door
      (`gibss.methods.fit_glm_susie` / new twogroup marginal). Expect **numbers to move** -
      twogroup is the one family whose parity map says results *change*, not match.
- [ ] Re-run `008-*-oracle-em-*` collections (currently empty scaffolding) - this is the
      `f1` oracle-vs-EM story; needed for slide 5 / backup.
- [ ] Re-run `003-hallmark-loc-snr` under migrated code for the power/resolution spotlight.
- [ ] Add the not-yet-run collections: t-error misspecification, n-sweep, loc-vs-scale,
      threshold sweep (specs already sketched in `notes/ser_simulations.md`).
- [ ] Regenerate all backup PNGs from migrated results; replace seeds in `slides/figures/`.

**Do the §3 local spotlights first** - they don't depend on any of this and already make
the argument.
