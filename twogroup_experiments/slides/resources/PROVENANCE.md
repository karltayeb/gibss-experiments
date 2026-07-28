# Redundancy slide - 4-stage reveal (covid, then simulation)

The redundancy slide is one frame with five overlay stages (`\only<1..5>`). In
stages 1-3 the real covid volcano stays fixed on the LEFT while the gene-coverage
curve on the RIGHT progressively reveals how few sets the signal really is;
stages 4-5 are the simulation, full width:

  1. `covid_volcano.pdf` + `covid_redundancy.pdf`  - volcano (540/4,815, 11%)
     beside the coverage curve (254 significance-ordered sets, or 36 greedy, to
     cover 90% of the significant genes).
  2. `covid_volcano.pdf` + `covid_redundancy1.pdf` - adds the 6 theme reps point
     (6 representatives cover 39% of significant genes).
  3. `covid_volcano.pdf` + `covid_redundancy2.pdf` - adds the 4 SuSiE credible
     sets point (4 components cover 29%): the joint-model punchline.
  4. `fig_ora_redundancy_marginal.pdf` - the simulated marginal cloud (below).
  5. `fig_ora_redundancy_causal.pdf`   - the simulated reveal: 10 causal sets.

`covid_volcano.pdf` and `covid_redundancy{,1,2}.pdf` are copied verbatim from
`gsea_examples/covid/eval/` (outputs of the covid logistic-SuSiE example's eval
step). Regenerate them there, then re-copy. (`covid_reveal.pdf`, the themed
volcano used earlier, was retired upstream in favour of this redundancy-panel
progression.) Simulated stages 4-5 build via the pipeline below.

# Redundancy simulation - stages 3-4 of the slide

- `gobp_collection.gmt` / `gobp_collection.meta.json` - a 4,870-set GO:BP
  collection (Entrez ids) built by `scripts/gobp_prep.py` from the MSigDB C5
  GO:BP sets (`c5.all.v2026.1.Hs.entrez.gmt`, shipped inside the covid example's
  gseasusie venv). Seeded (SEED=20260730): all GOBP sets of size 12-800 (matches
  the real covid GO:BP ORA scale, 4,815 sets), with 10 recorded "causal" sets
  (2 large / 3 medium / 5 small). Regenerate with `uv run python
  scripts/gobp_prep.py`; the MSigDB GMT is needed only to regenerate, never to
  build the figure.
- Figures `figures/fig_ora_redundancy_marginal.pdf` and `..._causal.pdf`
  (script `fig_ora_redundancy.py`): simulate gene z-scores under the two-group
  model (10 causal sets enriched; per-class activation so large sets don't
  dominate the top), run ORA (Fisher + BH), and plot the volcano (-log10 FDR vs
  log odds ratio) as two reveal stages - the marginal cloud (556/4,870 ~= 11%
  significant, matching covid) and the same volcano with the 10 causal sets
  highlighted by size class. Deck slide overlays the two with `\only<1..2>`.
  Seeded; provenance sidecars as usual.

# Legacy live-example images (no longer in the deck)

Copied from `~/Documents/presentations/2022_05_06_PCB/resources/` (the 2022-05-06
PCB talk, "A Multivariate Gene Set Enrichment Analysis with Logistic SuSiE").
Real data from the logistic-SuSiE GSEA analysis; not regenerated here.

- `burried.png`     - marginal enrichment volcano: hundreds of significant (red) gene
                      sets, the truly "active" ones (triangles) buried in the cloud.
- `tcellenrich.png` - T-cell GO:BP enrichment (logistic vs linear SuSiE): the blue
                      cloud of redundant marginal enrichments, with SuSiE credible-set
                      components L1-L4 picking representatives.
- `stepwise4.png`   - forward selection peeling signals off one at a time (Step 1-4;
                      residual significance recomputed after each pick).
- `go_example.png`  - GO DAG: one signal nested under many overlapping parent terms
                      (why enrichments are redundant).
- `R01renewal2022.pdf` - the funded R01 renewal (Stephens lab). Provides the framing:
                      Aim 2 = "joint GSEA" replacing thousands of marginal tests with one
                      SuSiE regression. Aim 2a = logistic SuSiE on a hit-list; Aim 2b = the
                      two-group model on effect sizes (this deck's model). The redundancy
                      numbers on slide 2 (CD19+ B cells: 1,117 / 9,896 GO sets enriched;
                      DAVID->538, WebGestalt->235 clusters) are the R01 preliminary study.
