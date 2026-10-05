# changepoints

GLM SuSiE (gibss) for changepoint detection with credible intervals:
British coal-mining disasters (Poisson, core) and lambda phage GC content (binomial,
extension), benchmarked against exact Bayesian changepoint posteriors. See `HANDOFF.md`
for the brief and `results/summary.md` for the findings.

Self-contained uv project (pins its own gibss-mono rev in `pyproject.toml`). Run from
this directory:

    uv sync
    bash scripts/00_download.sh              # boot::coal (R) + pymc copy
    uv run python scripts/01_build.py        # yearly / monthly counts, cross-checks
    uv run python scripts/02_exact_posterior.py   # Fearnhead 2006 recursions, prior grid
    uv run python scripts/03_fit_susie.py    # Poisson SuSiE, L x sensitivities
    uv run python scripts/04_simulate_calibration.py   # designs A and B (~15 min)
    uv run python scripts/05_summarize.py    # figures + comparison tables
    # lambda phage (binomial; the FASTA/GenBank fetch is in 00_download.sh)
    uv run python scripts/01_build_lambda.py
    uv run python scripts/02_exact_lambda.py        # Beta-Binomial exact posterior + 2-state HMM
    uv run python scripts/03_fit_lambda.py          # binomial SuSiE, widths x L (~10 min)
    uv run python scripts/04_calibration_lambda.py  # binomial / beta-binomial / autocorrelated truth
    uv run python scripts/05_summarize_lambda.py
    uv run pytest tests                      # exact recursions vs brute force; gibss Binomial (trials=)

Layout: `src/` (step basis, exact posterior, HMM, fit wrapper), `scripts/` (numbered steps),
`data/raw`, `data/processed`, `results/coal/{year,month}/{exact,fits,calibration}`,
`results/coal/figures`, `docs/data_notes.md`.
