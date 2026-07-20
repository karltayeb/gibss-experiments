# MGSA core: model-based gene set analysis (Bauer, Gagneur & Robinson 2010, NAR
# 38:3523; Bioconductor `mgsa`). A Bayesian multiset model over BINARY gene/gene-set
# activity: a gene is "on" if any containing set is active, and the observed hit list
# is a noisy (alpha=FP, beta=FN) readout. MCMC returns, for each set, the marginal
# posterior probability it is active (`estimate`), plus posteriors over the global
# alpha/beta/p parameters. Unlike SLPR this needs a discretized hit list, so we run
# it once per top-fraction threshold (the same hit lists ORA/logistic use, produced
# in Python for cross-method consistency). I/O via the workdir from fit_mgsa.py.
suppressMessages(library(mgsa))

args <- commandArgs(trailingOnly = TRUE)
workdir <- args[1]
rd <- function(f) file.path(workdir, f)

p <- read.table(rd("params.tsv"), header = TRUE, sep = "\t",
                stringsAsFactors = FALSE, colClasses = "character")
par <- setNames(p$value, p$key)
n_genes  <- as.integer(par[["n_genes"]])
steps    <- as.numeric(par[["steps"]])
restarts <- as.integer(par[["restarts"]])
thin     <- as.integer(par[["thin"]])
threads  <- as.integer(par[["threads"]])
seed     <- as.integer(par[["seed"]])

# sets: gene-id (index) vectors, named by set index; population: all tested genes.
long <- read.table(rd("membership_long.tsv"), header = TRUE, colClasses = "integer")
sets <- split(as.character(long$gene_idx), as.character(long$set_idx))
population <- as.character(seq_len(n_genes) - 1L)

obs <- read.table(rd("obs.tsv"), header = TRUE)  # columns: threshold, gene_idx
pmean <- function(df) sum(df$value * df$estimate)  # posterior mean over the grid

res_all <- list()
for (t in sort(unique(obs$threshold))) {
  o <- as.character(obs$gene_idx[obs$threshold == t])
  set.seed(seed)
  r <- mgsa(o, sets, population = population,
            steps = steps, restarts = restarts, thin = thin, threads = threads)
  sr <- setsResults(r)
  res_all[[length(res_all) + 1L]] <- data.frame(
    set_idx      = as.integer(rownames(sr)),
    threshold    = t,
    estimate     = sr$estimate,            # posterior P(set active)
    std_error    = sr$std.error,           # NA unless restarts > 1
    in_study_set = sr$inStudySet,
    in_population = sr$inPopulation,
    alpha_post   = pmean(r@alphaPost),     # posterior-mean global params (repeated)
    beta_post    = pmean(r@betaPost),
    p_post       = pmean(r@pPost),
    stringsAsFactors = FALSE
  )
  message(sprintf("MGSA threshold=%.3g: %d study genes, alpha=%.3f beta=%.3f p=%.4f",
                  t, length(o), pmean(r@alphaPost), pmean(r@betaPost), pmean(r@pPost)))
}
write.table(do.call(rbind, res_all), rd("mgsa_out.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE)
