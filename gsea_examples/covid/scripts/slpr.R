#!/usr/bin/env Rscript
# SLPR core: gene set Selection via LASSO Penalized Regression (Frost & Amos 2017,
# NAR 45(12):e114). Faithful port of the reference `slpr()` in that paper's SI
# (Section 3): regress the per-gene summary statistic on the genes x sets membership
# matrix under a LASSO penalty, then refit an unpenalized OLS on the selected sets
# (the two-stage "Gauss-Lasso"). Key reference choices kept here:
#   - response Z = gene-level EFFECT SIZE (log2FC), not a z/t-statistic (SI 1.2.1);
#   - cv.glmnet with standardize=FALSE, alpha=1 (pure LASSO), unpenalized intercept;
#   - lambda = value minimizing 10-fold CV MSE (lambda.min), snapped to the glmnet
#     path exactly as the reference does;
#   - stage-2 OLS on the non-zero-coef sets -> unshrunken coefs + (non-inferential)
#     p-values. Ranking is by |coef|; the coef sign gives enrichment direction.
# I/O is via the per-job workdir written by fit_slpr.py (see _rbridge.py).
suppressMessages({ library(Matrix); library(glmnet) })

args <- commandArgs(trailingOnly = TRUE)
workdir <- args[1]
rd <- function(f) file.path(workdir, f)

p <- read.table(rd("params.tsv"), header = TRUE, sep = "\t",
                stringsAsFactors = FALSE, colClasses = "character")
par <- setNames(p$value, p$key)
abs_response <- as.integer(par[["abs_response"]]) == 1L
alpha    <- as.numeric(par[["alpha"]])
nfolds   <- as.integer(par[["nfolds"]])
n_cv     <- as.integer(par[["num_cv_iter"]])
crit     <- par[["cv_criteria"]]
seed     <- as.integer(par[["seed"]])
# glmnet convergence thresh (1e-7) and maxit (1e5) are left at glmnet's defaults,
# which are exactly the values the reference slpr() passed explicitly.

# X: genes (obs) x sets (predictors); y: per-gene effect size.
X <- as(as(readMM(rd("membership.mtx")), "CsparseMatrix"), "dgCMatrix")
y <- scan(rd("response.tsv"), what = double(), skip = 1, quiet = TRUE)
if (abs_response) y <- abs(y)              # |Z|: power vs scale alternatives (SI 1.2.1)
n_sets <- ncol(X)
stopifnot(length(y) == nrow(X))

set.seed(seed)

# ---- stage 1: LASSO with lambda via k-fold CV (averaged over n_cv splits) --------
lambda.sum <- 0
cvfit <- NULL
for (i in seq_len(n_cv)) {
  cvfit <- cv.glmnet(x = X, y = y, standardize = FALSE, alpha = alpha,
                     family = "gaussian", intercept = TRUE, nfolds = nfolds)
  lambda.sum <- lambda.sum + if (crit == "lambda.1se") cvfit$lambda.1se else cvfit$lambda.min
}
mean.lambda <- lambda.sum / n_cv

# snap the (averaged) lambda onto the glmnet path, as the reference slpr() does
fit <- cvfit$glmnet.fit
larger <- which(fit$lambda >= mean.lambda)
lambda.index <- if (length(larger) == 0) 1L else larger[length(larger)]
lambda <- fit$lambda[lambda.index]

coef.lasso <- as.numeric(coef(fit, s = lambda))[2:(n_sets + 1)]   # drop intercept
nz <- which(coef.lasso != 0)
message(sprintf("SLPR: %d/%d sets selected at lambda=%.5g", length(nz), n_sets, lambda))

# ---- stage 2: unpenalized OLS on the selected sets (Gauss-Lasso) -----------------
coef.ols <- rep(0, n_sets)
pval.ols <- rep(NA_real_, n_sets)
if (length(nz) > 0) {
  Xs <- as.matrix(X[, nz, drop = FALSE])
  colnames(Xs) <- paste0("s", nz)
  dat <- data.frame(y = y, Xs, check.names = FALSE)
  ols <- lm(y ~ ., data = dat)
  sm <- summary(ols)$coefficients            # rows: (Intercept), s<idx>...
  for (k in seq_along(nz)) {
    nm <- paste0("s", nz[k])
    if (nm %in% rownames(sm)) {               # dropped rows => aliased (collinear) set
      coef.ols[nz[k]] <- sm[nm, "Estimate"]
      pval.ols[nz[k]] <- sm[nm, "Pr(>|t|)"]
    }
  }
}

out <- data.frame(
  set_idx     = seq_len(n_sets) - 1L,          # 0-based, aligns to Python set order
  coef_lasso  = coef.lasso,
  coef_ols    = coef.ols,
  ols_pvalue  = pval.ols,
  selected    = coef.lasso != 0,
  lambda      = lambda,
  n_selected  = length(nz),
  stringsAsFactors = FALSE
)
write.table(out, rd("slpr_out.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
