#!/usr/bin/env Rscript
# Convert Lifan's factor sumstats RDS into tidy per-factor TSVs and map the
# gene ids (a mix of HGNC symbols and Ensembl gene ids) to Entrez ids, which is
# what the MSigDB collections use.
#
# Usage:
#   Rscript convert_rds.R <rds> <gene_info.gz> <n_factors> <mapping.tsv> <factors_dir>
#
# Emits:
#   <mapping.tsv>            original_id, entrez, match_type  (one row per gene)
#   <factors_dir>/factor_XXX.tsv   original_id, entrez, match_type,
#                                  posterior_mean, posterior_sd, lfsr

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
  stop("expected: <rds> <gene_info.gz> <n_factors> <mapping.tsv> <factors_dir>")
}
rds_path    <- args[[1]]
geneinfo    <- args[[2]]
n_factors   <- as.integer(args[[3]])
mapping_out <- args[[4]]
factors_dir <- args[[5]]

dir.create(factors_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(mapping_out), recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------------
# Build id -> Entrez lookups from NCBI Homo_sapiens.gene_info
# columns (1-indexed): 1 tax_id 2 GeneID 3 Symbol 4 LocusTag 5 Synonyms 6 dbXrefs
# ---------------------------------------------------------------------------
message("reading gene_info ...")
gi <- read.delim(
  gzfile(geneinfo), header = FALSE, skip = 1, sep = "\t", quote = "",
  comment.char = "", stringsAsFactors = FALSE, colClasses = "character"
)
geneid   <- gi[[2]]
symbol   <- gi[[3]]
synonyms <- gi[[5]]
dbxrefs  <- gi[[6]]

# Symbol -> Entrez (primary official symbol; keep first on the rare duplicate).
symbol2entrez <- geneid[!duplicated(symbol)]
names(symbol2entrez) <- symbol[!duplicated(symbol)]

# Ensembl -> Entrez, parsed out of the dbXrefs field ("...|Ensembl:ENSG...").
ens <- regmatches(dbxrefs, regexpr("Ensembl:ENSG[0-9]+", dbxrefs))
ens_geneid <- geneid[regexpr("Ensembl:ENSG[0-9]+", dbxrefs) > 0]
ens <- sub("Ensembl:", "", ens)
ensembl2entrez <- ens_geneid[!duplicated(ens)]
names(ensembl2entrez) <- ens[!duplicated(ens)]

# Synonym -> Entrez, but only unambiguous synonyms that are not themselves a
# primary symbol (fallback for retired/alias symbols).
syn_list <- strsplit(synonyms, "\\|", perl = TRUE)
syn_flat <- unlist(syn_list, use.names = FALSE)
syn_gene <- rep(geneid, lengths(syn_list))
keep <- syn_flat != "-" & !is.na(syn_flat)
syn_flat <- syn_flat[keep]; syn_gene <- syn_gene[keep]
# drop synonyms mapping to >1 distinct gene, and those equal to a primary symbol
amb <- tapply(syn_gene, syn_flat, function(g) length(unique(g)) > 1)
ambiguous <- names(amb)[amb]
bad <- syn_flat %in% ambiguous | syn_flat %in% names(symbol2entrez)
syn_flat <- syn_flat[!bad]; syn_gene <- syn_gene[!bad]
synonym2entrez <- syn_gene[!duplicated(syn_flat)]
names(synonym2entrez) <- syn_flat[!duplicated(syn_flat)]

# ---------------------------------------------------------------------------
# Map the factor gene ids
# ---------------------------------------------------------------------------
message("reading RDS ...")
x <- readRDS(rds_path)
if (length(x) != n_factors) {
  stop(sprintf("expected %d factors, RDS has %d", n_factors, length(x)))
}
ids <- rownames(x[[1]])

is_ens  <- grepl("^ENSG[0-9]+", ids)
ens_ids <- sub("\\..*$", "", ids)  # strip any version suffix
entrez     <- rep(NA_character_, length(ids))
match_type <- rep("unmapped", length(ids))

hit <- is_ens & ens_ids %in% names(ensembl2entrez)
entrez[hit] <- ensembl2entrez[ens_ids[hit]]; match_type[hit] <- "ensembl"

hit <- match_type == "unmapped" & !is_ens & ids %in% names(symbol2entrez)
entrez[hit] <- symbol2entrez[ids[hit]]; match_type[hit] <- "symbol"

hit <- match_type == "unmapped" & !is_ens & ids %in% names(synonym2entrez)
entrez[hit] <- synonym2entrez[ids[hit]]; match_type[hit] <- "synonym"

message(sprintf(
  "mapped %d/%d genes (%.1f%%): ensembl=%d symbol=%d synonym=%d unmapped=%d",
  sum(match_type != "unmapped"), length(ids),
  100 * mean(match_type != "unmapped"),
  sum(match_type == "ensembl"), sum(match_type == "symbol"),
  sum(match_type == "synonym"), sum(match_type == "unmapped")
))

mapping <- data.frame(
  original_id = ids, entrez = entrez, match_type = match_type,
  stringsAsFactors = FALSE
)
write.table(mapping, mapping_out, sep = "\t", quote = FALSE, row.names = FALSE)

# ---------------------------------------------------------------------------
# Per-factor tidy TSVs
# ---------------------------------------------------------------------------
message("writing per-factor tables ...")
for (i in seq_len(n_factors)) {
  m <- x[[i]]
  stopifnot(identical(rownames(m), ids))
  df <- data.frame(
    original_id    = ids,
    entrez         = entrez,
    match_type     = match_type,
    posterior_mean = as.numeric(m[, "posterior_mean"]),
    posterior_sd   = as.numeric(m[, "posterior_sd"]),
    lfsr           = as.numeric(m[, "LFSR"]),
    stringsAsFactors = FALSE
  )
  out <- file.path(factors_dir, sprintf("factor_%03d.tsv", i))
  write.table(df, out, sep = "\t", quote = FALSE, row.names = FALSE)
}
message("done.")
