#!/usr/bin/env Rscript
# pbmc_purified `convert` step (bespoke): turn the purified-PBMC topic-model DE
# object into tidy per-topic TSVs that the shared `prep` step consumes.
#
# The source (`de-pbmc-purified-noshrink.RData`) is a `topic_model_de_analysis`:
# per-gene DE of a 6-topic (k1..k6) model of FACS-purified PBMCs (Zheng et al.
# 2017). Genes are Ensembl ids; per topic it carries `postmean` (signed loading,
# complete) and `z` (Wald z, ~250 NA/topic); `lfsr` is degenerate here so we rank
# on z. This step maps Ensembl -> Entrez (MSigDB's id space) via NCBI gene_info
# and, per topic, writes results/units/{topic}.tsv with columns [original_id,
# entrez, score, signed_effect].
#
# Usage:
#   Rscript convert.R <rdata> <gene_info.gz> <units_csv> <score_kind> <mapping.tsv> <units_dir>
#     score_kind: abs_z (|Wald z|, two-sided) | signed_postmean | abs_postmean

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 6) {
  stop("expected: <rdata> <gene_info.gz> <units_csv> <score_kind> <mapping.tsv> <units_dir>")
}
rdata_path  <- args[[1]]
geneinfo    <- args[[2]]
units       <- strsplit(args[[3]], ",")[[1]]
score_kind  <- args[[4]]
mapping_out <- args[[5]]
units_dir   <- args[[6]]

dir.create(units_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(mapping_out), recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------------
# id -> Entrez lookups from NCBI Homo_sapiens.gene_info
# columns (1-indexed): 1 tax_id 2 GeneID 3 Symbol 4 LocusTag 5 Synonyms 6 dbXrefs
# ---------------------------------------------------------------------------
message("reading gene_info ...")
gi <- read.delim(gzfile(geneinfo), header = FALSE, skip = 1, sep = "\t", quote = "",
                 comment.char = "", stringsAsFactors = FALSE, colClasses = "character")
geneid <- gi[[2]]; symbol <- gi[[3]]; synonyms <- gi[[5]]; dbxrefs <- gi[[6]]

symbol2entrez <- geneid[!duplicated(symbol)]; names(symbol2entrez) <- symbol[!duplicated(symbol)]

ens <- regmatches(dbxrefs, regexpr("Ensembl:ENSG[0-9]+", dbxrefs))
ens_geneid <- geneid[regexpr("Ensembl:ENSG[0-9]+", dbxrefs) > 0]
ens <- sub("Ensembl:", "", ens)
ensembl2entrez <- ens_geneid[!duplicated(ens)]; names(ensembl2entrez) <- ens[!duplicated(ens)]

# ---------------------------------------------------------------------------
# Load the topic-model DE object
# ---------------------------------------------------------------------------
message("reading RData ...")
e <- new.env(); load(rdata_path, envir = e)
de <- get("de", envir = e)
genes <- get("genes", envir = e)                 # data.frame(ensembl, symbol)

postmean <- de$postmean
z        <- de$z
ids      <- rownames(postmean)                   # Ensembl ids
stopifnot(all(units %in% colnames(postmean)))

# Ensembl -> Entrez, with a symbol fallback via the paired `genes` table.
ens_base <- sub("\\..*$", "", ids)
entrez     <- rep(NA_character_, length(ids))
match_type <- rep("unmapped", length(ids))

hit <- ens_base %in% names(ensembl2entrez)
entrez[hit] <- ensembl2entrez[ens_base[hit]]; match_type[hit] <- "ensembl"

sym <- genes$symbol[match(ids, genes$ensembl)]   # aligned to ids
hit <- match_type == "unmapped" & !is.na(sym) & sym %in% names(symbol2entrez)
entrez[hit] <- symbol2entrez[sym[hit]]; match_type[hit] <- "symbol"

message(sprintf("mapped %d/%d genes (%.1f%%): ensembl=%d symbol=%d unmapped=%d",
  sum(match_type != "unmapped"), length(ids), 100 * mean(match_type != "unmapped"),
  sum(match_type == "ensembl"), sum(match_type == "symbol"), sum(match_type == "unmapped")))

write.table(data.frame(original_id = ids, entrez = entrez, match_type = match_type),
            mapping_out, sep = "\t", quote = FALSE, row.names = FALSE)

# ---------------------------------------------------------------------------
# Per-topic tidy TSVs: score (ranking) + signed_effect (= postmean)
# ---------------------------------------------------------------------------
score_for <- function(k) {
  pm <- postmean[, k]; zz <- z[, k]
  if (score_kind == "abs_z")          return(abs(zz))
  if (score_kind == "signed_postmean") return(pm)
  if (score_kind == "abs_postmean")    return(abs(pm))
  stop(sprintf("unknown score_kind: %s", score_kind))
}

message("writing per-topic tables ...")
for (k in units) {
  score  <- score_for(k)
  signed <- postmean[, k]
  keep <- match_type != "unmapped" & is.finite(score) & is.finite(signed)
  df <- data.frame(original_id = ids[keep], entrez = entrez[keep],
                   score = score[keep], signed_effect = signed[keep],
                   stringsAsFactors = FALSE)
  out <- file.path(units_dir, sprintf("%s.tsv", k))
  write.table(df, out, sep = "\t", quote = FALSE, row.names = FALSE)
  message(sprintf("  %s: %d genes -> %s", k, nrow(df), out))
}
message("done.")
