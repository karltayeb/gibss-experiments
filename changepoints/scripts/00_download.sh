#!/usr/bin/env bash
# Fetch the coal-mining disaster dates from two independent copies.
#   data/raw/coal_boot.csv  - R boot::coal (Jarrett 1979 corrected series, decimal dates)
#   data/raw/coal_pymc.csv  - pymc-examples copy of the same series (cross-check)
# Run from changepoints/:  bash scripts/00_download.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw

Rscript -e '
suppressPackageStartupMessages(library(boot))
write.csv(data.frame(date = sprintf("%.6f", coal$date)), "data/raw/coal_boot.csv",
          row.names = FALSE, quote = FALSE)
cat("boot::coal rows:", nrow(coal), "\n")
'

curl -fsSL -o data/raw/coal_pymc.csv \
  https://raw.githubusercontent.com/pymc-devs/pymc-examples/main/examples/data/coal.csv
echo "pymc coal.csv rows: $(wc -l < data/raw/coal_pymc.csv)"

# lambda phage genome: FASTA + GenBank flat file from NCBI E-utilities
EUTILS="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_001416.1"
curl -fsSL "${EUTILS}&rettype=fasta&retmode=text" -o data/raw/NC_001416.fasta
curl -fsSL "${EUTILS}&rettype=gb&retmode=text" -o data/raw/NC_001416.gb
echo "NC_001416.1: $(grep -v '>' data/raw/NC_001416.fasta | tr -d '\n' | wc -c | tr -d ' ') bp"
