#!/usr/bin/env bash
# Data discovery + download. Paths below are best guesses — list the tree first
# and edit. Writes into data/raw/.
set -euo pipefail
cd "$(dirname "$0")/.."
RAW=data/raw; mkdir -p "$RAW"
BASE=https://ftp.ebi.ac.uk/pub/databases/cryptic

# 1. Look before downloading
curl -sL "$BASE/" | sed 's/<[^>]*>/ /g' | grep -v '^\s*$' > "$RAW/ftp_root_listing.txt"
curl -sL "$BASE/release_june2022/" | sed 's/<[^>]*>/ /g' | grep -v '^\s*$' > "$RAW/ftp_release_listing.txt" || true
cat "$RAW"/ftp_*listing.txt

# 2. Reuse table + reproducibility tables (EDIT names after inspecting listings)
# wget -c -P "$RAW" "$BASE/release_june2022/CRyPTIC_reuse_table_<DATE>.csv"
# wget -c -r -np -nH --cut-dirs=4 -P "$RAW/reproducibility" "$BASE/release_june2022/reproducibility/"

# 3. WHO catalogue 2023 (truth set)
# git clone --depth 1 https://github.com/GTB-tbsequencing/mutation-catalogue-2023 "$RAW/who_catalogue_2023"
