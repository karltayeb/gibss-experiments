# Data notes

## Coal-mining disasters (Jarrett 1979 corrected series)

Dates of explosions in British coal mines killing 10 or more men, 15 Mar 1851 to
22 Mar 1962. The original list is Maguire, Pearson & Wynn (1952); Jarrett (1979,
Biometrika 66:191-193) corrected it and the corrected list is what every later analysis
uses.

Sources fetched by `scripts/00_download.sh`:

| file | source | events | first | last |
|---|---|---|---|---|
| `data/raw/coal_boot.csv` | R package `boot`, dataset `coal` (decimal dates) | 191 | 1851.203 | 1962.220 |
| `data/raw/coal_pymc.csv` | pymc-examples `examples/data/coal.csv` | 191 | 1851.203 | 1962.220 |

Cross-check (`scripts/01_build.py`): the two copies agree event for event, the largest
difference between sorted dates is 5e-5 years (print precision). The `boot` copy is the
one used downstream.

Binning: calendar year = floor(decimal date); calendar month = floor(12 * fractional
part). Yearly series 1851-1962: 112 values, total 191. Monthly series: 1344 values,
total 191, maximum 3 events in a month. 1851 and 1962 are partial years (the series
starts in March 1851 and ends in March 1962); they are kept as full-exposure years, as in
the classic analyses.

Comparison with the yearly series circulated from Carlin, Gelfand & Smith (1992): that
series was transcribed from memory into `src/coal.py` as a plausibility check, not from
the paper. It sums to 190 and differs from the `boot` binning in three years (1891: 2 vs
1; 1941: 3 vs 4; 1942: 3 vs 2). Those are the kind of off-by-one-year differences that
year-boundary conventions produce, and the transcription itself is not trustworthy to the
single count, so nothing here rests on it. The `boot` dates are authoritative.

Decade totals from the `boot` binning:

| decade | events |
|---|---|
| 1850s (9 yr) | 25 |
| 1860s | 35 |
| 1870s | 35 |
| 1880s | 28 |
| 1890s | 12 |
| 1900s | 12 |
| 1910s | 10 |
| 1920s | 5 |
| 1930s | 16 |
| 1940s | 13 |
| 1950s | 2 |
| 1960s (3 yr) | 2 |

## Lambda phage genome (NC_001416.1)

Enterobacteria phage lambda complete genome, NCBI RefSeq NC_001416.1 (GenBank J02459),
fetched by `scripts/00_download.sh` through NCBI E-utilities efetch as FASTA
(`data/raw/NC_001416.fasta`) and GenBank flat file (`data/raw/NC_001416.gb`, record
dated 11-JAN-2023).

| quantity | value |
|---|---|
| length | 48,502 bp |
| A / C / G / T | 12,334 / 11,362 / 12,820 / 11,986 (no N) |
| GC fraction | 0.4986 |
| GC-indicator autocorrelation, lags 1 / 2 / 3 / 6 | 0.029 / 0.038 / 0.024 / 0.040 |
| CDS features parsed from the GenBank record | 73 |

Binning (`scripts/01_build_lambda.py`): y_t = 1 for G or C; non-overlapping windows of
50, 100, 250, 500 and 1000 bp from position 1; a trailing remainder shorter than half a
window is merged into the last window (so 100 bp gives 485 bins, the last 102 bp; 1000 bp
gives 49 bins, the last 502 bp is kept separate because it is longer than half). Each bin
is (k = GC count, m = bases). Bin start positions are 1-based; a step column between bins
j and j+1 is labelled by the start of bin j+1, the first base of the new regime.

The CDS table (`data/processed/lambda/cds.csv`) keeps start, end, strand, gene and
product. The functional blocks in `config.yaml` (head 191-7965, tail 7977-22557, b2/ea
22686-26973, int-xis 27812-29078, ea/exo-bet-gam 29374-33330, cIII-N-rex-cI 33299-37940,
cro-cII-O-P 38041-40570, Nin 40644-43889, Q 43886-44509, lysis 45186-46427, bor/right
end 46459-48502) are read off that table by gene name; they are the textbook grouping,
not an independent annotation.
