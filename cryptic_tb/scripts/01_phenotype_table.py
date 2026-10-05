"""Per-drug counts: isolates with a binary phenotype, and number resistant,
overall and by phenotype quality. Run first after download:
    uv run python scripts/01_phenotype_table.py data/raw/CRyPTIC_reuse_table_20221019.csv
Column names are assumed to be {DRUG}_BINARY_PHENOTYPE (R/S) and
{DRUG}_PHENOTYPE_QUALITY (HIGH/MEDIUM/LOW); the script lists what it finds.
"""
import sys
import polars as pl

path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/CRyPTIC_reuse_table_20221019.csv"
df = pl.read_csv(path, infer_schema_length=0)
drugs = sorted(c.removesuffix("_BINARY_PHENOTYPE") for c in df.columns if c.endswith("_BINARY_PHENOTYPE"))
print(f"{df.height} isolates; drugs found: {drugs}")

rows = []
for d in drugs:
    ph, q = f"{d}_BINARY_PHENOTYPE", f"{d}_PHENOTYPE_QUALITY"
    sub = df.filter(pl.col(ph).is_in(["R", "S"]))
    for label, keep in [("all", None), ("high+medium", ["HIGH", "MEDIUM"]), ("high", ["HIGH"])]:
        s = sub if keep is None or q not in df.columns else sub.filter(pl.col(q).is_in(keep))
        n, r = s.height, s.filter(pl.col(ph) == "R").height
        rows.append({"drug": d, "quality": label, "n_tested": n, "n_resistant": r,
                     "pct_resistant": round(100 * r / n, 1) if n else None})
out = pl.DataFrame(rows)
out.write_csv("results/phenotype_counts.csv")
print(out.filter(pl.col("quality") == "high+medium"))
