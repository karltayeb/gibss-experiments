"""Tables and derived numbers for the changepoints notebook, computed from saved results.

Everything here reads `results/` and `data/processed/`; nothing refits a model except the
exact-posterior segmentation samples, which are cheap and seeded.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import polars as pl
from great_tables import GT, md

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from exact import PoissonGamma, sample_segmentations
from stepbasis import cs_interval

RESULTS = ROOT / "results"
PROC = ROOT / "data/processed"


def gt(df: pl.DataFrame, legend: str, *, decimals: int = 2, title: str | None = None,
       per_column: dict[str, int] | None = None) -> GT:
    """great_tables view with a column legend as the source note. Floats are rounded to
    `decimals` (`per_column` overrides), with no thousands separators (years, bp), and
    missing values are blank."""
    per_column = per_column or {}
    floats = [c for c, t in df.schema.items() if t in (pl.Float64, pl.Float32)]
    g = GT(df)
    if title:
        g = g.tab_header(title=title)
    for c in floats:
        g = g.fmt_number(columns=c, decimals=per_column.get(c, decimals), use_seps=False)
    ints = [c for c, t in df.schema.items() if t.is_integer()]
    if ints:
        g = g.fmt_integer(columns=ints, use_seps=False)
    g = g.sub_missing(missing_text="")
    return g.tab_source_note(md(legend)).tab_options(table_font_size="13px")


def load_pickle(path: Path) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------- coal

def coal_years():
    d = pl.read_csv(PROC / "coal/year.csv")
    return d["label"].to_numpy(), d["count"].to_numpy().astype(float)


def coal_fit(tag: str, res: str = "year") -> dict:
    return load_pickle(RESULTS / f"coal/{res}/fits/{tag}.pkl")


def blocks(cs, alpha):
    """Contiguous runs of a credible set: list of (lo, hi, alpha mass)."""
    cols = np.array(sorted(cs))
    runs = np.split(cols, np.flatnonzero(np.diff(cols) > 1) + 1)
    return [(int(r[0]), int(r[-1]), float(alpha[r].sum())) for r in runs]


def coal_component_table(tag: str) -> pl.DataFrame:
    """SuSiE components with log BF > 0.5, with the main contiguous block of each CS."""
    years, _ = coal_years()
    fit = coal_fit(tag)
    rows = []
    for l in range(fit["L"]):
        lbf = float(fit["ser_log_bf"][l])
        if lbf < 0.5:
            continue
        a = fit["alpha"][l]
        cs = fit["cs"][l]
        lo, hi, _ = cs_interval(cs)
        main = max(blocks(cs, a), key=lambda b: b[2])
        top = int(np.argmax(a))
        rows.append({
            "effect": l + 1, "log BF": lbf, "declared": lbf >= 2,
            "CS size": len(cs),
            "CS hull": f"{years[lo + 1]}-{years[hi + 1]}",
            "main block": f"{years[main[0] + 1]}-{years[main[1] + 1]}",
            "block mass": main[2],
            "top": int(years[top + 1]), "top alpha": float(a[top]),
            "rate ratio": float(np.exp(fit["mu"][l, top])),
        })
    return pl.DataFrame(rows)


def coal_exact_windows(p_values, shape=1.0, n_samples=4000, seed=0) -> pl.DataFrame:
    """Exact posterior probability of at least one changepoint in each window, from
    perfect posterior samples of the segmentation."""
    years, y = coal_years()
    model = PoissonGamma(shape, shape / y.mean())
    M = model.segment_logml(y)
    grid = pl.read_csv(RESULTS / "coal/year/exact/grid_summary.csv").filter(
        pl.col("shape") == shape)
    rng = np.random.default_rng(seed)
    rows = []
    for p in p_values:
        S = sample_segmentations(M, p, n_samples, rng)
        g = grid.filter(pl.col("p_per_year") == p).row(0, named=True)

        def frac(lo, hi, S=S):
            return float(np.mean([np.any((years[s] >= lo) & (years[s] <= hi)) for s in S]))

        rows.append({
            "p": p, "prior E[k]": g["prior_k_mean"], "E[k | y]": g["k_mean"],
            "P(k=1)": g["P_k1"], "P(k=2)": g["P_k2"], "P(k>=3)": g["P_k_ge3"],
            "P(cp 1886-1897)": frac(1886, 1897), "P(cp 1940-1955)": frac(1940, 1955),
            "P(cp 1925-1935)": frac(1925, 1935),
        })
    return pl.DataFrame(rows)


def coal_sensitivity() -> pl.DataFrame:
    t = pl.read_csv(RESULTS / "coal/year/cs_table.csv")
    variant = (pl.col("tag").str.replace(r"_L\d+$", ""))
    e1 = t.filter(pl.col("component") == 1).with_columns(variant.alias("variant"))
    e2 = (t.filter(pl.col("component") == 2).with_columns(variant.alias("variant"))
          .select(["variant", "L", pl.col("ser_log_bf").alias("effect 2 log BF")]))
    names = {"poisson": "default (raw basis, EB prior variance)",
             "poisson_std": "standardized step basis",
             "poisson_pv0.25": "fixed prior variance 0.25",
             "poisson_pv1.0": "fixed prior variance 1.0",
             "cf_cavi": "cf_cavi (exact CAVI, Gaussian q)",
             "irls": "irls (plug-in IRLS)"}
    out = (e1.join(e2, on=["variant", "L"], how="left")
           .with_columns(pl.col("variant").replace_strict(names, default=pl.col("variant")),
                         pl.format("{}-{}", pl.col("cs_start").cast(pl.Int64),
                                   pl.col("cs_end").cast(pl.Int64)).alias("CS hull"))
           .select([pl.col("variant"), "L", pl.col("ser_log_bf").alias("log BF"),
                    pl.col("cs_size").alias("CS size"), "CS hull",
                    pl.col("top").cast(pl.Int64).alias("top"), "effect 2 log BF"]))
    order = list(names.values())
    return out.with_columns(pl.col("variant").map_elements(order.index, return_dtype=pl.Int64)
                            .alias("_o")).sort(["_o", "L"]).drop("_o")


def coal_design_a() -> pl.DataFrame:
    a = pl.read_csv(RESULTS / "coal/year/calibration/design_a_summary.csv")
    return a.select([
        "L", pl.col("declared_cs").alias("declared CSs"),
        pl.col("cs_coverage").alias("CS coverage"), pl.col("false_cs").alias("false CSs"),
        pl.col("median_cs_size").alias("median CS size"),
        pl.col("mean_declared").alias("declared per rep"),
        pl.col("recall_1892").alias("recall 1892"), pl.col("recall_1948").alias("recall 1948"),
        pl.col("exact_k_mean").alias("exact E[k] (truth 2)")])


def coal_design_b() -> pl.DataFrame:
    b = pl.read_csv(RESULTS / "coal/year/calibration/design_b_summary.csv")
    return b.sort(["rate_ratio", "position"]).select([
        (1852 + pl.col("position")).alias("new regime from"),
        pl.col("dist_to_edge").alias("years to end"),
        pl.col("rate_ratio").alias("rate ratio"),
        pl.col("L1_power").alias("power"), pl.col("L1_coverage").alias("SuSiE coverage"),
        pl.col("exact_hpd_coverage").alias("exact coverage"),
        pl.col("L1_median_cs_size").alias("SuSiE set size"),
        pl.col("exact_median_hpd_size").alias("exact set size"),
        pl.col("median_exact_mass_in_L1_cs").alias("exact mass in SuSiE CS"),
        pl.col("L3_mean_false").alias("L=3 false per fit")])


# ---------------------------------------------------------------- lambda

def lambda_annotated() -> pl.DataFrame:
    t = pl.read_csv(RESULTS / "lambda/w100/declared_cs_annotated.csv")

    def kb(c):
        return (pl.col(c) / 1000).round(1).cast(pl.String)

    def two(c):
        return pl.col(c).map_elements(lambda v: f"{v:.2f}", return_dtype=pl.String)

    return t.select([
        pl.format("{}-{}", kb("cs_start_bp"), kb("cs_end_bp")).alias("CS (kb)"),
        pl.col("top_bp").alias("top (bp)"), pl.col("ser_log_bf").alias("log BF"),
        pl.col("cs_size").alias("bins"),
        pl.format("{} to {}", two("gc_before"), two("gc_after")).alias("GC"),
        pl.col("nearest_block_boundary_bp").alias("block boundary (bp)"),
        pl.col("dist_to_block_boundary_bp").alias("distance (bp)"),
        pl.col("block_boundary_in_cs").alias("in CS"),
        pl.col("exact_mass_in_hull").alias("exact mass"),
        pl.col("hmm_switch_mass_in_hull").alias("HMM mass")])


def lambda_variants() -> pl.DataFrame:
    """Declared CSs at 100 bp for every L and prior-variance treatment, one row per CS."""
    t = pl.read_csv(RESULTS / "lambda/w100/cs_table.csv").filter(pl.col("declared"))
    return (t.with_columns(pl.format("{}-{}", pl.col("cs_start").cast(pl.Int64),
                                     pl.col("cs_end").cast(pl.Int64)).alias("CS (bp)"))
            .sort(["tag", "top"])
            .select([pl.col("tag").alias("fit"), pl.col("top").cast(pl.Int64).alias("top"),
                     pl.col("ser_log_bf").alias("log BF"), pl.col("cs_size").alias("size"),
                     "CS (bp)"]))


def lambda_presence() -> pl.DataFrame:
    """Which top locations (to the nearest kb-ish cluster) each fit declares."""
    t = pl.read_csv(RESULTS / "lambda/w100/cs_table.csv").filter(pl.col("declared"))
    tops = sorted(t["top"].unique().to_list())
    clusters = []
    for x in tops:
        if clusters and x - clusters[-1][-1] <= 500:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    label = {x: f"{min(c) / 1000:.1f} kb" for c in clusters for x in c}
    rows = []
    for tag in ["binomial_L5", "binomial_L10", "binomial_L20", "binomial_pv0.05_L20",
                "binomial_pv0.25_L20"]:
        sub = t.filter(pl.col("tag") == tag)
        row = {"fit": tag}
        for c in clusters:
            hit = sub.filter(pl.col("top").is_in(c))
            row[label[c[0]]] = f"{hit['ser_log_bf'].max():.0f}" if hit.height else ""
        rows.append(row)
    return pl.DataFrame(rows)


def lambda_scale() -> pl.DataFrame:
    s = pl.read_csv(RESULTS / "lambda/scale_table.csv")
    wide = s.pivot(on="L", index="width", values="declared_cs")
    meta = s.filter(pl.col("L") == s["L"].max()).select([
        "width", "n_bins", pl.col("median_cs_width_bp").alias("median CS width (bp)"),
        pl.format("{} ({}-{})", pl.col("exact_k_mean").round(1), "exact_k_q05", "exact_k_q95")
        .alias("exact E[k] (90%)"),
        pl.col("hmm_expected_switches").alias("HMM switches")])
    return (meta.join(wide.rename({c: f"CSs L={c}" for c in wide.columns if c != "width"}),
                      on="width")
            .select(["width", "n_bins", "CSs L=5", "CSs L=10", "CSs L=20",
                     "median CS width (bp)", "exact E[k] (90%)", "HMM switches"])
            .rename({"width": "bin width", "n_bins": "bins"}))


def lambda_calibration() -> pl.DataFrame:
    c = pl.read_csv(RESULTS / "lambda/calibration/summary.csv")
    names = {"binomial": "binomial", "betabinom": "beta-binomial (phi 1.07)",
             "autocorr": "per-base Markov (rho 0.03)"}
    order = list(names)
    return (c.with_columns(pl.col("design").map_elements(order.index, return_dtype=pl.Int64)
                           .alias("_o"))
            .sort(["_o", "L"])
            .select([pl.col("design").replace_strict(names).alias("truth"), "L",
                     pl.col("mean_declared").alias("declared per fit"),
                     pl.col("mean_false").alias("false per fit"),
                     pl.col("cs_coverage").alias("CS coverage"),
                     pl.col("recall").alias("recall of 7"),
                     pl.col("median_cs_size").alias("median CS size"),
                     pl.col("median_phi_hat").alias("residual dispersion")]))


def lambda_dispersion() -> tuple[float, float]:
    c = pl.read_csv(RESULTS / "lambda/calibration/summary.csv")
    return float(c["phi_data"][0]), float(c["rho_data"][0])
