"""Genotype resource: one stdpopsim population per spec, cached as a tree sequence, and
per-replicate cis-window dosage matrices of a random sample of its individuals.

Mirrors the GWAS design note (notes/gwas_simulation_design.md): HomSap, OutOfAfrica_3G09,
CEU, a chr22 window on the HapMapII_GRCh38 map, so the design has a real allele-frequency
spectrum and LD decay. Replicates share the population and differ in which individuals are
sampled, which sites are causal and the expression draw. Allele frequencies come from the
trees (no genotype decode); the n x p dosage matrix is built by `simplify` on the sampled
individuals only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import polars as pl
import tskit

__all__ = ["PopulationSpec", "Dosage", "simulate_population", "write_population",
           "load_population", "site_table", "sample_dosage"]


@dataclass(frozen=True)
class PopulationSpec:
    species: str = "HomSap"
    model: str = "OutOfAfrica_3G09"
    population: str = "CEU"
    chrom: str = "chr22"
    left: int = 30_000_000
    right: int = 30_500_000
    genetic_map: str | None = "HapMapII_GRCh38"  # None = the species' flat rate (tests)
    n_diploids: int = 20_000
    seed: int = 1

    @property
    def hash(self) -> str:
        """Content address over every field (16 hex chars)."""
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def simulate_population(spec: PopulationSpec) -> tskit.TreeSequence:
    """Run stdpopsim's msprime engine for `spec`. Downloads the genetic map on first use
    (~/.cache/stdpopsim, or $STDPOPSIM_CACHE)."""
    import stdpopsim

    species = stdpopsim.get_species(spec.species)
    model = species.get_demographic_model(spec.model)
    # stdpopsim's contig rate (1.29e-8) differs from the model's (2.35e-8); use the model's
    # so diversity matches the demography (gotcha from the GWAS spike).
    contig = species.get_contig(
        spec.chrom,
        genetic_map=spec.genetic_map,
        mutation_rate=model.mutation_rate,
        left=spec.left,
        right=spec.right,
    )
    engine = stdpopsim.get_engine("msprime")
    return engine.simulate(model, contig, samples={spec.population: spec.n_diploids}, seed=spec.seed)


def site_table(ts: tskit.TreeSequence) -> pl.DataFrame:
    """One row per site: id, position, derived-allele frequency in the population and
    whether it is biallelic. Frequency = tree.num_samples(mutation node), no decode."""
    ids, pos, freq, biallelic = [], [], [], []
    n = ts.num_samples
    multi: list[int] = []
    for tree in ts.trees():
        for site in tree.sites():
            ids.append(site.id)
            pos.append(site.position)
            one = len(site.mutations) == 1
            biallelic.append(one)
            # one mutation: derived count is its subtree size. Several (nested or back
            # mutations): subtree sizes double count, so decode those few sites below.
            freq.append(tree.num_samples(site.mutations[0].node) / n if one else float("nan"))
            if not one:
                multi.append(len(ids) - 1)
    if multi:
        var = tskit.Variant(ts)
        for i in multi:
            var.decode(ids[i])
            freq[i] = float(np.mean(var.genotypes > 0))
    return pl.DataFrame({
        "site_id": np.asarray(ids, dtype=np.int64),
        "position": np.asarray(pos, dtype=np.float64),
        "derived_freq": np.asarray(freq, dtype=np.float64),
        "biallelic": np.asarray(biallelic, dtype=bool),
    })


def write_population(spec: PopulationSpec, trees_path: str, sites_path: str) -> None:
    ts = simulate_population(spec)
    ts.dump(trees_path)
    site_table(ts).write_parquet(sites_path)


def load_population(trees_path: str) -> tskit.TreeSequence:
    return tskit.load(trees_path)


@dataclass(frozen=True)
class Dosage:
    """Sampled design: X (n, p) in {0, 1, 2} as uint8, population site ids, positions and
    the in-sample minor allele frequency per column."""

    X: np.ndarray
    site_id: np.ndarray
    position: np.ndarray
    maf: np.ndarray
    individuals: np.ndarray  # population individual ids, length n

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def p(self) -> int:
        return self.X.shape[1]


def sample_dosage(ts: tskit.TreeSequence, n: int, rng: np.random.Generator, *, maf_min: float) -> Dosage:
    """Draw `n` diploid individuals uniformly, simplify to them, and return the dosage
    matrix over biallelic sites with in-sample MAF >= `maf_min`."""
    n_ind = ts.num_individuals
    if n > n_ind:
        raise ValueError(f"asked for n={n} individuals from a population of {n_ind}")
    chosen = np.sort(rng.choice(n_ind, size=n, replace=False))
    nodes_table = ts.tables.nodes
    nodes = np.concatenate([np.flatnonzero(nodes_table.individual == i) for i in chosen])
    if nodes.shape[0] != 2 * n:
        raise ValueError("expected exactly two sample nodes per individual")
    sub = ts.simplify(nodes.tolist(), filter_sites=True, keep_unary=False)
    # simplify renumbers samples in the order given, so individual k owns columns 2k, 2k+1
    G = sub.genotype_matrix()  # (sites, 2n) haploid
    dosage = (G[:, 0::2] + G[:, 1::2]).T.astype(np.int16)  # (n, p)
    biallelic = G.max(axis=1) <= 1
    position = sub.sites_position
    # map simplified site ids back to population ids by position (positions are unique)
    pop_pos = ts.sites_position
    site_id = np.searchsorted(pop_pos, position)
    assert np.allclose(pop_pos[site_id], position)
    freq = dosage.mean(axis=0) / 2.0
    maf = np.minimum(freq, 1.0 - freq)
    keep = biallelic & (maf >= maf_min)
    return Dosage(
        X=dosage[:, keep].astype(np.uint8),
        site_id=site_id[keep].astype(np.int64),
        position=position[keep].astype(np.float64),
        maf=maf[keep].astype(np.float64),
        individuals=chosen.astype(np.int64),
    )
