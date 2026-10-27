import numpy as np

from eqtl.genotypes import PopulationSpec, sample_dosage, site_table


def test_population_hash_is_content_addressed():
    a, b = PopulationSpec(seed=1), PopulationSpec(seed=2)
    assert a.hash != b.hash and a.hash == PopulationSpec(seed=1).hash and len(a.hash) == 16


def test_site_table_frequencies_match_decoded_genotypes(small_population):
    st = site_table(small_population)
    G = small_population.genotype_matrix()
    decoded = (G > 0).mean(axis=1)
    assert np.allclose(st["derived_freq"].to_numpy(), decoded)


def test_sample_dosage_filters_and_shapes(small_population):
    d = sample_dosage(small_population, 120, np.random.default_rng(0), maf_min=0.05)
    assert d.X.shape[0] == 120 and d.X.dtype == np.uint8
    assert set(np.unique(d.X)) <= {0, 1, 2}
    assert d.maf.min() >= 0.05
    freq = d.X.mean(axis=0) / 2
    assert np.allclose(np.minimum(freq, 1 - freq), d.maf)
    assert np.all(np.diff(d.position) > 0)
    assert np.allclose(small_population.sites_position[d.site_id], d.position)


def test_sample_dosage_is_reproducible(small_population):
    a = sample_dosage(small_population, 50, np.random.default_rng(5), maf_min=0.05)
    b = sample_dosage(small_population, 50, np.random.default_rng(5), maf_min=0.05)
    assert np.array_equal(a.X, b.X) and np.array_equal(a.individuals, b.individuals)
