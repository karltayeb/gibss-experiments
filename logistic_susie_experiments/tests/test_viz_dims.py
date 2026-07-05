from viz_dims import method_dims, sim_dims


def test_method_dims_irls_converged_fixed_profiled():
    coord = {"function": "run_irls_method",
             "kwargs": {"ser_cadence": "block", "n_outer": 50, "L": 1,
                        "profile": True, "estimate_prior_variance": False,
                        "prior_variance": 100.0}}
    d = method_dims(coord)
    assert d["family"] == "irls"
    assert d["step"] == "converged"
    assert d["prior"] == "fixed"
    assert d["profile"] is True
    assert d["cadence"] == "block"
    assert d["L"] == 1
    assert d["function"] == "run_irls_method"
    assert d["m_n_outer"] == 50
    assert d["m_prior_variance"] == 100.0


def test_method_dims_one_step_eb_globaljj():
    coord = {"function": "run_globaljj_method",
             "kwargs": {"ser_cadence": "block", "n_outer": 1, "L": 1}}
    d = method_dims(coord)
    assert d["family"] == "globaljj"
    assert d["step"] == "one_step"
    assert d["prior"] == "eb"      # estimate_prior_variance absent -> EB
    assert d["profile"] is False   # absent -> False


def test_method_dims_logistic_impl_family():
    coord = {"function": "run_logistic_method", "kwargs": {"impl": "globaljj", "L": 1}}
    assert method_dims(coord)["family"] == "globaljj"


def test_sim_dims_signal_and_raw():
    coord = {"design": {"function": "gaussian_markov_X",
                        "arguments": {"n": 500, "p": 100, "rho": 0.9}},
             "enrichment": {"function": "uniform_single_effect",
                            "arguments": {"causal_effect": 1.0}, "intercept": -2.0},
             "base_seed": 1}
    d = sim_dims(coord)
    assert d["design"] == "gaussian"
    assert d["intercept"] == -2.0
    assert d["b"] == 1.0
    assert d["signal"] is True
    assert d["d_rho"] == 0.9
    assert d["e_causal_effect"] == 1.0


def test_sim_dims_null_is_not_signal():
    coord = {"design": {"function": "c4_gene_sets_X", "arguments": {}},
             "enrichment": {"function": "uniform_single_effect",
                            "arguments": {"causal_effect": 0.0}, "intercept": -2.0},
             "base_seed": 1}
    d = sim_dims(coord)
    assert d["design"] == "c4"
    assert d["signal"] is False
    assert d["b"] == 0.0


def test_sim_dims_rho_binary_logbf():
    coord = {
        "design": {"function": "binary_markov_X",
                   "arguments": {"n": 1000, "p": 500, "rho": 0.9, "freq": 0.1}},
        "enrichment": {"arguments": {"causal_effect": 0.968}, "intercept": -2.0},
    }
    d = sim_dims(coord)
    assert d["design"] == "binary_q0.1"
    assert d["rho"] == 0.9
    assert d["logbf"] == 32          # frozen b for binary q=0.1, b0=-2, L32 (centered)
    assert d["signal"] is True


def test_sim_dims_null_logbf_zero():
    coord = {
        "design": {"function": "gaussian_markov_X",
                   "arguments": {"n": 1000, "p": 500, "rho": 0.0}},
        "enrichment": {"arguments": {"causal_effect": 0.0}, "intercept": 0.0},
    }
    assert sim_dims(coord)["logbf"] == 0


def test_method_dims_exposes_declared_name_no_inference():
    # family stays the function-derived value (no special-casing); the declared
    # method name (config key, sans over-suffix) is its own dim for plot color.
    coord = {"name": "null_score__center=true", "function": "run_irls_method",
             "kwargs": {"ser_cadence": "block", "n_outer": 1, "L": 1, "center": True}}
    d = method_dims(coord)
    assert d["family"] == "irls"          # no null_score inference
    assert d["step"] == "one_step"
    assert d["method"] == "null_score"    # alias comes from the config name


def test_sim_dims_logbf_from_target_arg():
    coord = {"design": {"function": "gaussian_markov_X", "arguments": {"n": 1000, "p": 500, "rho": 0.9}},
             "enrichment": {"function": "logbf_single_effect",
                            "arguments": {"design": "gaussian", "b0": -2.0, "target_logbf": 16},
                            "intercept": -2.0}}
    d = sim_dims(coord)
    assert d["logbf"] == 16 and d["signal"] is True and d["rho"] == 0.9
