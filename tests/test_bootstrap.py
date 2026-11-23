"""Bootstrap correctness: Rust draws vs analytic robust standard errors, and determinism."""

import numpy as np
import pytest
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

import causal_rust_core as crc


def test_did_bootstrap_se_matches_robust_se(synthetic_data):
    df = synthetic_data
    draws = crc.did_bootstrap(df, "outcome_did", "group", "time", n_boot=2000, seed=1)
    g, t, y = (df[c].to_numpy() for c in ["group", "time", "outcome_did"])
    fit = sm.OLS(y, np.column_stack([np.ones_like(g), g, t, g * t])).fit(cov_type="HC1")
    assert draws.mean() == pytest.approx(fit.params[3], abs=0.05)
    assert draws.std(ddof=1) == pytest.approx(fit.bse[3], rel=0.1)


def test_iv_bootstrap_se_matches_robust_se(synthetic_data):
    df = synthetic_data
    draws = crc.iv_bootstrap(df, "outcome", "treatment", "instrument", n_boot=2000, seed=2)
    y, t, z = (df[c].to_numpy() for c in ["outcome", "treatment", "instrument"])
    fit = IV2SLS(y, np.ones_like(y), t, z).fit(cov_type="robust")
    assert np.median(draws) == pytest.approx(fit.params.iloc[1], abs=0.1)
    assert draws.std(ddof=1) == pytest.approx(fit.std_errors.iloc[1], rel=0.1)


def test_bootstrap_is_deterministic_and_thread_independent(synthetic_data):
    a = crc.ols_bootstrap(synthetic_data, "outcome", ["treatment", "w0"], n_boot=200, seed=7, n_threads=1)
    b = crc.ols_bootstrap(synthetic_data, "outcome", ["treatment", "w0"], n_boot=200, seed=7, n_threads=8)
    assert a.shape == (200, 3)
    np.testing.assert_array_equal(a, b)
