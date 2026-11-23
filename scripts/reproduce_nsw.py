"""Reproduce every NSW number in the README from the data.

    python scripts/reproduce_nsw.py            # writes results/nsw_results.json and prints a table

The NSW experimental sample is a randomised trial, so the difference in means is unbiased; the
other estimators are checks that adjustment does not move the answer much.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # causal_lib lives at the project root

import numpy as np
import polars as pl

import causal_rust_core as crc
from causal_lib import causal, data_loader, foundational

COVARIATES = ["age", "education", "black", "hispanic", "married", "nodegree", "re75"]
SEED = 42


def bootstrap_ci(df: pl.DataFrame, fn, n_boot: int = 500) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    stats = [fn(df[rng.integers(0, df.height, df.height)]) for _ in range(n_boot)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main() -> None:
    df = data_loader.load_nsw()
    neyman = lambda d: float(foundational.estimate_ate_neyman(d, "treat", "re78"))
    acme, ade, total = causal.estimate_mediation(df, "treat", "re75", "re78", ["age", "education"])
    results = {
        "n": df.height,
        "n_treated": int(df["treat"].sum()),
        "neyman_diff_in_means": neyman(df),
        "neyman_95ci_bootstrap": bootstrap_ci(df, neyman),
        "ols_sklearn": float(foundational.estimate_ate_mle(df, "treat", "re78", COVARIATES)),
        "ols_rust": float(crc.ols_fit(df.with_columns(pl.all().cast(pl.Float64)), "re78", ["treat", *COVARIATES])),
        "ipw": float(foundational.estimate_ate_rubin(df, "treat", "re78", COVARIATES)),
        "causal_forest_ate": float(causal.estimate_forest(df, "re78", "treat", COVARIATES)),
        "placebo_mediation_via_re75": {"acme": float(acme), "ade": float(ade), "total": float(total)},
    }
    out = Path("results/nsw_results.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
