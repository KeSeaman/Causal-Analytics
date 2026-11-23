"""Like-for-like benchmark: Rust estimators (via PyO3) against numpy, statsmodels and linearmodels.

Each estimator is timed on the same synthetic data at several sizes. Four implementations:

- ``rust_wrapper``: ``causal_rust_core.ols_fit(df, ...)``, the public API. Includes converting the
  Polars DataFrame to Python lists (``df.to_dict(as_series=False)``) and copying them into Rust.
- ``rust_core``: the compiled function called with a prebuilt dict of lists, so the Polars -> list
  conversion is excluded. Rust still extracts the Python lists into ``Vec<f64>``.
- ``numpy``: point estimate only via ``numpy.linalg.lstsq`` (the fair comparison: Rust returns
  point estimates only).
- ``statsmodels`` / ``linearmodels``: full fit including standard errors (what an analyst would
  actually call; more work than the Rust functions do).

Usage::

    python benchmarks/bench_estimators.py --sizes 10000 100000 1000000 10000000 --out benchmarks/results
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

import causal_rust_core as crc
from causal_rust_core import _causal_rust_core as core

TRUE_ATE, TRUE_DID = 2.0, 3.0


@dataclass(frozen=True)
class Row:
    estimator: str
    implementation: str
    n: int
    median_s: float
    min_s: float
    repeats: int
    estimate: float


def make_data(n: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Synthetic data with a confounder, a valid instrument and a 2x2 DiD design."""
    rng = np.random.default_rng(seed)
    w0 = rng.normal(size=n)
    z = rng.binomial(1, 0.5, size=n).astype(float)
    treatment = ((0.8 * z + 0.3 * w0 + rng.normal(0, 0.5, size=n)) > 0).astype(float)
    group = rng.binomial(1, 0.5, size=n).astype(float)
    time_ = rng.binomial(1, 0.5, size=n).astype(float)
    return {
        "outcome": TRUE_ATE * treatment + 0.5 * w0 + rng.normal(size=n),
        "treatment": treatment,
        "instrument": z,
        "w0": w0,
        "group": group,
        "time": time_,
        "outcome_did": TRUE_DID * group * time_
        + time_
        + 0.5 * group
        + rng.normal(size=n),
    }


def with_const(*cols: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones_like(cols[0]), *cols])


def lstsq(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(x, y, rcond=None)[0]


def numpy_2sls(y: np.ndarray, t: np.ndarray, z: np.ndarray) -> float:
    zc = with_const(z)
    t_hat = zc @ lstsq(t, zc)
    return float(lstsq(y, with_const(t_hat))[1])


def cases(d: dict[str, np.ndarray]) -> dict[str, dict[str, Callable[[], float]]]:
    """Every implementation of every estimator, as zero-argument callables."""
    cols = {
        "ols": ["outcome", "treatment", "w0"],
        "iv": ["outcome", "treatment", "instrument"],
        "did": ["outcome_did", "group", "time"],
    }
    dfs = {k: pl.DataFrame({c: d[c] for c in v}) for k, v in cols.items()}
    dicts = {k: df.to_dict(as_series=False) for k, df in dfs.items()}
    y, t, w0, z = d["outcome"], d["treatment"], d["w0"], d["instrument"]
    yd, g, tm = d["outcome_did"], d["group"], d["time"]
    x_ols, x_did = with_const(t, w0), with_const(g, tm, g * tm)
    return {
        "OLS": {
            "rust_wrapper": lambda: crc.ols_fit(
                dfs["ols"], "outcome", ["treatment", "w0"]
            ),
            "rust_core": lambda: core.ols_fit(
                dicts["ols"], "outcome", ["treatment", "w0"]
            ),
            "numpy": lambda: float(lstsq(y, x_ols)[1]),
            "statsmodels": lambda: float(sm.OLS(y, x_ols).fit().params[1]),
        },
        "IV/2SLS": {
            "rust_wrapper": lambda: crc.iv_fit(
                dfs["iv"], "outcome", "treatment", "instrument"
            ),
            "rust_core": lambda: core.iv_fit(
                dicts["iv"], "outcome", "treatment", "instrument"
            ),
            "numpy": lambda: numpy_2sls(y, t, z),
            "linearmodels": lambda: float(
                IV2SLS(y, np.ones_like(y), t, z)
                .fit(cov_type="unadjusted")
                .params.iloc[1]
            ),
        },
        "DiD": {
            "rust_wrapper": lambda: crc.did_fit(
                dfs["did"], "outcome_did", "group", "time"
            ),
            "rust_core": lambda: core.did_fit(
                dicts["did"], "outcome_did", "group", "time"
            ),
            "numpy": lambda: float(lstsq(yd, x_did)[3]),
            "statsmodels": lambda: float(sm.OLS(yd, x_did).fit().params[3]),
        },
    }


def time_call(
    fn: Callable[[], float], budget_s: float, max_repeats: int
) -> tuple[list[float], float]:
    estimate = fn()  # warm-up, also the reported estimate
    times: list[float] = []
    while len(times) < max_repeats and (len(times) < 3 or sum(times) < budget_s):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return times, estimate


def run(sizes: list[int], budget_s: float, max_repeats: int) -> list[Row]:
    rows: list[Row] = []
    for n in sizes:
        for est, impls in cases(make_data(n)).items():
            for impl, fn in impls.items():
                times, estimate = time_call(fn, budget_s, max_repeats)
                rows.append(
                    Row(
                        est,
                        impl,
                        n,
                        statistics.median(times),
                        min(times),
                        len(times),
                        estimate,
                    )
                )
                print(
                    f"{est:8s} {impl:13s} n={n:>10,d} median={rows[-1].median_s:.4f}s est={estimate:.6f}"
                )
    return rows


def environment() -> dict[str, object]:
    cpu = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        platform.processor(),
    )
    rustc = subprocess.run(
        ["rustc", "--version"], capture_output=True, text=True
    ).stdout.strip()
    return {
        "cpu": cpu,
        "cpus_available": len(os.sched_getaffinity(0)),
        "os": platform.platform(),
        "python": platform.python_version(),
        "rustc": rustc,
        "packages": {
            p: version(p)
            for p in [
                "numpy",
                "polars",
                "statsmodels",
                "linearmodels",
                "causal_rust_core",
            ]
        },
        "rust_build": "maturin develop --release",
    }


def to_markdown(rows: list[Row]) -> str:
    sizes = sorted({r.n for r in rows})
    lines = []
    for est in dict.fromkeys(r.estimator for r in rows):
        lines += [
            f"### {est}",
            "",
            "| Implementation | " + " | ".join(f"n={n:,}" for n in sizes) + " |",
            "| --- |" + " --- |" * len(sizes),
        ]
        for impl in dict.fromkeys(r.implementation for r in rows if r.estimator == est):
            cells = {
                r.n: r.median_s
                for r in rows
                if r.estimator == est and r.implementation == impl
            }
            lines.append(
                f"| {impl} | "
                + " | ".join(f"{cells[n] * 1e3:,.2f} ms" for n in sizes)
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sizes", type=int, nargs="+", default=[10_000, 100_000, 1_000_000, 10_000_000]
    )
    ap.add_argument(
        "--budget",
        type=float,
        default=2.0,
        help="seconds of timing per case before stopping",
    )
    ap.add_argument("--max-repeats", type=int, default=50)
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    args = ap.parse_args()

    rows = run(args.sizes, args.budget, args.max_repeats)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps(
            {"environment": environment(), "rows": [asdict(r) for r in rows]}, indent=2
        )
    )
    (args.out / "results.md").write_text(to_markdown(rows))
    print(to_markdown(rows))


if __name__ == "__main__":
    main()
