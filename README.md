# Causal Rust Core: Causal Inference in Rust + Python

A learning project that implements core causal estimators (OLS, IV/2SLS, 2×2 difference-in-differences) in Rust, exposes them to Python with PyO3, and checks them against standard Python estimators on the NSW job-training experiment.

- **Rust estimators** (~160 lines, `linfa-linear`): OLS, IV/2SLS (two-stage OLS) and DiD (OLS with a group × time interaction). They return point estimates only, with no standard errors.
- **Python layer** (`causal_lib/`): difference in means, OLS, inverse probability weighting, an EconML causal forest, and a Baron–Kenny mediation helper.
- **Benchmark** (`benchmarks/`): Rust against numpy, statsmodels and linearmodels from 10k to 10M rows.

## Results: NSW job-training experiment

Dehejia–Wahba NSW experimental sample: 722 men (297 treated), outcome 1978 earnings. Reproduce with `python scripts/reproduce_nsw.py`, which downloads the data from NBER and writes `results/nsw_results.json`.

| Estimator | Effect on 1978 earnings |
| --- | --- |
| Difference in means | $886 (bootstrap 95% CI −$79 to $1,932) |
| OLS with covariates (scikit-learn) | $807 |
| OLS with covariates (Rust) | $807 (identical to 10 significant figures) |
| Inverse probability weighting | $799 |
| Causal forest (EconML `CausalForestDML`) | $811 |

- Because treatment was randomised, the difference in means is unbiased; adjusting for covariates moves the estimate by under 10%, which is what a working randomisation should look like.
- The interval is wide: with 722 people the data are consistent with no effect at all. The point estimates agree; the precision does not support a strong claim.
- **Placebo check**: the mediation helper with 1975 earnings as the "mediator" gives an indirect effect of −$3. Since 1975 earnings were recorded before the programme, the true indirect effect is zero by design; the near-zero result is a sanity check, not a finding about mechanisms.

## Benchmark: Rust against numpy, statsmodels and linearmodels

Synthetic data with a known effect; every implementation returns the same estimate (largest disagreement 1e-11). Median wall time, `python benchmarks/bench_estimators.py`:

| Estimator | n | Rust (public API) | Rust (core only) | numpy `lstsq` | statsmodels / linearmodels |
| --- | --- | --- | --- | --- | --- |
| OLS | 100k | 8.8 ms | 3.9 ms | 1.3 ms | 6.1 ms |
| OLS | 10M | 1,340 ms | 647 ms | 314 ms | 994 ms |
| IV/2SLS | 100k | 7.1 ms | 2.9 ms | 2.1 ms | 110 ms |
| IV/2SLS | 10M | 1,370 ms | 702 ms | 530 ms | 11,730 ms |
| DiD | 100k | 9.5 ms | 3.3 ms | 1.8 ms | 6.5 ms |
| DiD | 10M | 1,496 ms | 913 ms | 461 ms | 1,190 ms |

Full table at 10k, 100k, 1M and 10M rows: [`benchmarks/results/results.md`](benchmarks/results/results.md).

- **The Rust path is slower than numpy**: 2.2–6.7× through the public API and 1.3–2.9× even when the Polars-to-list conversion is excluded (one exception: IV at 1M rows, 0.8×).
- The cost is data movement, not arithmetic: the wrapper converts the DataFrame to Python lists and Rust copies each list into a `Vec<f64>`. Reading numpy buffers zero-copy (the `numpy` crate) is the obvious next step.
- statsmodels and linearmodels do more work (standard errors, diagnostics), so their times are not like-for-like with the Rust point estimates. linearmodels' IV is 9–18× slower than Rust here for that reason.

Hardware and versions: Intel Core Ultra 9 285H (10 CPUs available), Fedora Linux, Python 3.13.12, rustc 1.91.1, numpy 2.5.5 (OpenBLAS), polars 2.1.0, statsmodels 0.15.0, linearmodels 7.1, Rust extension built with `maturin develop --release`.

## Usage

```python
import polars as pl
import causal_rust_core as crc

df = pl.read_csv("data.csv")
crc.ols_fit(df, "outcome", ["treatment", "age"])   # coefficient on the first feature
crc.iv_fit(df, "outcome", "treatment", "instrument")
crc.did_fit(df, "outcome", "group", "time")          # coefficient on group × time
```

All columns must be numeric (Float64). Each function returns a float; no standard errors.

## Setup

Requires Python 3.12+ and a Rust toolchain.

```bash
uv sync                                  # installs dependencies and builds the extension
uv run maturin develop --release         # optimised build (use before benchmarking)
uv run pytest                            # estimator recovery tests on synthetic data
uv run python scripts/reproduce_nsw.py   # NSW results
uv run python benchmarks/bench_estimators.py --sizes 10000 100000 1000000 10000000
uv run marimo edit notebook.py           # IHDP notebook (needs ihdp.csv with treatment, y_factual, x1..x25)
```

## Project structure

```text
src/                       Rust: OLS (foundational_methods.rs), IV (structural_methods.rs), DiD (longitudinal_methods.rs)
python/causal_rust_core/   Python wrapper: Polars DataFrame -> Rust
causal_lib/                Python estimators, data loaders (NSW, IHDP), plotting
scripts/reproduce_nsw.py   Regenerates the NSW results table
benchmarks/                Benchmark script and results
tests/                     Recovery tests (OLS, IV, DiD) on synthetic data with known effects
notebook.py                Marimo notebook for IHDP
```

## Limitations

- Point estimates only: no standard errors, clustering or weak-instrument diagnostics in the Rust layer.
- IV supports one instrument and no exogenous controls; DiD is the 2×2 case only.
- The causal forest runs entirely in EconML; nothing about it is accelerated by the Rust layer.

## License

MIT

## Acknowledgments

- Dehejia & Wahba NSW data, hosted by NBER
- [EconML](https://github.com/microsoft/EconML), [PyO3](https://pyo3.rs/), [linfa](https://github.com/rust-ml/linfa), [Polars](https://pola.rs/), [Marimo](https://marimo.io/)
