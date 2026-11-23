# Causal Rust Core: Causal Inference in Rust + Python

A learning project that implements core causal estimators (OLS, IV/2SLS, 2×2 difference-in-differences) in Rust, exposes them to Python with PyO3, and checks them against standard Python estimators on the NSW job-training experiment.

- **Rust estimators** (~160 lines, `linfa-linear`): OLS, IV/2SLS (two-stage OLS) and DiD (OLS with a group × time interaction). They return point estimates only, with no standard errors.
- **Python layer** (`causal_lib/`): difference in means, OLS, inverse probability weighting, an EconML causal forest, and a Baron–Kenny mediation helper.
- **Bootstrap in Rust** (`src/bootstrap.rs`): pairs-bootstrap standard errors for OLS/DiD and IV, on zero-copy numpy input, parallel across replicates.
- **Benchmarks** (`benchmarks/`): single fits against numpy, statsmodels and linearmodels from 10k to 10M rows; bootstrap against a numpy loop and a vectorised numpy implementation.

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
| OLS | 100k | 8.0 ms | 3.7 ms | 1.3 ms | 6.1 ms |
| OLS | 10M | 1,290 ms | 593 ms | 313 ms | 986 ms |
| IV/2SLS | 100k | 6.7 ms | 2.7 ms | 2.1 ms | 116 ms |
| IV/2SLS | 10M | 1,309 ms | 677 ms | 514 ms | 11,997 ms |
| DiD | 100k | 6.6 ms | 3.1 ms | 1.8 ms | 6.6 ms |
| DiD | 10M | 1,488 ms | 800 ms | 445 ms | 1,179 ms |

Full table at 10k, 100k, 1M and 10M rows: [`benchmarks/results/results.md`](benchmarks/results/results.md).

- **The Rust path is slower than numpy**: 2.0–6.0× through the public API and 1.2–2.8× even when the Polars-to-list conversion is excluded (one exception: IV at 1M rows, 0.8×).
- The cost is data movement, not arithmetic: the wrapper converts the DataFrame to Python lists and Rust copies each list into a `Vec<f64>`. For a single least-squares solve numpy is the right tool: it is one call into optimised BLAS/LAPACK.
- statsmodels and linearmodels do more work (standard errors, diagnostics), so their times are not like-for-like with the Rust point estimates. linearmodels' IV is 9–18× slower than Rust here for that reason.

Hardware and versions: Intel Core Ultra 9 285H (10 CPUs available), Fedora Linux, Python 3.14.10, rustc 1.91.1, numpy 2.5.5 (OpenBLAS), polars 2.1.0, statsmodels 0.15.0, linearmodels 7.1, Rust extension built with `maturin develop --release`.

## Bootstrap: where the Rust layer does pay off

A bootstrap refits the estimator on B resamples, which numpy cannot do in one call. The Rust kernels read numpy arrays without copying, accumulate the sufficient statistics (X'X and X'y, or five sums for IV) directly from drawn row indices instead of building resampled arrays, and run replicates in parallel with Rayon. They are compared with a plain numpy loop and with the strongest numpy version found: per-replicate resampling counts, then every replicate's X'WX in one matrix product per chunk and a batched solve. B = 1,000, median wall time (`python benchmarks/bench_bootstrap.py`):

| Estimator | n | numpy loop | numpy vectorised | Rust, 10 threads | Rust, 1 thread |
| --- | --- | --- | --- | --- | --- |
| DiD (4 coefficients) | 10k | 0.27 s | 0.098 s | 0.031 s | 0.098 s |
| DiD (4 coefficients) | 100k | 2.98 s | 0.89 s | 0.21 s | 1.14 s |
| DiD (4 coefficients) | 1M | 101.6 s | 8.84 s | 7.53 s | 60.6 s |
| IV slope | 10k | 0.049 s | 0.093 s | 0.016 s | 0.027 s |
| IV slope | 100k | 0.93 s | 0.88 s | 0.044 s | 0.16 s |
| IV slope | 1M | 17.2 s | 8.20 s | 1.49 s | 7.14 s |

Full results, including each implementation's bootstrap SE: [`benchmarks/results/bootstrap_results.md`](benchmarks/results/bootstrap_results.md).

- **Against the best numpy version, Rust is 3–20× faster** at 10k and 100k rows, and 5.5× faster for IV at 1M.
- **DiD at 1M rows is the exception (1.17×).** Each replicate gathers rows at random from a 32 MB array, which is limited by memory latency (one thread takes 61 s); numpy's vectorised version streams through memory in one dense matrix product. The Rust advantage there comes from parallelism, not from the kernel.
- **Correctness**: every implementation's bootstrap SE is within 7% of the analytic robust SE (statsmodels HC1 for DiD, linearmodels robust for IV). Over 40 seeds the Rust and numpy SE ratios both average 1.00 (sd 0.021 and 0.026), and successive Rust replicates are uncorrelated (lag-1 autocorrelation 0.009). Draws are identical for a given seed whatever the thread count.
- Hardware and versions as above.

## Usage

```python
import polars as pl
import causal_rust_core as crc

df = pl.read_csv("data.csv")
crc.ols_fit(df, "outcome", ["treatment", "age"])   # coefficient on the first feature
crc.iv_fit(df, "outcome", "treatment", "instrument")
crc.did_fit(df, "outcome", "group", "time")          # coefficient on group × time

# Pairs bootstrap (numpy arrays of draws); SE = draws.std(ddof=1)
crc.ols_bootstrap(df, "outcome", ["treatment", "age"], n_boot=1000, seed=0)  # (1000, 3), intercept first
crc.iv_bootstrap(df, "outcome", "treatment", "instrument", n_boot=1000)
crc.did_bootstrap(df, "outcome", "group", "time", n_boot=1000)
```

All columns must be numeric. The single-fit functions return a float; use the bootstrap functions for standard errors.

## Setup

Requires Python 3.12+ (the project defaults to 3.14) and a Rust toolchain.

```bash
uv sync                                  # installs dependencies and builds the extension
uv run maturin develop --release         # optimised build (use before benchmarking)
uv run pytest                            # estimator recovery tests on synthetic data
uv run python scripts/reproduce_nsw.py   # NSW results
uv run python benchmarks/bench_estimators.py --sizes 10000 100000 1000000 10000000
uv run python benchmarks/bench_bootstrap.py --sizes 10000 100000 1000000 --n-boot 1000
uv run marimo edit notebook.py           # IHDP notebook (needs ihdp.csv with treatment, y_factual, x1..x25)
```

## Project structure

```text
src/                       Rust: OLS (foundational_methods.rs), IV (structural_methods.rs), DiD (longitudinal_methods.rs), bootstrap (bootstrap.rs)
python/causal_rust_core/   Python wrapper: Polars DataFrame -> Rust
causal_lib/                Python estimators, data loaders (NSW, IHDP), plotting
scripts/reproduce_nsw.py   Regenerates the NSW results table
benchmarks/                Benchmark script and results
tests/                     Recovery tests (OLS, IV, DiD) on synthetic data with known effects
notebook.py                Marimo notebook for IHDP
```

## Limitations

- Standard errors come from the pairs bootstrap only: no analytic, clustered or wild-cluster variants, and no weak-instrument diagnostics.
- The bootstrap solves normal equations by Cholesky: fine for intercepts, dummies and standardised covariates, less robust than QR for ill-conditioned designs.
- IV supports one instrument and no exogenous controls; DiD is the 2×2 case only.
- The causal forest runs entirely in EconML; nothing about it is accelerated by the Rust layer.

## License

MIT

## Acknowledgments

- Dehejia & Wahba NSW data, hosted by NBER
- [EconML](https://github.com/microsoft/EconML), [PyO3](https://pyo3.rs/), [linfa](https://github.com/rust-ml/linfa), [Polars](https://pola.rs/), [Marimo](https://marimo.io/)
