"""Bootstrap benchmark: where a Rust kernel can beat numpy.

Pairs bootstrap (B resamples, refit each) of the 2x2 DiD regression (4 coefficients) and the
just-identified IV slope. Implementations:

- ``numpy_loop``: resample with fancy indexing, refit (lstsq / closed form). How most people write it.
- ``numpy_vectorised``: per-replicate resampling counts (``bincount``), then all replicates' X'WX
  and X'Wy as one matrix product per chunk, and a batched solve. The strongest numpy version.
- ``rust`` / ``rust_1_thread``: causal_rust_core's kernels (zero-copy input, sufficient statistics
  accumulated from drawn indices, Rayon across replicates; or one thread to separate the
  algorithmic gain from parallelism).

Every implementation's bootstrap SE is reported next to the analytic robust SE (statsmodels HC1,
linearmodels robust), so speed is never shown without correctness.

    python benchmarks/bench_bootstrap.py --sizes 10000 100000 1000000 --n-boot 1000 --out benchmarks/results
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
from importlib.metadata import version
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

from causal_rust_core import _causal_rust_core as core

CHUNK_BYTES = 256 * 2**20  # memory cap for the vectorised count matrix


def make_data(n: int, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    w0 = rng.normal(size=n)
    z = rng.binomial(1, 0.5, size=n).astype(float)
    t = ((0.8 * z + 0.3 * w0 + rng.normal(0, 0.5, size=n)) > 0).astype(float)
    g = rng.binomial(1, 0.5, size=n).astype(float)
    tm = rng.binomial(1, 0.5, size=n).astype(float)
    return {
        "y": 2.0 * t + 0.5 * w0 + rng.normal(size=n),
        "t": t,
        "z": z,
        "x_did": np.ascontiguousarray(np.column_stack([np.ones(n), g, tm, g * tm])),
        "y_did": 3.0 * g * tm
        + tm
        + 0.5 * g
        + rng.normal(size=n) * (1 + g),  # heteroskedastic
    }


def iv_slope(y: np.ndarray, t: np.ndarray, z: np.ndarray) -> float:
    n = len(y)
    return (n * (z @ y) - z.sum() * y.sum()) / (n * (z @ t) - z.sum() * t.sum())


# --- implementations ---------------------------------------------------------------------------


def did_numpy_loop(d, n_boot, seed):
    rng, x, y = np.random.default_rng(seed), d["x_did"], d["y_did"]
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        out[b] = np.linalg.lstsq(x[idx], y[idx], rcond=None)[0][3]
    return out


def iv_numpy_loop(d, n_boot, seed):
    rng, y, t, z = np.random.default_rng(seed), d["y"], d["t"], d["z"]
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        out[b] = iv_slope(y[idx], t[idx], z[idx])
    return out


def _count_chunks(n: int, n_boot: int, seed: int):
    rng, chunk = np.random.default_rng(seed), max(1, CHUNK_BYTES // (8 * n))
    for start in range(0, n_boot, chunk):
        m = min(chunk, n_boot - start)
        yield np.stack(
            [np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(m)]
        ).astype(np.float64)


def did_numpy_vectorised(d, n_boot, seed):
    x, y = d["x_did"], d["y_did"]
    k = x.shape[1]
    xx = (x[:, :, None] * x[:, None, :]).reshape(len(y), k * k)
    xy = x * y[:, None]
    out = []
    for w in _count_chunks(len(y), n_boot, seed):
        a, b = (w @ xx).reshape(-1, k, k), w @ xy
        out.append(np.linalg.solve(a, b[..., None])[:, 3, 0])
    return np.concatenate(out)


def iv_numpy_vectorised(d, n_boot, seed):
    y, t, z = d["y"], d["t"], d["z"]
    stats = np.column_stack([z, t, y, z * y, z * t])
    n = len(y)
    out = []
    for w in _count_chunks(n, n_boot, seed):
        sz, st, sy, szy, szt = (w @ stats).T
        out.append((n * szy - sz * sy) / (n * szt - sz * st))
    return np.concatenate(out)


IMPLS: dict[str, dict[str, Callable]] = {
    "DiD (4 coefficients)": {
        "numpy_loop": did_numpy_loop,
        "numpy_vectorised": did_numpy_vectorised,
        "rust": lambda d, B, s: core.ols_bootstrap(d["x_did"], d["y_did"], B, s)[:, 3],
        "rust_1_thread": lambda d, B, s: core.ols_bootstrap(
            d["x_did"], d["y_did"], B, s, 1
        )[:, 3],
    },
    "IV slope": {
        "numpy_loop": iv_numpy_loop,
        "numpy_vectorised": iv_numpy_vectorised,
        "rust": lambda d, B, s: core.iv_bootstrap(d["y"], d["t"], d["z"], B, s),
        "rust_1_thread": lambda d, B, s: core.iv_bootstrap(
            d["y"], d["t"], d["z"], B, s, 1
        ),
    },
}


def robust_se(est: str, d) -> float:
    if est.startswith("DiD"):
        return float(sm.OLS(d["y_did"], d["x_did"]).fit(cov_type="HC1").bse[3])
    return float(
        IV2SLS(d["y"], np.ones_like(d["y"]), d["t"], d["z"])
        .fit(cov_type="robust")
        .std_errors.iloc[1]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sizes", type=int, nargs="+", default=[10_000, 100_000, 1_000_000]
    )
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    args = ap.parse_args()

    rows = []
    for n in args.sizes:
        d = make_data(n)
        repeats = 3 if n <= 100_000 else 1
        for est, impls in IMPLS.items():
            se_ref = robust_se(est, d)
            for name, fn in impls.items():
                fn(d, min(args.n_boot, 20), 99)  # warm-up
                times, draws = [], None
                for r in range(repeats):
                    t0 = time.perf_counter()
                    draws = fn(d, args.n_boot, 1)
                    times.append(time.perf_counter() - t0)
                se = float(np.std(draws, ddof=1))
                rows.append(
                    {
                        "estimator": est,
                        "implementation": name,
                        "n": n,
                        "n_boot": args.n_boot,
                        "median_s": statistics.median(times),
                        "repeats": repeats,
                        "bootstrap_se": se,
                        "robust_se": se_ref,
                    }
                )
                print(
                    f"{est:20s} {name:17s} n={n:>9,d} {statistics.median(times):8.3f}s  SE {se:.5f} (robust {se_ref:.5f})"
                )

    rustc = subprocess.run(
        ["rustc", "--version"], capture_output=True, text=True
    ).stdout.strip()
    env = {
        "cpu": next(
            l.split(":", 1)[1].strip()
            for l in Path("/proc/cpuinfo").read_text().splitlines()
            if l.startswith("model name")
        ),
        "cpus_available": len(os.sched_getaffinity(0)),
        "python": platform.python_version(),
        "rustc": rustc,
        "packages": {
            p: version(p)
            for p in ["numpy", "statsmodels", "linearmodels", "causal_rust_core"]
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "bootstrap_results.json").write_text(
        json.dumps({"environment": env, "rows": rows}, indent=2)
    )

    sizes = sorted({r["n"] for r in rows})
    md = []
    for est in IMPLS:
        md += [
            f"### {est}, B = {args.n_boot}",
            "",
            "| Implementation | " + " | ".join(f"n={n:,}" for n in sizes) + " |",
            "| --- |" + " --- |" * len(sizes),
        ]
        for name in IMPLS[est]:
            cell = {
                r["n"]: r
                for r in rows
                if r["estimator"] == est and r["implementation"] == name
            }
            md.append(
                f"| {name} | "
                + " | ".join(f"{cell[n]['median_s']:.3f} s" for n in sizes)
                + " |"
            )
        md += [
            "",
            "| Bootstrap SE / robust SE | "
            + " | ".join(f"n={n:,}" for n in sizes)
            + " |",
            "| --- |" + " --- |" * len(sizes),
        ]
        for name in IMPLS[est]:
            cell = {
                r["n"]: r
                for r in rows
                if r["estimator"] == est and r["implementation"] == name
            }
            md.append(
                f"| {name} | "
                + " | ".join(
                    f"{cell[n]['bootstrap_se'] / cell[n]['robust_se']:.3f}"
                    for n in sizes
                )
                + " |"
            )
        md.append("")
    (args.out / "bootstrap_results.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
