from ._causal_rust_core import *
from ._causal_rust_core import ols_fit as _ols_fit
from ._causal_rust_core import iv_fit as _iv_fit
from ._causal_rust_core import did_fit as _did_fit
import polars as pl

from . import _causal_rust_core

def ols_fit(df: pl.DataFrame, target: str, features: list[str]) -> float:
    """
    Fit OLS using Rust backend.
    """
    data_dict = df.to_dict(as_series=False)
    return _ols_fit(data_dict, target, features)

def iv_fit(df: pl.DataFrame, target: str, treatment: str, instrument: str) -> float:
    """
    Fit IV/2SLS using Rust backend.
    """
    data_dict = df.to_dict(as_series=False)
    return _iv_fit(data_dict, target, treatment, instrument)

def did_fit(df: pl.DataFrame, target: str, group: str, time: str) -> float:
    """
    Fit DiD using Rust backend.
    """
    data_dict = df.to_dict(as_series=False)
    return _did_fit(data_dict, target, group, time)


def ols_bootstrap(
    df: pl.DataFrame, target: str, features: list[str], n_boot: int = 1000, seed: int = 0, n_threads: int | None = None
):
    """Pairs-bootstrap OLS (intercept added) in Rust. Returns (n_boot, 1 + len(features)) coefficient draws."""
    import numpy as np

    x = np.column_stack([np.ones(df.height), df.select(features).to_numpy().astype(np.float64)])
    y = df.get_column(target).to_numpy().astype(np.float64)
    return _causal_rust_core.ols_bootstrap(np.ascontiguousarray(x), np.ascontiguousarray(y), n_boot, seed, n_threads)


def did_bootstrap(
    df: pl.DataFrame, target: str, group: str, time: str, n_boot: int = 1000, seed: int = 0, n_threads: int | None = None
):
    """Pairs-bootstrap draws of the 2x2 DiD effect (coefficient on group x time)."""
    with_inter = df.with_columns((pl.col(group) * pl.col(time)).alias("__did"))
    return ols_bootstrap(with_inter, target, [group, time, "__did"], n_boot, seed, n_threads)[:, 3]


def iv_bootstrap(
    df: pl.DataFrame, target: str, treatment: str, instrument: str, n_boot: int = 1000, seed: int = 0, n_threads: int | None = None
):
    """Pairs-bootstrap draws of the just-identified IV slope (with intercept)."""
    import numpy as np

    col = lambda c: np.ascontiguousarray(df.get_column(c).to_numpy().astype(np.float64))
    return _causal_rust_core.iv_bootstrap(col(target), col(treatment), col(instrument), n_boot, seed, n_threads)
