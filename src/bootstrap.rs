//! Pairs bootstrap for OLS/DiD and just-identified IV, on zero-copy numpy input.
//!
//! Each replicate draws n row indices with replacement and accumulates the sufficient statistics
//! (X'X and X'y for OLS; five sums for IV) directly from the drawn rows, so no resampled array
//! is ever materialised. Replicates run in parallel with Rayon with the GIL released, and each
//! replicate has its own seed, so results do not depend on thread scheduling.
//!
//! OLS solves the normal equations by Cholesky, which is fine for well-conditioned designs
//! (intercept, dummies, standardised covariates) but less robust than QR/SVD for ill-conditioned
//! ones; a singular replicate returns NaN.

use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// SplitMix64 output function: a strong 64-bit mixer.
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// SplitMix64: small, fast, reproducible; good enough to drive bootstrap resampling.
struct SplitMix64(u64);

impl SplitMix64 {
    /// Each replicate starts at a hashed point of the 2^64 cycle. (Seeding with `seed ^ b * gamma`
    /// would make replicate b+1's stream replicate b's stream shifted by one draw, because the
    /// state advances by the same gamma: the replicates would share resamples and the bootstrap
    /// variance would collapse.)
    fn for_replicate(seed: u64, b: usize) -> Self {
        SplitMix64(mix64(
            seed.wrapping_add(mix64(b as u64 ^ 0xD1B5_4A32_D192_ED03)),
        ))
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Unbiased integer in [0, n) (Lemire's method).
    fn below(&mut self, n: u64) -> usize {
        let mut m = (self.next() as u128) * (n as u128);
        if (m as u64) < n {
            let threshold = n.wrapping_neg() % n;
            while (m as u64) < threshold {
                m = (self.next() as u128) * (n as u128);
            }
        }
        (m >> 64) as usize
    }
}

/// Solves A beta = b for symmetric positive-definite A (k x k, row-major) by Cholesky.
fn solve_spd(a: &[f64], b: &[f64], k: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..=i {
            let s: f64 = a[i * k + j] - (0..j).map(|p| l[i * k + p] * l[j * k + p]).sum::<f64>();
            if i == j {
                if s <= 1e-12 * a[i * k + i].abs().max(1.0) {
                    return None;
                }
                l[i * k + i] = s.sqrt();
            } else {
                l[i * k + j] = s / l[j * k + j];
            }
        }
    }
    let mut z = vec![0.0; k];
    for i in 0..k {
        z[i] = (b[i] - (0..i).map(|p| l[i * k + p] * z[p]).sum::<f64>()) / l[i * k + i];
    }
    let mut beta = vec![0.0; k];
    for i in (0..k).rev() {
        beta[i] = (z[i] - (i + 1..k).map(|p| l[p * k + i] * beta[p]).sum::<f64>()) / l[i * k + i];
    }
    Some(beta)
}

fn with_threads<T: Send>(
    n_threads: Option<usize>,
    f: impl FnOnce() -> T + Send,
) -> Result<T, String> {
    match n_threads {
        None => Ok(f()),
        Some(t) => rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .map(|pool| pool.install(f))
            .map_err(|e| e.to_string()),
    }
}

/// B x k coefficient draws, row-major. `x` is n x k row-major.
pub fn ols_bootstrap_core(x: &[f64], y: &[f64], k: usize, n_boot: usize, seed: u64) -> Vec<f64> {
    let n = y.len();
    (0..n_boot)
        .into_par_iter()
        .flat_map_iter(|b| {
            let mut rng = SplitMix64::for_replicate(seed, b);
            let mut xtx = vec![0.0; k * k];
            let mut xty = vec![0.0; k];
            for _ in 0..n {
                let i = rng.below(n as u64);
                let row = &x[i * k..(i + 1) * k];
                for a in 0..k {
                    xty[a] += row[a] * y[i];
                    for c in a..k {
                        xtx[a * k + c] += row[a] * row[c];
                    }
                }
            }
            for a in 0..k {
                for c in 0..a {
                    xtx[a * k + c] = xtx[c * k + a];
                }
            }
            solve_spd(&xtx, &xty, k).unwrap_or_else(|| vec![f64::NAN; k])
        })
        .collect()
}

/// B draws of the just-identified IV slope with intercept: cov(z, y) / cov(z, t).
pub fn iv_bootstrap_core(y: &[f64], t: &[f64], z: &[f64], n_boot: usize, seed: u64) -> Vec<f64> {
    let n = y.len();
    (0..n_boot)
        .into_par_iter()
        .map(|b| {
            let mut rng = SplitMix64::for_replicate(seed, b);
            let (mut sz, mut st, mut sy, mut szy, mut szt) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for _ in 0..n {
                let i = rng.below(n as u64);
                sz += z[i];
                st += t[i];
                sy += y[i];
                szy += z[i] * y[i];
                szt += z[i] * t[i];
            }
            let nf = n as f64;
            let denom = nf * szt - sz * st;
            if denom.abs() < 1e-12 {
                f64::NAN
            } else {
                (nf * szy - sz * sy) / denom
            }
        })
        .collect()
}

/// Pairs-bootstrap OLS coefficients. `x` (n x k, float64, C-contiguous) must include any intercept column.
/// Returns an (n_boot, k) array of coefficient draws.
#[pyfunction]
#[pyo3(signature = (x, y, n_boot, seed=0, n_threads=None))]
pub fn ols_bootstrap<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    n_boot: usize,
    seed: u64,
    n_threads: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (n, k) = (x.shape()[0], x.shape()[1]);
    let xs = x
        .as_slice()
        .map_err(|_| PyValueError::new_err("x must be a C-contiguous float64 array"))?;
    let ys = y
        .as_slice()
        .map_err(|_| PyValueError::new_err("y must be a contiguous float64 array"))?;
    if ys.len() != n || n == 0 {
        return Err(PyValueError::new_err(
            "x and y must have the same, non-zero number of rows",
        ));
    }
    let out = py
        .detach(|| with_threads(n_threads, || ols_bootstrap_core(xs, ys, k, n_boot, seed)))
        .map_err(PyValueError::new_err)?;
    PyArray1::from_vec(py, out).reshape([n_boot, k])
}

/// Pairs-bootstrap just-identified IV slope (one endogenous treatment, one instrument, intercept).
#[pyfunction]
#[pyo3(signature = (y, t, z, n_boot, seed=0, n_threads=None))]
pub fn iv_bootstrap<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    t: PyReadonlyArray1<'py, f64>,
    z: PyReadonlyArray1<'py, f64>,
    n_boot: usize,
    seed: u64,
    n_threads: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let err = |_| PyValueError::new_err("inputs must be contiguous float64 arrays");
    let (ys, ts, zs) = (
        y.as_slice().map_err(err)?,
        t.as_slice().map_err(err)?,
        z.as_slice().map_err(err)?,
    );
    if ys.is_empty() || ys.len() != ts.len() || ys.len() != zs.len() {
        return Err(PyValueError::new_err(
            "y, t and z must have the same, non-zero length",
        ));
    }
    let out = py
        .detach(|| with_threads(n_threads, || iv_bootstrap_core(ys, ts, zs, n_boot, seed)))
        .map_err(PyValueError::new_err)?;
    Ok(PyArray1::from_vec(py, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbouring_replicates_do_not_share_streams() {
        let draws = |b| { let mut r = SplitMix64::for_replicate(0, b); (0..100).map(|_| r.below(1000)).collect::<Vec<_>>() };
        let (a, b) = (draws(0), draws(1));
        assert_ne!(a[1..], b[..99], "replicate 1 must not be replicate 0 shifted by one draw");
        let shared = a.iter().zip(&b).filter(|(x, y)| x == y).count();
        assert!(shared < 5, "{shared} identical positions");
    }

    #[test]
    fn below_stays_in_range_and_covers_it() {
        let mut rng = SplitMix64::for_replicate(1, 0);
        let mut seen = [false; 7];
        for _ in 0..1000 {
            let v = rng.below(7);
            assert!(v < 7);
            seen[v] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn solve_spd_matches_known_solution() {
        // [[4, 2], [2, 3]] beta = [10, 8] -> beta = [1.75, 1.5]
        let beta = solve_spd(&[4.0, 2.0, 2.0, 3.0], &[10.0, 8.0], 2).unwrap();
        assert!((beta[0] - 1.75).abs() < 1e-12 && (beta[1] - 1.5).abs() < 1e-12);
    }

    #[test]
    fn same_seed_same_draws_regardless_of_threads() {
        let n = 500;
        let x: Vec<f64> = (0..n).flat_map(|i| [1.0, (i % 10) as f64]).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 2.0 + 0.5 * (i % 10) as f64 + ((i * 37) % 7) as f64 * 0.1)
            .collect();
        let a = with_threads(Some(1), || ols_bootstrap_core(&x, &y, 2, 50, 42)).unwrap();
        let b = with_threads(Some(4), || ols_bootstrap_core(&x, &y, 2, 50, 42)).unwrap();
        assert_eq!(a, b);
    }
}
