### OLS

| Implementation | n=10,000 | n=100,000 | n=1,000,000 | n=10,000,000 |
| --- | --- | --- | --- | --- |
| rust_wrapper | 0.56 ms | 7.95 ms | 108.44 ms | 1,289.52 ms |
| rust_core | 0.25 ms | 3.68 ms | 53.67 ms | 592.59 ms |
| numpy | 0.15 ms | 1.32 ms | 25.39 ms | 313.04 ms |
| statsmodels | 0.73 ms | 6.06 ms | 103.29 ms | 986.48 ms |

### IV/2SLS

| Implementation | n=10,000 | n=100,000 | n=1,000,000 | n=10,000,000 |
| --- | --- | --- | --- | --- |
| rust_wrapper | 0.56 ms | 6.70 ms | 107.87 ms | 1,308.61 ms |
| rust_core | 0.26 ms | 2.68 ms | 41.71 ms | 676.82 ms |
| numpy | 0.22 ms | 2.09 ms | 53.73 ms | 514.12 ms |
| linearmodels | 10.28 ms | 116.01 ms | 1,054.66 ms | 11,997.31 ms |

### DiD

| Implementation | n=10,000 | n=100,000 | n=1,000,000 | n=10,000,000 |
| --- | --- | --- | --- | --- |
| rust_wrapper | 0.53 ms | 6.58 ms | 116.01 ms | 1,487.63 ms |
| rust_core | 0.27 ms | 3.12 ms | 70.63 ms | 799.96 ms |
| numpy | 0.18 ms | 1.79 ms | 47.78 ms | 445.35 ms |
| statsmodels | 0.73 ms | 6.63 ms | 123.78 ms | 1,179.17 ms |
