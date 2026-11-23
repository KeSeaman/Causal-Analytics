### OLS

| Implementation | n=10,000 | n=100,000 | n=1,000,000 | n=10,000,000 |
| --- | --- | --- | --- | --- |
| rust_wrapper | 0.55 ms | 8.83 ms | 120.83 ms | 1,340.39 ms |
| rust_core | 0.27 ms | 3.87 ms | 59.05 ms | 646.84 ms |
| numpy | 0.13 ms | 1.32 ms | 25.84 ms | 314.16 ms |
| statsmodels | 0.69 ms | 6.10 ms | 109.04 ms | 993.66 ms |

### IV/2SLS

| Implementation | n=10,000 | n=100,000 | n=1,000,000 | n=10,000,000 |
| --- | --- | --- | --- | --- |
| rust_wrapper | 0.55 ms | 7.14 ms | 119.10 ms | 1,369.76 ms |
| rust_core | 0.27 ms | 2.89 ms | 44.57 ms | 701.71 ms |
| numpy | 0.22 ms | 2.05 ms | 54.43 ms | 530.28 ms |
| linearmodels | 9.80 ms | 109.80 ms | 1,079.29 ms | 11,730.32 ms |

### DiD

| Implementation | n=10,000 | n=100,000 | n=1,000,000 | n=10,000,000 |
| --- | --- | --- | --- | --- |
| rust_wrapper | 0.57 ms | 9.47 ms | 129.64 ms | 1,495.60 ms |
| rust_core | 0.30 ms | 3.33 ms | 73.60 ms | 912.81 ms |
| numpy | 0.18 ms | 1.78 ms | 48.45 ms | 460.56 ms |
| statsmodels | 0.69 ms | 6.48 ms | 127.69 ms | 1,190.19 ms |
