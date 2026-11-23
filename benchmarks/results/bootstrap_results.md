### DiD (4 coefficients), B = 1000

| Implementation | n=10,000 | n=100,000 | n=1,000,000 |
| --- | --- | --- | --- |
| numpy_loop | 0.273 s | 2.978 s | 101.579 s |
| numpy_vectorised | 0.098 s | 0.890 s | 8.836 s |
| rust | 0.031 s | 0.210 s | 7.534 s |
| rust_1_thread | 0.098 s | 1.138 s | 60.575 s |

| Bootstrap SE / robust SE | n=10,000 | n=100,000 | n=1,000,000 |
| --- | --- | --- | --- |
| numpy_loop | 1.032 | 0.994 | 1.049 |
| numpy_vectorised | 1.032 | 0.994 | 1.049 |
| rust | 1.002 | 1.020 | 1.048 |
| rust_1_thread | 1.002 | 1.020 | 1.048 |

### IV slope, B = 1000

| Implementation | n=10,000 | n=100,000 | n=1,000,000 |
| --- | --- | --- | --- |
| numpy_loop | 0.049 s | 0.929 s | 17.188 s |
| numpy_vectorised | 0.093 s | 0.883 s | 8.198 s |
| rust | 0.016 s | 0.044 s | 1.486 s |
| rust_1_thread | 0.027 s | 0.164 s | 7.139 s |

| Bootstrap SE / robust SE | n=10,000 | n=100,000 | n=1,000,000 |
| --- | --- | --- | --- |
| numpy_loop | 1.007 | 0.975 | 0.973 |
| numpy_vectorised | 1.007 | 0.975 | 0.973 |
| rust | 0.931 | 0.989 | 1.014 |
| rust_1_thread | 0.931 | 0.989 | 1.014 |
