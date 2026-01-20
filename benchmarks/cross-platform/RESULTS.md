# NumPy vs numpy-node Benchmark Results

- **NumPy**: Python 2.4.1
- **numpy-node**: Node.js v24.12.0
- **Date**: 2026-01-20
- **Platform**: darwin arm64

## Results

| Benchmark              | NumPy (ms) | numpy-node (ms) | Speedup           |
| ---------------------- | ---------- | --------------- | ----------------- |
| zeros(1000x1000)       | 0.087      | 0.003           | **25.64x faster** |
| ones(1000x1000)        | 0.129      | 0.756           | 5.84x slower      |
| arange(100000)         | 0.014      | 0.080           | 5.60x slower      |
| linspace(0, 1, 100000) | 0.064      | 0.081           | 1.26x slower      |
| add(1000x1000)         | 0.358      | 1.012           | 2.83x slower      |
| multiply(1000x1000)    | 0.341      | 1.065           | 3.13x slower      |
| sqrt(1000x1000)        | 0.313      | 0.932           | 2.98x slower      |
| exp(1000x1000)         | 2.341      | 1.890           | **1.24x faster**  |
| sum(1000x1000)         | 0.188      | 0.102           | **1.85x faster**  |
| mean(1000x1000)        | 0.189      | 0.102           | **1.86x faster**  |
| std(1000x1000)         | 0.712      | 0.282           | **2.53x faster**  |
| min(1000x1000)         | 0.107      | 0.101           | **1.05x faster**  |
| max(1000x1000)         | 0.106      | 0.102           | **1.04x faster**  |
| matmul(500x500)        | 0.256      | 0.424           | 1.65x slower      |
| dot(500x500)           | 0.280      | 0.436           | 1.56x slower      |
| inv(100x100)           | 0.099      | 0.115           | 1.16x slower      |
| det(100x100)           | 0.040      | 0.036           | **1.12x faster**  |
| svd(100x100)           | 0.914      | 1.740           | 1.90x slower      |
| qr(100x100)            | 0.210      | 0.188           | **1.12x faster**  |
| eig(100x100)           | 5.031      | 4.074           | **1.23x faster**  |
| solve(100x100)         | 0.055      | 0.041           | **1.33x faster**  |
| 1000x small array ops  | 0.974      | 4.144           | 4.25x slower      |

## Interpretation

- **Speedup > 1**: numpy-node is faster
- **Speedup < 1**: NumPy (Python) is faster
- Large matrix operations (BLAS/LAPACK) should be similar as both use the same underlying libraries
- Small operations and loops should favor Node.js due to V8's JIT compilation
