# NumPy vs numpy-node Benchmark Results

- **NumPy**: Python 2.4.1
- **numpy-node**: Node.js v24.12.0
- **Date**: 2026-01-17
- **Platform**: darwin arm64

## Results

| Benchmark              | NumPy (ms) | numpy-node (ms) | Speedup           |
| ---------------------- | ---------- | --------------- | ----------------- |
| zeros(1000x1000)       | 0.087      | 0.003           | **25.84x faster** |
| ones(1000x1000)        | 0.129      | 0.701           | 5.42x slower      |
| arange(100000)         | 0.014      | 0.077           | 5.31x slower      |
| linspace(0, 1, 100000) | 0.064      | 0.077           | 1.19x slower      |
| add(1000x1000)         | 0.353      | 0.979           | 2.77x slower      |
| multiply(1000x1000)    | 0.351      | 0.974           | 2.78x slower      |
| sqrt(1000x1000)        | 0.312      | 0.935           | 3.00x slower      |
| exp(1000x1000)         | 2.364      | 1.876           | **1.26x faster**  |
| sum(1000x1000)         | 0.185      | 0.103           | **1.79x faster**  |
| mean(1000x1000)        | 0.186      | 0.103           | **1.80x faster**  |
| std(1000x1000)         | 0.746      | 0.286           | **2.61x faster**  |
| min(1000x1000)         | 0.101      | 0.103           | 1.02x slower      |
| max(1000x1000)         | 0.102      | 0.104           | 1.02x slower      |
| matmul(500x500)        | 0.267      | 0.472           | 1.77x slower      |
| dot(500x500)           | 0.272      | 0.450           | 1.65x slower      |
| inv(100x100)           | 0.099      | 0.119           | 1.21x slower      |
| det(100x100)           | 0.040      | 0.036           | **1.12x faster**  |
| svd(100x100)           | 0.884      | 1.676           | 1.89x slower      |
| qr(100x100)            | 0.199      | 0.187           | **1.07x faster**  |
| eig(100x100)           | 4.868      | 3.785           | **1.29x faster**  |
| solve(100x100)         | 0.054      | 0.041           | **1.31x faster**  |
| 1000x small array ops  | 0.959      | 3.947           | 4.11x slower      |

## Interpretation

- **Speedup > 1**: numpy-node is faster
- **Speedup < 1**: NumPy (Python) is faster
- Large matrix operations (BLAS/LAPACK) should be similar as both use the same underlying libraries
- Small operations and loops should favor Node.js due to V8's JIT compilation
