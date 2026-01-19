# NumPy vs numpy-node Benchmark Results

- **NumPy**: Python 2.4.1
- **numpy-node**: Node.js v24.12.0
- **Date**: 2026-01-19
- **Platform**: darwin arm64

## Results

| Benchmark              | NumPy (ms) | numpy-node (ms) | Speedup           |
| ---------------------- | ---------- | --------------- | ----------------- |
| zeros(1000x1000)       | 0.087      | 0.003           | **25.27x faster** |
| ones(1000x1000)        | 0.134      | 0.695           | 5.20x slower      |
| arange(100000)         | 0.015      | 0.074           | 4.87x slower      |
| linspace(0, 1, 100000) | 0.071      | 0.078           | 1.11x slower      |
| add(1000x1000)         | 0.352      | 0.986           | 2.80x slower      |
| multiply(1000x1000)    | 0.350      | 1.028           | 2.94x slower      |
| sqrt(1000x1000)        | 0.312      | 0.914           | 2.93x slower      |
| exp(1000x1000)         | 2.352      | 1.890           | **1.24x faster**  |
| sum(1000x1000)         | 0.204      | 0.115           | **1.77x faster**  |
| mean(1000x1000)        | 0.228      | 0.117           | **1.94x faster**  |
| std(1000x1000)         | 0.945      | 0.284           | **3.32x faster**  |
| min(1000x1000)         | 0.145      | 0.103           | **1.41x faster**  |
| max(1000x1000)         | 0.131      | 0.103           | **1.27x faster**  |
| matmul(500x500)        | 0.330      | 0.432           | 1.31x slower      |
| dot(500x500)           | 0.426      | 0.442           | 1.04x slower      |
| inv(100x100)           | 0.123      | 0.159           | 1.29x slower      |
| det(100x100)           | 0.047      | 0.038           | **1.22x faster**  |
| svd(100x100)           | 1.034      | 1.899           | 1.84x slower      |
| qr(100x100)            | 0.243      | 0.202           | **1.20x faster**  |
| eig(100x100)           | 5.341      | 4.487           | **1.19x faster**  |
| solve(100x100)         | 0.057      | 0.042           | **1.37x faster**  |
| 1000x small array ops  | 1.032      | 4.185           | 4.05x slower      |

## Interpretation

- **Speedup > 1**: numpy-node is faster
- **Speedup < 1**: NumPy (Python) is faster
- Large matrix operations (BLAS/LAPACK) should be similar as both use the same underlying libraries
- Small operations and loops should favor Node.js due to V8's JIT compilation
