# NumPy vs numpy-node Benchmark Results

- **NumPy**: Python 2.4.1
- **numpy-node**: Node.js v24.12.0
- **Date**: 2026-01-17
- **Platform**: darwin arm64

## Results

| Benchmark              | NumPy (ms) | numpy-node (ms) | Speedup           |
| ---------------------- | ---------- | --------------- | ----------------- |
| zeros(1000x1000)       | 0.087      | 0.004           | **24.90x faster** |
| ones(1000x1000)        | 0.129      | 0.705           | 5.44x slower      |
| arange(100000)         | 0.015      | 0.082           | 5.40x slower      |
| linspace(0, 1, 100000) | 0.070      | 0.082           | 1.18x slower      |
| add(1000x1000)         | 0.354      | 0.991           | 2.80x slower      |
| multiply(1000x1000)    | 0.384      | 1.020           | 2.66x slower      |
| sqrt(1000x1000)        | 0.317      | 0.993           | 3.13x slower      |
| exp(1000x1000)         | 2.454      | 1.910           | **1.28x faster**  |
| sum(1000x1000)         | 0.184      | 0.104           | **1.77x faster**  |
| mean(1000x1000)        | 0.186      | 0.105           | **1.77x faster**  |
| std(1000x1000)         | 0.710      | 0.287           | **2.48x faster**  |
| min(1000x1000)         | 0.101      | 0.104           | 1.03x slower      |
| max(1000x1000)         | 0.100      | 0.104           | 1.04x slower      |
| matmul(500x500)        | 0.293      | 0.451           | 1.54x slower      |
| dot(500x500)           | 0.313      | 0.438           | 1.40x slower      |
| inv(100x100)           | 0.100      | 0.119           | 1.19x slower      |
| det(100x100)           | 0.041      | 0.037           | **1.12x faster**  |
| svd(100x100)           | 0.941      | 1.815           | 1.93x slower      |
| qr(100x100)            | 0.201      | 0.192           | **1.04x faster**  |
| eig(100x100)           | 4.706      | 4.198           | **1.12x faster**  |
| solve(100x100)         | 0.059      | 0.042           | **1.43x faster**  |
| 1000x small array ops  | 0.949      | 4.014           | 4.23x slower      |

## Interpretation

- **Speedup > 1**: numpy-node is faster
- **Speedup < 1**: NumPy (Python) is faster
- Large matrix operations (BLAS/LAPACK) should be similar as both use the same underlying libraries
- Small operations and loops should favor Node.js due to V8's JIT compilation
