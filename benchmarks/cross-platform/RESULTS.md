# NumPy vs numpy-node Benchmark Results

- **NumPy**: Python 2.4.1
- **numpy-node**: Node.js v24.12.0
- **Date**: 2026-01-17
- **Platform**: darwin arm64

## Results

| Benchmark              | NumPy (ms) | numpy-node (ms) | Speedup           |
| ---------------------- | ---------- | --------------- | ----------------- |
| zeros(1000x1000)       | 0.090      | 0.003           | **27.39x faster** |
| ones(1000x1000)        | 0.132      | 0.698           | 5.27x slower      |
| arange(100000)         | 0.015      | 0.084           | 5.55x slower      |
| linspace(0, 1, 100000) | 0.068      | 0.086           | 1.25x slower      |
| add(1000x1000)         | 0.399      | 0.971           | 2.44x slower      |
| multiply(1000x1000)    | 0.361      | 1.054           | 2.92x slower      |
| sqrt(1000x1000)        | 0.313      | 1.040           | 3.33x slower      |
| exp(1000x1000)         | 2.363      | 3.035           | 1.28x slower      |
| sum(1000x1000)         | 0.198      | 0.094           | **2.10x faster**  |
| mean(1000x1000)        | 0.192      | 0.094           | **2.04x faster**  |
| std(1000x1000)         | 0.724      | 0.280           | **2.58x faster**  |
| min(1000x1000)         | 0.103      | 0.097           | **1.07x faster**  |
| max(1000x1000)         | 0.103      | 0.094           | **1.09x faster**  |
| matmul(500x500)        | 0.248      | 0.447           | 1.80x slower      |
| dot(500x500)           | 0.281      | 0.445           | 1.58x slower      |
| inv(100x100)           | 0.099      | 0.122           | 1.23x slower      |
| det(100x100)           | 0.041      | 0.036           | **1.13x faster**  |
| svd(100x100)           | 0.893      | 1.697           | 1.90x slower      |
| qr(100x100)            | 0.200      | 0.189           | **1.06x faster**  |
| eig(100x100)           | 4.863      | 4.546           | **1.07x faster**  |
| solve(100x100)         | 0.055      | 0.041           | **1.33x faster**  |
| 1000x small array ops  | 0.940      | 3.895           | 4.15x slower      |

## Interpretation

- **Speedup > 1**: numpy-node is faster
- **Speedup < 1**: NumPy (Python) is faster
- Large matrix operations (BLAS/LAPACK) should be similar as both use the same underlying libraries
- Small operations and loops should favor Node.js due to V8's JIT compilation
