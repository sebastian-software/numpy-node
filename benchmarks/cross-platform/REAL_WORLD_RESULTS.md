# Real-World Benchmark Results

These benchmarks represent actual use cases rather than isolated method calls.

- **NumPy**: Python 2.4.1
- **numpy-node**: Node.js v24.12.0
- **Date**: 2026-01-17
- **Platform**: darwin arm64

## Scenarios

| Scenario                                   | NumPy   | numpy-node | Speedup          |
| ------------------------------------------ | ------- | ---------- | ---------------- |
| Data Normalization (10k x 100)             | 2.19ms  | 2.69ms     | 1.23x slower     |
| Linear Regression (5k x 50)                | 160.8µs | 431.0µs    | 2.68x slower     |
| PCA via SVD (2k x 100 -> 10)               | 11.33ms | 13.05ms    | 1.15x slower     |
| Correlation Matrix (5k x 50)               | 1.07ms  | 1.04ms     | **1.03x faster** |
| Neural Net Forward (1k x 256->128->64->10) | 442.2µs | 445.0µs    | 1.01x slower     |
| Covariance + Eigendecomp (3k x 80)         | 4.07ms  | 2.67ms     | **1.52x faster** |
| Least Squares QR (8k x 100)                | 36.05ms | 38.28ms    | 1.06x slower     |
| Batch Statistics (100 x 1k x 20)           | 7.25ms  | 2.11ms     | **3.43x faster** |
| Pairwise Distances (1k x 50)               | 2.72ms  | 2.47ms     | **1.10x faster** |
| Polynomial Fit (1k points, deg 10)         | 148.7µs | 32.9µs     | **4.52x faster** |

**Summary: numpy-node wins 5/10 scenarios (50%)**

## Interpretation

- These scenarios combine multiple operations, amortizing N-API overhead
- Compute-heavy operations (SVD, eigendecomposition) benefit from same BLAS/LAPACK backend
- Real applications should see performance closer to these results than micro-benchmarks
