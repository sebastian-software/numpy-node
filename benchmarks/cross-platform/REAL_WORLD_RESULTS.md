# Real-World Benchmark Results

These benchmarks represent actual use cases rather than isolated method calls.

- **NumPy**: Python 2.4.1
- **numpy-node**: Node.js v24.12.0
- **Date**: 2026-01-18
- **Platform**: darwin arm64

## Scenarios

| Scenario                                   | NumPy   | numpy-node | Speedup           |
| ------------------------------------------ | ------- | ---------- | ----------------- |
| Data Normalization (10k x 100)             | 1.97ms  | 1.45ms     | **1.36x faster**  |
| Linear Regression (5k x 50)                | 165.6µs | 164.3µs    | **1.01x faster**  |
| PCA via SVD (2k x 100 -> 10)               | 11.51ms | 13.02ms    | 1.13x slower      |
| Correlation Matrix (5k x 50)               | 883.4µs | 762.2µs    | **1.16x faster**  |
| Neural Net Forward (1k x 256->128->64->10) | 404.3µs | 447.0µs    | 1.11x slower      |
| Covariance + Eigendecomp (3k x 80)         | 4.14ms  | 3.15ms     | **1.31x faster**  |
| Least Squares QR (8k x 100)                | 37.95ms | 39.60ms    | 1.04x slower      |
| Batch Statistics (100 x 1k x 20)           | 7.60ms  | 2.23ms     | **3.41x faster**  |
| Pairwise Distances (1k x 50)               | 2.65ms  | 1.61ms     | **1.65x faster**  |
| Polynomial Fit (1k points, deg 10)         | 147.0µs | 11.9µs     | **12.34x faster** |
| Min-Max Scaling (10k x 100)                | 1.26ms  | 1.81ms     | 1.43x slower      |
| Outer Product Sum (1k x 100)               | 58.7µs  | 61.5µs     | 1.05x slower      |
| Weighted Mean (5k x 200)                   | 200.8µs | 230.7µs    | 1.15x slower      |
| Gradient Descent Step (10k x 50)           | 239.8µs | 271.1µs    | 1.13x slower      |
| Cross-Entropy Loss (1k x 10)               | 44.9µs  | 54.4µs     | 1.21x slower      |
| Cosine Similarity (500 x 128)              | 661.8µs | 518.8µs    | **1.28x faster**  |
| K-Means Step (5k x 20, k=10)               | 187.0µs | 126.2µs    | **1.48x faster**  |
| Image Filter (256 x 256)                   | 58.8µs  | 59.6µs     | 1.01x slower      |
| Power Iteration (200 x 200)                | 71.8µs  | 78.4µs     | 1.09x slower      |
| Rolling Std (10k, window=50)               | 6.49ms  | 6.26ms     | **1.04x faster**  |
| Log Returns (5k x 100)                     | 2.81ms  | 2.21ms     | **1.27x faster**  |
| Sharpe Ratio (1k x 50)                     | 89.2µs  | 32.5µs     | **2.74x faster**  |
| Portfolio Variance (1k x 50)               | 90.2µs  | 101.4µs    | 1.12x slower      |
| Attention Scores (128 x 64)                | 86.6µs  | 112.3µs    | 1.30x slower      |
| Layer Normalization (1k x 512)             | 1.25ms  | 1.17ms     | **1.07x faster**  |
| Batch Matmul (32 x 64 x 64)                | 100.2µs | 248.5µs    | 2.48x slower      |
| Softmax + Cross-Entropy (1k x 100)         | 754.3µs | 841.0µs    | 1.11x slower      |
| 1D Convolution (10k, kernel=51)            | 19.16ms | 533.1µs    | **35.93x faster** |
| Histogram (100k, 100 bins)                 | 722.3µs | 311.5µs    | **2.32x faster**  |
| Percentiles (10k x 50)                     | 11.14ms | 10.34ms    | **1.08x faster**  |
| Jacobi Iteration (200 x 200)               | 310.2µs | 301.6µs    | **1.03x faster**  |
| Trapezoidal Integration (10k pts)          | 35.5µs  | 35.7µs     | 1.01x slower      |
| Finite Difference (200 x 200)              | 108.4µs | 91.7µs     | **1.18x faster**  |
| Matrix Exponential (100 x 100)             | 200.3µs | 190.0µs    | **1.05x faster**  |
| Ridge Regression (5k x 100)                | 423.0µs | 439.5µs    | 1.04x slower      |
| Gram-Schmidt (200 x 50)                    | 2.11ms  | 473.0µs    | **4.46x faster**  |
| LU Solve (300 x 300)                       | 419.2µs | 426.3µs    | 1.02x slower      |
| Outer Product (1k x 1k)                    | 390.0µs | 839.4µs    | 2.15x slower      |
| Kronecker Product (50x50 ⊗ 20x20)          | 547.6µs | 1.00ms     | 1.83x slower      |

**Summary: numpy-node wins 20/39 scenarios (51%)**

## Interpretation

- These scenarios combine multiple operations, amortizing N-API overhead
- Compute-heavy operations (SVD, eigendecomposition) benefit from same BLAS/LAPACK backend
- Real applications should see performance closer to these results than micro-benchmarks
