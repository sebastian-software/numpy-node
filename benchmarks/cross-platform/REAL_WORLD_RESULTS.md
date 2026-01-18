# Real-World Benchmark Results

These benchmarks represent actual use cases rather than isolated method calls.

- **NumPy**: Python 2.4.1
- **numpy-node**: Node.js v24.12.0
- **Date**: 2026-01-18
- **Platform**: darwin arm64

## Scenarios

| Scenario                                   | NumPy   | numpy-node | Speedup           |
| ------------------------------------------ | ------- | ---------- | ----------------- |
| Data Normalization (10k x 100)             | 2.19ms  | 1.35ms     | **1.63x faster**  |
| Linear Regression (5k x 50)                | 170.5µs | 179.5µs    | 1.05x slower      |
| PCA via SVD (2k x 100 -> 10)               | 11.71ms | 14.10ms    | 1.20x slower      |
| Correlation Matrix (5k x 50)               | 1.17ms  | 805.2µs    | **1.46x faster**  |
| Neural Net Forward (1k x 256->128->64->10) | 406.3µs | 451.9µs    | 1.11x slower      |
| Covariance + Eigendecomp (3k x 80)         | 4.23ms  | 3.20ms     | **1.32x faster**  |
| Least Squares QR (8k x 100)                | 35.96ms | 40.40ms    | 1.12x slower      |
| Batch Statistics (100 x 1k x 20)           | 7.19ms  | 2.15ms     | **3.35x faster**  |
| Pairwise Distances (1k x 50)               | 2.64ms  | 1.69ms     | **1.56x faster**  |
| Polynomial Fit (1k points, deg 10)         | 145.8µs | 11.3µs     | **12.87x faster** |
| Min-Max Scaling (10k x 100)                | 1.51ms  | 3.30ms     | 2.18x slower      |
| Outer Product Sum (1k x 100)               | 53.6µs  | 60.9µs     | 1.14x slower      |
| Weighted Mean (5k x 200)                   | 239.6µs | 210.1µs    | **1.14x faster**  |
| Gradient Descent Step (10k x 50)           | 233.4µs | 270.2µs    | 1.16x slower      |
| Cross-Entropy Loss (1k x 10)               | 41.7µs  | 49.1µs     | 1.18x slower      |
| Cosine Similarity (500 x 128)              | 643.7µs | 1.22ms     | 1.90x slower      |
| K-Means Step (5k x 20, k=10)               | 152.7µs | 236.5µs    | 1.55x slower      |
| Image Filter (256 x 256)                   | 54.8µs  | 259.2µs    | 4.73x slower      |
| Power Iteration (200 x 200)                | 69.1µs  | 121.8µs    | 1.76x slower      |
| Rolling Std (10k, window=50)               | 6.07ms  | 6.28ms     | 1.04x slower      |
| Log Returns (5k x 100)                     | 2.85ms  | 2.27ms     | **1.26x faster**  |
| Sharpe Ratio (1k x 50)                     | 91.0µs  | 30.9µs     | **2.95x faster**  |
| Portfolio Variance (1k x 50)               | 108.5µs | 98.5µs     | **1.10x faster**  |
| Attention Scores (128 x 64)                | 91.6µs  | 124.7µs    | 1.36x slower      |
| Layer Normalization (1k x 512)             | 1.36ms  | 1.14ms     | **1.20x faster**  |
| Batch Matmul (32 x 64 x 64)                | 97.9µs  | 247.7µs    | 2.53x slower      |
| Softmax + Cross-Entropy (1k x 100)         | 786.0µs | 810.1µs    | 1.03x slower      |
| 1D Convolution (10k, kernel=51)            | 18.95ms | 523.8µs    | **36.18x faster** |
| Histogram (100k, 100 bins)                 | 724.5µs | 309.8µs    | **2.34x faster**  |
| Percentiles (10k x 50)                     | 11.38ms | 32.24ms    | 2.83x slower      |
| Jacobi Iteration (200 x 200)               | 304.2µs | 577.0µs    | 1.90x slower      |
| Trapezoidal Integration (10k pts)          | 35.5µs  | 31.8µs     | **1.12x faster**  |
| Finite Difference (200 x 200)              | 114.9µs | 133.1µs    | 1.16x slower      |
| Matrix Exponential (100 x 100)             | 203.2µs | 461.8µs    | 2.27x slower      |
| Ridge Regression (5k x 100)                | 515.9µs | 445.3µs    | **1.16x faster**  |
| Gram-Schmidt (200 x 50)                    | 2.16ms  | 452.5µs    | **4.78x faster**  |
| LU Solve (300 x 300)                       | 425.1µs | 393.7µs    | **1.08x faster**  |
| Outer Product (1k x 1k)                    | 417.3µs | 846.1µs    | 2.03x slower      |
| Kronecker Product (50x50 ⊗ 20x20)          | 550.7µs | 1.02ms     | 1.84x slower      |

**Summary: numpy-node wins 17/39 scenarios (44%)**

## Interpretation

- These scenarios combine multiple operations, amortizing N-API overhead
- Compute-heavy operations (SVD, eigendecomposition) benefit from same BLAS/LAPACK backend
- Real applications should see performance closer to these results than micro-benchmarks
