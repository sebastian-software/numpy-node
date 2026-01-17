# Real-World Benchmark Results

These benchmarks represent actual use cases rather than isolated method calls.

- **NumPy**: Python 2.4.1
- **numpy-node**: Node.js v24.12.0
- **Date**: 2026-01-17
- **Platform**: darwin arm64

## Scenarios

| Scenario                                   | NumPy   | numpy-node | Speedup           |
| ------------------------------------------ | ------- | ---------- | ----------------- |
| Data Normalization (10k x 100)             | 2.29ms  | 1.36ms     | **1.69x faster**  |
| Linear Regression (5k x 50)                | 173.1µs | 164.2µs    | **1.05x faster**  |
| PCA via SVD (2k x 100 -> 10)               | 10.18ms | 12.29ms    | 1.21x slower      |
| Correlation Matrix (5k x 50)               | 1.24ms  | 766.2µs    | **1.62x faster**  |
| Neural Net Forward (1k x 256->128->64->10) | 412.5µs | 417.2µs    | 1.01x slower      |
| Covariance + Eigendecomp (3k x 80)         | 4.12ms  | 2.65ms     | **1.56x faster**  |
| Least Squares QR (8k x 100)                | 33.17ms | 37.92ms    | 1.14x slower      |
| Batch Statistics (100 x 1k x 20)           | 7.36ms  | 2.11ms     | **3.48x faster**  |
| Pairwise Distances (1k x 50)               | 2.64ms  | 1.62ms     | **1.63x faster**  |
| Polynomial Fit (1k points, deg 10)         | 146.3µs | 11.2µs     | **13.06x faster** |
| Min-Max Scaling (10k x 100)                | 1.51ms  | 2.55ms     | 1.68x slower      |
| Outer Product Sum (1k x 100)               | 53.6µs  | 205.3µs    | 3.83x slower      |
| Weighted Mean (5k x 200)                   | 157.2µs | 229.7µs    | 1.46x slower      |
| Gradient Descent Step (10k x 50)           | 208.4µs | 948.8µs    | 4.55x slower      |
| Cross-Entropy Loss (1k x 10)               | 41.5µs  | 47.1µs     | 1.13x slower      |
| Cosine Similarity (500 x 128)              | 652.3µs | 1.09ms     | 1.67x slower      |
| K-Means Step (5k x 20, k=10)               | 159.7µs | 223.3µs    | 1.40x slower      |
| Image Filter (256 x 256)                   | 56.3µs  | 240.0µs    | 4.26x slower      |
| Power Iteration (200 x 200)                | 69.4µs  | 123.7µs    | 1.78x slower      |
| Rolling Std (10k, window=50)               | 5.95ms  | 6.18ms     | 1.04x slower      |
| Log Returns (5k x 100)                     | 2.78ms  | 2.08ms     | **1.33x faster**  |
| Sharpe Ratio (1k x 50)                     | 85.7µs  | 30.7µs     | **2.79x faster**  |
| Portfolio Variance (1k x 50)               | 79.5µs  | 156.8µs    | 1.97x slower      |
| Attention Scores (128 x 64)                | 88.5µs  | 107.9µs    | 1.22x slower      |
| Layer Normalization (1k x 512)             | 1.28ms  | 5.89ms     | 4.58x slower      |
| Batch Matmul (32 x 64 x 64)                | 99.8µs  | 222.5µs    | 2.23x slower      |
| Softmax + Cross-Entropy (1k x 100)         | 752.5µs | 749.7µs    | **1.00x faster**  |
| 1D Convolution (10k, kernel=51)            | 18.29ms | 508.8µs    | **35.96x faster** |
| Histogram (100k, 100 bins)                 | 699.7µs | 301.3µs    | **2.32x faster**  |
| Percentiles (10k x 50)                     | 10.73ms | 31.12ms    | 2.90x slower      |
| Jacobi Iteration (200 x 200)               | 311.4µs | 586.2µs    | 1.88x slower      |
| Trapezoidal Integration (10k pts)          | 34.4µs  | 29.9µs     | **1.15x faster**  |
| Finite Difference (200 x 200)              | 109.7µs | 116.3µs    | 1.06x slower      |
| Matrix Exponential (100 x 100)             | 199.4µs | 458.1µs    | 2.30x slower      |
| Ridge Regression (5k x 100)                | 432.0µs | 1.34ms     | 3.10x slower      |
| Gram-Schmidt (200 x 50)                    | 2.09ms  | 445.0µs    | **4.69x faster**  |
| LU Solve (300 x 300)                       | 416.8µs | 398.8µs    | **1.05x faster**  |
| Outer Product (1k x 1k)                    | 392.8µs | 739.4µs    | 1.88x slower      |
| Kronecker Product (50x50 ⊗ 20x20)          | 544.7µs | 919.9µs    | 1.69x slower      |

**Summary: numpy-node wins 15/39 scenarios (38%)**

## Interpretation

- These scenarios combine multiple operations, amortizing N-API overhead
- Compute-heavy operations (SVD, eigendecomposition) benefit from same BLAS/LAPACK backend
- Real applications should see performance closer to these results than micro-benchmarks
