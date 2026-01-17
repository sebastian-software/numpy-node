# Real-World Benchmark Results

These benchmarks represent actual use cases rather than isolated method calls.

- **NumPy**: Python 2.4.1
- **numpy-node**: Node.js v24.12.0
- **Date**: 2026-01-17
- **Platform**: darwin arm64

## Scenarios

| Scenario                                   | NumPy   | numpy-node | Speedup           |
| ------------------------------------------ | ------- | ---------- | ----------------- |
| Data Normalization (10k x 100)             | 2.34ms  | 1.29ms     | **1.81x faster**  |
| Linear Regression (5k x 50)                | 172.0µs | 161.6µs    | **1.06x faster**  |
| PCA via SVD (2k x 100 -> 10)               | 10.39ms | 12.04ms    | 1.16x slower      |
| Correlation Matrix (5k x 50)               | 1.03ms  | 761.5µs    | **1.35x faster**  |
| Neural Net Forward (1k x 256->128->64->10) | 397.1µs | 409.5µs    | 1.03x slower      |
| Covariance + Eigendecomp (3k x 80)         | 4.01ms  | 2.59ms     | **1.55x faster**  |
| Least Squares QR (8k x 100)                | 34.64ms | 37.51ms    | 1.08x slower      |
| Batch Statistics (100 x 1k x 20)           | 7.17ms  | 2.03ms     | **3.53x faster**  |
| Pairwise Distances (1k x 50)               | 2.64ms  | 1.62ms     | **1.63x faster**  |
| Polynomial Fit (1k points, deg 10)         | 146.3µs | 11.8µs     | **12.36x faster** |
| Min-Max Scaling (10k x 100)                | 1.51ms  | 2.54ms     | 1.68x slower      |
| Outer Product Sum (1k x 100)               | 53.6µs  | 198.9µs    | 3.71x slower      |
| Weighted Mean (5k x 200)                   | 183.1µs | 173.0µs    | **1.06x faster**  |
| Gradient Descent Step (10k x 50)           | 208.9µs | 903.9µs    | 4.33x slower      |
| Cross-Entropy Loss (1k x 10)               | 41.8µs  | 49.8µs     | 1.19x slower      |
| Cosine Similarity (500 x 128)              | 653.2µs | 1.10ms     | 1.68x slower      |
| K-Means Step (5k x 20, k=10)               | 159.3µs | 217.5µs    | 1.37x slower      |
| Image Filter (256 x 256)                   | 57.6µs  | 260.7µs    | 4.53x slower      |
| Power Iteration (200 x 200)                | 69.5µs  | 122.7µs    | 1.76x slower      |
| Rolling Std (10k, window=50)               | 6.08ms  | 6.05ms     | **1.01x faster**  |
| Log Returns (5k x 100)                     | 2.76ms  | 2.11ms     | **1.31x faster**  |
| Sharpe Ratio (1k x 50)                     | 88.6µs  | 30.7µs     | **2.88x faster**  |
| Portfolio Variance (1k x 50)               | 75.9µs  | 146.7µs    | 1.93x slower      |
| Attention Scores (128 x 64)                | 88.3µs  | 109.8µs    | 1.24x slower      |
| Layer Normalization (1k x 512)             | 1.27ms  | 5.91ms     | 4.65x slower      |
| Batch Matmul (32 x 64 x 64)                | 100.4µs | 237.5µs    | 2.37x slower      |
| Softmax + Cross-Entropy (1k x 100)         | 763.3µs | 773.2µs    | 1.01x slower      |
| 1D Convolution (10k, kernel=51)            | 18.50ms | 504.9µs    | **36.64x faster** |
| Histogram (100k, 100 bins)                 | 659.6µs | 301.3µs    | **2.19x faster**  |
| Percentiles (10k x 50)                     | 10.84ms | 30.44ms    | 2.81x slower      |
| Jacobi Iteration (200 x 200)               | 306.8µs | 2.38ms     | 7.75x slower      |
| Trapezoidal Integration (10k pts)          | 34.4µs  | 30.0µs     | **1.15x faster**  |
| Finite Difference (200 x 200)              | 108.9µs | 137.3µs    | 1.26x slower      |
| Matrix Exponential (100 x 100)             | 198.4µs | 457.3µs    | 2.30x slower      |
| Ridge Regression (5k x 100)                | 425.7µs | 1.28ms     | 3.01x slower      |
| Gram-Schmidt (200 x 50)                    | 2.09ms  | 436.9µs    | **4.79x faster**  |
| LU Solve (300 x 300)                       | 417.9µs | 395.9µs    | **1.06x faster**  |
| Outer Product (1k x 1k)                    | 393.5µs | 739.9µs    | 1.88x slower      |
| Kronecker Product (50x50 ⊗ 20x20)          | 541.5µs | 911.4µs    | 1.68x slower      |

**Summary: numpy-node wins 16/39 scenarios (41%)**

## Interpretation

- These scenarios combine multiple operations, amortizing N-API overhead
- Compute-heavy operations (SVD, eigendecomposition) benefit from same BLAS/LAPACK backend
- Real applications should see performance closer to these results than micro-benchmarks
