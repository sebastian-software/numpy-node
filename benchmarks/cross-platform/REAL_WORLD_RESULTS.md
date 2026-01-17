# Real-World Benchmark Results

These benchmarks represent actual use cases rather than isolated method calls.

- **NumPy**: Python 2.4.1
- **numpy-node**: Node.js v24.12.0
- **Date**: 2026-01-17
- **Platform**: darwin arm64

## Scenarios

| Scenario                                   | NumPy   | numpy-node | Speedup           |
| ------------------------------------------ | ------- | ---------- | ----------------- |
| Data Normalization (10k x 100)             | 2.32ms  | 1.33ms     | **1.75x faster**  |
| Linear Regression (5k x 50)                | 173.1µs | 180.3µs    | 1.04x slower      |
| PCA via SVD (2k x 100 -> 10)               | 11.05ms | 14.38ms    | 1.30x slower      |
| Correlation Matrix (5k x 50)               | 1.02ms  | 758.7µs    | **1.35x faster**  |
| Neural Net Forward (1k x 256->128->64->10) | 428.0µs | 398.9µs    | **1.07x faster**  |
| Covariance + Eigendecomp (3k x 80)         | 4.09ms  | 2.65ms     | **1.54x faster**  |
| Least Squares QR (8k x 100)                | 34.91ms | 37.63ms    | 1.08x slower      |
| Batch Statistics (100 x 1k x 20)           | 7.20ms  | 2.09ms     | **3.44x faster**  |
| Pairwise Distances (1k x 50)               | 2.69ms  | 1.62ms     | **1.66x faster**  |
| Polynomial Fit (1k points, deg 10)         | 146.5µs | 10.8µs     | **13.57x faster** |
| Min-Max Scaling (10k x 100)                | 1.52ms  | 2.52ms     | 1.66x slower      |
| Outer Product Sum (1k x 100)               | 53.7µs  | 61.5µs     | 1.15x slower      |
| Weighted Mean (5k x 200)                   | 151.7µs | 186.4µs    | 1.23x slower      |
| Gradient Descent Step (10k x 50)           | 216.5µs | 288.7µs    | 1.33x slower      |
| Cross-Entropy Loss (1k x 10)               | 43.2µs  | 50.4µs     | 1.17x slower      |
| Cosine Similarity (500 x 128)              | 653.8µs | 1.12ms     | 1.71x slower      |
| K-Means Step (5k x 20, k=10)               | 161.6µs | 214.2µs    | 1.33x slower      |
| Image Filter (256 x 256)                   | 58.2µs  | 229.1µs    | 3.94x slower      |
| Power Iteration (200 x 200)                | 67.6µs  | 123.3µs    | 1.82x slower      |
| Rolling Std (10k, window=50)               | 6.20ms  | 6.14ms     | **1.01x faster**  |
| Log Returns (5k x 100)                     | 2.83ms  | 2.20ms     | **1.28x faster**  |
| Sharpe Ratio (1k x 50)                     | 88.7µs  | 31.4µs     | **2.82x faster**  |
| Portfolio Variance (1k x 50)               | 104.5µs | 155.1µs    | 1.48x slower      |
| Attention Scores (128 x 64)                | 90.4µs  | 112.8µs    | 1.25x slower      |
| Layer Normalization (1k x 512)             | 1.34ms  | 1.04ms     | **1.28x faster**  |
| Batch Matmul (32 x 64 x 64)                | 102.8µs | 232.8µs    | 2.26x slower      |
| Softmax + Cross-Entropy (1k x 100)         | 755.8µs | 742.7µs    | **1.02x faster**  |
| 1D Convolution (10k, kernel=51)            | 18.57ms | 521.5µs    | **35.61x faster** |
| Histogram (100k, 100 bins)                 | 682.3µs | 294.4µs    | **2.32x faster**  |
| Percentiles (10k x 50)                     | 10.90ms | 31.17ms    | 2.86x slower      |
| Jacobi Iteration (200 x 200)               | 309.3µs | 606.7µs    | 1.96x slower      |
| Trapezoidal Integration (10k pts)          | 35.1µs  | 30.0µs     | **1.17x faster**  |
| Finite Difference (200 x 200)              | 119.9µs | 95.1µs     | **1.26x faster**  |
| Matrix Exponential (100 x 100)             | 198.8µs | 452.8µs    | 2.28x slower      |
| Ridge Regression (5k x 100)                | 434.3µs | 443.7µs    | 1.02x slower      |
| Gram-Schmidt (200 x 50)                    | 2.11ms  | 442.9µs    | **4.76x faster**  |
| LU Solve (300 x 300)                       | 424.7µs | 400.3µs    | **1.06x faster**  |
| Outer Product (1k x 1k)                    | 398.5µs | 748.4µs    | 1.88x slower      |
| Kronecker Product (50x50 ⊗ 20x20)          | 539.5µs | 890.8µs    | 1.65x slower      |

**Summary: numpy-node wins 18/39 scenarios (46%)**

## Interpretation

- These scenarios combine multiple operations, amortizing N-API overhead
- Compute-heavy operations (SVD, eigendecomposition) benefit from same BLAS/LAPACK backend
- Real applications should see performance closer to these results than micro-benchmarks
