# Real-World Benchmark Results

These benchmarks represent actual use cases rather than isolated method calls.

- **NumPy**: Python 2.4.1
- **numpy-node**: Node.js v24.12.0
- **Date**: 2026-01-19
- **Platform**: darwin arm64

## Scenarios

| Scenario                                   | NumPy    | numpy-node | Speedup            |
| ------------------------------------------ | -------- | ---------- | ------------------ |
| Data Normalization (10k x 100)             | 2.34ms   | 1.31ms     | **1.78x faster**   |
| Linear Regression (5k x 50)                | 185.8µs  | 167.5µs    | ~1.11x             |
| PCA via SVD (2k x 100 -> 10)               | 12.23ms  | 13.55ms    | ~1.11x             |
| Correlation Matrix (5k x 50)               | 1.25ms   | 804.6µs    | **1.55x faster**   |
| Neural Net Forward (1k x 256->128->64->10) | 429.8µs  | 451.5µs    | ~1.05x             |
| Covariance + Eigendecomp (3k x 80)         | 4.28ms   | 3.16ms     | **1.35x faster**   |
| Least Squares QR (8k x 100)                | 40.68ms  | 40.35ms    | ~1.01x             |
| Batch Statistics (100 x 1k x 20)           | 7.64ms   | 2.15ms     | **3.56x faster**   |
| Pairwise Distances (1k x 50)               | 2.75ms   | 1.75ms     | **1.57x faster**   |
| Polynomial Fit (1k points, deg 10)         | 148.5µs  | 12.0µs     | **12.34x faster**  |
| Min-Max Scaling (10k x 100)                | 1.59ms   | 1.86ms     | ~1.17x             |
| Outer Product Sum (1k x 100)               | 53.7µs   | 63.5µs     | 1.18x slower       |
| Weighted Mean (5k x 200)                   | 182.1µs  | 206.8µs    | ~1.14x             |
| Gradient Descent Step (10k x 50)           | 227.1µs  | 238.8µs    | ~1.05x             |
| Cross-Entropy Loss (1k x 10)               | 45.2µs   | 49.0µs     | ~1.08x             |
| Cosine Similarity (500 x 128)              | 651.3µs  | 524.2µs    | **1.24x faster**   |
| K-Means Step (5k x 20, k=10)               | 132.9µs  | 131.3µs    | ~1.01x             |
| Image Filter (256 x 256)                   | 57.2µs   | 61.7µs     | ~1.08x             |
| Power Iteration (200 x 200)                | 69.5µs   | 75.2µs     | ~1.08x             |
| Rolling Std (10k, window=50)               | 6.19ms   | 6.28ms     | ~1.01x             |
| Log Returns (5k x 100)                     | 2.85ms   | 2.21ms     | **1.29x faster**   |
| Sharpe Ratio (1k x 50)                     | 86.1µs   | 31.0µs     | **2.78x faster**   |
| Portfolio Variance (1k x 50)               | 77.5µs   | 99.9µs     | 1.29x slower       |
| Attention Scores (128 x 64)                | 88.7µs   | 95.4µs     | ~1.07x             |
| Layer Normalization (1k x 512)             | 1.32ms   | 1.17ms     | ~1.13x             |
| Batch Matmul (32 x 64 x 64)                | 100.8µs  | 177.7µs    | 1.76x slower       |
| Softmax + Cross-Entropy (1k x 100)         | 759.9µs  | 778.9µs    | ~1.03x             |
| 1D Convolution (10k, kernel=51)            | 19.18ms  | 538.8µs    | **35.59x faster**  |
| Histogram (100k, 100 bins)                 | 710.0µs  | 307.8µs    | **2.31x faster**   |
| Percentiles (10k x 50)                     | 10.95ms  | 10.19ms    | ~1.07x             |
| Jacobi Iteration (200 x 200)               | 310.5µs  | 292.3µs    | ~1.06x             |
| Trapezoidal Integration (10k pts)          | 35.5µs   | 31.2µs     | ~1.14x             |
| Finite Difference (200 x 200)              | 105.2µs  | 85.9µs     | **1.22x faster**   |
| Matrix Exponential (100 x 100)             | 200.2µs  | 200.9µs    | ~1.00x             |
| Ridge Regression (5k x 100)                | 424.3µs  | 442.8µs    | ~1.04x             |
| Gram-Schmidt (200 x 50)                    | 2.20ms   | 455.9µs    | **4.83x faster**   |
| LU Solve (300 x 300)                       | 422.8µs  | 408.1µs    | ~1.04x             |
| Outer Product (1k x 1k)                    | 425.2µs  | 837.8µs    | 1.97x slower       |
| Kronecker Product (50x50 ⊗ 20x20)          | 579.0µs  | 1.06ms     | 1.83x slower       |
| Batch Normalization (256x64x32x32)         | 42.58ms  | 73.98ms    | 1.74x slower       |
| Dropout Forward (1k x 512)                 | 2.28ms   | 22.20ms    | 9.76x slower       |
| Xavier Init (4 layers)                     | 8.98ms   | 40.85ms    | 4.55x slower       |
| Adam Optimizer Step (100k params)          | 297.7µs  | 2.03ms     | 6.80x slower       |
| Confusion Matrix (10k samples)             | 2.50ms   | 22.8µs     | **109.63x faster** |
| Bootstrap Mean (1k samples, 1k resamples)  | 9.83ms   | 6.33ms     | **1.55x faster**   |
| Welch t-Test (500 vs 600)                  | 16.1µs   | 6.5µs      | **2.47x faster**   |
| KDE (1k points, 200 eval)                  | 1.28ms   | 1.38ms     | ~1.08x             |
| Moving Window Stats (10k, w=100)           | 91.96ms  | 1.75ms     | **52.66x faster**  |
| Autocorrelation (5k, lag=100)              | 473.6µs  | 488.8µs    | ~1.03x             |
| N-Body Step (500 bodies)                   | 18.92ms  | 548.0µs    | **34.53x faster**  |
| Heat Equation (100x100, 50 steps)          | 2.20ms   | 3.74ms     | 1.70x slower       |
| Monte Carlo Pi (1M samples)                | 7.91ms   | 15.19ms    | 1.92x slower       |
| Black-Scholes (10k options)                | 193.1µs  | 570.9µs    | 2.96x slower       |
| VaR Historical (1k days, 50 assets)        | 32.3µs   | 190.3µs    | 5.89x slower       |
| EWMA Volatility (2k returns)               | 592.2µs  | 25.0µs     | **23.73x faster**  |
| Matrix Factorization Step (1k×500)         | 56.58ms  | 1.33ms     | **42.68x faster**  |
| TF-IDF (500 docs, 1k terms)                | 874.0µs  | 3.10ms     | 3.55x slower       |
| Bilinear Interpolation (256→512)           | 266.44ms | 1.01ms     | **264.85x faster** |

**Summary: numpy-node wins 29/58 scenarios (50%)**

## Interpretation

- These scenarios combine multiple operations, amortizing N-API overhead
- Compute-heavy operations (SVD, eigendecomposition) benefit from same BLAS/LAPACK backend
- Real applications should see performance closer to these results than micro-benchmarks
