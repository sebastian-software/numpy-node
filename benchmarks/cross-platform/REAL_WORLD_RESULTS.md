# Real-World Benchmark Results

These benchmarks represent actual use cases rather than isolated method calls.

- **NumPy**: Python 2.4.1
- **numpy-node**: Node.js v24.12.0
- **Date**: 2026-01-19
- **Platform**: darwin arm64

## Scenarios

| Scenario                                   | NumPy    | numpy-node | Speedup            |
| ------------------------------------------ | -------- | ---------- | ------------------ |
| Data Normalization (10k x 100)             | 2.17ms   | 1.33ms     | **1.63x faster**   |
| Linear Regression (5k x 50)                | 170.0µs  | 161.9µs    | ~1.05x             |
| PCA via SVD (2k x 100 -> 10)               | 10.36ms  | 13.16ms    | 1.27x slower       |
| Correlation Matrix (5k x 50)               | 941.7µs  | 781.1µs    | **1.21x faster**   |
| Neural Net Forward (1k x 256->128->64->10) | 404.9µs  | 427.8µs    | ~1.06x             |
| Covariance + Eigendecomp (3k x 80)         | 4.12ms   | 3.15ms     | **1.31x faster**   |
| Least Squares QR (8k x 100)                | 34.27ms  | 39.55ms    | ~1.15x             |
| Batch Statistics (100 x 1k x 20)           | 7.39ms   | 2.09ms     | **3.53x faster**   |
| Pairwise Distances (1k x 50)               | 2.64ms   | 1.66ms     | **1.59x faster**   |
| Polynomial Fit (1k points, deg 10)         | 147.2µs  | 11.7µs     | **12.61x faster**  |
| Min-Max Scaling (10k x 100)                | 1.51ms   | 1.70ms     | ~1.12x             |
| Outer Product Sum (1k x 100)               | 53.5µs   | 57.8µs     | ~1.08x             |
| Weighted Mean (5k x 200)                   | 195.5µs  | 208.7µs    | ~1.07x             |
| Gradient Descent Step (10k x 50)           | 222.8µs  | 245.2µs    | ~1.10x             |
| Cross-Entropy Loss (1k x 10)               | 43.1µs   | 46.1µs     | ~1.07x             |
| Cosine Similarity (500 x 128)              | 672.1µs  | 508.4µs    | **1.32x faster**   |
| K-Means Step (5k x 20, k=10)               | 166.3µs  | 120.2µs    | **1.38x faster**   |
| Image Filter (256 x 256)                   | 55.0µs   | 53.8µs     | ~1.02x             |
| Power Iteration (200 x 200)                | 69.0µs   | 76.4µs     | ~1.11x             |
| Rolling Std (10k, window=50)               | 6.43ms   | 6.12ms     | ~1.05x             |
| Log Returns (5k x 100)                     | 2.91ms   | 2.20ms     | **1.33x faster**   |
| Sharpe Ratio (1k x 50)                     | 93.0µs   | 32.8µs     | **2.84x faster**   |
| Portfolio Variance (1k x 50)               | 113.1µs  | 109.4µs    | ~1.03x             |
| Attention Scores (128 x 64)                | 95.1µs   | 96.1µs     | ~1.01x             |
| Layer Normalization (1k x 512)             | 1.32ms   | 1.08ms     | **1.22x faster**   |
| Batch Matmul (32 x 64 x 64)                | 100.8µs  | 145.4µs    | 1.44x slower       |
| Softmax + Cross-Entropy (1k x 100)         | 785.3µs  | 764.3µs    | ~1.03x             |
| 1D Convolution (10k, kernel=51)            | 18.96ms  | 522.1µs    | **36.31x faster**  |
| Histogram (100k, 100 bins)                 | 722.5µs  | 330.1µs    | **2.19x faster**   |
| Percentiles (10k x 50)                     | 11.18ms  | 10.12ms    | ~1.11x             |
| Jacobi Iteration (200 x 200)               | 323.1µs  | 299.6µs    | ~1.08x             |
| Trapezoidal Integration (10k pts)          | 33.4µs   | 31.2µs     | ~1.07x             |
| Finite Difference (200 x 200)              | 112.8µs  | 82.2µs     | **1.37x faster**   |
| Matrix Exponential (100 x 100)             | 210.4µs  | 189.3µs    | ~1.11x             |
| Ridge Regression (5k x 100)                | 426.3µs  | 445.4µs    | ~1.04x             |
| Gram-Schmidt (200 x 50)                    | 2.21ms   | 452.5µs    | **4.87x faster**   |
| LU Solve (300 x 300)                       | 440.0µs  | 396.9µs    | ~1.11x             |
| Outer Product (1k x 1k)                    | 396.2µs  | 747.8µs    | 1.89x slower       |
| Kronecker Product (50x50 ⊗ 20x20)          | 586.5µs  | 929.8µs    | 1.59x slower       |
| Batch Normalization (256x64x32x32)         | 42.80ms  | 76.54ms    | 1.79x slower       |
| Dropout Forward (1k x 512)                 | 2.28ms   | 7.23ms     | 3.18x slower       |
| Xavier Init (4 layers)                     | 8.85ms   | 36.47ms    | 4.12x slower       |
| Adam Optimizer Step (100k params)          | 308.3µs  | 1.05ms     | 3.42x slower       |
| Confusion Matrix (10k samples)             | 2.50ms   | 13.5µs     | **185.66x faster** |
| Bootstrap Mean (1k samples, 1k resamples)  | 9.69ms   | 6.55ms     | **1.48x faster**   |
| Welch t-Test (500 vs 600)                  | 15.6µs   | 6.5µs      | **2.39x faster**   |
| KDE (1k points, 200 eval)                  | 1.26ms   | 1.40ms     | ~1.11x             |
| Moving Window Stats (10k, w=100)           | 89.97ms  | 1.91ms     | **47.08x faster**  |
| Autocorrelation (5k, lag=100)              | 470.2µs  | 476.1µs    | ~1.01x             |
| N-Body Step (500 bodies)                   | 18.56ms  | 530.0µs    | **35.02x faster**  |
| Heat Equation (100x100, 50 steps)          | 2.13ms   | 3.66ms     | 1.72x slower       |
| Monte Carlo Pi (1M samples)                | 8.30ms   | 15.02ms    | 1.81x slower       |
| Black-Scholes (10k options)                | 196.7µs  | 555.3µs    | 2.82x slower       |
| VaR Historical (1k days, 50 assets)        | 40.4µs   | 185.5µs    | 4.59x slower       |
| EWMA Volatility (2k returns)               | 604.4µs  | 25.7µs     | **23.55x faster**  |
| Matrix Factorization Step (1k×500)         | 55.52ms  | 1.32ms     | **42.08x faster**  |
| TF-IDF (500 docs, 1k terms)                | 891.5µs  | 2.95ms     | 3.31x slower       |
| Bilinear Interpolation (256→512)           | 262.64ms | 930.7µs    | **282.18x faster** |

**Summary: numpy-node wins 33/58 scenarios (57%)**

## Interpretation

- These scenarios combine multiple operations, amortizing N-API overhead
- Compute-heavy operations (SVD, eigendecomposition) benefit from same BLAS/LAPACK backend
- Real applications should see performance closer to these results than micro-benchmarks
