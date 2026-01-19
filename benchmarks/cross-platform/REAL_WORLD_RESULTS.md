# Real-World Benchmark Results

These benchmarks represent actual use cases rather than isolated method calls.

- **NumPy**: Python 2.4.1
- **numpy-node**: Node.js v24.12.0
- **Date**: 2026-01-19
- **Platform**: darwin arm64

## Scenarios

| Scenario                                   | NumPy    | numpy-node | Speedup            |
| ------------------------------------------ | -------- | ---------- | ------------------ |
| Data Normalization (10k x 100)             | 1.91ms   | 1.22ms     | **1.56x faster**   |
| Linear Regression (5k x 50)                | 170.6µs  | 161.3µs    | ~1.06x             |
| PCA via SVD (2k x 100 -> 10)               | 11.24ms  | 12.34ms    | ~1.10x             |
| Correlation Matrix (5k x 50)               | 1.09ms   | 779.5µs    | **1.39x faster**   |
| Neural Net Forward (1k x 256->128->64->10) | 401.8µs  | 489.8µs    | 1.22x slower       |
| Covariance + Eigendecomp (3k x 80)         | 4.04ms   | 3.37ms     | **1.20x faster**   |
| Least Squares QR (8k x 100)                | 36.16ms  | 40.35ms    | ~1.12x             |
| Batch Statistics (100 x 1k x 20)           | 7.24ms   | 2.10ms     | **3.45x faster**   |
| Pairwise Distances (1k x 50)               | 2.95ms   | 3.29ms     | ~1.12x             |
| Polynomial Fit (1k points, deg 10)         | 165.8µs  | 11.5µs     | **14.47x faster**  |
| Min-Max Scaling (10k x 100)                | 1.23ms   | 2.52ms     | 2.05x slower       |
| Outer Product Sum (1k x 100)               | 53.7µs   | 473.7µs    | 8.83x slower       |
| Weighted Mean (5k x 200)                   | 195.2µs  | 177.5µs    | ~1.10x             |
| Gradient Descent Step (10k x 50)           | 214.1µs  | 2.85ms     | 13.31x slower      |
| Cross-Entropy Loss (1k x 10)               | 45.2µs   | 51.3µs     | ~1.13x             |
| Cosine Similarity (500 x 128)              | 693.0µs  | 564.5µs    | **1.23x faster**   |
| K-Means Step (5k x 20, k=10)               | 171.1µs  | 217.1µs    | 1.27x slower       |
| Image Filter (256 x 256)                   | 58.5µs   | 53.9µs     | ~1.09x             |
| Power Iteration (200 x 200)                | 70.0µs   | 107.6µs    | 1.54x slower       |
| Rolling Std (10k, window=50)               | 6.10ms   | 6.06ms     | ~1.01x             |
| Log Returns (5k x 100)                     | 2.78ms   | 2.13ms     | **1.30x faster**   |
| Sharpe Ratio (1k x 50)                     | 86.3µs   | 30.9µs     | **2.79x faster**   |
| Portfolio Variance (1k x 50)               | 69.3µs   | 320.4µs    | 4.62x slower       |
| Attention Scores (128 x 64)                | 87.8µs   | 105.1µs    | 1.20x slower       |
| Layer Normalization (1k x 512)             | 1.31ms   | 1.48ms     | ~1.13x             |
| Batch Matmul (32 x 64 x 64)                | 100.3µs  | 144.6µs    | 1.44x slower       |
| Softmax + Cross-Entropy (1k x 100)         | 737.0µs  | 813.8µs    | ~1.10x             |
| 1D Convolution (10k, kernel=51)            | 18.84ms  | 520.8µs    | **36.17x faster**  |
| Histogram (100k, 100 bins)                 | 699.9µs  | 302.1µs    | **2.32x faster**   |
| Percentiles (10k x 50)                     | 10.88ms  | 9.93ms     | ~1.10x             |
| Trapezoidal Integration (10k pts)          | 35.0µs   | 31.9µs     | ~1.10x             |
| Finite Difference (200 x 200)              | 96.0µs   | 261.6µs    | 2.73x slower       |
| Ridge Regression (5k x 100)                | 478.4µs  | 5.33ms     | 11.15x slower      |
| LU Solve (300 x 300)                       | 422.3µs  | 411.0µs    | ~1.03x             |
| Outer Product (1k x 1k)                    | 384.6µs  | 736.6µs    | 1.92x slower       |
| Kronecker Product (50x50 ⊗ 20x20)          | 553.0µs  | 907.3µs    | 1.64x slower       |
| Batch Normalization (256x64x32x32)         | 42.23ms  | 95.03ms    | 2.25x slower       |
| Dropout Forward (1k x 512)                 | 2.27ms   | 7.37ms     | 3.24x slower       |
| Xavier Init (4 layers)                     | 8.66ms   | 36.36ms    | 4.20x slower       |
| Adam Optimizer Step (100k params)          | 272.7µs  | 1.55ms     | 5.70x slower       |
| Confusion Matrix (10k samples)             | 2.46ms   | 14.5µs     | **169.25x faster** |
| Bootstrap Mean (1k samples, 1k resamples)  | 9.63ms   | 6.36ms     | **1.51x faster**   |
| Welch t-Test (500 vs 600)                  | 16.3µs   | 6.9µs      | **2.36x faster**   |
| KDE (1k points, 200 eval)                  | 1.27ms   | 1.40ms     | ~1.10x             |
| Moving Window Stats (10k, w=100)           | 91.06ms  | 1.63ms     | **55.96x faster**  |
| Autocorrelation (5k, lag=100)              | 475.9µs  | 473.5µs    | ~1.01x             |
| FFT Signal Processing (4k samples)         | 83.0µs   | 539.2µs    | 6.49x slower       |
| Batched Attention (32x64x64)               | 751.8µs  | 7.78ms     | 10.34x slower      |
| N-Body Step (500 bodies)                   | 7.75ms   | 848.1µs    | **9.14x faster**   |
| Heat Equation (100x100, 50 steps)          | 2.12ms   | 3.78ms     | 1.78x slower       |
| Black-Scholes (10k options)                | 196.5µs  | 570.0µs    | 2.90x slower       |
| VaR Historical (1k days, 50 assets)        | 42.5µs   | 179.5µs    | 4.23x slower       |
| EWMA Volatility (2k returns)               | 612.7µs  | 25.4µs     | **24.11x faster**  |
| Matrix Factorization Step (1k×500)         | 54.77ms  | 1.36ms     | **40.15x faster**  |
| TF-IDF (500 docs, 1k terms)                | 885.2µs  | 3.03ms     | 3.42x slower       |
| Bilinear Interpolation (256→512)           | 261.05ms | 1.06ms     | **245.77x faster** |

**Summary: numpy-node wins 26/56 scenarios (46%)**

## Interpretation

- These scenarios combine multiple operations, amortizing N-API overhead
- Compute-heavy operations (SVD, eigendecomposition) benefit from same BLAS/LAPACK backend
- Real applications should see performance closer to these results than micro-benchmarks
