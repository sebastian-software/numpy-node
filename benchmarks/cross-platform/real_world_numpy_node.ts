#!/usr/bin/env npx tsx
/**
 * Real-world numpy-node benchmark scenarios.
 * These represent actual use cases rather than isolated method calls.
 */

import {
  array,
  zeros,
  linspace,
  add,
  subtract,
  multiply,
  divide,
  matmul,
  dot,
  sqrt,
  exp,
  log,
  sin,
  cos,
  sum,
  mean,
  std,
  min,
  max,
  svd,
  qr,
  eig,
  solve,
  eye,
  normal_equations,
  zscore,
  corrcoef,
  gram_matrix,
  softmax,
  pdist_sq,
  NDArray,
} from '../../src/index.js';

interface BenchmarkResult {
  name: string;
  mean: number;
  min: number;
  max: number;
  median: number;
  p95: number;
  iterations: number;
}

function benchmark(name: string, fn: () => void, warmup = 3, iterations = 50): BenchmarkResult {
  // Warmup
  for (let i = 0; i < warmup; i++) {
    fn();
  }

  // Timed runs
  const times: number[] = [];
  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    fn();
    const end = performance.now();
    times.push(end - start);
  }

  times.sort((a, b) => a - b);

  return {
    name,
    mean: times.reduce((a, b) => a + b, 0) / times.length,
    min: times[0],
    max: times[times.length - 1],
    median: times[Math.floor(times.length / 2)],
    p95: times[Math.floor(times.length * 0.95)],
    iterations,
  };
}

// Helper to create random matrix
function randomMatrix(rows: number, cols: number): NDArray {
  const data: number[][] = [];
  for (let i = 0; i < rows; i++) {
    const row: number[] = [];
    for (let j = 0; j < cols; j++) {
      // Box-Muller transform for normal distribution
      const u1 = Math.random();
      const u2 = Math.random();
      row.push(Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2));
    }
    data.push(row);
  }
  return array(data);
}

function randomVector(n: number): NDArray {
  const data: number[] = [];
  for (let i = 0; i < n; i++) {
    const u1 = Math.random();
    const u2 = Math.random();
    data.push(Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2));
  }
  return array(data);
}

// ============================================
// Real-World Scenarios
// ============================================

function scenarioDataNormalization() {
  /**
   * Z-Score normalization of a dataset.
   * Common in ML preprocessing.
   * Uses fused zscore for optimal performance.
   */
  // Simulate a dataset: 10000 samples, 100 features
  const data = randomMatrix(10000, 100);

  return function normalize() {
    // Fused operation: computes mean, std, and normalizes in one native call
    const normalized = zscore(data, 0);
    return normalized;
  };
}

function scenarioLinearRegression() {
  /**
   * Ordinary Least Squares linear regression.
   * β = (X'X)^(-1) X'y
   * Uses fused normal_equations for optimal performance.
   */
  // 5000 samples, 50 features
  const X = randomMatrix(5000, 50);
  const y = randomMatrix(5000, 1); // Column vector

  return function fit() {
    // Fused operation: computes (X'X)^(-1) X'y in one native call
    const beta = normal_equations(X, y);
    return beta;
  };
}

function scenarioPCA() {
  /**
   * Principal Component Analysis - dimensionality reduction.
   * Uses SVD for numerical stability.
   */
  // 2000 samples, 100 features -> reduce to 10 components
  const data = randomMatrix(2000, 100);
  const nComponents = 10;

  return function pca() {
    // Center the data
    const μ = mean(data, 0) as NDArray;
    const centered = subtract(data, μ);

    // SVD - returns { u, s, vh } where vh is V transposed
    const { vh } = svd(centered);

    // Get first n_components rows of vh (= first n components)
    // For now, we'll use the full vh and let the matmul handle it
    // In a real implementation, we'd slice vh

    // Project to lower dimensions: X @ vh.T[:, :n_components]
    const V = vh.T;
    const projected = matmul(centered, V);
    return projected;
  };
}

function scenarioCorrelationMatrix() {
  /**
   * Compute correlation matrix for a dataset.
   * Common in exploratory data analysis.
   * Uses fused corrcoef for optimal performance.
   */
  // 5000 samples, 50 variables
  const data = randomMatrix(5000, 50);

  return function correlation() {
    // Fused operation: standardizes and computes X'X / (n-1) in one native call
    const corr = corrcoef(data);
    return corr;
  };
}

function scenarioMatrixChain() {
  /**
   * Chain of matrix multiplications - common in neural network forward pass.
   * Uses fused softmax for optimal performance.
   */
  // Simulate layer dimensions
  const X = randomMatrix(1000, 256); // Input: 1000 samples, 256 features
  const W1 = randomMatrix(256, 128); // Layer 1 weights
  const W2 = randomMatrix(128, 64); // Layer 2 weights
  const W3 = randomMatrix(64, 10); // Output layer

  return function forward() {
    // Chain of matrix multiplications
    const h1 = matmul(X, W1);
    const h2 = matmul(h1, W2);
    const out = matmul(h2, W3);

    // Fused softmax (exp, sum, divide in one native call)
    const result = softmax(out);

    return result;
  };
}

function scenarioCovarianceEigendecomp() {
  /**
   * Compute covariance matrix and its eigendecomposition.
   * Used in PCA, factor analysis, etc.
   */
  // 3000 samples, 80 features
  const data = randomMatrix(3000, 80);
  const n = 3000;

  return function eigenAnalysis() {
    // Center
    const μ = mean(data, 0) as NDArray;
    const centered = subtract(data, μ);

    // Covariance
    const Xt = centered.T;
    const XtX = matmul(Xt, centered);
    const cov = divide(XtX, n - 1);

    // Eigendecomposition
    const { values, vectors } = eig(cov);
    return { values, vectors };
  };
}

function scenarioLeastSquaresQR() {
  /**
   * Solve overdetermined system using QR decomposition.
   * More numerically stable than normal equations.
   */
  // 8000 equations, 100 unknowns
  const A = randomMatrix(8000, 100);
  const b = randomMatrix(8000, 1); // Column vector

  return function solveQR() {
    const { q, r } = qr(A);
    // x = R^(-1) Q' b
    const Qt = q.T;
    const Qtb = matmul(Qt, b);
    const x = solve(r, Qtb);
    return x;
  };
}

function scenarioBatchStatistics() {
  /**
   * Compute various statistics on batched data.
   * Common in data analysis pipelines.
   */
  // Create 100 batches of data
  const batches: NDArray[] = [];
  for (let i = 0; i < 100; i++) {
    batches.push(randomMatrix(1000, 20));
  }

  return function computeStats() {
    const means: NDArray[] = [];
    const stds: NDArray[] = [];

    // Per-batch statistics
    for (const batch of batches) {
      means.push(mean(batch, 0) as NDArray);
      stds.push(std(batch, 0) as NDArray);
    }

    // Global mean of means (simplified)
    let globalMean = means[0];
    for (let i = 1; i < means.length; i++) {
      globalMean = add(globalMean, means[i]);
    }
    globalMean = divide(globalMean, 100);

    return globalMean;
  };
}

function scenarioDistanceMatrix() {
  /**
   * Compute pairwise Euclidean distances.
   * Used in clustering, nearest neighbors, etc.
   * Uses gram_matrix for efficient X @ X.T computation.
   */
  // 1000 points in 50-dimensional space
  const points = randomMatrix(1000, 50);

  return function distances() {
    // Fused operation: computes ||x_i - x_j||^2 for all pairs in one native call
    const distSq = pdist_sq(points);
    return distSq;
  };
}

function scenarioPolynomialFit() {
  /**
   * Fit a polynomial using Vandermonde matrix.
   */
  // 1000 data points, degree 10 polynomial
  const xArr = linspace(-5, 5, 1000);
  const xData = xArr.data as Float64Array;

  // Create noisy sine wave as column vector
  const yData: number[][] = [];
  for (let i = 0; i < 1000; i++) {
    yData.push([Math.sin(xData[i]) + 0.1 * (Math.random() - 0.5) * 2]);
  }
  const y = array(yData);

  const degree = 10;

  // Build Vandermonde matrix
  const vData: number[][] = [];
  for (let i = 0; i < 1000; i++) {
    const row: number[] = [];
    for (let j = degree; j >= 0; j--) {
      row.push(Math.pow(xData[i], j));
    }
    vData.push(row);
  }
  const V = array(vData);

  return function fitPoly() {
    // Solve least squares via fused normal equations
    const coeffs = normal_equations(V, y);

    // Evaluate
    const yPred = matmul(V, coeffs);
    return { coeffs, yPred };
  };
}

// ============================================
// Additional Real-World Scenarios
// ============================================

function scenarioMovingAverage() {
  /**
   * Simple moving average - common in time series analysis.
   * Uses cumsum approach for efficiency.
   */
  // 100k time series, window size 50
  const data = randomVector(100000);
  const windowSize = 50;

  return function movingAvg() {
    // Cumulative sum approach: MA[i] = (cumsum[i+w] - cumsum[i]) / w
    const cumsum = zeros([100001]);
    let acc = 0;
    for (let i = 0; i < 100000; i++) {
      acc += data.toFlatArray()[i];
      // Note: This is simplified - in practice we'd use a native cumsum
    }
    return acc; // Return something to prevent optimization
  };
}

function scenarioMinMaxScaling() {
  /**
   * Min-max feature scaling to [0, 1] range.
   * Common in ML preprocessing.
   */
  const data = randomMatrix(10000, 100);

  return function scale() {
    const minVals = min(data, 0) as NDArray;
    const maxVals = max(data, 0) as NDArray;
    const range = subtract(maxVals, minVals);
    const centered = subtract(data, minVals);
    const scaled = divide(centered, range);
    return scaled;
  };
}

function scenarioOuterProductSum() {
  /**
   * Sum of outer products - used in covariance estimation.
   * Computes sum of x_i * x_i^T for all samples.
   */
  const vectors = randomMatrix(1000, 100); // 1000 vectors of dim 100

  return function outerSum() {
    // X^T @ X gives sum of outer products
    const Xt = vectors.T;
    const result = matmul(Xt, vectors);
    return result;
  };
}

function scenarioWeightedMean() {
  /**
   * Weighted mean computation.
   * Common in ensemble methods and attention mechanisms.
   */
  const values = randomMatrix(5000, 200);
  const weights = randomVector(5000);

  return function weightedMean() {
    // Normalize weights
    const totalWeight = sum(weights) as number;
    const normWeights = divide(weights, totalWeight);

    // Weighted sum: (w.T @ values) - reshape for matmul
    const wReshaped = normWeights.reshape([1, 5000]);
    const result = matmul(wReshaped, values);
    return result;
  };
}

function scenarioGradientDescent() {
  /**
   * Single gradient descent step for linear regression.
   * X: features, y: target, theta: parameters
   */
  const X = randomMatrix(10000, 50);
  const y = randomMatrix(10000, 1);
  const theta = randomMatrix(50, 1);
  const learningRate = 0.01;
  const m = 10000;

  return function gradientStep() {
    // predictions = X @ theta
    const predictions = matmul(X, theta);

    // errors = predictions - y
    const errors = subtract(predictions, y);

    // gradient = (1/m) * X.T @ errors
    const Xt = X.T;
    const grad = matmul(Xt, errors);
    const scaledGrad = divide(grad, m);

    // new_theta = theta - lr * gradient
    const update = multiply(scaledGrad, learningRate);
    const newTheta = subtract(theta, update);
    return newTheta;
  };
}

function scenarioCrossEntropy() {
  /**
   * Cross-entropy loss computation.
   * Common in classification tasks.
   */
  const predictions = randomMatrix(1000, 10); // 1000 samples, 10 classes
  const targets = randomMatrix(1000, 10); // One-hot encoded

  return function crossEntropy() {
    // Softmax predictions
    const softmaxPreds = softmax(predictions);

    // Cross-entropy: -sum(targets * log(preds))
    const logPreds = multiply(targets, softmaxPreds); // element-wise
    const ce = sum(logPreds) as number;
    return -ce;
  };
}

function scenarioCosineSimilarity() {
  /**
   * Pairwise cosine similarity matrix.
   * Used in recommendation systems and NLP.
   */
  const vectors = randomMatrix(500, 128); // 500 vectors of dim 128

  return function cosineSim() {
    // Normalize vectors: x / ||x||
    const sq = multiply(vectors, vectors);
    const norms = sqrt(sum(sq, 1) as NDArray);

    // Similarity = (X @ X.T) / (||x|| * ||y||)
    const dotProds = gram_matrix(vectors);

    // Outer product of norms
    const normCol = norms.reshape([500, 1]);
    const normRow = norms.reshape([1, 500]);
    const normOuter = matmul(normCol, normRow);

    const similarity = divide(dotProds, normOuter);
    return similarity;
  };
}

function scenarioKMeansStep() {
  /**
   * Single K-means iteration: assign + update centroids.
   */
  const data = randomMatrix(5000, 20); // 5000 points, 20 dims
  const centroids = randomMatrix(10, 20); // 10 clusters

  return function kmeansStep() {
    // Compute distances to all centroids
    // dist[i,k] = ||x_i - c_k||^2

    // For each point, compute squared distance to each centroid
    // Using: ||x - c||^2 = ||x||^2 + ||c||^2 - 2*x.c

    const xSq = sum(multiply(data, data), 1) as NDArray; // [5000]
    const cSq = sum(multiply(centroids, centroids), 1) as NDArray; // [10]

    // data @ centroids.T = [5000, 10]
    const centroidsT = centroids.T;
    const dotProds = matmul(data, centroidsT);

    // Would need broadcasting for full distance computation
    // Simplified: just return dot products
    return dotProds;
  };
}

function scenarioImageFilter() {
  /**
   * Simple 3x3 filter applied to an image (as matrix operation).
   * Simplified convolution using matrix multiplication.
   */
  // Simulate 256x256 grayscale image
  const image = randomMatrix(256, 256);

  return function filter() {
    // Apply simple averaging by reshaping and matrix ops
    // This is a simplified version - real conv would be different

    // Just do element-wise operations as proxy
    const blurred = add(image, multiply(image, 0.1));
    const sharpened = subtract(multiply(image, 2), blurred);
    return sharpened;
  };
}

function scenarioPowerIteration() {
  /**
   * Power iteration for dominant eigenvalue.
   * Used in PageRank and spectral methods.
   */
  const A = randomMatrix(200, 200);
  // Make symmetric positive definite
  const At = A.T;
  const M = matmul(At, A); // M = A^T A is symmetric PSD

  let v = randomVector(200);

  return function powerIter() {
    // 10 iterations of power method
    for (let i = 0; i < 10; i++) {
      // v = M @ v
      const vCol = v.reshape([200, 1]);
      const Mv = matmul(M, vCol);

      // Normalize
      const norm = Math.sqrt(sum(multiply(Mv, Mv)) as number);
      v = divide(Mv, norm).reshape([200]);
    }
    return v;
  };
}

// ============================================
// Additional Real-World Scenarios
// ============================================

function scenarioRollingStd() {
  /**
   * Rolling standard deviation - common in finance/time series.
   * Compute std over sliding windows.
   */
  const data = randomVector(10000);
  const windowSize = 50;

  return function rollingStd() {
    const n = data.shape[0] - windowSize + 1;
    const results: number[] = [];

    // Sliding window std
    for (let i = 0; i < n; i += 10) {
      // Sample every 10th window for speed
      let sum = 0;
      let sumSq = 0;
      for (let j = 0; j < windowSize; j++) {
        const idx = i + j;
        const val = (data.data as Float64Array)[idx];
        sum += val;
        sumSq += val * val;
      }
      const mean = sum / windowSize;
      const variance = sumSq / windowSize - mean * mean;
      results.push(Math.sqrt(variance));
    }
    return results;
  };
}

function scenarioLogReturns() {
  /**
   * Calculate log returns from price series.
   * Common in quantitative finance.
   */
  // Simulate 5000 days of 100 stock prices
  const prices = randomMatrix(5000, 100);

  return function logReturns() {
    // log(P_t / P_{t-1}) = log(P_t) - log(P_{t-1})
    const logPrices = log(prices);
    const logData = logPrices.data as Float64Array;
    const rows = 5000;
    const cols = 100;

    // Compute differences (simplified - just the computation)
    const returns: number[] = new Array((rows - 1) * cols);
    for (let i = 1; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        returns[(i - 1) * cols + j] = logData[i * cols + j] - logData[(i - 1) * cols + j];
      }
    }
    return returns;
  };
}

function scenarioSharpeRatio() {
  /**
   * Calculate Sharpe ratio for portfolio returns.
   * Sharpe = (mean(returns) - rf) / std(returns)
   */
  // 1000 days of returns for 50 assets
  const returns = randomMatrix(1000, 50);
  const riskFreeRate = 0.02 / 252; // Daily risk-free rate

  return function sharpeRatio() {
    const meanReturns = mean(returns, 0) as NDArray;
    const stdReturns = std(returns, 0) as NDArray;

    // Excess returns
    const excessReturns = subtract(meanReturns, riskFreeRate);

    // Sharpe ratios
    const sharpe = divide(excessReturns, stdReturns);
    return sharpe;
  };
}

function scenarioPortfolioVariance() {
  /**
   * Calculate portfolio variance using covariance matrix.
   * var(portfolio) = w' @ Cov @ w
   */
  // 50 assets, 1000 observations
  const returns = randomMatrix(1000, 50);
  const weights = randomVector(50);

  return function portfolioVar() {
    // Compute covariance matrix
    const μ = mean(returns, 0) as NDArray;
    const centered = subtract(returns, μ);
    const cov = divide(matmul(centered.T, centered), 999);

    // Portfolio variance: w' @ Cov @ w
    const wCol = weights.reshape([50, 1]);
    const covW = matmul(cov, wCol);
    const variance = matmul(weights.reshape([1, 50]), covW);
    return variance;
  };
}

function scenarioAttentionScores() {
  /**
   * Scaled dot-product attention scores.
   * scores = softmax(Q @ K.T / sqrt(d_k))
   */
  // Batch of 32, sequence length 128, dimension 64
  const Q = randomMatrix(128, 64);
  const K = randomMatrix(128, 64);
  const scale = 1 / Math.sqrt(64);

  return function attention() {
    // Q @ K.T
    const Kt = K.T;
    const scores = matmul(Q, Kt);

    // Scale
    const scaled = multiply(scores, scale);

    // Softmax (row-wise would be ideal, but we do global for simplicity)
    const attended = softmax(scaled);
    return attended;
  };
}

function scenarioLayerNorm() {
  /**
   * Layer normalization - normalize across features.
   * Used in transformers.
   */
  // 1000 samples, 512 features
  const data = randomMatrix(1000, 512);
  const gamma = randomVector(512); // Scale
  const beta = randomVector(512); // Shift

  return function layerNorm() {
    // Normalize along axis=1 (each row independently)
    const normalized = zscore(data, 1);

    // Apply scale and shift: gamma * normalized + beta
    const scaled = multiply(normalized, gamma);
    const result = add(scaled, beta);
    return result;
  };
}

function scenarioBatchMatmul() {
  /**
   * Batch matrix multiplication - multiple small matrices.
   * Common in attention mechanisms.
   */
  // 32 matrices of 64x64
  const batchSize = 32;
  const As: NDArray[] = [];
  const Bs: NDArray[] = [];
  for (let i = 0; i < batchSize; i++) {
    As.push(randomMatrix(64, 64));
    Bs.push(randomMatrix(64, 64));
  }

  return function batchMM() {
    const results: NDArray[] = [];
    for (let i = 0; i < batchSize; i++) {
      results.push(matmul(As[i], Bs[i]));
    }
    return results;
  };
}

function scenarioConvolution1D() {
  /**
   * 1D convolution - signal processing.
   * Using matrix multiplication approach.
   */
  const signal = randomVector(10000);
  const kernel = randomVector(51); // Kernel size 51

  return function conv1d() {
    // Simple convolution via loop (real impl would use FFT or im2col)
    const signalData = signal.data as Float64Array;
    const kernelData = kernel.data as Float64Array;
    const outLen = signalData.length - kernelData.length + 1;
    const result = new Float64Array(outLen);

    for (let i = 0; i < outLen; i++) {
      let sum = 0;
      for (let j = 0; j < kernelData.length; j++) {
        sum += signalData[i + j] * kernelData[j];
      }
      result[i] = sum;
    }
    return result;
  };
}

function scenarioHistogram() {
  /**
   * Compute histogram bins.
   * Common in data analysis.
   */
  const data = randomVector(100000);
  const numBins = 100;

  return function histogram() {
    const dataArr = data.data as Float64Array;
    const bins = new Float64Array(numBins);

    // Find min/max
    let minVal = dataArr[0];
    let maxVal = dataArr[0];
    for (let i = 1; i < dataArr.length; i++) {
      if (dataArr[i] < minVal) minVal = dataArr[i];
      if (dataArr[i] > maxVal) maxVal = dataArr[i];
    }

    const binWidth = (maxVal - minVal) / numBins;

    // Count
    for (let i = 0; i < dataArr.length; i++) {
      const binIdx = Math.min(Math.floor((dataArr[i] - minVal) / binWidth), numBins - 1);
      bins[binIdx]++;
    }
    return bins;
  };
}

function scenarioPercentiles() {
  /**
   * Calculate multiple percentiles.
   * Common in statistical analysis.
   */
  const data = randomMatrix(10000, 50);
  const percentiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99];

  return function computePercentiles() {
    const dataArr = data.data as Float64Array;
    const cols = 50;
    const rows = 10000;
    const results: number[][] = [];

    for (let col = 0; col < cols; col++) {
      // Extract column
      const column = new Float64Array(rows);
      for (let row = 0; row < rows; row++) {
        column[row] = dataArr[row * cols + col];
      }

      // Sort
      column.sort();

      // Get percentiles
      const colPercentiles = percentiles.map((p) => {
        const idx = Math.floor(p * (rows - 1));
        return column[idx];
      });
      results.push(colPercentiles);
    }
    return results;
  };
}

function scenarioJacobiIteration() {
  /**
   * Jacobi iterative method for solving Ax = b.
   * Common in numerical linear algebra.
   */
  // Create diagonally dominant matrix
  const n = 200;
  const A = randomMatrix(n, n);
  const b = randomVector(n);

  // Make diagonally dominant
  const Adata = A.data as Float64Array;
  for (let i = 0; i < n; i++) {
    let rowSum = 0;
    for (let j = 0; j < n; j++) {
      if (i !== j) rowSum += Math.abs(Adata[i * n + j]);
    }
    Adata[i * n + i] = rowSum + 1;
  }

  return function jacobi() {
    const bData = b.data as Float64Array;
    let x = new Float64Array(n); // Initial guess = 0
    const xNew = new Float64Array(n);

    // 50 iterations
    for (let iter = 0; iter < 50; iter++) {
      for (let i = 0; i < n; i++) {
        let sigma = 0;
        for (let j = 0; j < n; j++) {
          if (i !== j) {
            sigma += Adata[i * n + j] * x[j];
          }
        }
        xNew[i] = (bData[i] - sigma) / Adata[i * n + i];
      }
      // Swap
      const tmp = x;
      x = xNew;
      // xNew = tmp; // Not needed, will be overwritten
    }
    return x;
  };
}

function scenarioTrapezoidalIntegration() {
  /**
   * Numerical integration using trapezoidal rule.
   */
  // Integrate sin(x) from 0 to pi (should be ~2)
  const n = 10000;
  const x = linspace(0, Math.PI, n);

  return function integrate() {
    const y = sin(x);
    const yData = y.data as Float64Array;
    const h = Math.PI / (n - 1);

    // Trapezoidal rule: h/2 * (y[0] + 2*sum(y[1:-1]) + y[-1])
    let integral = (yData[0] + yData[n - 1]) / 2;
    for (let i = 1; i < n - 1; i++) {
      integral += yData[i];
    }
    integral *= h;
    return integral;
  };
}

function scenarioFiniteDifference() {
  /**
   * Compute gradient using finite differences.
   * Common in optimization when analytical gradients unavailable.
   */
  // 2D function evaluated on grid
  const gridSize = 200;
  const f = randomMatrix(gridSize, gridSize);

  return function gradient() {
    const fData = f.data as Float64Array;
    const h = 1.0;

    // Compute partial derivatives
    const dfdx = new Float64Array(gridSize * gridSize);
    const dfdy = new Float64Array(gridSize * gridSize);

    // Central differences for interior points
    for (let i = 1; i < gridSize - 1; i++) {
      for (let j = 1; j < gridSize - 1; j++) {
        const idx = i * gridSize + j;
        dfdx[idx] = (fData[idx + 1] - fData[idx - 1]) / (2 * h);
        dfdy[idx] = (fData[(i + 1) * gridSize + j] - fData[(i - 1) * gridSize + j]) / (2 * h);
      }
    }
    return { dfdx, dfdy };
  };
}

function scenarioMatrixExponential() {
  /**
   * Approximate matrix exponential using Taylor series.
   * exp(A) ≈ I + A + A²/2! + A³/3! + ...
   */
  const A = randomMatrix(100, 100);
  // Scale down to ensure convergence
  const scaledA = multiply(A, 0.01);

  return function matrixExp() {
    const n = 100;
    let result = eye(n);
    let term = eye(n);

    // 10 terms of Taylor series
    for (let k = 1; k <= 10; k++) {
      term = divide(matmul(term, scaledA), k);
      result = add(result, term);
    }
    return result;
  };
}

function scenarioRidgeRegression() {
  /**
   * Ridge regression: (X'X + λI)^{-1} X'y
   * Regularized least squares.
   */
  const X = randomMatrix(5000, 100);
  const y = randomMatrix(5000, 1);
  const lambda = 0.1;
  const n = 100;

  return function ridge() {
    const Xt = X.T;
    const XtX = matmul(Xt, X);

    // Add regularization: XtX + λI
    const reg = multiply(eye(n), lambda);
    const XtXreg = add(XtX, reg);

    const Xty = matmul(Xt, y);
    const beta = solve(XtXreg, Xty);
    return beta;
  };
}

function scenarioSoftmaxCrossEntropy() {
  /**
   * Softmax + Cross-entropy loss (combined).
   * Core of neural network classification.
   */
  // 1000 samples, 100 classes
  const logits = randomMatrix(1000, 100);
  // Create one-hot labels (simplified - just use random)
  const labels = randomMatrix(1000, 100);

  return function softmaxCE() {
    // Softmax per row (using our fused softmax on flattened, simplified)
    const probs = softmax(logits);

    // Cross-entropy: -sum(labels * log(probs))
    const logProbs = log(add(probs, 1e-10)); // Add epsilon for numerical stability
    const ce = multiply(labels, logProbs);
    const loss = -(sum(ce) as number);
    return loss;
  };
}

function scenarioGramSchmidt() {
  /**
   * Gram-Schmidt orthogonalization.
   * QR decomposition alternative.
   */
  const A = randomMatrix(200, 50);

  return function gramSchmidt() {
    const m = 200;
    const n = 50;
    const Q = new Float64Array(m * n);
    const Adata = A.data as Float64Array;

    // Copy first column and normalize
    let norm = 0;
    for (let i = 0; i < m; i++) {
      norm += Adata[i * n] * Adata[i * n];
    }
    norm = Math.sqrt(norm);
    for (let i = 0; i < m; i++) {
      Q[i * n] = Adata[i * n] / norm;
    }

    // Orthogonalize remaining columns
    for (let j = 1; j < n; j++) {
      // Copy column j
      for (let i = 0; i < m; i++) {
        Q[i * n + j] = Adata[i * n + j];
      }

      // Subtract projections onto previous columns
      for (let k = 0; k < j; k++) {
        let dot = 0;
        for (let i = 0; i < m; i++) {
          dot += Q[i * n + j] * Q[i * n + k];
        }
        for (let i = 0; i < m; i++) {
          Q[i * n + j] -= dot * Q[i * n + k];
        }
      }

      // Normalize
      norm = 0;
      for (let i = 0; i < m; i++) {
        norm += Q[i * n + j] * Q[i * n + j];
      }
      norm = Math.sqrt(norm);
      for (let i = 0; i < m; i++) {
        Q[i * n + j] /= norm;
      }
    }
    return Q;
  };
}

function scenarioLUDecomposition() {
  /**
   * LU decomposition with partial pivoting.
   * Using our solve function internally.
   */
  const A = randomMatrix(300, 300);
  const b = randomVector(300);

  return function luSolve() {
    // Solve Ax = b using our native solve
    const bCol = b.reshape([300, 1]);
    const x = solve(A, bCol);
    return x;
  };
}

function scenarioOuterProduct() {
  /**
   * Outer product: a ⊗ b = a * b.T
   * Common in rank-1 updates.
   */
  const a = randomVector(1000);
  const b = randomVector(1000);

  return function outerProduct() {
    const aCol = a.reshape([1000, 1]);
    const bRow = b.reshape([1, 1000]);
    const outer = matmul(aCol, bRow);
    return outer;
  };
}

function scenarioKroneckerProduct() {
  /**
   * Kronecker product of two matrices.
   * Used in quantum computing, signal processing.
   */
  const A = randomMatrix(50, 50);
  const B = randomMatrix(20, 20);

  return function kronecker() {
    const am = 50,
      an = 50;
    const bm = 20,
      bn = 20;
    const Adata = A.data as Float64Array;
    const Bdata = B.data as Float64Array;

    const result = new Float64Array(am * bm * an * bn);
    const resultRows = am * bm;
    const resultCols = an * bn;

    for (let i = 0; i < am; i++) {
      for (let j = 0; j < an; j++) {
        const aij = Adata[i * an + j];
        for (let k = 0; k < bm; k++) {
          for (let l = 0; l < bn; l++) {
            const row = i * bm + k;
            const col = j * bn + l;
            result[row * resultCols + col] = aij * Bdata[k * bn + l];
          }
        }
      }
    }
    return result;
  };
}

function runBenchmarks(): BenchmarkResult[] {
  const results: BenchmarkResult[] = [];

  const scenarios: [string, () => () => unknown][] = [
    // Original scenarios
    ['Data Normalization (10k x 100)', scenarioDataNormalization],
    ['Linear Regression (5k x 50)', scenarioLinearRegression],
    ['PCA via SVD (2k x 100 -> 10)', scenarioPCA],
    ['Correlation Matrix (5k x 50)', scenarioCorrelationMatrix],
    ['Neural Net Forward (1k x 256->128->64->10)', scenarioMatrixChain],
    ['Covariance + Eigendecomp (3k x 80)', scenarioCovarianceEigendecomp],
    ['Least Squares QR (8k x 100)', scenarioLeastSquaresQR],
    ['Batch Statistics (100 x 1k x 20)', scenarioBatchStatistics],
    ['Pairwise Distances (1k x 50)', scenarioDistanceMatrix],
    ['Polynomial Fit (1k points, deg 10)', scenarioPolynomialFit],
    // New scenarios
    ['Min-Max Scaling (10k x 100)', scenarioMinMaxScaling],
    ['Outer Product Sum (1k x 100)', scenarioOuterProductSum],
    ['Weighted Mean (5k x 200)', scenarioWeightedMean],
    ['Gradient Descent Step (10k x 50)', scenarioGradientDescent],
    ['Cross-Entropy Loss (1k x 10)', scenarioCrossEntropy],
    ['Cosine Similarity (500 x 128)', scenarioCosineSimilarity],
    ['K-Means Step (5k x 20, k=10)', scenarioKMeansStep],
    ['Image Filter (256 x 256)', scenarioImageFilter],
    ['Power Iteration (200 x 200)', scenarioPowerIteration],
    // Finance / Time Series
    ['Rolling Std (10k, window=50)', scenarioRollingStd],
    ['Log Returns (5k x 100)', scenarioLogReturns],
    ['Sharpe Ratio (1k x 50)', scenarioSharpeRatio],
    ['Portfolio Variance (1k x 50)', scenarioPortfolioVariance],
    // Deep Learning
    ['Attention Scores (128 x 64)', scenarioAttentionScores],
    ['Layer Normalization (1k x 512)', scenarioLayerNorm],
    ['Batch Matmul (32 x 64 x 64)', scenarioBatchMatmul],
    ['Softmax + Cross-Entropy (1k x 100)', scenarioSoftmaxCrossEntropy],
    // Signal Processing / Statistics
    ['1D Convolution (10k, kernel=51)', scenarioConvolution1D],
    ['Histogram (100k, 100 bins)', scenarioHistogram],
    ['Percentiles (10k x 50)', scenarioPercentiles],
    // Numerical Methods
    ['Jacobi Iteration (200 x 200)', scenarioJacobiIteration],
    ['Trapezoidal Integration (10k pts)', scenarioTrapezoidalIntegration],
    ['Finite Difference (200 x 200)', scenarioFiniteDifference],
    ['Matrix Exponential (100 x 100)', scenarioMatrixExponential],
    // Regularization / Decomposition
    ['Ridge Regression (5k x 100)', scenarioRidgeRegression],
    ['Gram-Schmidt (200 x 50)', scenarioGramSchmidt],
    ['LU Solve (300 x 300)', scenarioLUDecomposition],
    // Matrix Operations
    ['Outer Product (1k x 1k)', scenarioOuterProduct],
    ['Kronecker Product (50x50 ⊗ 20x20)', scenarioKroneckerProduct],
  ];

  for (const [name, scenarioFn] of scenarios) {
    const fn = scenarioFn();
    const result = benchmark(name, fn);
    results.push(result);
  }

  return results;
}

// Main
console.error('numpy-node real-world benchmark');
console.error('Running benchmarks...');

const results = runBenchmarks();

const output = {
  runtime: 'node',
  version: process.version,
  type: 'real-world',
  results,
};

console.log(JSON.stringify(output, null, 2));
