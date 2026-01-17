#!/usr/bin/env python3
"""
Real-world NumPy benchmark scenarios.
These represent actual use cases rather than isolated method calls.
"""

import json
import time
import numpy as np
from typing import Callable, List, Dict, Any

def benchmark(name: str, fn: Callable, warmup: int = 3, iterations: int = 50) -> Dict[str, Any]:
    """Run a benchmark with warmup and multiple iterations."""
    # Warmup
    for _ in range(warmup):
        fn()

    # Timed runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    times.sort()
    return {
        "name": name,
        "mean": sum(times) / len(times),
        "min": times[0],
        "max": times[-1],
        "median": times[len(times) // 2],
        "p95": times[int(len(times) * 0.95)],
        "iterations": iterations
    }

# ============================================
# Real-World Scenarios
# ============================================

def scenario_data_normalization():
    """
    Z-Score normalization of a dataset.
    Common in ML preprocessing.
    """
    # Simulate a dataset: 10000 samples, 100 features
    data = np.random.randn(10000, 100)

    def normalize():
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        normalized = (data - mean) / std
        return normalized

    return normalize

def scenario_linear_regression():
    """
    Ordinary Least Squares linear regression.
    β = (X'X)^(-1) X'y
    """
    # 5000 samples, 50 features
    X = np.random.randn(5000, 50)
    y = np.random.randn(5000)

    def fit():
        XtX = X.T @ X
        Xty = X.T @ y
        beta = np.linalg.solve(XtX, Xty)
        return beta

    return fit

def scenario_pca():
    """
    Principal Component Analysis - dimensionality reduction.
    Uses SVD for numerical stability.
    """
    # 2000 samples, 100 features -> reduce to 10 components
    data = np.random.randn(2000, 100)
    n_components = 10

    def pca():
        # Center the data
        mean = np.mean(data, axis=0)
        centered = data - mean

        # SVD
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)

        # Project to lower dimensions
        projected = centered @ Vt[:n_components].T
        return projected

    return pca

def scenario_correlation_matrix():
    """
    Compute correlation matrix for a dataset.
    Common in exploratory data analysis.
    """
    # 5000 samples, 50 variables
    data = np.random.randn(5000, 50)

    def correlation():
        # Center
        mean = np.mean(data, axis=0)
        centered = data - mean

        # Standardize
        std = np.std(data, axis=0)
        standardized = centered / std

        # Correlation = (X'X) / (n-1)
        n = data.shape[0]
        corr = (standardized.T @ standardized) / (n - 1)
        return corr

    return correlation

def scenario_matrix_chain():
    """
    Chain of matrix operations - common in neural network forward pass.
    """
    # Simulate layer dimensions
    X = np.random.randn(1000, 256)   # Input: 1000 samples, 256 features
    W1 = np.random.randn(256, 128)   # Layer 1 weights
    W2 = np.random.randn(128, 64)    # Layer 2 weights
    W3 = np.random.randn(64, 10)     # Output layer

    def forward():
        h1 = np.maximum(0, X @ W1)          # ReLU activation
        h2 = np.maximum(0, h1 @ W2)         # ReLU activation
        out = h2 @ W3                        # Linear output
        # Softmax
        exp_out = np.exp(out - np.max(out, axis=1, keepdims=True))
        softmax = exp_out / np.sum(exp_out, axis=1, keepdims=True)
        return softmax

    return forward

def scenario_covariance_eigendecomp():
    """
    Compute covariance matrix and its eigendecomposition.
    Used in PCA, factor analysis, etc.
    """
    # 3000 samples, 80 features
    data = np.random.randn(3000, 80)

    def eigen_analysis():
        # Center
        mean = np.mean(data, axis=0)
        centered = data - mean

        # Covariance
        n = data.shape[0]
        cov = (centered.T @ centered) / (n - 1)

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        return eigenvalues, eigenvectors

    return eigen_analysis

def scenario_least_squares_qr():
    """
    Solve overdetermined system using QR decomposition.
    More numerically stable than normal equations.
    """
    # 8000 equations, 100 unknowns
    A = np.random.randn(8000, 100)
    b = np.random.randn(8000)

    def solve_qr():
        Q, R = np.linalg.qr(A)
        # x = R^(-1) Q' b
        x = np.linalg.solve(R, Q.T @ b)
        return x

    return solve_qr

def scenario_batch_statistics():
    """
    Compute various statistics on batched data.
    Common in data analysis pipelines.
    """
    # 100 batches of 1000 samples each, 20 features
    data = np.random.randn(100, 1000, 20)

    def compute_stats():
        # Per-batch statistics
        means = np.mean(data, axis=1)
        stds = np.std(data, axis=1)
        mins = np.min(data, axis=1)
        maxs = np.max(data, axis=1)

        # Global statistics
        global_mean = np.mean(means, axis=0)
        global_std = np.mean(stds, axis=0)

        return global_mean, global_std

    return compute_stats

def scenario_distance_matrix():
    """
    Compute pairwise Euclidean distances.
    Used in clustering, nearest neighbors, etc.
    """
    # 1000 points in 50-dimensional space
    points = np.random.randn(1000, 50)

    def distances():
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        sq_norms = np.sum(points ** 2, axis=1)
        # Broadcasting: (n,1) + (1,n) - 2*(n,n)
        dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * (points @ points.T)
        dist = np.sqrt(np.maximum(dist_sq, 0))  # Clamp negative values from numerical errors
        return dist

    return distances

def scenario_polynomial_fit():
    """
    Fit a polynomial using Vandermonde matrix.
    """
    # 1000 data points, degree 10 polynomial
    x = np.linspace(-5, 5, 1000)
    y = np.sin(x) + 0.1 * np.random.randn(1000)  # Noisy sine wave
    degree = 10

    def fit_poly():
        # Build Vandermonde matrix
        V = np.vander(x, degree + 1)
        # Solve least squares
        coeffs = np.linalg.lstsq(V, y, rcond=None)[0]
        # Evaluate
        y_pred = V @ coeffs
        return coeffs, y_pred

    return fit_poly


# ============================================
# Additional Real-World Scenarios
# ============================================

def scenario_min_max_scaling():
    """
    Min-max feature scaling to [0, 1] range.
    Common in ML preprocessing.
    """
    data = np.random.randn(10000, 100)

    def scale():
        min_vals = data.min(axis=0)
        max_vals = data.max(axis=0)
        range_vals = max_vals - min_vals
        scaled = (data - min_vals) / range_vals
        return scaled

    return scale

def scenario_outer_product_sum():
    """
    Sum of outer products - used in covariance estimation.
    Computes sum of x_i * x_i^T for all samples.
    """
    vectors = np.random.randn(1000, 100)

    def outer_sum():
        # X^T @ X gives sum of outer products
        result = vectors.T @ vectors
        return result

    return outer_sum

def scenario_weighted_mean():
    """
    Weighted mean computation.
    Common in ensemble methods and attention mechanisms.
    """
    values = np.random.randn(5000, 200)
    weights = np.abs(np.random.randn(5000))

    def weighted_mean():
        # Normalize weights
        norm_weights = weights / weights.sum()
        # Weighted sum
        result = norm_weights.reshape(1, -1) @ values
        return result

    return weighted_mean

def scenario_gradient_descent():
    """
    Single gradient descent step for linear regression.
    """
    X = np.random.randn(10000, 50)
    y = np.random.randn(10000, 1)
    theta = np.random.randn(50, 1)
    learning_rate = 0.01
    m = 10000

    def gradient_step():
        predictions = X @ theta
        errors = predictions - y
        gradient = X.T @ errors / m
        new_theta = theta - learning_rate * gradient
        return new_theta

    return gradient_step

def scenario_cross_entropy():
    """
    Cross-entropy loss computation.
    Common in classification tasks.
    """
    predictions = np.random.randn(1000, 10)
    targets = np.random.randn(1000, 10)

    def cross_entropy():
        # Softmax
        exp_preds = np.exp(predictions - predictions.max())
        softmax_preds = exp_preds / exp_preds.sum()
        # Cross-entropy
        log_preds = targets * softmax_preds
        return -log_preds.sum()

    return cross_entropy

def scenario_cosine_similarity():
    """
    Pairwise cosine similarity matrix.
    Used in recommendation systems and NLP.
    """
    vectors = np.random.randn(500, 128)

    def cosine_sim():
        # Normalize
        norms = np.sqrt((vectors ** 2).sum(axis=1))
        # Dot products
        dot_prods = vectors @ vectors.T
        # Outer product of norms
        norm_outer = norms.reshape(-1, 1) @ norms.reshape(1, -1)
        similarity = dot_prods / norm_outer
        return similarity

    return cosine_sim

def scenario_kmeans_step():
    """
    Single K-means iteration: compute distances to centroids.
    """
    data = np.random.randn(5000, 20)
    centroids = np.random.randn(10, 20)

    def kmeans_step():
        # Using: ||x - c||^2 = ||x||^2 + ||c||^2 - 2*x.c
        x_sq = (data ** 2).sum(axis=1)
        c_sq = (centroids ** 2).sum(axis=1)
        dot_prods = data @ centroids.T
        return dot_prods

    return kmeans_step

def scenario_image_filter():
    """
    Simple filter applied to an image.
    """
    image = np.random.randn(256, 256)

    def filter_img():
        blurred = image + image * 0.1
        sharpened = image * 2 - blurred
        return sharpened

    return filter_img

def scenario_power_iteration():
    """
    Power iteration for dominant eigenvalue.
    Used in PageRank and spectral methods.
    """
    A = np.random.randn(200, 200)
    M = A.T @ A  # Symmetric PSD
    v = np.random.randn(200)

    def power_iter():
        v_curr = v.copy()
        for _ in range(10):
            Mv = M @ v_curr
            v_curr = Mv / np.sqrt((Mv ** 2).sum())
        return v_curr

    return power_iter


# ============================================
# Finance / Time Series Scenarios
# ============================================

def scenario_rolling_std():
    """
    Rolling standard deviation - common in finance/time series.
    """
    data = np.random.randn(10000)
    window_size = 50

    def rolling_std():
        n = len(data) - window_size + 1
        results = []
        # Sample every 10th window for speed
        for i in range(0, n, 10):
            window = data[i:i + window_size]
            results.append(np.std(window))
        return results

    return rolling_std


def scenario_log_returns():
    """
    Calculate log returns from price series.
    Common in quantitative finance.
    """
    prices = np.abs(np.random.randn(5000, 100)) + 1  # Positive prices

    def log_returns():
        log_prices = np.log(prices)
        returns = np.diff(log_prices, axis=0)
        return returns

    return log_returns


def scenario_sharpe_ratio():
    """
    Calculate Sharpe ratio for portfolio returns.
    """
    returns = np.random.randn(1000, 50) * 0.02  # Daily returns
    risk_free_rate = 0.02 / 252

    def sharpe_ratio():
        mean_returns = np.mean(returns, axis=0)
        std_returns = np.std(returns, axis=0)
        excess_returns = mean_returns - risk_free_rate
        sharpe = excess_returns / std_returns
        return sharpe

    return sharpe_ratio


def scenario_portfolio_variance():
    """
    Calculate portfolio variance using covariance matrix.
    """
    returns = np.random.randn(1000, 50) * 0.02
    weights = np.abs(np.random.randn(50))
    weights = weights / weights.sum()

    def portfolio_var():
        mean = np.mean(returns, axis=0)
        centered = returns - mean
        cov = (centered.T @ centered) / (len(returns) - 1)
        variance = weights @ cov @ weights
        return variance

    return portfolio_var


# ============================================
# Deep Learning Scenarios
# ============================================

def scenario_attention_scores():
    """
    Scaled dot-product attention scores.
    """
    Q = np.random.randn(128, 64)
    K = np.random.randn(128, 64)
    scale = 1 / np.sqrt(64)

    def attention():
        scores = Q @ K.T * scale
        # Row-wise softmax
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        attended = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        return attended

    return attention


def scenario_layer_norm():
    """
    Layer normalization - normalize across features.
    """
    data = np.random.randn(1000, 512)
    gamma = np.random.randn(512)
    beta = np.random.randn(512)

    def layer_norm():
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        normalized = (data - mean) / std
        result = gamma * normalized + beta
        return result

    return layer_norm


def scenario_batch_matmul():
    """
    Batch matrix multiplication.
    """
    batch_size = 32
    As = [np.random.randn(64, 64) for _ in range(batch_size)]
    Bs = [np.random.randn(64, 64) for _ in range(batch_size)]

    def batch_mm():
        results = [A @ B for A, B in zip(As, Bs)]
        return results

    return batch_mm


def scenario_softmax_cross_entropy():
    """
    Softmax + Cross-entropy loss.
    """
    logits = np.random.randn(1000, 100)
    labels = np.random.randn(1000, 100)

    def softmax_ce():
        # Row-wise softmax
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        # Cross-entropy
        log_probs = np.log(probs + 1e-10)
        loss = -(labels * log_probs).sum()
        return loss

    return softmax_ce


# ============================================
# Signal Processing / Statistics Scenarios
# ============================================

def scenario_convolution_1d():
    """
    1D convolution - signal processing.
    """
    signal = np.random.randn(10000)
    kernel = np.random.randn(51)

    def conv1d():
        out_len = len(signal) - len(kernel) + 1
        result = np.zeros(out_len)
        for i in range(out_len):
            result[i] = np.sum(signal[i:i + len(kernel)] * kernel)
        return result

    return conv1d


def scenario_histogram():
    """
    Compute histogram bins.
    """
    data = np.random.randn(100000)
    num_bins = 100

    def histogram():
        hist, _ = np.histogram(data, bins=num_bins)
        return hist

    return histogram


def scenario_percentiles():
    """
    Calculate multiple percentiles.
    """
    data = np.random.randn(10000, 50)
    percentiles = [10, 25, 50, 75, 90, 95, 99]

    def compute_percentiles():
        results = np.percentile(data, percentiles, axis=0)
        return results

    return compute_percentiles


# ============================================
# Numerical Methods Scenarios
# ============================================

def scenario_jacobi_iteration():
    """
    Jacobi iterative method for solving Ax = b.
    """
    n = 200
    A = np.random.randn(n, n)
    # Make diagonally dominant
    A = A + np.eye(n) * (np.abs(A).sum(axis=1) + 1)
    b = np.random.randn(n)

    def jacobi():
        x = np.zeros(n)
        D = np.diag(A)
        R = A - np.diag(D)
        for _ in range(50):
            x = (b - R @ x) / D
        return x

    return jacobi


def scenario_trapezoidal_integration():
    """
    Numerical integration using trapezoidal rule.
    """
    n = 10000
    x = np.linspace(0, np.pi, n)

    def integrate():
        y = np.sin(x)
        h = np.pi / (n - 1)
        integral = h * (y[0]/2 + y[1:-1].sum() + y[-1]/2)
        return integral

    return integrate


def scenario_finite_difference():
    """
    Compute gradient using finite differences.
    """
    grid_size = 200
    f = np.random.randn(grid_size, grid_size)
    h = 1.0

    def gradient():
        dfdx = np.zeros_like(f)
        dfdy = np.zeros_like(f)
        # Central differences for interior
        dfdx[1:-1, 1:-1] = (f[1:-1, 2:] - f[1:-1, :-2]) / (2 * h)
        dfdy[1:-1, 1:-1] = (f[2:, 1:-1] - f[:-2, 1:-1]) / (2 * h)
        return dfdx, dfdy

    return gradient


def scenario_matrix_exponential():
    """
    Approximate matrix exponential using Taylor series.
    """
    A = np.random.randn(100, 100) * 0.01

    def matrix_exp():
        result = np.eye(100)
        term = np.eye(100)
        for k in range(1, 11):
            term = term @ A / k
            result = result + term
        return result

    return matrix_exp


# ============================================
# Regularization / Decomposition Scenarios
# ============================================

def scenario_ridge_regression():
    """
    Ridge regression: (X'X + λI)^{-1} X'y
    """
    X = np.random.randn(5000, 100)
    y = np.random.randn(5000, 1)
    lambda_reg = 0.1

    def ridge():
        XtX = X.T @ X
        XtX_reg = XtX + lambda_reg * np.eye(100)
        Xty = X.T @ y
        beta = np.linalg.solve(XtX_reg, Xty)
        return beta

    return ridge


def scenario_gram_schmidt():
    """
    Gram-Schmidt orthogonalization.
    """
    A = np.random.randn(200, 50)

    def gram_schmidt():
        Q = np.zeros_like(A)
        for j in range(A.shape[1]):
            v = A[:, j].copy()
            for k in range(j):
                v = v - np.dot(Q[:, k], v) * Q[:, k]
            Q[:, j] = v / np.linalg.norm(v)
        return Q

    return gram_schmidt


def scenario_lu_solve():
    """
    Solve using LU decomposition.
    """
    A = np.random.randn(300, 300)
    b = np.random.randn(300, 1)

    def lu_solve():
        x = np.linalg.solve(A, b)
        return x

    return lu_solve


# ============================================
# Matrix Operations Scenarios
# ============================================

def scenario_outer_product():
    """
    Outer product: a ⊗ b
    """
    a = np.random.randn(1000)
    b = np.random.randn(1000)

    def outer_product():
        result = np.outer(a, b)
        return result

    return outer_product


def scenario_kronecker_product():
    """
    Kronecker product of two matrices.
    """
    A = np.random.randn(50, 50)
    B = np.random.randn(20, 20)

    def kronecker():
        result = np.kron(A, B)
        return result

    return kronecker


def run_benchmarks() -> List[Dict[str, Any]]:
    """Run all real-world benchmarks."""
    results = []

    scenarios = [
        # Original scenarios
        ("Data Normalization (10k x 100)", scenario_data_normalization),
        ("Linear Regression (5k x 50)", scenario_linear_regression),
        ("PCA via SVD (2k x 100 -> 10)", scenario_pca),
        ("Correlation Matrix (5k x 50)", scenario_correlation_matrix),
        ("Neural Net Forward (1k x 256->128->64->10)", scenario_matrix_chain),
        ("Covariance + Eigendecomp (3k x 80)", scenario_covariance_eigendecomp),
        ("Least Squares QR (8k x 100)", scenario_least_squares_qr),
        ("Batch Statistics (100 x 1k x 20)", scenario_batch_statistics),
        ("Pairwise Distances (1k x 50)", scenario_distance_matrix),
        ("Polynomial Fit (1k points, deg 10)", scenario_polynomial_fit),
        # Additional scenarios
        ("Min-Max Scaling (10k x 100)", scenario_min_max_scaling),
        ("Outer Product Sum (1k x 100)", scenario_outer_product_sum),
        ("Weighted Mean (5k x 200)", scenario_weighted_mean),
        ("Gradient Descent Step (10k x 50)", scenario_gradient_descent),
        ("Cross-Entropy Loss (1k x 10)", scenario_cross_entropy),
        ("Cosine Similarity (500 x 128)", scenario_cosine_similarity),
        ("K-Means Step (5k x 20, k=10)", scenario_kmeans_step),
        ("Image Filter (256 x 256)", scenario_image_filter),
        ("Power Iteration (200 x 200)", scenario_power_iteration),
        # Finance / Time Series
        ("Rolling Std (10k, window=50)", scenario_rolling_std),
        ("Log Returns (5k x 100)", scenario_log_returns),
        ("Sharpe Ratio (1k x 50)", scenario_sharpe_ratio),
        ("Portfolio Variance (1k x 50)", scenario_portfolio_variance),
        # Deep Learning
        ("Attention Scores (128 x 64)", scenario_attention_scores),
        ("Layer Normalization (1k x 512)", scenario_layer_norm),
        ("Batch Matmul (32 x 64 x 64)", scenario_batch_matmul),
        ("Softmax + Cross-Entropy (1k x 100)", scenario_softmax_cross_entropy),
        # Signal Processing / Statistics
        ("1D Convolution (10k, kernel=51)", scenario_convolution_1d),
        ("Histogram (100k, 100 bins)", scenario_histogram),
        ("Percentiles (10k x 50)", scenario_percentiles),
        # Numerical Methods
        ("Jacobi Iteration (200 x 200)", scenario_jacobi_iteration),
        ("Trapezoidal Integration (10k pts)", scenario_trapezoidal_integration),
        ("Finite Difference (200 x 200)", scenario_finite_difference),
        ("Matrix Exponential (100 x 100)", scenario_matrix_exponential),
        # Regularization / Decomposition
        ("Ridge Regression (5k x 100)", scenario_ridge_regression),
        ("Gram-Schmidt (200 x 50)", scenario_gram_schmidt),
        ("LU Solve (300 x 300)", scenario_lu_solve),
        # Matrix Operations
        ("Outer Product (1k x 1k)", scenario_outer_product),
        ("Kronecker Product (50x50 ⊗ 20x20)", scenario_kronecker_product),
    ]

    for name, scenario_fn in scenarios:
        fn = scenario_fn()
        result = benchmark(name, fn)
        results.append(result)

    return results


if __name__ == "__main__":
    print("NumPy version:", np.__version__, file=__import__('sys').stderr)
    print("Running real-world benchmarks...", file=__import__('sys').stderr)

    results = run_benchmarks()

    output = {
        "runtime": "python",
        "version": np.__version__,
        "type": "real-world",
        "results": results
    }

    print(json.dumps(output, indent=2))
