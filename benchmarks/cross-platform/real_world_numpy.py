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
        # New scenarios
        ("Min-Max Scaling (10k x 100)", scenario_min_max_scaling),
        ("Outer Product Sum (1k x 100)", scenario_outer_product_sum),
        ("Weighted Mean (5k x 200)", scenario_weighted_mean),
        ("Gradient Descent Step (10k x 50)", scenario_gradient_descent),
        ("Cross-Entropy Loss (1k x 10)", scenario_cross_entropy),
        ("Cosine Similarity (500 x 128)", scenario_cosine_similarity),
        ("K-Means Step (5k x 20, k=10)", scenario_kmeans_step),
        ("Image Filter (256 x 256)", scenario_image_filter),
        ("Power Iteration (200 x 200)", scenario_power_iteration),
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
