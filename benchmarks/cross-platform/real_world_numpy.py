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


# ============================================
# Machine Learning / Deep Learning Scenarios
# ============================================

def scenario_batch_normalization():
    """
    Batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    Common in CNNs and deep networks.
    """
    x = np.random.randn(256, 64, 32, 32)  # NCHW format
    gamma = np.random.randn(64)
    beta = np.random.randn(64)
    eps = 1e-5

    def batch_norm():
        # Compute mean and var over batch, height, width (axes 0, 2, 3)
        mean = x.mean(axis=(0, 2, 3), keepdims=True)
        var = x.var(axis=(0, 2, 3), keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        # Scale and shift (broadcast gamma and beta)
        out = x_norm * gamma.reshape(1, -1, 1, 1) + beta.reshape(1, -1, 1, 1)
        return out

    return batch_norm


def scenario_dropout_forward():
    """
    Dropout forward pass - generate mask and apply.
    Common regularization technique.
    """
    x = np.random.randn(1000, 512)
    p = 0.5  # dropout probability

    def dropout():
        # Generate binary mask
        mask = (np.random.rand(*x.shape) > p).astype(np.float64)
        # Apply mask and scale
        out = x * mask / (1 - p)
        return out, mask

    return dropout


def scenario_xavier_init():
    """
    Xavier/Glorot weight initialization for neural networks.
    """
    layers = [(784, 512), (512, 256), (256, 128), (128, 10)]

    def xavier():
        weights = []
        for fan_in, fan_out in layers:
            std = np.sqrt(2.0 / (fan_in + fan_out))
            W = np.random.randn(fan_in, fan_out) * std
            weights.append(W)
        return weights

    return xavier


def scenario_adam_optimizer_step():
    """
    Adam optimizer update step.
    Most popular optimizer for deep learning.
    """
    n_params = 100000
    params = np.random.randn(n_params)
    grads = np.random.randn(n_params)
    m = np.zeros(n_params)  # First moment
    v = np.zeros(n_params)  # Second moment
    lr = 0.001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    t = 1

    def adam_step():
        nonlocal m, v, t
        # Update biased moments
        m = beta1 * m + (1 - beta1) * grads
        v = beta2 * v + (1 - beta2) * (grads ** 2)
        # Bias correction
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        # Update params
        new_params = params - lr * m_hat / (np.sqrt(v_hat) + eps)
        t += 1
        return new_params

    return adam_step


def scenario_confusion_matrix():
    """
    Compute confusion matrix from predictions.
    Essential for classification evaluation.
    """
    n_samples = 10000
    n_classes = 10
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)

    def compute_confusion():
        cm = np.zeros((n_classes, n_classes), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        return cm

    return compute_confusion


# ============================================
# Statistics Scenarios
# ============================================

def scenario_bootstrap_mean():
    """
    Bootstrap resampling for confidence intervals.
    Common in statistical inference.
    """
    data = np.random.randn(1000)
    n_bootstrap = 1000

    def bootstrap():
        means = np.zeros(n_bootstrap)
        n = len(data)
        for i in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.randint(0, n, n)
            sample = data[indices]
            means[i] = sample.mean()
        # Return 95% CI
        ci_low = np.percentile(means, 2.5)
        ci_high = np.percentile(means, 97.5)
        return means.mean(), ci_low, ci_high

    return bootstrap


def scenario_welch_ttest():
    """
    Welch's t-test for comparing two samples.
    """
    sample1 = np.random.randn(500) * 1.5 + 2.0
    sample2 = np.random.randn(600) * 2.0 + 2.5

    def ttest():
        n1, n2 = len(sample1), len(sample2)
        mean1, mean2 = sample1.mean(), sample2.mean()
        var1, var2 = sample1.var(ddof=1), sample2.var(ddof=1)

        # Welch's t-statistic
        se = np.sqrt(var1/n1 + var2/n2)
        t_stat = (mean1 - mean2) / se

        # Degrees of freedom (Welch-Satterthwaite)
        num = (var1/n1 + var2/n2) ** 2
        denom = (var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1)
        df = num / denom

        return t_stat, df

    return ttest


def scenario_kde():
    """
    Kernel Density Estimation with Gaussian kernel.
    """
    data = np.random.randn(1000)
    x_eval = np.linspace(-4, 4, 200)
    bandwidth = 0.3

    def kde():
        n = len(data)
        # Gaussian kernel: K(u) = exp(-u^2/2) / sqrt(2*pi)
        # For each eval point, sum contributions from all data points
        density = np.zeros(len(x_eval))
        for i, x in enumerate(x_eval):
            u = (x - data) / bandwidth
            density[i] = np.exp(-0.5 * u**2).sum()
        density /= (n * bandwidth * np.sqrt(2 * np.pi))
        return density

    return kde


def scenario_moving_window_stats():
    """
    Moving window statistics: mean, std, min, max.
    Common in time series analysis.
    """
    data = np.random.randn(10000)
    window = 100

    def moving_stats():
        n = len(data) - window + 1
        means = np.zeros(n)
        stds = np.zeros(n)
        mins = np.zeros(n)
        maxs = np.zeros(n)

        for i in range(n):
            w = data[i:i+window]
            means[i] = w.mean()
            stds[i] = w.std()
            mins[i] = w.min()
            maxs[i] = w.max()

        return means, stds, mins, maxs

    return moving_stats


# ============================================
# Signal Processing / Physics Scenarios
# ============================================

def scenario_autocorrelation():
    """
    Autocorrelation of a time series.
    Important for time series analysis.
    """
    signal = np.random.randn(5000)
    max_lag = 100

    def autocorr():
        n = len(signal)
        mean = signal.mean()
        var = signal.var()
        signal_centered = signal - mean

        result = np.zeros(max_lag)
        for lag in range(max_lag):
            if lag == 0:
                result[lag] = 1.0
            else:
                result[lag] = np.sum(signal_centered[:-lag] * signal_centered[lag:]) / ((n - lag) * var)
        return result

    return autocorr


def scenario_fft_signal_processing():
    """
    FFT-based signal processing: compute frequency spectrum,
    filter frequencies, and inverse transform.
    Common in audio processing, communications, vibration analysis.
    """
    # Create signal with multiple frequency components + noise
    n = 4096
    t = np.linspace(0, 1, n)
    # Signal: 50 Hz + 120 Hz + noise
    signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t) + 0.2 * np.random.randn(n)

    def fft_process():
        # Forward FFT
        spectrum = np.fft.fft(signal)
        # Compute power spectrum
        power = np.abs(spectrum) ** 2
        # Apply simple low-pass filter (zero out high frequencies)
        filtered = spectrum.copy()
        filtered[200:-200] = 0
        # Inverse FFT
        reconstructed = np.fft.ifft(filtered)
        return np.real(reconstructed), power[:n//2]

    return fft_process


def scenario_batched_attention():
    """
    Batched attention computation using matrix multiplication.
    Essential for transformer architectures and deep learning.
    Uses np.matmul (@ operator) for batched operations.
    """
    batch_size = 32
    seq_len = 64
    d_model = 64
    Q = np.random.randn(batch_size, seq_len, d_model).astype(np.float64)
    K = np.random.randn(batch_size, seq_len, d_model).astype(np.float64)
    V = np.random.randn(batch_size, seq_len, d_model).astype(np.float64)

    # Pre-transpose K to (batch, d_model, seq) for Q @ K^T
    KT = K.transpose(0, 2, 1)

    def batched_attention():
        # Attention scores: Q @ K^T for each batch -> (batch, seq, seq)
        scores = Q @ KT
        # Apply softmax (simplified)
        scores_exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attention = scores_exp / scores_exp.sum(axis=-1, keepdims=True)
        # Weighted sum of values: attention @ V -> (batch, seq, d_model)
        output = attention @ V
        return output

    return batched_attention


def scenario_nbody_step():
    """
    N-body gravitational simulation step.
    Compute pairwise forces and update velocities.
    """
    n_bodies = 500
    positions = np.random.randn(n_bodies, 3) * 10
    velocities = np.random.randn(n_bodies, 3) * 0.1
    masses = np.random.rand(n_bodies) + 0.1
    G = 1.0
    dt = 0.01
    softening = 0.1

    def nbody():
        # Compute pairwise displacements
        dx = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # (n, n, 3)
        # Compute distances
        r2 = (dx ** 2).sum(axis=2) + softening ** 2  # (n, n)
        r3 = r2 * np.sqrt(r2)
        # Compute accelerations
        # a_i = G * sum_j m_j * (x_j - x_i) / |x_j - x_i|^3
        accel = G * np.sum(masses[np.newaxis, :, np.newaxis] * (-dx) / r3[:, :, np.newaxis], axis=1)
        # Update velocities
        new_velocities = velocities + accel * dt
        return new_velocities

    return nbody


def scenario_heat_equation():
    """
    2D heat equation using finite differences.
    Laplacian: d²T/dx² + d²T/dy²
    """
    n = 100
    T = np.random.rand(n, n)
    T[0, :] = T[-1, :] = T[:, 0] = T[:, -1] = 0  # Boundary conditions
    alpha = 0.25  # Diffusion coefficient
    n_steps = 50

    def heat_step():
        T_new = T.copy()
        for _ in range(n_steps):
            # Laplacian using 5-point stencil
            laplacian = (
                T_new[:-2, 1:-1] + T_new[2:, 1:-1] +
                T_new[1:-1, :-2] + T_new[1:-1, 2:] -
                4 * T_new[1:-1, 1:-1]
            )
            T_new[1:-1, 1:-1] += alpha * laplacian
        return T_new

    return heat_step


def scenario_monte_carlo_pi():
    """
    Monte Carlo estimation of Pi.
    Classic example of MC simulation.
    """
    n_samples = 1000000

    def monte_carlo():
        # Generate random points in unit square
        x = np.random.rand(n_samples)
        y = np.random.rand(n_samples)
        # Count points inside quarter circle
        inside = (x**2 + y**2) <= 1.0
        pi_estimate = 4.0 * inside.sum() / n_samples
        return pi_estimate

    return monte_carlo


# ============================================
# Finance Scenarios (Additional)
# ============================================

def scenario_black_scholes():
    """
    Black-Scholes option pricing.
    """
    n_options = 10000
    S = np.random.uniform(80, 120, n_options)  # Stock price
    K = 100 * np.ones(n_options)  # Strike price
    T = np.random.uniform(0.1, 2.0, n_options)  # Time to maturity
    r = 0.05  # Risk-free rate
    sigma = np.random.uniform(0.1, 0.5, n_options)  # Volatility

    def black_scholes():
        # d1 and d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Normal CDF approximation
        def norm_cdf(x):
            return 0.5 * (1 + np.tanh(x * 0.7978845608))  # Approximation

        # Call price
        call = S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
        return call

    return black_scholes


def scenario_var_historical():
    """
    Value at Risk using historical simulation.
    """
    n_days = 1000
    n_assets = 50
    returns = np.random.randn(n_days, n_assets) * 0.02  # Daily returns
    weights = np.random.rand(n_assets)
    weights /= weights.sum()  # Normalize
    confidence = 0.95

    def var():
        # Portfolio returns
        portfolio_returns = returns @ weights
        # Sort returns
        sorted_returns = np.sort(portfolio_returns)
        # VaR at confidence level
        var_idx = int((1 - confidence) * n_days)
        var_value = -sorted_returns[var_idx]
        # Expected Shortfall (CVaR)
        cvar = -sorted_returns[:var_idx].mean()
        return var_value, cvar

    return var


def scenario_ewma_volatility():
    """
    Exponentially Weighted Moving Average volatility.
    Common in risk management.
    """
    returns = np.random.randn(2000) * 0.02
    lambda_param = 0.94  # Decay factor

    def ewma():
        n = len(returns)
        variance = np.zeros(n)
        variance[0] = returns[0] ** 2

        for i in range(1, n):
            variance[i] = lambda_param * variance[i-1] + (1 - lambda_param) * returns[i-1]**2

        volatility = np.sqrt(variance)
        return volatility

    return ewma


# ============================================
# Miscellaneous Scenarios
# ============================================

def scenario_matrix_factorization_step():
    """
    Matrix factorization SGD step for recommender systems.
    """
    n_users = 1000
    n_items = 500
    n_factors = 50
    n_ratings = 10000

    # Latent factors
    P = np.random.randn(n_users, n_factors) * 0.1
    Q = np.random.randn(n_items, n_factors) * 0.1

    # Sparse ratings (user_id, item_id, rating)
    users = np.random.randint(0, n_users, n_ratings)
    items = np.random.randint(0, n_items, n_ratings)
    ratings = np.random.rand(n_ratings) * 4 + 1  # 1-5 scale

    lr = 0.01
    reg = 0.02

    def mf_step():
        P_new = P.copy()
        Q_new = Q.copy()

        for u, i, r in zip(users, items, ratings):
            pred = P_new[u] @ Q_new[i]
            error = r - pred
            # Update factors
            P_new[u] += lr * (error * Q_new[i] - reg * P_new[u])
            Q_new[i] += lr * (error * P_new[u] - reg * Q_new[i])

        return P_new, Q_new

    return mf_step


def scenario_tfidf():
    """
    TF-IDF computation for text processing.
    """
    n_docs = 500
    n_terms = 1000
    # Simulated term frequencies (sparse-ish)
    tf = np.random.poisson(2, (n_docs, n_terms)).astype(np.float64)

    def tfidf():
        # Term frequency normalization
        tf_norm = tf / (tf.sum(axis=1, keepdims=True) + 1e-10)
        # Document frequency
        df = (tf > 0).sum(axis=0)
        # Inverse document frequency
        idf = np.log(n_docs / (df + 1))
        # TF-IDF
        tfidf_matrix = tf_norm * idf
        return tfidf_matrix

    return tfidf


def scenario_bilinear_interpolation():
    """
    Bilinear interpolation for image resizing.
    """
    # Source image
    src_h, src_w = 256, 256
    src = np.random.rand(src_h, src_w)
    # Target size
    dst_h, dst_w = 512, 512

    def bilinear():
        dst = np.zeros((dst_h, dst_w))

        # Scale factors
        scale_y = src_h / dst_h
        scale_x = src_w / dst_w

        for y in range(dst_h):
            for x in range(dst_w):
                # Source coordinates
                src_y = y * scale_y
                src_x = x * scale_x

                # Integer parts
                y0 = int(src_y)
                x0 = int(src_x)
                y1 = min(y0 + 1, src_h - 1)
                x1 = min(x0 + 1, src_w - 1)

                # Fractional parts
                fy = src_y - y0
                fx = src_x - x0

                # Bilinear interpolation
                dst[y, x] = (
                    src[y0, x0] * (1 - fx) * (1 - fy) +
                    src[y0, x1] * fx * (1 - fy) +
                    src[y1, x0] * (1 - fx) * fy +
                    src[y1, x1] * fx * fy
                )

        return dst

    return bilinear


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
        ("Trapezoidal Integration (10k pts)", scenario_trapezoidal_integration),
        ("Finite Difference (200 x 200)", scenario_finite_difference),
        # Regularization / Decomposition
        ("Ridge Regression (5k x 100)", scenario_ridge_regression),
        ("LU Solve (300 x 300)", scenario_lu_solve),
        # Matrix Operations
        ("Outer Product (1k x 1k)", scenario_outer_product),
        ("Kronecker Product (50x50 ⊗ 20x20)", scenario_kronecker_product),
        # Machine Learning / Deep Learning
        ("Batch Normalization (256x64x32x32)", scenario_batch_normalization),
        ("Dropout Forward (1k x 512)", scenario_dropout_forward),
        ("Xavier Init (4 layers)", scenario_xavier_init),
        ("Adam Optimizer Step (100k params)", scenario_adam_optimizer_step),
        ("Confusion Matrix (10k samples)", scenario_confusion_matrix),
        # Statistics
        ("Bootstrap Mean (1k samples, 1k resamples)", scenario_bootstrap_mean),
        ("Welch t-Test (500 vs 600)", scenario_welch_ttest),
        ("KDE (1k points, 200 eval)", scenario_kde),
        ("Moving Window Stats (10k, w=100)", scenario_moving_window_stats),
        # Signal Processing / Physics
        ("Autocorrelation (5k, lag=100)", scenario_autocorrelation),
        ("FFT Signal Processing (4k samples)", scenario_fft_signal_processing),
        ("Batched Attention (32x64x64)", scenario_batched_attention),
        ("N-Body Step (500 bodies)", scenario_nbody_step),
        ("Heat Equation (100x100, 50 steps)", scenario_heat_equation),
        # Finance (Additional)
        ("Black-Scholes (10k options)", scenario_black_scholes),
        ("VaR Historical (1k days, 50 assets)", scenario_var_historical),
        ("EWMA Volatility (2k returns)", scenario_ewma_volatility),
        # Miscellaneous
        ("Matrix Factorization Step (1k×500)", scenario_matrix_factorization_step),
        ("TF-IDF (500 docs, 1k terms)", scenario_tfidf),
        ("Bilinear Interpolation (256→512)", scenario_bilinear_interpolation),
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
