#!/usr/bin/env python3
"""
NumPy benchmark suite for comparison with numpy-node.
Run with: python numpy_benchmark.py
"""

import numpy as np
import time
import json
import sys

def benchmark(name, fn, warmup=3, iterations=100):
    """Run a benchmark and return timing statistics."""
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

def run_benchmarks():
    results = []

    # ============================================
    # Array Creation
    # ============================================

    results.append(benchmark(
        "zeros(1000x1000)",
        lambda: np.zeros((1000, 1000))
    ))

    results.append(benchmark(
        "ones(1000x1000)",
        lambda: np.ones((1000, 1000))
    ))

    results.append(benchmark(
        "arange(100000)",
        lambda: np.arange(100000)
    ))

    results.append(benchmark(
        "linspace(0, 1, 100000)",
        lambda: np.linspace(0, 1, 100000)
    ))

    # ============================================
    # Element-wise Operations
    # ============================================

    a = np.random.rand(1000, 1000)
    b = np.random.rand(1000, 1000)

    results.append(benchmark(
        "add(1000x1000)",
        lambda: np.add(a, b)
    ))

    results.append(benchmark(
        "multiply(1000x1000)",
        lambda: np.multiply(a, b)
    ))

    results.append(benchmark(
        "sqrt(1000x1000)",
        lambda: np.sqrt(a)
    ))

    results.append(benchmark(
        "exp(1000x1000)",
        lambda: np.exp(a)
    ))

    # ============================================
    # Reductions
    # ============================================

    results.append(benchmark(
        "sum(1000x1000)",
        lambda: np.sum(a)
    ))

    results.append(benchmark(
        "mean(1000x1000)",
        lambda: np.mean(a)
    ))

    results.append(benchmark(
        "std(1000x1000)",
        lambda: np.std(a)
    ))

    results.append(benchmark(
        "min(1000x1000)",
        lambda: np.min(a)
    ))

    results.append(benchmark(
        "max(1000x1000)",
        lambda: np.max(a)
    ))

    # ============================================
    # Linear Algebra (BLAS/LAPACK)
    # ============================================

    m1 = np.random.rand(500, 500)
    m2 = np.random.rand(500, 500)

    results.append(benchmark(
        "matmul(500x500)",
        lambda: np.matmul(m1, m2),
        iterations=50
    ))

    results.append(benchmark(
        "dot(500x500)",
        lambda: np.dot(m1, m2),
        iterations=50
    ))

    m_small = np.random.rand(100, 100)

    results.append(benchmark(
        "inv(100x100)",
        lambda: np.linalg.inv(m_small),
        iterations=50
    ))

    results.append(benchmark(
        "det(100x100)",
        lambda: np.linalg.det(m_small),
        iterations=50
    ))

    results.append(benchmark(
        "svd(100x100)",
        lambda: np.linalg.svd(m_small),
        iterations=50
    ))

    results.append(benchmark(
        "qr(100x100)",
        lambda: np.linalg.qr(m_small),
        iterations=50
    ))

    results.append(benchmark(
        "eig(100x100)",
        lambda: np.linalg.eig(m_small),
        iterations=50
    ))

    # Solve linear system
    A = np.random.rand(100, 100)
    b_vec = np.random.rand(100)

    results.append(benchmark(
        "solve(100x100)",
        lambda: np.linalg.solve(A, b_vec),
        iterations=50
    ))

    # ============================================
    # Many Small Operations (Loop Overhead)
    # ============================================

    def many_small_ops():
        for _ in range(1000):
            x = np.array([1, 2, 3, 4, 5])
            y = np.array([5, 4, 3, 2, 1])
            _ = np.add(x, y)

    results.append(benchmark(
        "1000x small array ops",
        many_small_ops,
        iterations=20
    ))

    return results

if __name__ == "__main__":
    print(f"NumPy version: {np.__version__}", file=sys.stderr)
    print(f"Running benchmarks...", file=sys.stderr)

    results = run_benchmarks()

    output = {
        "runtime": "python",
        "version": np.__version__,
        "results": results
    }

    print(json.dumps(output, indent=2))
