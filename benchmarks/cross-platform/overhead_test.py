#!/usr/bin/env python3
"""
Test to measure Python/NumPy call overhead vs actual computation.
"""
import time
import numpy as np

# Create test data
data = np.random.randn(10000, 100)

# Measure individual operation overhead
def measure_overhead():
    results = {}

    # 1. Just the function call overhead (tiny array)
    tiny = np.array([1.0])

    start = time.perf_counter()
    for _ in range(100000):
        np.mean(tiny)
    end = time.perf_counter()
    results['mean(tiny) per call'] = (end - start) / 100000 * 1e6  # microseconds

    # 2. Mean on large array
    start = time.perf_counter()
    for _ in range(1000):
        np.mean(data)
    end = time.perf_counter()
    results['mean(10k x 100) per call'] = (end - start) / 1000 * 1e6

    # 3. Z-score as separate operations
    start = time.perf_counter()
    for _ in range(100):
        m = np.mean(data, axis=0)
        s = np.std(data, axis=0)
        result = (data - m) / s
    end = time.perf_counter()
    results['zscore separate ops'] = (end - start) / 100 * 1000  # ms

    # 4. Count the operations
    results['zscore op count'] = '4 ops: mean, std, subtract, divide'

    return results

results = measure_overhead()

print("Python/NumPy Call Overhead Analysis")
print("=" * 50)
for k, v in results.items():
    if isinstance(v, float):
        print(f"{k}: {v:.2f} µs" if v < 1000 else f"{k}: {v/1000:.2f} ms")
    else:
        print(f"{k}: {v}")

print("\nKey insight:")
print(f"  - Tiny array mean: ~{results['mean(tiny) per call']:.1f} µs (mostly overhead)")
print(f"  - Large array mean: ~{results['mean(10k x 100) per call']:.0f} µs (computation dominates)")
print(f"  - Overhead ratio: ~{results['mean(10k x 100) per call'] / results['mean(tiny) per call']:.1f}x more work, but overhead is amortized")
