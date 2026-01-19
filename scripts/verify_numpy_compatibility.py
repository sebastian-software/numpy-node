#!/usr/bin/env python3
"""
NumPy Compatibility Verification Script

This script generates test cases by running NumPy operations and outputs
the expected results in a format that can be used to verify numpy-ts compatibility.

Run: python3 scripts/verify_numpy_compatibility.py
"""

import numpy as np
import json
from typing import Any

def to_list(arr):
    """Convert numpy array to nested list for JSON serialization."""
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    elif isinstance(arr, (np.bool_, bool)):
        return bool(arr)
    elif isinstance(arr, (np.integer, int)):
        return int(arr)
    elif isinstance(arr, (np.floating, float)):
        return float(arr)
    return arr

def print_test(category: str, func_name: str, inputs: str, result: Any, dtype: str = None):
    """Print a test case in a standardized format."""
    result_str = to_list(result)
    dtype_str = f", dtype={result.dtype}" if hasattr(result, 'dtype') else ""
    shape_str = f", shape={list(result.shape)}" if hasattr(result, 'shape') else ""
    print(f"  {func_name}: {inputs} => {result_str}{dtype_str}{shape_str}")

print("=" * 70)
print("NumPy Compatibility Verification")
print(f"NumPy Version: {np.__version__}")
print("=" * 70)

# =============================================================================
# ARRAY CREATION
# =============================================================================
print("\n### ARRAY CREATION ###\n")

# zeros
print("zeros([3]):", to_list(np.zeros(3)), f"dtype={np.zeros(3).dtype}")
print("zeros([2,3]):", to_list(np.zeros((2,3))), f"dtype={np.zeros((2,3)).dtype}")
print("zeros([3], dtype=int32):", to_list(np.zeros(3, dtype=np.int32)), f"dtype={np.zeros(3, dtype=np.int32).dtype}")

# ones
print("ones([3]):", to_list(np.ones(3)), f"dtype={np.ones(3).dtype}")
print("ones([2,3]):", to_list(np.ones((2,3))), f"dtype={np.ones((2,3)).dtype}")

# full
print("full([3], 5):", to_list(np.full(3, 5)), f"dtype={np.full(3, 5).dtype}")
print("full([2,3], 3.14):", to_list(np.full((2,3), 3.14)), f"dtype={np.full((2,3), 3.14).dtype}")

# arange
print("arange(5):", to_list(np.arange(5)))
print("arange(2, 7):", to_list(np.arange(2, 7)))
print("arange(0, 1, 0.2):", to_list(np.arange(0, 1, 0.2)))

# linspace
print("linspace(0, 1, 5):", to_list(np.linspace(0, 1, 5)))
print("linspace(0, 10, 3):", to_list(np.linspace(0, 10, 3)))

# eye
print("eye(3):", to_list(np.eye(3)))
print("eye(3, 4):", to_list(np.eye(3, 4)))
print("eye(3, 4, k=1):", to_list(np.eye(3, 4, k=1)))

# =============================================================================
# ARITHMETIC OPERATIONS
# =============================================================================
print("\n### ARITHMETIC OPERATIONS ###\n")

a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
print(f"a = {to_list(a)}")
print(f"b = {to_list(b)}")

print("add(a, b):", to_list(np.add(a, b)))
print("add(a, 10):", to_list(np.add(a, 10)))
print("subtract(a, b):", to_list(np.subtract(a, b)))
print("multiply(a, b):", to_list(np.multiply(a, b)))
print("divide(a, b):", to_list(np.divide(a, b)))
print("power(a, 2):", to_list(np.power(a, 2)))
print("power(a, b):", to_list(np.power(a, b)))

# Broadcasting
a2d = np.array([[1, 2, 3], [4, 5, 6]])
b1d = np.array([1, 2, 3])
print(f"\na2d = {to_list(a2d)}")
print(f"b1d = {to_list(b1d)}")
print("add(a2d, b1d):", to_list(np.add(a2d, b1d)))
print("subtract(a2d, b1d):", to_list(np.subtract(a2d, b1d)))

# =============================================================================
# UNARY MATH FUNCTIONS
# =============================================================================
print("\n### UNARY MATH FUNCTIONS ###\n")

x = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
print(f"x = {to_list(x)}")
print("sqrt(x):", to_list(np.sqrt(x)))
print("exp(x):", [round(v, 10) for v in to_list(np.exp(x))])
print("log([1, e, e^2]):", to_list(np.log([1, np.e, np.e**2])))

angles = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
print(f"\nangles = [0, pi/6, pi/4, pi/3, pi/2]")
print("sin(angles):", [round(v, 10) for v in to_list(np.sin(angles))])
print("cos(angles):", [round(v, 10) for v in to_list(np.cos(angles))])

print("\nabs([-1, 2, -3, 4]):", to_list(np.abs([-1, 2, -3, 4])))
print("negative([1, -2, 3]):", to_list(np.negative([1, -2, 3])))

# =============================================================================
# REDUCTION OPERATIONS
# =============================================================================
print("\n### REDUCTION OPERATIONS ###\n")

arr = np.array([1, 2, 3, 4, 5])
print(f"arr = {to_list(arr)}")
print("sum(arr):", to_list(np.sum(arr)))
print("prod(arr):", to_list(np.prod(arr)))
print("mean(arr):", to_list(np.mean(arr)))
print("std(arr):", round(to_list(np.std(arr)), 10))
print("var(arr):", to_list(np.var(arr)))
print("min(arr):", to_list(np.min(arr)))
print("max(arr):", to_list(np.max(arr)))
print("median(arr):", to_list(np.median(arr)))

# With axis
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\narr2d = {to_list(arr2d)}")
print("sum(arr2d, axis=0):", to_list(np.sum(arr2d, axis=0)))
print("sum(arr2d, axis=1):", to_list(np.sum(arr2d, axis=1)))
print("mean(arr2d, axis=0):", to_list(np.mean(arr2d, axis=0)))
print("mean(arr2d, axis=1):", to_list(np.mean(arr2d, axis=1)))
print("min(arr2d, axis=0):", to_list(np.min(arr2d, axis=0)))
print("max(arr2d, axis=1):", to_list(np.max(arr2d, axis=1)))

# =============================================================================
# COMPARISON OPERATORS
# =============================================================================
print("\n### COMPARISON OPERATORS ###\n")

a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 4, 3, 2, 1])
print(f"a = {to_list(a)}")
print(f"b = {to_list(b)}")
print("equal(a, b):", to_list(np.equal(a, b)), f"dtype={np.equal(a, b).dtype}")
print("equal(a, 3):", to_list(np.equal(a, 3)))
print("not_equal(a, b):", to_list(np.not_equal(a, b)))
print("less(a, 3):", to_list(np.less(a, 3)))
print("less_equal(a, 3):", to_list(np.less_equal(a, 3)))
print("greater(a, 3):", to_list(np.greater(a, 3)))
print("greater_equal(a, 3):", to_list(np.greater_equal(a, 3)))

# =============================================================================
# LOGICAL OPERATORS
# =============================================================================
print("\n### LOGICAL OPERATORS ###\n")

x = np.array([True, True, False, False])
y = np.array([True, False, True, False])
print(f"x = {to_list(x)}")
print(f"y = {to_list(y)}")
print("logical_and(x, y):", to_list(np.logical_and(x, y)), f"dtype={np.logical_and(x, y).dtype}")
print("logical_or(x, y):", to_list(np.logical_or(x, y)))
print("logical_xor(x, y):", to_list(np.logical_xor(x, y)))
print("logical_not(x):", to_list(np.logical_not(x)))

# Broadcasting
print("\nBroadcasting:")
a2d = np.array([[1, 0], [1, 1]])
b1d = np.array([1, 0])
print(f"a2d = {to_list(a2d)}")
print(f"b1d = {to_list(b1d)}")
print("logical_and(a2d, b1d):", to_list(np.logical_and(a2d, b1d)))

# =============================================================================
# BOOLEAN REDUCTIONS
# =============================================================================
print("\n### BOOLEAN REDUCTIONS ###\n")

arr = np.array([0, 0, 1, 0])
print(f"arr = {to_list(arr)}")
print("any(arr):", to_list(np.any(arr)), f"type={type(np.any(arr))}")
print("all(arr):", to_list(np.all(arr)))

arr2d = np.array([[0, 1, 0], [0, 0, 1]])
print(f"\narr2d = {to_list(arr2d)}")
print("any(arr2d, axis=0):", to_list(np.any(arr2d, axis=0)), f"dtype={np.any(arr2d, axis=0).dtype}")
print("any(arr2d, axis=1):", to_list(np.any(arr2d, axis=1)))
print("all(arr2d, axis=0):", to_list(np.all(arr2d, axis=0)))
print("all(arr2d, axis=1):", to_list(np.all(arr2d, axis=1)))

# =============================================================================
# LINEAR ALGEBRA
# =============================================================================
print("\n### LINEAR ALGEBRA ###\n")

# matmul / dot
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(f"A = {to_list(A)}")
print(f"B = {to_list(B)}")
print("matmul(A, B):", to_list(np.matmul(A, B)))
print("dot(A, B):", to_list(np.dot(A, B)))

# Vector dot product
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
print(f"\nv1 = {to_list(v1)}")
print(f"v2 = {to_list(v2)}")
print("dot(v1, v2):", to_list(np.dot(v1, v2)))

# inv, det
print(f"\nA = {to_list(A)}")
print("linalg.inv(A):", to_list(np.linalg.inv(A)))
print("linalg.det(A):", round(to_list(np.linalg.det(A)), 10))

# solve: Ax = b
b = np.array([1, 2])
print(f"\nb = {to_list(b)}")
print("linalg.solve(A, b):", to_list(np.linalg.solve(A, b)))

# eig
eigenvalues, eigenvectors = np.linalg.eig(A)
print("\nlinalg.eig(A):")
print("  eigenvalues:", [round(v, 10) for v in to_list(eigenvalues)])

# svd
U, S, Vh = np.linalg.svd(A)
print("\nlinalg.svd(A):")
print("  S (singular values):", [round(v, 10) for v in to_list(S)])

# qr
Q, R = np.linalg.qr(A.astype(float))
print("\nlinalg.qr(A):")
print("  R:", [[round(v, 10) for v in row] for row in to_list(R)])

# cholesky (needs positive definite matrix)
P = np.array([[4, 2], [2, 3]])
print(f"\nP (positive definite) = {to_list(P)}")
print("linalg.cholesky(P):", to_list(np.linalg.cholesky(P)))

# norm
v = np.array([3, 4])
print(f"\nv = {to_list(v)}")
print("linalg.norm(v):", to_list(np.linalg.norm(v)))
print("linalg.norm(v, ord=1):", to_list(np.linalg.norm(v, ord=1)))
print("linalg.norm(v, ord=np.inf):", to_list(np.linalg.norm(v, ord=np.inf)))

# matrix_rank
print(f"\nA = {to_list(A)}")
print("linalg.matrix_rank(A):", to_list(np.linalg.matrix_rank(A)))

# trace
print("trace(A):", to_list(np.trace(A)))

# cond
print("linalg.cond(A):", round(to_list(np.linalg.cond(A)), 10))

# =============================================================================
# ADVANCED MATH
# =============================================================================
print("\n### ADVANCED MATH ###\n")

# outer
a = np.array([1, 2, 3])
b = np.array([4, 5])
print(f"a = {to_list(a)}")
print(f"b = {to_list(b)}")
print("outer(a, b):", to_list(np.outer(a, b)))

# kron
A = np.array([[1, 2], [3, 4]])
B = np.array([[0, 5], [6, 7]])
print(f"\nA = {to_list(A)}")
print(f"B = {to_list(B)}")
print("kron(A, B):", to_list(np.kron(A, B)))

# percentile
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"\narr = {to_list(arr)}")
print("percentile(arr, 50):", to_list(np.percentile(arr, 50)))
print("percentile(arr, [25, 50, 75]):", to_list(np.percentile(arr, [25, 50, 75])))

# corrcoef
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\nX = {to_list(X)}")
print("corrcoef(X):", [[round(v, 10) for v in row] for row in to_list(np.corrcoef(X))])

# =============================================================================
# RANDOM (deterministic with seed)
# =============================================================================
print("\n### RANDOM (with seed=42) ###\n")

np.random.seed(42)
print("random.random([3]):", to_list(np.random.random(3)))
np.random.seed(42)
print("random.uniform(0, 10, [3]):", to_list(np.random.uniform(0, 10, 3)))
np.random.seed(42)
print("random.normal(0, 1, [3]):", [round(v, 10) for v in to_list(np.random.normal(0, 1, 3))])
np.random.seed(42)
print("random.randint(0, 10, [5]):", to_list(np.random.randint(0, 10, 5)))

print("\n" + "=" * 70)
print("Verification Complete")
print("=" * 70)
