#!/usr/bin/env python3
"""
NumPy Reference Value Generator

Generates JSON file with expected NumPy outputs for conformity testing.
Run: python3 scripts/generate_numpy_reference.py

Output: scripts/numpy_reference.json
"""

import numpy as np
import json
from typing import Any

def to_serializable(obj) -> Any:
    """Convert NumPy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return {
            "data": obj.tolist(),
            "dtype": str(obj.dtype),
            "shape": list(obj.shape)
        }
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj):
            return "NaN"
        elif np.isinf(obj):
            return "Inf" if obj > 0 else "-Inf"
        return float(obj)
    elif isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    return obj

def generate_reference():
    """Generate all reference values."""
    ref = {
        "numpy_version": np.__version__,
        "tests": {}
    }

    # =========================================================================
    # ARRAY CREATION
    # =========================================================================
    ref["tests"]["array_creation"] = {
        "zeros_1d": to_serializable(np.zeros(5)),
        "zeros_2d": to_serializable(np.zeros((2, 3))),
        "zeros_int32": to_serializable(np.zeros(3, dtype=np.int32)),
        "ones_1d": to_serializable(np.ones(5)),
        "ones_2d": to_serializable(np.ones((2, 3))),
        "full_int": to_serializable(np.full(5, 7)),
        "full_float": to_serializable(np.full((2, 3), 3.14)),
        "arange_simple": to_serializable(np.arange(5)),
        "arange_start_stop": to_serializable(np.arange(2, 7)),
        "arange_step": to_serializable(np.arange(0, 1, 0.2)),
        "linspace": to_serializable(np.linspace(0, 1, 5)),
        "linspace_10": to_serializable(np.linspace(0, 10, 3)),
        "eye_3": to_serializable(np.eye(3)),
        "eye_3x4": to_serializable(np.eye(3, 4)),
        "eye_3x4_k1": to_serializable(np.eye(3, 4, k=1)),
    }

    # =========================================================================
    # ARITHMETIC OPERATIONS
    # =========================================================================
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

    ref["tests"]["arithmetic"] = {
        "input_a": to_serializable(a),
        "input_b": to_serializable(b),
        "add_arrays": to_serializable(np.add(a, b)),
        "add_scalar": to_serializable(np.add(a, 10)),
        "subtract": to_serializable(np.subtract(a, b)),
        "multiply": to_serializable(np.multiply(a, b)),
        "divide": to_serializable(np.divide(a, b)),
        "power_scalar": to_serializable(np.power(a, 2)),
        "power_array": to_serializable(np.power(a, b)),
    }

    # Broadcasting
    a2d = np.array([[1, 2, 3], [4, 5, 6]])
    b1d = np.array([1, 2, 3])
    ref["tests"]["arithmetic"]["broadcast_add"] = to_serializable(np.add(a2d, b1d))
    ref["tests"]["arithmetic"]["broadcast_subtract"] = to_serializable(np.subtract(a2d, b1d))

    # =========================================================================
    # UNARY MATH FUNCTIONS
    # =========================================================================
    x = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
    angles = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])

    ref["tests"]["unary_math"] = {
        "input_x": to_serializable(x),
        "sqrt": to_serializable(np.sqrt(x)),
        "exp": to_serializable(np.exp(x)),
        "log_e": to_serializable(np.log([1, np.e, np.e**2])),
        "sin": to_serializable(np.sin(angles)),
        "cos": to_serializable(np.cos(angles)),
        "abs": to_serializable(np.abs([-1, 2, -3, 4])),
        "negative": to_serializable(np.negative([1, -2, 3])),
        # Rounding functions
        "round": to_serializable(np.round([-1.7, -0.5, 0.5, 1.3, 2.5])),
        "floor": to_serializable(np.floor([-1.7, -0.5, 0.5, 1.3, 2.5])),
        "ceil": to_serializable(np.ceil([-1.7, -0.5, 0.5, 1.3, 2.5])),
    }

    # =========================================================================
    # REDUCTION OPERATIONS
    # =========================================================================
    arr = np.array([1, 2, 3, 4, 5])
    arr2d = np.array([[1, 2, 3], [4, 5, 6]])

    ref["tests"]["reductions"] = {
        "input_1d": to_serializable(arr),
        "input_2d": to_serializable(arr2d),
        "sum": to_serializable(np.sum(arr)),
        "prod": to_serializable(np.prod(arr)),
        "mean": to_serializable(np.mean(arr)),
        "std": to_serializable(np.std(arr)),
        "var": to_serializable(np.var(arr)),
        "min": to_serializable(np.min(arr)),
        "max": to_serializable(np.max(arr)),
        "median": to_serializable(np.median(arr)),
        "argmin": to_serializable(np.argmin(arr)),
        "argmax": to_serializable(np.argmax(arr)),
        "sum_axis0": to_serializable(np.sum(arr2d, axis=0)),
        "sum_axis1": to_serializable(np.sum(arr2d, axis=1)),
        "mean_axis0": to_serializable(np.mean(arr2d, axis=0)),
        "mean_axis1": to_serializable(np.mean(arr2d, axis=1)),
        "min_axis0": to_serializable(np.min(arr2d, axis=0)),
        "max_axis1": to_serializable(np.max(arr2d, axis=1)),
        "argmin_axis0": to_serializable(np.argmin(arr2d, axis=0)),
        "argmin_axis1": to_serializable(np.argmin(arr2d, axis=1)),
        "argmax_axis0": to_serializable(np.argmax(arr2d, axis=0)),
        "argmax_axis1": to_serializable(np.argmax(arr2d, axis=1)),
        # Cumulative operations
        "cumsum_1d": to_serializable(np.cumsum(arr)),
        "cumsum_2d_flat": to_serializable(np.cumsum(arr2d)),
        "cumsum_axis0": to_serializable(np.cumsum(arr2d, axis=0)),
        "cumsum_axis1": to_serializable(np.cumsum(arr2d, axis=1)),
        "cumprod_1d": to_serializable(np.cumprod(arr)),
        "cumprod_2d_flat": to_serializable(np.cumprod(arr2d)),
        "cumprod_axis0": to_serializable(np.cumprod(arr2d, axis=0)),
        "cumprod_axis1": to_serializable(np.cumprod(arr2d, axis=1)),
    }

    # =========================================================================
    # COMPARISON OPERATORS
    # =========================================================================
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([5, 4, 3, 2, 1])

    ref["tests"]["comparison"] = {
        "input_a": to_serializable(a),
        "input_b": to_serializable(b),
        "equal_arrays": to_serializable(np.equal(a, b)),
        "equal_scalar": to_serializable(np.equal(a, 3)),
        "not_equal": to_serializable(np.not_equal(a, b)),
        "less_scalar": to_serializable(np.less(a, 3)),
        "less_equal_scalar": to_serializable(np.less_equal(a, 3)),
        "greater_scalar": to_serializable(np.greater(a, 3)),
        "greater_equal_scalar": to_serializable(np.greater_equal(a, 3)),
    }

    # =========================================================================
    # LOGICAL OPERATORS
    # =========================================================================
    x = np.array([True, True, False, False])
    y = np.array([True, False, True, False])

    ref["tests"]["logical"] = {
        "input_x": to_serializable(x),
        "input_y": to_serializable(y),
        "and": to_serializable(np.logical_and(x, y)),
        "or": to_serializable(np.logical_or(x, y)),
        "xor": to_serializable(np.logical_xor(x, y)),
        "not": to_serializable(np.logical_not(x)),
    }

    # Broadcasting for logical
    a2d = np.array([[1, 0], [1, 1]])
    b1d = np.array([1, 0])
    ref["tests"]["logical"]["broadcast_and"] = to_serializable(np.logical_and(a2d, b1d))

    # =========================================================================
    # ARRAY MANIPULATION
    # =========================================================================
    arr = np.array([-2, 5, 3, -1, 8])
    arr2d = np.array([[1, 8, 3], [7, 2, 9]])

    ref["tests"]["array_manipulation"] = {
        "input_1d": to_serializable(arr),
        "input_2d": to_serializable(arr2d),
        # clip
        "clip_basic": to_serializable(np.clip(arr, 0, 5)),
        "clip_2d": to_serializable(np.clip(arr2d, 2, 7)),
        # where - same shape
        "where_basic": to_serializable(np.where(
            np.array([True, False, True, False, True]),
            np.array([1, 2, 3, 4, 5]),
            np.array([10, 20, 30, 40, 50])
        )),
        # where - with comparison
        "where_comparison": to_serializable(np.where(
            arr > 0,
            arr,
            np.zeros(5)
        )),
        # where - broadcasting scalar-like
        "where_broadcast": to_serializable(np.where(
            np.array([[True, False], [False, True]]),
            np.array([[1, 2], [3, 4]]),
            np.array([10, 20])  # broadcasts along axis 0
        )),
        # squeeze
        "squeeze_all": to_serializable(np.squeeze(np.array([[[1, 2, 3]]]))),  # (1, 1, 3) -> (3,)
        "squeeze_axis": to_serializable(np.squeeze(np.array([[1], [2], [3]]), axis=1)),  # (3, 1) -> (3,)
        # expand_dims
        "expand_dims_0": to_serializable(np.expand_dims(np.array([1, 2, 3]), axis=0)),  # (3,) -> (1, 3)
        "expand_dims_1": to_serializable(np.expand_dims(np.array([1, 2, 3]), axis=1)),  # (3,) -> (3, 1)
        "expand_dims_neg": to_serializable(np.expand_dims(np.array([1, 2, 3]), axis=-1)),  # (3,) -> (3, 1)
    }

    # =========================================================================
    # ARRAY JOINING
    # =========================================================================
    a1d_1 = np.array([1, 2, 3])
    a1d_2 = np.array([4, 5, 6])
    a2d_1 = np.array([[1, 2], [3, 4]])
    a2d_2 = np.array([[5, 6], [7, 8]])

    ref["tests"]["array_joining"] = {
        "input_1d_1": to_serializable(a1d_1),
        "input_1d_2": to_serializable(a1d_2),
        "input_2d_1": to_serializable(a2d_1),
        "input_2d_2": to_serializable(a2d_2),
        # concatenate
        "concat_1d": to_serializable(np.concatenate([a1d_1, a1d_2])),
        "concat_2d_axis0": to_serializable(np.concatenate([a2d_1, a2d_2], axis=0)),
        "concat_2d_axis1": to_serializable(np.concatenate([a2d_1, a2d_2], axis=1)),
        # stack
        "stack_1d_axis0": to_serializable(np.stack([a1d_1, a1d_2], axis=0)),
        "stack_1d_axis1": to_serializable(np.stack([a1d_1, a1d_2], axis=1)),
        "stack_2d_axis0": to_serializable(np.stack([a2d_1, a2d_2], axis=0)),
        "stack_2d_axis1": to_serializable(np.stack([a2d_1, a2d_2], axis=1)),
        # vstack
        "vstack_1d": to_serializable(np.vstack([a1d_1, a1d_2])),
        "vstack_2d": to_serializable(np.vstack([a2d_1, a2d_2])),
        # hstack
        "hstack_1d": to_serializable(np.hstack([a1d_1, a1d_2])),
        "hstack_2d": to_serializable(np.hstack([a2d_1, a2d_2])),
    }

    # =========================================================================
    # SORTING AND SEARCHING
    # =========================================================================
    sort_arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
    sort_2d = np.array([[3, 1, 2], [6, 4, 5]])
    diff_arr = np.array([1, 3, 6, 10, 15])
    unique_arr = np.array([3, 1, 2, 1, 3, 2, 4, 1])

    ref["tests"]["sorting_searching"] = {
        "input_1d": to_serializable(sort_arr),
        "input_2d": to_serializable(sort_2d),
        "diff_input": to_serializable(diff_arr),
        "unique_input": to_serializable(unique_arr),
        # diff
        "diff_1d": to_serializable(np.diff(diff_arr)),
        "diff_1d_n2": to_serializable(np.diff(diff_arr, n=2)),
        "diff_2d_axis0": to_serializable(np.diff(sort_2d, axis=0)),
        "diff_2d_axis1": to_serializable(np.diff(sort_2d, axis=1)),
        # sort
        "sort_1d": to_serializable(np.sort(sort_arr)),
        "sort_2d_axis0": to_serializable(np.sort(sort_2d, axis=0)),
        "sort_2d_axis1": to_serializable(np.sort(sort_2d, axis=1)),
        # argsort
        "argsort_1d": to_serializable(np.argsort(sort_arr)),
        "argsort_2d_axis0": to_serializable(np.argsort(sort_2d, axis=0)),
        "argsort_2d_axis1": to_serializable(np.argsort(sort_2d, axis=1)),
        # unique
        "unique": to_serializable(np.unique(unique_arr)),
        # searchsorted
        "searchsorted_left": to_serializable(np.searchsorted([1, 2, 4, 5, 7], [0, 2, 3, 6, 8], side='left')),
        "searchsorted_right": to_serializable(np.searchsorted([1, 2, 4, 5, 7], [0, 2, 3, 6, 8], side='right')),
    }

    # =========================================================================
    # ARRAY MANIPULATION (Tier 3+4)
    # =========================================================================
    arr_1d = np.array([1, 2, 3])
    arr_2d = np.array([[1, 2], [3, 4]])
    arr_sparse = np.array([0, 1, 0, 2, 0, 3])

    ref["tests"]["array_manipulation_advanced"] = {
        "input_1d": to_serializable(arr_1d),
        "input_2d": to_serializable(arr_2d),
        # tile
        "tile_1d_2": to_serializable(np.tile(arr_1d, 2)),
        "tile_1d_tuple": to_serializable(np.tile(arr_1d, (2, 3))),
        "tile_2d_rows": to_serializable(np.tile(arr_2d, (2, 1))),
        "tile_2d_cols": to_serializable(np.tile(arr_2d, (1, 3))),
        # repeat
        "repeat_1d": to_serializable(np.repeat(arr_1d, 3)),
        "repeat_2d_axis0": to_serializable(np.repeat(arr_2d, 2, axis=0)),
        "repeat_2d_axis1": to_serializable(np.repeat(arr_2d, 2, axis=1)),
        # flip
        "flip_1d": to_serializable(np.flip(arr_1d)),
        "flip_2d": to_serializable(np.flip(arr_2d)),
        "flip_2d_axis0": to_serializable(np.flip(arr_2d, axis=0)),
        "flip_2d_axis1": to_serializable(np.flip(arr_2d, axis=1)),
        # rot90
        "rot90_k1": to_serializable(np.rot90(arr_2d)),
        "rot90_k2": to_serializable(np.rot90(arr_2d, k=2)),
        "rot90_k3": to_serializable(np.rot90(arr_2d, k=3)),
        # sign
        "sign": to_serializable(np.sign([-5, 0, 3, -2, 7])),
        # mod
        "mod_scalar": to_serializable(np.mod([5, 7, 9, 11], 3)),
        "mod_array": to_serializable(np.mod([5, 7, 9, 11], [2, 3, 4, 5])),
        # split
        "split_equal": to_serializable([x.tolist() for x in np.split(np.arange(9), 3)]),
        "split_indices": to_serializable([x.tolist() for x in np.split(np.arange(9), [2, 5])]),
        # nonzero
        "nonzero_1d": to_serializable(np.nonzero(arr_sparse)),
        "nonzero_2d": to_serializable(np.nonzero(np.array([[0, 1, 0], [2, 0, 3]]))),
    }

    # =========================================================================
    # APPROXIMATE COMPARISON
    # =========================================================================
    a_close = np.array([1.0, 2.0, 3.0])
    b_close = np.array([1.0, 2.00001, 3.0001])
    b_not_close = np.array([1.0, 2.1, 3.5])

    ref["tests"]["approximate_comparison"] = {
        "input_a": to_serializable(a_close),
        "input_b_close": to_serializable(b_close),
        "input_b_not_close": to_serializable(b_not_close),
        "allclose_true": to_serializable(np.allclose(a_close, b_close)),
        "allclose_false": to_serializable(np.allclose(a_close, b_not_close)),
        "allclose_custom_tol": to_serializable(np.allclose(a_close, b_not_close, rtol=0.5, atol=0.5)),
        "isclose_close": to_serializable(np.isclose(a_close, b_close)),
        "isclose_not_close": to_serializable(np.isclose(a_close, b_not_close)),
    }

    # =========================================================================
    # BOOLEAN REDUCTIONS
    # =========================================================================
    arr = np.array([0, 0, 1, 0])
    arr2d = np.array([[0, 1, 0], [0, 0, 1]])

    ref["tests"]["boolean_reductions"] = {
        "input_1d": to_serializable(arr),
        "input_2d": to_serializable(arr2d),
        "any": to_serializable(np.any(arr)),
        "all": to_serializable(np.all(arr)),
        "any_axis0": to_serializable(np.any(arr2d, axis=0)),
        "any_axis1": to_serializable(np.any(arr2d, axis=1)),
        "all_axis0": to_serializable(np.all(arr2d, axis=0)),
        "all_axis1": to_serializable(np.all(arr2d, axis=1)),
    }

    # =========================================================================
    # LINEAR ALGEBRA
    # =========================================================================
    A = np.array([[1, 2], [3, 4]], dtype=float)
    B = np.array([[5, 6], [7, 8]], dtype=float)
    v1 = np.array([1, 2, 3], dtype=float)
    v2 = np.array([4, 5, 6], dtype=float)
    b_vec = np.array([1, 2], dtype=float)
    P = np.array([[4, 2], [2, 3]], dtype=float)  # positive definite
    v = np.array([3, 4], dtype=float)

    ref["tests"]["linalg"] = {
        "input_A": to_serializable(A),
        "input_B": to_serializable(B),
        "matmul": to_serializable(np.matmul(A, B)),
        "dot_matrix": to_serializable(np.dot(A, B)),
        "dot_vector": to_serializable(np.dot(v1, v2)),
        "inv": to_serializable(np.linalg.inv(A)),
        "det": to_serializable(np.linalg.det(A)),
        "solve": to_serializable(np.linalg.solve(A, b_vec)),
        "trace": to_serializable(np.trace(A)),
        "norm_l2": to_serializable(np.linalg.norm(v)),
        "norm_l1": to_serializable(np.linalg.norm(v, ord=1)),
        "norm_inf": to_serializable(np.linalg.norm(v, ord=np.inf)),
        "matrix_rank": to_serializable(np.linalg.matrix_rank(A)),
        "cond": to_serializable(np.linalg.cond(A)),
    }

    # Eigenvalues (sorted for comparison)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    idx = np.argsort(eigenvalues)
    ref["tests"]["linalg"]["eigenvalues"] = to_serializable(eigenvalues[idx])

    # SVD singular values
    U, S, Vh = np.linalg.svd(A)
    ref["tests"]["linalg"]["svd_S"] = to_serializable(S)

    # Cholesky
    ref["tests"]["linalg"]["cholesky"] = to_serializable(np.linalg.cholesky(P))

    # =========================================================================
    # ADVANCED MATH
    # =========================================================================
    a = np.array([1, 2, 3])
    b = np.array([4, 5])
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[0, 5], [6, 7]])
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    ref["tests"]["advanced_math"] = {
        "outer": to_serializable(np.outer(a, b)),
        "kron": to_serializable(np.kron(A, B)),
        "percentile_50": to_serializable(np.percentile(arr, 50)),
        "percentile_quartiles": to_serializable(np.percentile(arr, [25, 50, 75])),
    }

    # Correlation coefficient
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    ref["tests"]["advanced_math"]["corrcoef"] = to_serializable(np.corrcoef(X))

    # =========================================================================
    # RANDOM (deterministic with seed)
    # =========================================================================
    np.random.seed(42)
    random_vals = np.random.random(3)
    np.random.seed(42)
    uniform_vals = np.random.uniform(0, 10, 3)
    np.random.seed(42)
    normal_vals = np.random.normal(0, 1, 3)
    np.random.seed(42)
    randint_vals = np.random.randint(0, 10, 5)

    ref["tests"]["random"] = {
        "seed": 42,
        "random": to_serializable(random_vals),
        "uniform": to_serializable(uniform_vals),
        "normal": to_serializable(normal_vals),
        "randint": to_serializable(randint_vals),
    }

    # =========================================================================
    # FFT (Fast Fourier Transform)
    # =========================================================================
    fft_input = np.array([1.0, 2.0, 3.0, 4.0])
    fft_result = np.fft.fft(fft_input)

    ref["tests"]["fft"] = {
        "input": to_serializable(fft_input),
        "fft_real": to_serializable(fft_result.real),
        "fft_imag": to_serializable(fft_result.imag),
        "ifft_real": to_serializable(np.fft.ifft(fft_result).real),
        "ifft_imag": to_serializable(np.fft.ifft(fft_result).imag),
        "rfft_real": to_serializable(np.fft.rfft(fft_input).real),
        "rfft_imag": to_serializable(np.fft.rfft(fft_input).imag),
        "irfft": to_serializable(np.fft.irfft(np.fft.rfft(fft_input), n=4)),
        "fftfreq": to_serializable(np.fft.fftfreq(4, 1)),
        "rfftfreq": to_serializable(np.fft.rfftfreq(4, 1)),
    }

    # =========================================================================
    # EINSUM (Einstein Summation)
    # =========================================================================
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])

    ref["tests"]["einsum"] = {
        "input_A": to_serializable(A),
        "input_B": to_serializable(B),
        "input_a": to_serializable(a),
        "input_b": to_serializable(b),
        "matmul": to_serializable(np.einsum('ij,jk->ik', A, B)),
        "trace": to_serializable(np.einsum('ii', A)),
        "diagonal": to_serializable(np.einsum('ii->i', A)),
        "inner": to_serializable(np.einsum('i,i', a, b)),
        "outer": to_serializable(np.einsum('i,j->ij', a, b)),
        "transpose": to_serializable(np.einsum('ij->ji', A)),
        "batch_diag": to_serializable(np.einsum('ij,ij->i', A, B)),
    }

    return ref

if __name__ == "__main__":
    import os

    reference = generate_reference()

    # Write to JSON file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "numpy_reference.json")

    with open(output_path, "w") as f:
        json.dump(reference, f, indent=2)

    print(f"NumPy Reference Generated")
    print(f"  Version: {reference['numpy_version']}")
    print(f"  Output: {output_path}")
    print(f"  Test categories: {len(reference['tests'])}")

    total_tests = sum(len(v) for v in reference['tests'].values())
    print(f"  Total test values: {total_tests}")
