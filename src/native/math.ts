/**
 * Math operations - thin wrapper around native math module
 */

import { native } from './loader.js';
import { NDArray } from './ndarray.js';

const { math } = native;

// ============================================================
// Arithmetic Operations
// ============================================================

/**
 * Element-wise addition
 */
export function add(a: NDArray, b: NDArray | number): NDArray {
  return new NDArray(math.add(a._native, b instanceof NDArray ? b._native : b));
}

/**
 * Element-wise subtraction
 */
export function subtract(a: NDArray, b: NDArray | number): NDArray {
  return new NDArray(math.subtract(a._native, b instanceof NDArray ? b._native : b));
}

/**
 * Element-wise multiplication
 */
export function multiply(a: NDArray, b: NDArray | number): NDArray {
  return new NDArray(math.multiply(a._native, b instanceof NDArray ? b._native : b));
}

/**
 * Element-wise division
 */
export function divide(a: NDArray, b: NDArray | number): NDArray {
  return new NDArray(math.divide(a._native, b instanceof NDArray ? b._native : b));
}

/**
 * Element-wise power
 */
export function power(a: NDArray, b: NDArray | number): NDArray {
  return new NDArray(math.power(a._native, b instanceof NDArray ? b._native : b));
}

// ============================================================
// Unary Math Functions
// ============================================================

/**
 * Element-wise square root
 */
export function sqrt(a: NDArray): NDArray {
  return new NDArray(math.sqrt(a._native));
}

/**
 * Element-wise exponential
 */
export function exp(a: NDArray): NDArray {
  return new NDArray(math.exp(a._native));
}

/**
 * Element-wise natural logarithm
 */
export function log(a: NDArray): NDArray {
  return new NDArray(math.log(a._native));
}

/**
 * Element-wise sine
 */
export function sin(a: NDArray): NDArray {
  return new NDArray(math.sin(a._native));
}

/**
 * Element-wise cosine
 */
export function cos(a: NDArray): NDArray {
  return new NDArray(math.cos(a._native));
}

/**
 * Element-wise tangent
 */
export function tan(a: NDArray): NDArray {
  return new NDArray(math.tan(a._native));
}

// ============================================================
// Reduction Operations
// ============================================================

/**
 * Sum of array elements
 */
export function sum(a: NDArray, axis?: number): NDArray | number {
  const result = math.sum(a._native, axis);
  if (typeof result === 'number') {
    return result;
  }
  return new NDArray(result);
}

/**
 * Mean of array elements
 */
export function mean(a: NDArray, axis?: number): NDArray | number {
  const result = math.mean(a._native, axis);
  if (typeof result === 'number') {
    return result;
  }
  return new NDArray(result);
}

/**
 * Standard deviation of array elements
 */
export function std(a: NDArray, axis?: number): NDArray | number {
  const result = math.std(a._native, axis);
  if (typeof result === 'number') {
    return result;
  }
  return new NDArray(result);
}

/**
 * Variance of array elements
 */
export function variance(a: NDArray, axis?: number): NDArray | number {
  const result = math.var(a._native, axis);
  if (typeof result === 'number') {
    return result;
  }
  return new NDArray(result);
}

/**
 * Median of array elements
 */
export function median(a: NDArray): number {
  return math.median(a._native);
}

/**
 * Minimum of array elements
 */
export function min(a: NDArray, axis?: number): NDArray | number {
  const result = math.min(a._native, axis);
  if (typeof result === 'number') {
    return result;
  }
  return new NDArray(result);
}

/**
 * Maximum of array elements
 */
export function max(a: NDArray, axis?: number): NDArray | number {
  const result = math.max(a._native, axis);
  if (typeof result === 'number') {
    return result;
  }
  return new NDArray(result);
}

/**
 * Product of array elements
 */
export function prod(a: NDArray, axis?: number): NDArray | number {
  const result = math.prod(a._native, axis);
  if (typeof result === 'number') {
    return result;
  }
  return new NDArray(result);
}

// ============================================================
// Additional Math Functions
// ============================================================

/**
 * Element-wise absolute value
 */
export function abs(a: NDArray): NDArray {
  return new NDArray(math.abs(a._native));
}

/**
 * Negative of array elements
 */
export function negative(a: NDArray): NDArray {
  return multiply(a, -1);
}

// ============================================================
// Fused Statistical Operations
// ============================================================

/**
 * Z-score normalization: (X - mean) / std
 * Computes mean, std, and normalizes in a single native call.
 */
export function zscore(a: NDArray, axis?: number): NDArray {
  return new NDArray(math.zscore(a._native, axis));
}

/**
 * Correlation coefficient matrix
 * Computes standardized X'X / (n-1) in a single native call.
 */
export function corrcoef(a: NDArray): NDArray {
  return new NDArray(math.corrcoef(a._native));
}

/**
 * Gram matrix: X @ X.T
 * Uses dsyrk for efficient symmetric matrix computation.
 */
export function gram_matrix(a: NDArray): NDArray {
  return new NDArray(math.gram_matrix(a._native));
}

/**
 * Softmax function: exp(x) / sum(exp(x))
 * Computed in a single native call with numerical stability.
 */
export function softmax(a: NDArray): NDArray {
  return new NDArray(math.softmax(a._native));
}

/**
 * Pairwise squared Euclidean distances.
 * Computes D_ij = ||x_i - x_j||^2 for all pairs of points.
 */
export function pdist_sq(a: NDArray): NDArray {
  return new NDArray(math.pdist_sq(a._native));
}

/**
 * Affine transform: gamma * x + beta
 * Fused multiply-add with row broadcasting.
 * Common in layer normalization and batch normalization.
 */
export function affine(x: NDArray, gamma: NDArray, beta: NDArray): NDArray {
  return new NDArray(math.affine(x._native, gamma._native, beta._native));
}

/**
 * Row-wise division: result[i,j] = x[i,j] / scales[i]
 * Useful for normalizing vectors (e.g., dividing each row by its norm).
 * @param x Input 2D array [m, n]
 * @param scales 1D array of scale values [m]
 */
export function row_divide(x: NDArray, scales: NDArray): NDArray {
  return new NDArray(math.row_divide(x._native, scales._native));
}

/**
 * Compute X.T @ X without explicit transpose.
 * Uses BLAS dsyrk for efficient symmetric matrix computation.
 */
export function xtx(x: NDArray): NDArray {
  return new NDArray(math.xtx(x._native));
}

/**
 * Compute X.T @ y without explicit transpose.
 * Uses BLAS dgemv with transpose flag.
 */
export function xty(x: NDArray, y: NDArray): NDArray {
  return new NDArray(math.xty(x._native, y._native));
}

/**
 * Compute percentiles using quickselect algorithm - O(n) per percentile.
 * @param a Input array
 * @param q Array of percentile values (0-100)
 * @param axis Axis along which to compute percentiles (default: flatten and compute globally)
 */
export function percentile(a: NDArray, q: number[], axis?: number): NDArray {
  return new NDArray(math.percentile(a._native, q, axis));
}

/**
 * Min-max scaling: (X - min) / (max - min)
 * Fused operation that computes min, max, and scales in a single native call.
 * @param a Input 2D array
 * @param axis Axis along which to scale (0=columns, 1=rows, default=0)
 */
export function minmax_scale(a: NDArray, axis?: number): NDArray {
  return new NDArray(math.minmax_scale(a._native, axis));
}

/**
 * Kronecker product: A ⊗ B
 * For A (m×n) and B (p×q), result is (m*p × n*q).
 * Each element a[i,j] is multiplied by the entire B matrix.
 */
export function kron(a: NDArray, b: NDArray): NDArray {
  return new NDArray(math.kron(a._native, b._native));
}

/**
 * Outer product: a ⊗ b = a * b.T
 * For vectors a (m,) and b (n,), result is (m × n).
 * Uses BLAS dger for optimal performance.
 */
export function outer(a: NDArray, b: NDArray): NDArray {
  return new NDArray(math.outer(a._native, b._native));
}

/**
 * Matrix exponential using Taylor series: exp(A) = I + A + A²/2! + A³/3! + ...
 * Fused native implementation with BLAS acceleration.
 * @param a Input square matrix
 * @param numTerms Number of Taylor series terms (default 10)
 */
export function matrix_exp(a: NDArray, numTerms?: number): NDArray {
  return new NDArray(math.matrix_exp(a._native, numTerms));
}

/**
 * BLAS-style axpby: result = alpha*x + beta*y
 * Fuses scalar multiply and addition into one operation.
 * If beta and y are not provided, computes alpha*x (scalar multiply).
 */
export function axpby(alpha: number, x: NDArray, beta?: number, y?: NDArray): NDArray {
  if (beta !== undefined && y !== undefined) {
    return new NDArray(math.axpby(alpha, x._native, beta, y._native));
  }
  return new NDArray(math.axpby(alpha, x._native));
}

/**
 * Matrix-vector multiply: y = A @ x
 * Uses BLAS dgemv for optimal performance.
 */
export function matvec(A: NDArray, x: NDArray): NDArray {
  return new NDArray(math.matvec(A._native, x._native));
}

/**
 * Compute squared L2 norms along an axis.
 * Fuses multiply and sum into one operation: sum(x^2, axis)
 * @param x Input array
 * @param axis Axis along which to compute (0=columns, 1=rows, default=1)
 */
export function norm_sq(x: NDArray, axis?: number): NDArray | number {
  const result = math.norm_sq(x._native, axis);
  if (typeof result === 'number') {
    return result;
  }
  return new NDArray(result);
}

/**
 * Fused Jacobi iteration step: x_new = (b - R @ x) / D
 * Combines matvec, subtract, and element-wise divide in one native call.
 * @param R Matrix (n x n) - the off-diagonal part of A
 * @param x Current solution vector (n)
 * @param b Right-hand side vector (n)
 * @param D Diagonal elements of A (n)
 */
export function jacobi_step(R: NDArray, x: NDArray, b: NDArray, D: NDArray): NDArray {
  return new NDArray(math.jacobi_step(R._native, x._native, b._native, D._native));
}

/**
 * Compute 2D gradients using central differences: df/dx and df/dy
 * Native implementation with loop unrolling for performance.
 * @param f Input 2D array
 * @param h Grid spacing (default 1.0)
 */
export function gradient_2d(f: NDArray, h?: number): { dfdx: NDArray; dfdy: NDArray } {
  const result = math.gradient_2d(f._native, h);
  return {
    dfdx: new NDArray(result.dfdx),
    dfdy: new NDArray(result.dfdy),
  };
}
