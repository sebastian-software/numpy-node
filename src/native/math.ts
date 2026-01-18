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
