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
// In-place Arithmetic Operations
// These modify the first argument and return it (for chaining)
// ============================================================

/**
 * In-place element-wise addition: a += b
 * Modifies `a` directly and returns it
 */
export function add_inplace(a: NDArray, b: NDArray | number): NDArray {
  math.add_inplace(a._native, b instanceof NDArray ? b._native : b);
  return a;
}

/**
 * In-place element-wise subtraction: a -= b
 * Modifies `a` directly and returns it
 */
export function subtract_inplace(a: NDArray, b: NDArray | number): NDArray {
  math.subtract_inplace(a._native, b instanceof NDArray ? b._native : b);
  return a;
}

/**
 * In-place element-wise multiplication: a *= b
 * Modifies `a` directly and returns it
 */
export function multiply_inplace(a: NDArray, b: NDArray | number): NDArray {
  math.multiply_inplace(a._native, b instanceof NDArray ? b._native : b);
  return a;
}

/**
 * In-place element-wise division: a /= b
 * Modifies `a` directly and returns it
 */
export function divide_inplace(a: NDArray, b: NDArray | number): NDArray {
  math.divide_inplace(a._native, b instanceof NDArray ? b._native : b);
  return a;
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
 * Compute percentiles using quickselect algorithm - O(n) per percentile.
 * @param a Input array
 * @param q Array of percentile values (0-100)
 * @param axis Axis along which to compute percentiles (default: flatten and compute globally)
 */
export function percentile(a: NDArray, q: number[], axis?: number): NDArray {
  return new NDArray(math.percentile(a._native, q, axis));
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

// ============================================================
// Comparison Operators (return boolean arrays)
// ============================================================

/**
 * Element-wise equality comparison
 * Returns a boolean array where each element is true if a[i] == b[i]
 */
export function equal(a: NDArray, b: NDArray | number): NDArray {
  return new NDArray(math.equal(a._native, b instanceof NDArray ? b._native : b));
}

/**
 * Element-wise inequality comparison
 * Returns a boolean array where each element is true if a[i] != b[i]
 */
export function not_equal(a: NDArray, b: NDArray | number): NDArray {
  return new NDArray(math.not_equal(a._native, b instanceof NDArray ? b._native : b));
}

/**
 * Element-wise less-than comparison
 * Returns a boolean array where each element is true if a[i] < b[i]
 */
export function less(a: NDArray, b: NDArray | number): NDArray {
  return new NDArray(math.less(a._native, b instanceof NDArray ? b._native : b));
}

/**
 * Element-wise less-than-or-equal comparison
 * Returns a boolean array where each element is true if a[i] <= b[i]
 */
export function less_equal(a: NDArray, b: NDArray | number): NDArray {
  return new NDArray(math.less_equal(a._native, b instanceof NDArray ? b._native : b));
}

/**
 * Element-wise greater-than comparison
 * Returns a boolean array where each element is true if a[i] > b[i]
 */
export function greater(a: NDArray, b: NDArray | number): NDArray {
  return new NDArray(math.greater(a._native, b instanceof NDArray ? b._native : b));
}

/**
 * Element-wise greater-than-or-equal comparison
 * Returns a boolean array where each element is true if a[i] >= b[i]
 */
export function greater_equal(a: NDArray, b: NDArray | number): NDArray {
  return new NDArray(math.greater_equal(a._native, b instanceof NDArray ? b._native : b));
}

// ============================================================
// Logical Operators (return boolean arrays)
// ============================================================

/**
 * Element-wise logical AND
 * Returns a boolean array where each element is true if a[i] && b[i]
 */
export function logical_and(a: NDArray, b: NDArray): NDArray {
  return new NDArray(math.logical_and(a._native, b._native));
}

/**
 * Element-wise logical OR
 * Returns a boolean array where each element is true if a[i] || b[i]
 */
export function logical_or(a: NDArray, b: NDArray): NDArray {
  return new NDArray(math.logical_or(a._native, b._native));
}

/**
 * Element-wise logical XOR
 * Returns a boolean array where each element is true if a[i] != b[i] (as booleans)
 */
export function logical_xor(a: NDArray, b: NDArray): NDArray {
  return new NDArray(math.logical_xor(a._native, b._native));
}

/**
 * Element-wise logical NOT
 * Returns a boolean array where each element is !a[i]
 */
export function logical_not(a: NDArray): NDArray {
  return new NDArray(math.logical_not(a._native));
}

// ============================================================
// Boolean Reductions
// ============================================================

/**
 * Test whether any element is truthy
 * @param a Input array
 * @param axis Axis along which to reduce (optional)
 * @returns true if any element is truthy, or an NDArray if axis is specified
 */
export function any(a: NDArray, axis?: number): NDArray | boolean {
  const result = math.any(a._native, axis);
  if (typeof result === 'boolean') {
    return result;
  }
  return new NDArray(result);
}

/**
 * Test whether all elements are truthy
 * @param a Input array
 * @param axis Axis along which to reduce (optional)
 * @returns true if all elements are truthy, or an NDArray if axis is specified
 */
export function all(a: NDArray, axis?: number): NDArray | boolean {
  const result = math.all(a._native, axis);
  if (typeof result === 'boolean') {
    return result;
  }
  return new NDArray(result);
}
