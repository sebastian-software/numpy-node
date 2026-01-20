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
 * Supports both (array, scalar) and (scalar, array) - commutative
 */
export function add(a: NDArray | number, b: NDArray | number): NDArray {
  // Commutative: swap if scalar-first
  if (typeof a === 'number') {
    if (!(b instanceof NDArray)) {
      throw new Error('At least one argument must be an NDArray');
    }
    return new NDArray(math.add(b._native, a));
  }
  return new NDArray(math.add(a._native, b instanceof NDArray ? b._native : b));
}

/**
 * Element-wise subtraction
 * Supports both (array, scalar) and (scalar, array)
 */
export function subtract(a: NDArray | number, b: NDArray | number): NDArray {
  // Non-commutative: need native support for scalar-first
  if (typeof a === 'number') {
    if (!(b instanceof NDArray)) {
      throw new Error('At least one argument must be an NDArray');
    }
    return new NDArray(math.subtract_scalar_first(a, b._native));
  }
  return new NDArray(math.subtract(a._native, b instanceof NDArray ? b._native : b));
}

/**
 * Element-wise multiplication
 * Supports both (array, scalar) and (scalar, array) - commutative
 */
export function multiply(a: NDArray | number, b: NDArray | number): NDArray {
  // Commutative: swap if scalar-first
  if (typeof a === 'number') {
    if (!(b instanceof NDArray)) {
      throw new Error('At least one argument must be an NDArray');
    }
    return new NDArray(math.multiply(b._native, a));
  }
  return new NDArray(math.multiply(a._native, b instanceof NDArray ? b._native : b));
}

/**
 * Element-wise division
 * Supports both (array, scalar) and (scalar, array)
 */
export function divide(a: NDArray | number, b: NDArray | number): NDArray {
  // Non-commutative: need native support for scalar-first
  if (typeof a === 'number') {
    if (!(b instanceof NDArray)) {
      throw new Error('At least one argument must be an NDArray');
    }
    return new NDArray(math.divide_scalar_first(a, b._native));
  }
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

/**
 * Element-wise hyperbolic tangent
 */
export function tanh(a: NDArray): NDArray {
  return new NDArray(math.tanh(a._native));
}

// ============================================================
// Reduction Operations
// ============================================================

/**
 * Sum of array elements
 */
export function sum(a: NDArray, axis?: number, keepdims?: boolean): NDArray | number {
  const result = math.sum(a._native, axis, keepdims);
  if (typeof result === 'number') {
    return result;
  }
  return new NDArray(result);
}

/**
 * Mean of array elements
 * @param a Input array
 * @param axis Axis along which to compute mean (optional)
 * @param keepdims If true, reduced axes are left with size 1 (optional)
 */
export function mean(a: NDArray, axis?: number, keepdims?: boolean): NDArray | number {
  const result = math.mean(a._native, axis, keepdims);
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
 * @param a Input array
 * @param axis Axis along which to find minimum (optional)
 * @param keepdims If true, reduced axes are left with size 1 (optional)
 */
export function min(a: NDArray, axis?: number, keepdims?: boolean): NDArray | number {
  const result = math.min(a._native, axis, keepdims);
  if (typeof result === 'number') {
    return result;
  }
  return new NDArray(result);
}

/**
 * Maximum of array elements
 * @param a Input array
 * @param axis Axis along which to find maximum (optional)
 * @param keepdims If true, reduced axes are left with size 1 (optional)
 */
export function max(a: NDArray, axis?: number, keepdims?: boolean): NDArray | number {
  const result = math.max(a._native, axis, keepdims);
  if (typeof result === 'number') {
    return result;
  }
  return new NDArray(result);
}

/**
 * Returns the indices of the minimum values along an axis.
 * In case of multiple occurrences of the minimum value, returns the index of the first occurrence.
 * @param a Input array
 * @param axis Axis along which to find indices (optional)
 * @returns Index of minimum as number (global) or NDArray (along axis)
 */
export function argmin(a: NDArray, axis?: number): NDArray | number {
  const result = math.argmin(a._native, axis);
  if (typeof result === 'number') {
    return result;
  }
  return new NDArray(result);
}

/**
 * Returns the indices of the maximum values along an axis.
 * In case of multiple occurrences of the maximum value, returns the index of the first occurrence.
 * @param a Input array
 * @param axis Axis along which to find indices (optional)
 * @returns Index of maximum as number (global) or NDArray (along axis)
 */
export function argmax(a: NDArray, axis?: number): NDArray | number {
  const result = math.argmax(a._native, axis);
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

/**
 * Cumulative sum along axis
 * @param a Input array
 * @param axis Axis along which to compute cumulative sum. If not specified, flattens array first.
 * @returns Array with cumulative sums
 */
export function cumsum(a: NDArray, axis?: number): NDArray {
  return new NDArray(math.cumsum(a._native, axis));
}

/**
 * Cumulative product along axis
 * @param a Input array
 * @param axis Axis along which to compute cumulative product. If not specified, flattens array first.
 * @returns Array with cumulative products
 */
export function cumprod(a: NDArray, axis?: number): NDArray {
  return new NDArray(math.cumprod(a._native, axis));
}

// ============================================================
// Array Joining
// ============================================================

/**
 * Join arrays along an existing axis
 * @param arrays Sequence of arrays to concatenate
 * @param axis Axis along which to concatenate (default: 0)
 * @returns Concatenated array
 */
export function concatenate(arrays: NDArray[], axis: number = 0): NDArray {
  return new NDArray(
    math.concatenate(
      arrays.map((a) => a._native),
      axis
    )
  );
}

/**
 * Join arrays along a new axis
 * @param arrays Sequence of arrays to stack (must have same shape)
 * @param axis Axis in the result array along which to stack (default: 0)
 * @returns Stacked array with one more dimension than input arrays
 */
export function stack(arrays: NDArray[], axis: number = 0): NDArray {
  return new NDArray(
    math.stack(
      arrays.map((a) => a._native),
      axis
    )
  );
}

/**
 * Stack arrays vertically (row-wise)
 * Equivalent to concatenate along axis 0 for 2D+ arrays,
 * or stack along axis 0 for 1D arrays (making them rows)
 * @param arrays Sequence of arrays to stack
 * @returns Vertically stacked array
 */
export function vstack(arrays: NDArray[]): NDArray {
  if (arrays.length === 0) {
    throw new Error('Need at least one array to vstack');
  }
  const first = arrays[0]!;
  // For 1D arrays, stack them as rows (same as np.row_stack)
  if (first.ndim === 1) {
    return stack(arrays, 0);
  }
  // For 2D+ arrays, concatenate along axis 0
  return concatenate(arrays, 0);
}

/**
 * Stack arrays horizontally (column-wise)
 * Equivalent to concatenate along axis 1 for 2D+ arrays,
 * or concatenate along axis 0 for 1D arrays
 * @param arrays Sequence of arrays to stack
 * @returns Horizontally stacked array
 */
export function hstack(arrays: NDArray[]): NDArray {
  if (arrays.length === 0) {
    throw new Error('Need at least one array to hstack');
  }
  const first = arrays[0]!;
  // For 1D arrays, concatenate along axis 0 (horizontally for 1D)
  if (first.ndim === 1) {
    return concatenate(arrays, 0);
  }
  // For 2D+ arrays, concatenate along axis 1
  return concatenate(arrays, 1);
}

// ============================================================
// Sorting and Searching
// ============================================================

/**
 * Calculate the n-th discrete difference along the given axis
 * @param a Input array
 * @param n Number of times to apply the difference (default: 1)
 * @param axis Axis along which to compute difference (default: -1, last axis)
 * @returns Array of differences with shape reduced by n along the given axis
 */
export function diff(a: NDArray, n: number = 1, axis: number = -1): NDArray {
  return new NDArray(math.diff(a._native, n, axis));
}

/**
 * Return a sorted copy of the array
 * @param a Input array
 * @param axis Axis along which to sort (default: -1, last axis)
 * @returns Sorted array
 */
export function sort(a: NDArray, axis: number = -1): NDArray {
  return new NDArray(math.sort(a._native, axis));
}

/**
 * Return the indices that would sort the array
 * @param a Input array
 * @param axis Axis along which to sort (default: -1, last axis)
 * @returns Array of indices that sort the input
 */
export function argsort(a: NDArray, axis: number = -1): NDArray {
  return new NDArray(math.argsort(a._native, axis));
}

/**
 * Find the unique elements of an array (sorted)
 * @param a Input array (will be flattened if not 1D)
 * @returns Sorted array of unique values
 */
export function unique(a: NDArray): NDArray {
  return new NDArray(math.unique(a._native));
}

/**
 * Find indices where elements should be inserted to maintain order
 * @param a Sorted input array
 * @param v Values to insert
 * @param side 'left' (default) or 'right'
 * @returns Array of insertion indices
 */
export function searchsorted(a: NDArray, v: NDArray, side: 'left' | 'right' = 'left'): NDArray {
  return new NDArray(math.searchsorted(a._native, v._native, side));
}

// ============================================================
// Array Manipulation (Tier 3+4)
// ============================================================

/**
 * Construct an array by repeating a the number of times given by reps
 * @param a Input array
 * @param reps Number of repetitions along each axis
 * @returns Tiled array
 */
export function tile(a: NDArray, reps: number | number[]): NDArray {
  const repsArray = Array.isArray(reps) ? reps : [reps];
  return new NDArray(math.tile(a._native, repsArray));
}

/**
 * Repeat elements of an array
 * @param a Input array
 * @param repeats Number of repetitions for each element
 * @param axis Axis along which to repeat. If undefined, flatten first.
 * @returns Array with repeated elements
 */
export function repeat(a: NDArray, repeats: number, axis?: number): NDArray {
  return new NDArray(math.repeat(a._native, repeats, axis));
}

/**
 * Reverse the order of elements along the given axis
 * @param a Input array
 * @param axis Axis or axes to flip. If undefined, flip all axes.
 * @returns Flipped array
 */
export function flip(a: NDArray, axis?: number | number[]): NDArray {
  return new NDArray(math.flip(a._native, axis));
}

/**
 * Rotate an array by 90 degrees in the plane specified by axes
 * @param a Input array (must be at least 2D)
 * @param k Number of times to rotate (default 1)
 * @param axes The plane of rotation (default [0, 1])
 * @returns Rotated array
 */
export function rot90(a: NDArray, k: number = 1, axes: [number, number] = [0, 1]): NDArray {
  return new NDArray(math.rot90(a._native, k, axes));
}

/**
 * Split an array into multiple sub-arrays
 * @param a Input array
 * @param indices_or_sections Number of equal parts or specific indices to split at
 * @param axis Axis along which to split (default 0)
 * @returns Array of sub-arrays
 */
export function split(
  a: NDArray,
  indices_or_sections: number | number[],
  axis: number = 0
): NDArray[] {
  const result = math.split(a._native, indices_or_sections, axis);
  return Array.from(result).map((arr) => new NDArray(arr));
}

/**
 * Slice specification type for multi-dimensional slicing
 * - number: single index (reduces dimension)
 * - [start, end]: slice from start to end
 * - [start, end, step]: slice with step
 * - null: all elements (:)
 */
export type SliceSpec = number | [number | null, number | null, number?] | null;

/**
 * Slice array along multiple dimensions (like NumPy's arr[1:-1, 2:])
 *
 * @param a Input array
 * @param slices Slice specification for each dimension
 * @returns Sliced array
 *
 * @example
 * // NumPy: arr[1:-1, 2:]
 * slice(arr, [[1, -1], [2, null]])
 *
 * // NumPy: arr[0, :]  (first row)
 * slice(arr, [0, null])
 *
 * // NumPy: arr[:, 1:-1]
 * slice(arr, [null, [1, -1]])
 */
export function slice(a: NDArray, ...slices: SliceSpec[]): NDArray {
  return new NDArray(math.slice(a._native, slices));
}

/**
 * Compute 2D gradient using central differences
 * Interior uses central differences, edges use forward/backward differences
 *
 * @param f 2D input array
 * @param h Grid spacing (default 1.0)
 * @returns Object with dfdx and dfdy gradient arrays
 */
export function gradient_2d(f: NDArray, h: number = 1.0): { dfdx: NDArray; dfdy: NDArray } {
  const result = math.gradient_2d(f._native, h);
  return {
    dfdx: new NDArray(result.dfdx),
    dfdy: new NDArray(result.dfdy),
  };
}

/**
 * Perform multiple heat equation time steps using 5-point stencil
 * T_new = T + alpha * (laplacian(T))
 *
 * @param T 2D temperature array
 * @param alpha Thermal diffusivity coefficient (dt * k / (dx^2))
 * @param nSteps Number of time steps to perform
 * @returns New temperature array after nSteps
 */
export function heat_step_2d(T: NDArray, alpha: number, nSteps: number): NDArray {
  return new NDArray(math.heat_step_2d(T._native, alpha, nSteps));
}

/**
 * Return the indices of non-zero elements
 * @param a Input array
 * @returns Tuple of arrays, one for each dimension
 */
export function nonzero(a: NDArray): NDArray[] {
  const result = math.nonzero(a._native);
  return Array.from(result).map((arr) => new NDArray(arr));
}

/**
 * Element-wise sign function
 * @param a Input array
 * @returns Array with -1, 0, or 1 for each element
 */
export function sign(a: NDArray): NDArray {
  return new NDArray(math.sign(a._native));
}

/**
 * Element-wise modulo operation (NumPy/Python behavior)
 * @param a Dividend array
 * @param b Divisor (array or scalar)
 * @returns Remainder with same sign as divisor
 */
export function mod(a: NDArray, b: NDArray | number): NDArray {
  const bValue = typeof b === 'number' ? b : b._native;
  return new NDArray(math.mod(a._native, bValue));
}

/**
 * Element-wise check if two arrays are close within tolerance
 * @param a First array
 * @param b Second array
 * @param rtol Relative tolerance (default 1e-5)
 * @param atol Absolute tolerance (default 1e-8)
 * @returns Boolean array
 */
export function isclose(a: NDArray, b: NDArray, rtol: number = 1e-5, atol: number = 1e-8): NDArray {
  return new NDArray(math.isclose(a._native, b._native, rtol, atol));
}

/**
 * Check if all elements of two arrays are close within tolerance
 * @param a First array
 * @param b Second array
 * @param rtol Relative tolerance (default 1e-5)
 * @param atol Absolute tolerance (default 1e-8)
 * @returns true if all elements are close
 */
export function allclose(
  a: NDArray,
  b: NDArray,
  rtol: number = 1e-5,
  atol: number = 1e-8
): boolean {
  return math.allclose(a._native, b._native, rtol, atol);
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

/**
 * Round elements to the nearest integer
 * @param a Input array
 * @returns Array with elements rounded to nearest integer
 */
export function round(a: NDArray): NDArray {
  return new NDArray(math.round(a._native));
}

/**
 * Floor of array elements (round down to nearest integer)
 * @param a Input array
 * @returns Array with floor of each element
 */
export function floor(a: NDArray): NDArray {
  return new NDArray(math.floor(a._native));
}

/**
 * Ceiling of array elements (round up to nearest integer)
 * @param a Input array
 * @returns Array with ceiling of each element
 */
export function ceil(a: NDArray): NDArray {
  return new NDArray(math.ceil(a._native));
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

// ============================================================
// Array Manipulation
// ============================================================

/**
 * Clip (limit) the values in an array.
 * Given an interval, values outside the interval are clipped to the interval edges.
 * @param a Array containing elements to clip
 * @param a_min Minimum value
 * @param a_max Maximum value
 * @returns An array with values clipped to [a_min, a_max]
 */
export function clip(a: NDArray, a_min: number, a_max: number): NDArray {
  return new NDArray(math.clip(a._native, a_min, a_max));
}

/**
 * Return elements chosen from x or y depending on condition.
 * @param condition Where True, yield x, otherwise yield y
 * @param x Values from which to choose where condition is True
 * @param y Values from which to choose where condition is False
 * @returns An array with elements from x where condition is truthy, and from y elsewhere
 */
export function where(condition: NDArray, x: NDArray, y: NDArray): NDArray {
  return new NDArray(math.where(condition._native, x._native, y._native));
}

/**
 * Remove axes of length one from an array.
 * @param a Input array
 * @param axis If specified, only squeeze this axis. Must have length 1.
 * @returns Array with squeezed dimensions removed
 */
export function squeeze(a: NDArray, axis?: number): NDArray {
  return new NDArray(math.squeeze(a._native, axis));
}

/**
 * Expand the shape of an array by inserting a new axis.
 * @param a Input array
 * @param axis Position in the expanded axes where the new axis is placed
 * @returns Array with expanded shape
 */
export function expand_dims(a: NDArray, axis: number): NDArray {
  return new NDArray(math.expand_dims(a._native, axis));
}

// ============================================================
// Einstein Summation
// ============================================================

/**
 * Evaluates the Einstein summation convention on the operands.
 *
 * Using the Einstein summation convention, many common multi-dimensional,
 * linear algebraic array operations can be represented in a simple fashion.
 *
 * @param subscripts Specifies the subscripts for summation as a string
 * @param operands The arrays for the operation
 * @returns The calculation based on the Einstein summation convention
 *
 * @example
 * ```typescript
 * // Matrix multiplication
 * const A = array([[1, 2], [3, 4]]);
 * const B = array([[5, 6], [7, 8]]);
 * const C = einsum('ij,jk->ik', A, B);
 *
 * // Trace (sum of diagonal)
 * const trace = einsum('ii', A);
 *
 * // Diagonal
 * const diag = einsum('ii->i', A);
 *
 * // Inner product
 * const a = array([1, 2, 3]);
 * const b = array([4, 5, 6]);
 * const dot = einsum('i,i', a, b);
 *
 * // Outer product
 * const outer = einsum('i,j->ij', a, b);
 * ```
 */
export function einsum(subscripts: string, ...operands: NDArray[]): NDArray {
  const nativeOperands = operands.map((op) => op._native);
  return new NDArray(native.einsum(subscripts, ...nativeOperands));
}

// ============================================================
// Fused Operations (reduce N-API overhead)
// ============================================================

/**
 * Normalize: (x - mean) / std
 *
 * Fused operation for batch normalization and z-score computation.
 * Replaces: divide(subtract(x, mean), std)
 *
 * @param x Input array
 * @param mean Mean value (scalar or array)
 * @param std Standard deviation (scalar or array)
 * @returns Normalized array
 */
export function normalize(x: NDArray, mean: NDArray | number, std: NDArray | number): NDArray {
  return new NDArray(
    math.normalize(
      x._native,
      mean instanceof NDArray ? mean._native : mean,
      std instanceof NDArray ? std._native : std
    )
  );
}

/**
 * Affine transformation: x * scale + bias
 *
 * Fused operation for linear transformations common in neural networks.
 * Replaces: add(multiply(x, scale), bias)
 *
 * @param x Input array
 * @param scale Scale factor (scalar or array)
 * @param bias Bias term (scalar or array)
 * @returns Transformed array
 */
export function affine(x: NDArray, scale: NDArray | number, bias: NDArray | number): NDArray {
  return new NDArray(
    math.affine(
      x._native,
      scale instanceof NDArray ? scale._native : scale,
      bias instanceof NDArray ? bias._native : bias
    )
  );
}

/**
 * Fused multiply-add: a * b + c
 *
 * Efficient computation of a * b + c in a single pass.
 * Replaces: add(multiply(a, b), c)
 *
 * @param a First operand (array)
 * @param b Second operand (scalar or array)
 * @param c Third operand (scalar or array)
 * @returns Result of a * b + c
 */
export function muladd(a: NDArray, b: NDArray | number, c: NDArray | number): NDArray {
  return new NDArray(
    math.muladd(
      a._native,
      b instanceof NDArray ? b._native : b,
      c instanceof NDArray ? c._native : c
    )
  );
}

/**
 * Numerically stable softmax
 *
 * Computes softmax along the specified axis:
 * softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
 *
 * @param x Input array
 * @param axis Axis along which to compute softmax (default: -1, last axis)
 * @returns Softmax probabilities (sum to 1 along axis)
 */
export function softmax(x: NDArray, axis?: number): NDArray {
  return new NDArray(math.softmax(x._native, axis));
}

/**
 * Min-max feature scaling
 *
 * Scales data to [0, 1] range along the specified axis:
 * minmax_scale(x) = (x - min(x)) / (max(x) - min(x))
 *
 * Fused operation that computes min, max, subtraction and division
 * in a single native call, reducing N-API overhead.
 *
 * @param x Input array
 * @param axis Axis along which to scale (default: 0)
 * @returns Scaled array with values in [0, 1]
 */
export function minmax_scale(x: NDArray, axis: number = 0): NDArray {
  return new NDArray(math.minmax_scale(x._native, axis));
}

/**
 * Batch normalization
 *
 * Normalizes input using: (x - mean) / sqrt(var + eps) * gamma + beta
 *
 * Fused operation that computes mean, variance, normalization, scale and shift
 * in a single native call, reducing N-API overhead from 6+ calls to 1.
 *
 * @param x Input array (2D: samples x features)
 * @param gamma Scale parameter (1D: features), optional
 * @param beta Shift parameter (1D: features), optional
 * @param axis Axis along which to normalize (default: 0)
 * @param eps Small constant for numerical stability (default: 1e-5)
 * @returns Normalized array
 */
export function batch_norm(
  x: NDArray,
  gamma?: NDArray,
  beta?: NDArray,
  axis: number = 0,
  eps: number = 1e-5
): NDArray {
  return new NDArray(math.batch_norm(x._native, gamma?._native, beta?._native, axis, eps));
}

// ============================================================
// Optimizer Operations (fused for performance)
// ============================================================

/**
 * Adam optimizer step - fused for maximum performance.
 * Updates m and v in-place and returns new parameters.
 *
 * @param params Current parameters (not modified)
 * @param grads Gradients
 * @param m First moment (modified in-place)
 * @param v Second moment (modified in-place)
 * @param lr Learning rate
 * @param beta1 First moment decay rate (default: 0.9)
 * @param beta2 Second moment decay rate (default: 0.999)
 * @param eps Epsilon for numerical stability (default: 1e-8)
 * @param t Current timestep
 * @returns New parameters
 */
export function adam_step(
  params: NDArray,
  grads: NDArray,
  m: NDArray,
  v: NDArray,
  lr: number,
  beta1: number = 0.9,
  beta2: number = 0.999,
  eps: number = 1e-8,
  t: number
): NDArray {
  return new NDArray(
    math.adam_step(params._native, grads._native, m._native, v._native, lr, beta1, beta2, eps, t)
  );
}

/**
 * Power iteration for computing dominant eigenvalue/eigenvector.
 * Runs n_iters iterations of: v = M @ v; v = v / ||v||
 *
 * @param M Square matrix (n x n)
 * @param v Initial vector (n,)
 * @param n_iters Number of iterations
 * @returns Final eigenvector estimate
 */
export function power_iter(M: NDArray, v: NDArray, n_iters: number): NDArray {
  return new NDArray(math.power_iter(M._native, v._native, n_iters));
}

/**
 * Fused batched attention: softmax(Q @ K^T) @ V
 * Performs the full attention computation in a single native call.
 *
 * @param Q Query tensor [batch, seq, dim]
 * @param K Key tensor [batch, seq, dim]
 * @param V Value tensor [batch, seq, dim]
 * @returns Attention output [batch, seq, dim]
 */
export function batched_attention(Q: NDArray, K: NDArray, V: NDArray): NDArray {
  return new NDArray(math.batched_attention(Q._native, K._native, V._native));
}

/**
 * Fused Black-Scholes option pricing
 *
 * Computes call option prices using the Black-Scholes formula with
 * logistic approximation for the normal CDF.
 *
 * @param S - Spot prices array
 * @param K - Strike price (scalar)
 * @param T - Time to expiry array
 * @param r - Risk-free rate (scalar)
 * @param sigma - Volatility array
 * @returns Call option prices
 */
export function black_scholes(
  S: NDArray,
  K: number,
  T: NDArray,
  r: number,
  sigma: NDArray
): NDArray {
  return new NDArray(math.black_scholes(S._native, K, T._native, r, sigma._native));
}

/**
 * Fused TF-IDF computation
 *
 * Computes TF-IDF (Term Frequency-Inverse Document Frequency) in a single
 * native call, avoiding N-API overhead from multiple operations.
 *
 * @param tf - Term frequency matrix [nDocs, nTerms]
 * @returns TF-IDF matrix [nDocs, nTerms]
 */
export function tfidf(tf: NDArray): NDArray {
  return new NDArray(math.tfidf(tf._native));
}

/**
 * Fused dropout forward pass
 *
 * Generates a random mask and applies it to the input in a single
 * native call, avoiding N-API overhead from multiple operations.
 *
 * @param x - Input array
 * @param p - Dropout probability (probability of setting to 0)
 * @param seed - Optional random seed
 * @returns Object with output and mask arrays
 */
export function dropout(x: NDArray, p: number, seed?: number): { output: NDArray; mask: NDArray } {
  const result = math.dropout(x._native, p, seed);
  return {
    output: new NDArray(result.output),
    mask: new NDArray(result.mask),
  };
}

/**
 * Fused cross-entropy loss
 *
 * Computes softmax cross-entropy loss in a single native call.
 *
 * @param predictions - Logits [nSamples, nClasses]
 * @param targets - One-hot encoded targets [nSamples, nClasses]
 * @returns Scalar loss value
 */
export function cross_entropy(predictions: NDArray, targets: NDArray): number {
  return math.cross_entropy(predictions._native, targets._native);
}

/**
 * Fused sum of squares: sum(x*x, axis)
 *
 * Replaces the common pattern: sum(multiply(x, x), axis)
 * Single N-API call instead of two.
 *
 * @param a - Input array
 * @param axis - Optional axis to reduce along
 * @param keepdims - Whether to keep the reduced dimension
 * @returns Sum of squares (scalar or array depending on axis)
 */
export function sumsq(a: NDArray, axis?: number, keepdims?: boolean): NDArray | number {
  const result = math.sumsq(a._native, axis, keepdims);
  if (typeof result === 'number') {
    return result;
  }
  return new NDArray(result);
}

/**
 * Fused layer normalization: (x - mean) / std * gamma + beta
 *
 * Replaces the common pattern: affine(zscore(x, axis), gamma, beta)
 * Single N-API call instead of two.
 *
 * @param x - Input array
 * @param gamma - Optional scale parameter (per-feature)
 * @param beta - Optional bias parameter (per-feature)
 * @param axis - Normalization axis (default 1)
 * @param eps - Epsilon for numerical stability (default 1e-5)
 * @returns Normalized array with same shape as input
 */
export function layer_norm(
  x: NDArray,
  gamma?: NDArray,
  beta?: NDArray,
  axis?: number,
  eps?: number
): NDArray {
  return new NDArray(math.layer_norm(x._native, gamma?._native, beta?._native, axis, eps));
}

/**
 * Fused softmax cross-entropy loss from logits
 *
 * Computes: mean(-sum(labels * log_softmax(logits), axis))
 * Numerically stable implementation using log-sum-exp trick.
 *
 * @param logits - Input logits [nSamples, nClasses]
 * @param labels - One-hot encoded labels [nSamples, nClasses]
 * @param axis - Softmax axis (default 1)
 * @returns Scalar loss value (mean over samples)
 */
export function softmax_cross_entropy(logits: NDArray, labels: NDArray, axis?: number): number {
  return math.softmax_cross_entropy(logits._native, labels._native, axis);
}
