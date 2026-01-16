/**
 * Comparison operations for np-ts
 * Provides element-wise comparison with broadcasting support
 */

import { NDArray, type DTypeName } from '../core/index.js';
import { applyComparison, applyComparisonScalar } from './ufunc.js';

// ============================================================
// Element-wise Comparison
// ============================================================

/**
 * Return (x1 == x2) element-wise
 */
export function equal<D1 extends DTypeName, D2 extends DTypeName>(
  x1: NDArray<D1>,
  x2: NDArray<D2> | number | bigint
): NDArray<'bool'> {
  if (typeof x2 === 'number' || typeof x2 === 'bigint') {
    return applyComparisonScalar(x1, x2, (a, b) => a === b);
  }
  return applyComparison(x1, x2, (a, b) => a === b);
}

/**
 * Return (x1 != x2) element-wise
 */
export function notEqual<D1 extends DTypeName, D2 extends DTypeName>(
  x1: NDArray<D1>,
  x2: NDArray<D2> | number | bigint
): NDArray<'bool'> {
  if (typeof x2 === 'number' || typeof x2 === 'bigint') {
    return applyComparisonScalar(x1, x2, (a, b) => a !== b);
  }
  return applyComparison(x1, x2, (a, b) => a !== b);
}

/**
 * Alias for notEqual
 */
export const not_equal = notEqual;

/**
 * Return (x1 < x2) element-wise
 */
export function less<D1 extends DTypeName, D2 extends DTypeName>(
  x1: NDArray<D1>,
  x2: NDArray<D2> | number | bigint
): NDArray<'bool'> {
  if (typeof x2 === 'number' || typeof x2 === 'bigint') {
    return applyComparisonScalar(x1, x2, (a, b) => a < b);
  }
  return applyComparison(x1, x2, (a, b) => a < b);
}

/**
 * Return (x1 <= x2) element-wise
 */
export function lessEqual<D1 extends DTypeName, D2 extends DTypeName>(
  x1: NDArray<D1>,
  x2: NDArray<D2> | number | bigint
): NDArray<'bool'> {
  if (typeof x2 === 'number' || typeof x2 === 'bigint') {
    return applyComparisonScalar(x1, x2, (a, b) => a <= b);
  }
  return applyComparison(x1, x2, (a, b) => a <= b);
}

/**
 * Alias for lessEqual
 */
export const less_equal = lessEqual;

/**
 * Return (x1 > x2) element-wise
 */
export function greater<D1 extends DTypeName, D2 extends DTypeName>(
  x1: NDArray<D1>,
  x2: NDArray<D2> | number | bigint
): NDArray<'bool'> {
  if (typeof x2 === 'number' || typeof x2 === 'bigint') {
    return applyComparisonScalar(x1, x2, (a, b) => a > b);
  }
  return applyComparison(x1, x2, (a, b) => a > b);
}

/**
 * Return (x1 >= x2) element-wise
 */
export function greaterEqual<D1 extends DTypeName, D2 extends DTypeName>(
  x1: NDArray<D1>,
  x2: NDArray<D2> | number | bigint
): NDArray<'bool'> {
  if (typeof x2 === 'number' || typeof x2 === 'bigint') {
    return applyComparisonScalar(x1, x2, (a, b) => a >= b);
  }
  return applyComparison(x1, x2, (a, b) => a >= b);
}

/**
 * Alias for greaterEqual
 */
export const greater_equal = greaterEqual;

// ============================================================
// Array-level Comparison
// ============================================================

/**
 * Returns True if two arrays are element-wise equal within a tolerance
 */
export function allclose<D1 extends DTypeName, D2 extends DTypeName>(
  a: NDArray<D1>,
  b: NDArray<D2>,
  rtol: number = 1e-5,
  atol: number = 1e-8,
  _equalNan: boolean = false
): boolean {
  return a.allClose(b, rtol, atol);
}

/**
 * True if two arrays have the same shape and elements
 */
export function arrayEqual<D1 extends DTypeName, D2 extends DTypeName>(
  a1: NDArray<D1>,
  a2: NDArray<D2>,
  _equalNan: boolean = false
): boolean {
  return a1.equals(a2);
}

/**
 * Alias for arrayEqual
 */
export const array_equal = arrayEqual;

/**
 * Returns True if input arrays are shape consistent and all elements equal
 */
export function arrayEquiv<D1 extends DTypeName, D2 extends DTypeName>(
  a1: NDArray<D1>,
  a2: NDArray<D2>
): boolean {
  // Check if shapes are broadcast-compatible
  try {
    const { broadcastShapes } = require('../core/broadcasting.js');
    broadcastShapes(a1.shape, a2.shape);
    // If shapes are compatible, compare with broadcasting
    const eqResult = equal(a1, a2);
    const allResult = all(eqResult);
    return typeof allResult === 'boolean' ? allResult : false;
  } catch {
    return false;
  }
}

/**
 * Alias for arrayEquiv
 */
export const array_equiv = arrayEquiv;

// ============================================================
// Logical Operations on Comparisons
// ============================================================

/**
 * Test whether all array elements evaluate to True
 */
export function all<D extends DTypeName>(
  a: NDArray<D>,
  axis?: number,
  keepdims: boolean = false
): boolean | NDArray<'bool'> {
  if (axis === undefined) {
    for (const val of a) {
      if (!Number(val)) {
        return false;
      }
    }
    return true;
  }

  const { reduceAxis } = require('./ufunc.js');
  // Reduce using logical AND
  const result = reduceAxis(
    a,
    (acc: number, val: number) => (acc !== 0 && val !== 0 ? 1 : 0),
    axis,
    1,
    keepdims
  );
  return result.astype('bool');
}

/**
 * Test whether any array element evaluates to True
 */
export function any<D extends DTypeName>(
  a: NDArray<D>,
  axis?: number,
  keepdims: boolean = false
): boolean | NDArray<'bool'> {
  if (axis === undefined) {
    for (const val of a) {
      if (Number(val)) {
        return true;
      }
    }
    return false;
  }

  const { reduceAxis } = require('./ufunc.js');
  // Reduce using logical OR
  const result = reduceAxis(
    a,
    (acc: number, val: number) => (acc !== 0 || val !== 0 ? 1 : 0),
    axis,
    0,
    keepdims
  );
  return result.astype('bool');
}

// ============================================================
// Logical Element-wise Operations
// ============================================================

/**
 * Compute the truth value of x1 AND x2 element-wise
 */
export function logicalAnd<D1 extends DTypeName, D2 extends DTypeName>(
  x1: NDArray<D1>,
  x2: NDArray<D2>
): NDArray<'bool'> {
  return applyComparison(x1, x2, (a, b) => Boolean(a) && Boolean(b));
}

/**
 * Alias for logicalAnd
 */
export const logical_and = logicalAnd;

/**
 * Compute the truth value of x1 OR x2 element-wise
 */
export function logicalOr<D1 extends DTypeName, D2 extends DTypeName>(
  x1: NDArray<D1>,
  x2: NDArray<D2>
): NDArray<'bool'> {
  return applyComparison(x1, x2, (a, b) => Boolean(a) || Boolean(b));
}

/**
 * Alias for logicalOr
 */
export const logical_or = logicalOr;

/**
 * Compute the truth value of x1 XOR x2 element-wise
 */
export function logicalXor<D1 extends DTypeName, D2 extends DTypeName>(
  x1: NDArray<D1>,
  x2: NDArray<D2>
): NDArray<'bool'> {
  return applyComparison(x1, x2, (a, b) => Boolean(a) !== Boolean(b));
}

/**
 * Alias for logicalXor
 */
export const logical_xor = logicalXor;

/**
 * Compute the truth value of NOT x element-wise
 */
export function logicalNot<D extends DTypeName>(x: NDArray<D>): NDArray<'bool'> {
  const result = NDArray.zeros<'bool'>(x.shape, { dtype: 'bool' });

  let i = 0;
  for (const val of x) {
    result.setFlat(i++, !Number(val) ? 1 : 0);
  }

  return result;
}

/**
 * Alias for logicalNot
 */
export const logical_not = logicalNot;

// ============================================================
// Where and Select
// ============================================================

/**
 * Return elements chosen from x or y depending on condition
 */
export function where<D1 extends DTypeName, D2 extends DTypeName>(
  condition: NDArray<'bool'>,
  x: NDArray<D1>,
  y: NDArray<D2>
): NDArray {
  const { broadcastShapes, getBroadcastStrides, needsBroadcast } = require('../core/broadcasting.js');
  const { IndexIterator, indicesToFlat } = require('../core/indexing.js');
  const { promoteDTypes } = require('../core/dtype.js');

  const broadcastShape = broadcastShapes(condition.shape, x.shape, y.shape);
  const dtype = promoteDTypes(x.dtype, y.dtype);
  const result = NDArray.zeros(broadcastShape, { dtype });

  const condStrides = needsBroadcast(condition.shape, broadcastShape)
    ? getBroadcastStrides(condition.shape, condition.strides, broadcastShape)
    : condition.strides;

  const xStrides = needsBroadcast(x.shape, broadcastShape)
    ? getBroadcastStrides(x.shape, x.strides, broadcastShape)
    : x.strides;

  const yStrides = needsBroadcast(y.shape, broadcastShape)
    ? getBroadcastStrides(y.shape, y.strides, broadcastShape)
    : y.strides;

  let i = 0;
  for (const indices of new IndexIterator(broadcastShape)) {
    const condOffset = condition.offset + indicesToFlat(indices, condStrides);
    const xOffset = x.offset + indicesToFlat(indices, xStrides);
    const yOffset = y.offset + indicesToFlat(indices, yStrides);

    const condVal = Number(condition.data[condOffset]);
    const value = condVal ? x.data[xOffset] : y.data[yOffset];
    result.setFlat(i++, Number(value));
  }

  return result;
}

/**
 * Return indices where condition is true
 */
export function nonzero<D extends DTypeName>(a: NDArray<D>): NDArray<'int32'>[] {
  const { IndexIterator } = require('../core/indexing.js');
  const indices: number[][] = Array.from({ length: a.ndim }, () => []);

  for (const idx of new IndexIterator(a.shape)) {
    const val = a.at(...idx);
    if (Number(val) !== 0) {
      for (let d = 0; d < a.ndim; d++) {
        indices[d]!.push(idx[d]!);
      }
    }
  }

  return indices.map((arr) => NDArray.from(arr, { dtype: 'int32' }));
}

/**
 * Return indices where condition is true (flattened)
 */
export function flatnonzero<D extends DTypeName>(a: NDArray<D>): NDArray<'int32'> {
  const indices: number[] = [];

  let i = 0;
  for (const val of a) {
    if (Number(val) !== 0) {
      indices.push(i);
    }
    i++;
  }

  return NDArray.from(indices, { dtype: 'int32' });
}

/**
 * Return the indices of the maximum values along an axis
 */
export function argmax<D extends DTypeName>(
  a: NDArray<D>,
  axis?: number
): number | NDArray<'int32'> {
  if (axis === undefined) {
    let maxIdx = 0;
    let maxVal = Number(a.getFlat(0));
    let i = 0;
    for (const val of a) {
      const numVal = Number(val);
      if (numVal > maxVal) {
        maxVal = numVal;
        maxIdx = i;
      }
      i++;
    }
    return maxIdx;
  }

  const { IndexIterator } = require('../core/indexing.js');
  const normalizedAxis = axis < 0 ? axis + a.ndim : axis;
  const axisSize = a.shape[normalizedAxis]!;

  const outShape = a.shape.filter((_, i) => i !== normalizedAxis);
  if (outShape.length === 0) {
    outShape.push(1);
  }

  const result = NDArray.zeros<'int32'>(outShape, { dtype: 'int32' });

  let outIdx = 0;
  for (const outIndices of new IndexIterator(outShape)) {
    const inIndices: number[] = [];
    let j = 0;
    for (let i = 0; i < a.ndim; i++) {
      if (i === normalizedAxis) {
        inIndices.push(0);
      } else {
        inIndices.push(outIndices[j]!);
        j++;
      }
    }

    let maxIdx = 0;
    let maxVal = Number(a.at(...inIndices));

    for (let k = 1; k < axisSize; k++) {
      inIndices[normalizedAxis] = k;
      const val = Number(a.at(...inIndices));
      if (val > maxVal) {
        maxVal = val;
        maxIdx = k;
      }
    }

    result.setFlat(outIdx++, maxIdx);
  }

  return result;
}

/**
 * Return the indices of the minimum values along an axis
 */
export function argmin<D extends DTypeName>(
  a: NDArray<D>,
  axis?: number
): number | NDArray<'int32'> {
  if (axis === undefined) {
    let minIdx = 0;
    let minVal = Number(a.getFlat(0));
    let i = 0;
    for (const val of a) {
      const numVal = Number(val);
      if (numVal < minVal) {
        minVal = numVal;
        minIdx = i;
      }
      i++;
    }
    return minIdx;
  }

  const { IndexIterator } = require('../core/indexing.js');
  const normalizedAxis = axis < 0 ? axis + a.ndim : axis;
  const axisSize = a.shape[normalizedAxis]!;

  const outShape = a.shape.filter((_, i) => i !== normalizedAxis);
  if (outShape.length === 0) {
    outShape.push(1);
  }

  const result = NDArray.zeros<'int32'>(outShape, { dtype: 'int32' });

  let outIdx = 0;
  for (const outIndices of new IndexIterator(outShape)) {
    const inIndices: number[] = [];
    let j = 0;
    for (let i = 0; i < a.ndim; i++) {
      if (i === normalizedAxis) {
        inIndices.push(0);
      } else {
        inIndices.push(outIndices[j]!);
        j++;
      }
    }

    let minIdx = 0;
    let minVal = Number(a.at(...inIndices));

    for (let k = 1; k < axisSize; k++) {
      inIndices[normalizedAxis] = k;
      const val = Number(a.at(...inIndices));
      if (val < minVal) {
        minVal = val;
        minIdx = k;
      }
    }

    result.setFlat(outIdx++, minIdx);
  }

  return result;
}

/**
 * Count number of non-zero elements
 */
export function countNonzero<D extends DTypeName>(
  a: NDArray<D>,
  axis?: number
): number | NDArray<'int32'> {
  if (axis === undefined) {
    let count = 0;
    for (const val of a) {
      if (Number(val) !== 0) {
        count++;
      }
    }
    return count;
  }

  const { reduceAxis } = require('./ufunc.js');
  return reduceAxis(
    a,
    (acc: number, val: number) => acc + (val !== 0 ? 1 : 0),
    axis,
    0,
    false
  ).astype('int32');
}

/**
 * Alias for countNonzero
 */
export const count_nonzero = countNonzero;
