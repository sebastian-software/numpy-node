/**
 * Arithmetic operations for numpy-node
 * Provides element-wise arithmetic with broadcasting support
 */

import { NDArray, type DTypeName, IndexIterator } from '../core/index.js';
import { applyBinary, applyBinaryScalar, reduceAxis, reduceAll } from './ufunc.js';

// ============================================================
// Addition
// ============================================================

/**
 * Add arguments element-wise
 */
export function add<D1 extends DTypeName, D2 extends DTypeName>(
  x1: NDArray<D1>,
  x2: NDArray<D2> | number | bigint
): NDArray {
  if (typeof x2 === 'number' || typeof x2 === 'bigint') {
    return applyBinaryScalar(x1, x2, (a, b) => a + b, (a, b) => a + b);
  }
  return applyBinary(x1, x2, (a, b) => a + b, (a, b) => a + b);
}

// ============================================================
// Subtraction
// ============================================================

/**
 * Subtract arguments element-wise
 */
export function subtract<D1 extends DTypeName, D2 extends DTypeName>(
  x1: NDArray<D1>,
  x2: NDArray<D2> | number | bigint
): NDArray {
  if (typeof x2 === 'number' || typeof x2 === 'bigint') {
    return applyBinaryScalar(x1, x2, (a, b) => a - b, (a, b) => a - b);
  }
  return applyBinary(x1, x2, (a, b) => a - b, (a, b) => a - b);
}

// ============================================================
// Multiplication
// ============================================================

/**
 * Multiply arguments element-wise
 */
export function multiply<D1 extends DTypeName, D2 extends DTypeName>(
  x1: NDArray<D1>,
  x2: NDArray<D2> | number | bigint
): NDArray {
  if (typeof x2 === 'number' || typeof x2 === 'bigint') {
    return applyBinaryScalar(x1, x2, (a, b) => a * b, (a, b) => a * b);
  }
  return applyBinary(x1, x2, (a, b) => a * b, (a, b) => a * b);
}

// ============================================================
// Division
// ============================================================

/**
 * Divide arguments element-wise (true division)
 */
export function divide<D1 extends DTypeName, D2 extends DTypeName>(
  x1: NDArray<D1>,
  x2: NDArray<D2> | number | bigint
): NDArray<'float64'> {
  if (typeof x2 === 'number' || typeof x2 === 'bigint') {
    return applyBinaryScalar(x1, x2, (a, b) => a / b, undefined, 'float64') as NDArray<'float64'>;
  }
  return applyBinary(x1, x2, (a, b) => a / b, undefined, 'float64') as NDArray<'float64'>;
}

/**
 * Divide arguments element-wise (true division) - alias for divide
 */
export const trueDivide = divide;

/**
 * Integer division element-wise
 */
export function floorDivide<D1 extends DTypeName, D2 extends DTypeName>(
  x1: NDArray<D1>,
  x2: NDArray<D2> | number | bigint
): NDArray {
  if (typeof x2 === 'number' || typeof x2 === 'bigint') {
    return applyBinaryScalar(
      x1,
      x2,
      (a, b) => Math.floor(a / b),
      (a, b) => a / b
    );
  }
  return applyBinary(
    x1,
    x2,
    (a, b) => Math.floor(a / b),
    (a, b) => a / b
  );
}

// ============================================================
// Modulo & Remainder
// ============================================================

/**
 * Return element-wise remainder of division
 */
export function mod<D1 extends DTypeName, D2 extends DTypeName>(
  x1: NDArray<D1>,
  x2: NDArray<D2> | number | bigint
): NDArray {
  // NumPy's mod always returns result with same sign as divisor
  const modOp = (a: number, b: number): number => ((a % b) + b) % b;
  const modOpBigInt = (a: bigint, b: bigint): bigint => ((a % b) + b) % b;

  if (typeof x2 === 'number' || typeof x2 === 'bigint') {
    return applyBinaryScalar(x1, x2, modOp, modOpBigInt);
  }
  return applyBinary(x1, x2, modOp, modOpBigInt);
}

/**
 * Return element-wise remainder of division (same as mod)
 */
export const remainder = mod;

/**
 * Return element-wise remainder of division (C-style)
 */
export function fmod<D1 extends DTypeName, D2 extends DTypeName>(
  x1: NDArray<D1>,
  x2: NDArray<D2> | number | bigint
): NDArray {
  if (typeof x2 === 'number' || typeof x2 === 'bigint') {
    return applyBinaryScalar(x1, x2, (a, b) => a % b, (a, b) => a % b);
  }
  return applyBinary(x1, x2, (a, b) => a % b, (a, b) => a % b);
}

/**
 * Return element-wise quotient and remainder simultaneously
 */
export function divmod<D1 extends DTypeName, D2 extends DTypeName>(
  x1: NDArray<D1>,
  x2: NDArray<D2> | number | bigint
): [NDArray, NDArray] {
  return [floorDivide(x1, x2), mod(x1, x2)];
}

// ============================================================
// Powers & Roots
// ============================================================

/**
 * First array elements raised to powers from second array
 */
export function power<D1 extends DTypeName, D2 extends DTypeName>(
  x1: NDArray<D1>,
  x2: NDArray<D2> | number | bigint
): NDArray {
  if (typeof x2 === 'number' || typeof x2 === 'bigint') {
    return applyBinaryScalar(
      x1,
      x2,
      (a, b) => Math.pow(a, b),
      (a, b) => {
        if (b < 0) throw new Error('Negative powers not supported for bigint');
        let result = BigInt(1);
        for (let i = BigInt(0); i < b; i++) {
          result *= a;
        }
        return result;
      },
      'float64'
    );
  }
  return applyBinary(
    x1,
    x2,
    (a, b) => Math.pow(a, b),
    undefined,
    'float64'
  );
}

/**
 * Return the non-negative square-root element-wise
 */
export function sqrt<D extends DTypeName>(x: NDArray<D>): NDArray<'float64'> {
  return power(x, 0.5) as NDArray<'float64'>;
}

/**
 * Return the cube-root element-wise
 */
export function cbrt<D extends DTypeName>(x: NDArray<D>): NDArray<'float64'> {
  const result = NDArray.zeros<'float64'>(x.shape, { dtype: 'float64' });

  let i = 0;
  for (const val of x) {
    result.setFlat(i++, Math.cbrt(Number(val)));
  }

  return result;
}

/**
 * Return the element-wise square
 */
export function square<D extends DTypeName>(x: NDArray<D>): NDArray {
  return multiply(x, x);
}

// ============================================================
// Sign & Absolute
// ============================================================

/**
 * Calculate the absolute value element-wise
 */
export function abs<D extends DTypeName>(x: NDArray<D>): NDArray<D> {
  const result = NDArray.zeros(x.shape, { dtype: x.dtype }) as NDArray<D>;

  let i = 0;
  for (const val of x) {
    if (typeof val === 'bigint') {
      result.setFlat(i++, val < BigInt(0) ? -val : val);
    } else {
      result.setFlat(i++, Math.abs(Number(val)));
    }
  }

  return result;
}

/**
 * Alias for abs
 */
export const absolute = abs;
export const fabs = abs;

/**
 * Returns an element-wise indication of the sign
 */
export function sign<D extends DTypeName>(x: NDArray<D>): NDArray<D> {
  const result = NDArray.zeros(x.shape, { dtype: x.dtype }) as NDArray<D>;

  let i = 0;
  for (const val of x) {
    if (typeof val === 'bigint') {
      result.setFlat(i++, val > BigInt(0) ? BigInt(1) : val < BigInt(0) ? BigInt(-1) : BigInt(0));
    } else {
      result.setFlat(i++, Math.sign(Number(val)));
    }
  }

  return result;
}

/**
 * Change sign to that of second argument
 */
export function copysign<D1 extends DTypeName, D2 extends DTypeName>(
  x1: NDArray<D1>,
  x2: NDArray<D2>
): NDArray<'float64'> {
  return applyBinary(
    x1,
    x2,
    (a, b) => Math.abs(a) * Math.sign(b),
    undefined,
    'float64'
  ) as NDArray<'float64'>;
}

// ============================================================
// Negation
// ============================================================

/**
 * Numerical negative element-wise
 */
export function negative<D extends DTypeName>(x: NDArray<D>): NDArray<D> {
  const result = NDArray.zeros(x.shape, { dtype: x.dtype }) as NDArray<D>;

  let i = 0;
  for (const val of x) {
    if (typeof val === 'bigint') {
      result.setFlat(i++, -val);
    } else {
      result.setFlat(i++, -Number(val));
    }
  }

  return result;
}

/**
 * Numerical positive element-wise (returns a copy)
 */
export function positive<D extends DTypeName>(x: NDArray<D>): NDArray<D> {
  return x.copy();
}

// ============================================================
// Reciprocal
// ============================================================

/**
 * Return the reciprocal element-wise
 */
export function reciprocal<D extends DTypeName>(x: NDArray<D>): NDArray<'float64'> {
  const result = NDArray.zeros<'float64'>(x.shape, { dtype: 'float64' });

  let i = 0;
  for (const val of x) {
    result.setFlat(i++, 1 / Number(val));
  }

  return result;
}

// ============================================================
// Min/Max Operations
// ============================================================

/**
 * Element-wise maximum of array elements
 */
export function maximum<D1 extends DTypeName, D2 extends DTypeName>(
  x1: NDArray<D1>,
  x2: NDArray<D2> | number | bigint
): NDArray {
  if (typeof x2 === 'number' || typeof x2 === 'bigint') {
    return applyBinaryScalar(
      x1,
      x2,
      (a, b) => Math.max(a, b),
      (a, b) => (a > b ? a : b)
    );
  }
  return applyBinary(
    x1,
    x2,
    (a, b) => Math.max(a, b),
    (a, b) => (a > b ? a : b)
  );
}

/**
 * Element-wise minimum of array elements
 */
export function minimum<D1 extends DTypeName, D2 extends DTypeName>(
  x1: NDArray<D1>,
  x2: NDArray<D2> | number | bigint
): NDArray {
  if (typeof x2 === 'number' || typeof x2 === 'bigint') {
    return applyBinaryScalar(
      x1,
      x2,
      (a, b) => Math.min(a, b),
      (a, b) => (a < b ? a : b)
    );
  }
  return applyBinary(
    x1,
    x2,
    (a, b) => Math.min(a, b),
    (a, b) => (a < b ? a : b)
  );
}

/**
 * Element-wise maximum ignoring NaNs
 */
export function fmax<D1 extends DTypeName, D2 extends DTypeName>(
  x1: NDArray<D1>,
  x2: NDArray<D2>
): NDArray {
  return applyBinary(x1, x2, (a, b) => {
    if (isNaN(a)) return b;
    if (isNaN(b)) return a;
    return Math.max(a, b);
  });
}

/**
 * Element-wise minimum ignoring NaNs
 */
export function fmin<D1 extends DTypeName, D2 extends DTypeName>(
  x1: NDArray<D1>,
  x2: NDArray<D2>
): NDArray {
  return applyBinary(x1, x2, (a, b) => {
    if (isNaN(a)) return b;
    if (isNaN(b)) return a;
    return Math.min(a, b);
  });
}

// ============================================================
// Reduction Operations
// ============================================================

/**
 * Sum of array elements
 */
export function sum<D extends DTypeName>(
  a: NDArray<D>,
  axis?: number,
  keepdims: boolean = false
): NDArray | number {
  if (axis === undefined) {
    return reduceAll(a, (x, y) => x + y, 0);
  }
  return reduceAxis(a, (x, y) => x + y, axis, 0, keepdims);
}

/**
 * Product of array elements
 */
export function prod<D extends DTypeName>(
  a: NDArray<D>,
  axis?: number,
  keepdims: boolean = false
): NDArray | number {
  if (axis === undefined) {
    return reduceAll(a, (x, y) => x * y, 1);
  }
  return reduceAxis(a, (x, y) => x * y, axis, 1, keepdims);
}

/**
 * Cumulative sum of array elements
 */
export function cumsum<D extends DTypeName>(
  a: NDArray<D>,
  axis?: number
): NDArray {
  if (axis === undefined) {
    // Flatten and cumsum
    const result = NDArray.zeros([a.size], { dtype: a.dtype });
    let acc = 0;
    let i = 0;
    for (const val of a) {
      acc += Number(val);
      result.setFlat(i++, acc);
    }
    return result;
  }

  // Cumsum along axis
  const result = a.copy();
  const normalizedAxis = axis < 0 ? axis + a.ndim : axis;
  const axisSize = a.shape[normalizedAxis]!;

  // Iterate in a way that accumulates along the axis
  const otherShape = a.shape.filter((_, i) => i !== normalizedAxis);

  if (otherShape.length === 0) {
    // 1D case
    let acc = 0;
    for (let k = 0; k < axisSize; k++) {
      acc += Number(result.at(k));
      result.set([k], acc);
    }
  } else {
    for (const otherIndices of new IndexIterator(otherShape)) {
      const indices: number[] = [];
      let j = 0;
      for (let i = 0; i < a.ndim; i++) {
        if (i === normalizedAxis) {
          indices.push(0);
        } else {
          indices.push(otherIndices[j]!);
          j++;
        }
      }

      let acc = 0;
      for (let k = 0; k < axisSize; k++) {
        indices[normalizedAxis] = k;
        acc += Number(result.at(...indices));
        result.set(indices, acc);
      }
    }
  }

  return result;
}

/**
 * Cumulative product of array elements
 */
export function cumprod<D extends DTypeName>(
  a: NDArray<D>,
  axis?: number
): NDArray {
  if (axis === undefined) {
    const result = NDArray.zeros([a.size], { dtype: a.dtype });
    let acc = 1;
    let i = 0;
    for (const val of a) {
      acc *= Number(val);
      result.setFlat(i++, acc);
    }
    return result;
  }

  const result = a.copy();
  const normalizedAxis = axis < 0 ? axis + a.ndim : axis;
  const axisSize = a.shape[normalizedAxis]!;
  const otherShape = a.shape.filter((_, i) => i !== normalizedAxis);

  if (otherShape.length === 0) {
    // 1D case
    let acc = 1;
    for (let k = 0; k < axisSize; k++) {
      acc *= Number(result.at(k));
      result.set([k], acc);
    }
  } else {
    for (const otherIndices of new IndexIterator(otherShape)) {
      const indices: number[] = [];
      let j = 0;
      for (let i = 0; i < a.ndim; i++) {
        if (i === normalizedAxis) {
          indices.push(0);
        } else {
          indices.push(otherIndices[j]!);
          j++;
        }
      }

      let acc = 1;
      for (let k = 0; k < axisSize; k++) {
        indices[normalizedAxis] = k;
        acc *= Number(result.at(...indices));
        result.set(indices, acc);
      }
    }
  }

  return result;
}

/**
 * Return the maximum of an array or maximum along an axis
 */
export function max<D extends DTypeName>(
  a: NDArray<D>,
  axis?: number,
  keepdims: boolean = false
): NDArray | number {
  if (axis === undefined) {
    return reduceAll(a, (x, y) => Math.max(x, y));
  }
  return reduceAxis(a, (x, y) => Math.max(x, y), axis, undefined, keepdims);
}

/**
 * Alias for max
 */
export const amax = max;

/**
 * Return the minimum of an array or minimum along an axis
 */
export function min<D extends DTypeName>(
  a: NDArray<D>,
  axis?: number,
  keepdims: boolean = false
): NDArray | number {
  if (axis === undefined) {
    return reduceAll(a, (x, y) => Math.min(x, y));
  }
  return reduceAxis(a, (x, y) => Math.min(x, y), axis, undefined, keepdims);
}

/**
 * Alias for min
 */
export const amin = min;

/**
 * Peak to peak (maximum - minimum) value along a given axis
 */
export function ptp<D extends DTypeName>(
  a: NDArray<D>,
  axis?: number,
  keepdims: boolean = false
): NDArray | number {
  const maxVal = max(a, axis, keepdims);
  const minVal = min(a, axis, keepdims);

  if (typeof maxVal === 'number' && typeof minVal === 'number') {
    return maxVal - minVal;
  }

  return subtract(maxVal as NDArray, minVal as NDArray);
}

/**
 * Clip (limit) the values in an array
 */
export function clip<D extends DTypeName>(
  a: NDArray<D>,
  aMin: number | null,
  aMax: number | null
): NDArray<D> {
  const result = a.copy();

  let i = 0;
  for (const val of a) {
    let clipped = Number(val);
    if (aMin !== null && clipped < aMin) clipped = aMin;
    if (aMax !== null && clipped > aMax) clipped = aMax;
    result.setFlat(i++, clipped);
  }

  return result;
}
