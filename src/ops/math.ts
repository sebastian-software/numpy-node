/**
 * Mathematical functions for numpy-node
 * Provides element-wise mathematical operations
 */

import { NDArray, type DTypeName } from '../core/index.js';
import { applyUnary } from './ufunc.js';

// ============================================================
// Trigonometric Functions
// ============================================================

/**
 * Trigonometric sine element-wise
 */
export function sin<D extends DTypeName>(x: NDArray<D>): NDArray<'float64'> {
  return applyUnary(x, Math.sin, undefined, 'float64') as NDArray<'float64'>;
}

/**
 * Trigonometric cosine element-wise
 */
export function cos<D extends DTypeName>(x: NDArray<D>): NDArray<'float64'> {
  return applyUnary(x, Math.cos, undefined, 'float64') as NDArray<'float64'>;
}

/**
 * Trigonometric tangent element-wise
 */
export function tan<D extends DTypeName>(x: NDArray<D>): NDArray<'float64'> {
  return applyUnary(x, Math.tan, undefined, 'float64') as NDArray<'float64'>;
}

/**
 * Inverse sine element-wise
 */
export function arcsin<D extends DTypeName>(x: NDArray<D>): NDArray<'float64'> {
  return applyUnary(x, Math.asin, undefined, 'float64') as NDArray<'float64'>;
}

/**
 * Alias for arcsin
 */
export const asin = arcsin;

/**
 * Inverse cosine element-wise
 */
export function arccos<D extends DTypeName>(x: NDArray<D>): NDArray<'float64'> {
  return applyUnary(x, Math.acos, undefined, 'float64') as NDArray<'float64'>;
}

/**
 * Alias for arccos
 */
export const acos = arccos;

/**
 * Inverse tangent element-wise
 */
export function arctan<D extends DTypeName>(x: NDArray<D>): NDArray<'float64'> {
  return applyUnary(x, Math.atan, undefined, 'float64') as NDArray<'float64'>;
}

/**
 * Alias for arctan
 */
export const atan = arctan;

/**
 * Element-wise arc tangent of x1/x2 choosing the quadrant correctly
 */
export function arctan2<D1 extends DTypeName, D2 extends DTypeName>(
  y: NDArray<D1>,
  x: NDArray<D2>
): NDArray<'float64'> {
  const { applyBinary } = require('./ufunc.js');
  return applyBinary(y, x, Math.atan2, undefined, 'float64') as NDArray<'float64'>;
}

/**
 * Alias for arctan2
 */
export const atan2 = arctan2;

/**
 * Given the "legs" of a right triangle, return its hypotenuse
 */
export function hypot<D1 extends DTypeName, D2 extends DTypeName>(
  x1: NDArray<D1>,
  x2: NDArray<D2>
): NDArray<'float64'> {
  const { applyBinary } = require('./ufunc.js');
  return applyBinary(x1, x2, Math.hypot, undefined, 'float64') as NDArray<'float64'>;
}

// ============================================================
// Hyperbolic Functions
// ============================================================

/**
 * Hyperbolic sine element-wise
 */
export function sinh<D extends DTypeName>(x: NDArray<D>): NDArray<'float64'> {
  return applyUnary(x, Math.sinh, undefined, 'float64') as NDArray<'float64'>;
}

/**
 * Hyperbolic cosine element-wise
 */
export function cosh<D extends DTypeName>(x: NDArray<D>): NDArray<'float64'> {
  return applyUnary(x, Math.cosh, undefined, 'float64') as NDArray<'float64'>;
}

/**
 * Hyperbolic tangent element-wise
 */
export function tanh<D extends DTypeName>(x: NDArray<D>): NDArray<'float64'> {
  return applyUnary(x, Math.tanh, undefined, 'float64') as NDArray<'float64'>;
}

/**
 * Inverse hyperbolic sine element-wise
 */
export function arcsinh<D extends DTypeName>(x: NDArray<D>): NDArray<'float64'> {
  return applyUnary(x, Math.asinh, undefined, 'float64') as NDArray<'float64'>;
}

/**
 * Alias for arcsinh
 */
export const asinh = arcsinh;

/**
 * Inverse hyperbolic cosine element-wise
 */
export function arccosh<D extends DTypeName>(x: NDArray<D>): NDArray<'float64'> {
  return applyUnary(x, Math.acosh, undefined, 'float64') as NDArray<'float64'>;
}

/**
 * Alias for arccosh
 */
export const acosh = arccosh;

/**
 * Inverse hyperbolic tangent element-wise
 */
export function arctanh<D extends DTypeName>(x: NDArray<D>): NDArray<'float64'> {
  return applyUnary(x, Math.atanh, undefined, 'float64') as NDArray<'float64'>;
}

/**
 * Alias for arctanh
 */
export const atanh = arctanh;

// ============================================================
// Rounding Functions
// ============================================================

/**
 * Round elements to the given number of decimals
 */
export function round<D extends DTypeName>(
  a: NDArray<D>,
  decimals: number = 0
): NDArray<'float64'> {
  const factor = Math.pow(10, decimals);
  return applyUnary(
    a,
    (x) => Math.round(x * factor) / factor,
    undefined,
    'float64'
  ) as NDArray<'float64'>;
}

/**
 * Alias for round
 */
export const around = round;

/**
 * Round to nearest even value
 */
export function rint<D extends DTypeName>(x: NDArray<D>): NDArray<'float64'> {
  return applyUnary(
    x,
    (val) => {
      const rounded = Math.round(val);
      // Handle halfway cases - round to even
      if (Math.abs(val - rounded + 0.5) < 1e-10) {
        return rounded % 2 === 0 ? rounded : rounded - 1;
      }
      return rounded;
    },
    undefined,
    'float64'
  ) as NDArray<'float64'>;
}

/**
 * Return the floor of the input element-wise
 */
export function floor<D extends DTypeName>(x: NDArray<D>): NDArray<'float64'> {
  return applyUnary(x, Math.floor, undefined, 'float64') as NDArray<'float64'>;
}

/**
 * Return the ceiling of the input element-wise
 */
export function ceil<D extends DTypeName>(x: NDArray<D>): NDArray<'float64'> {
  return applyUnary(x, Math.ceil, undefined, 'float64') as NDArray<'float64'>;
}

/**
 * Return the truncated value of the input element-wise
 */
export function trunc<D extends DTypeName>(x: NDArray<D>): NDArray<'float64'> {
  return applyUnary(x, Math.trunc, undefined, 'float64') as NDArray<'float64'>;
}

/**
 * Alias for trunc
 */
export const fix = trunc;

// ============================================================
// Exponential and Logarithmic Functions
// ============================================================

/**
 * Calculate the exponential of all elements
 */
export function exp<D extends DTypeName>(x: NDArray<D>): NDArray<'float64'> {
  return applyUnary(x, Math.exp, undefined, 'float64') as NDArray<'float64'>;
}

/**
 * Calculate exp(x) - 1 for all elements
 */
export function expm1<D extends DTypeName>(x: NDArray<D>): NDArray<'float64'> {
  return applyUnary(x, Math.expm1, undefined, 'float64') as NDArray<'float64'>;
}

/**
 * Calculate 2**p for all elements
 */
export function exp2<D extends DTypeName>(x: NDArray<D>): NDArray<'float64'> {
  return applyUnary(x, (val) => Math.pow(2, val), undefined, 'float64') as NDArray<'float64'>;
}

/**
 * Natural logarithm element-wise
 */
export function log<D extends DTypeName>(x: NDArray<D>): NDArray<'float64'> {
  return applyUnary(x, Math.log, undefined, 'float64') as NDArray<'float64'>;
}

/**
 * Return the natural logarithm of one plus the input array element-wise
 */
export function log1p<D extends DTypeName>(x: NDArray<D>): NDArray<'float64'> {
  return applyUnary(x, Math.log1p, undefined, 'float64') as NDArray<'float64'>;
}

/**
 * Base-2 logarithm element-wise
 */
export function log2<D extends DTypeName>(x: NDArray<D>): NDArray<'float64'> {
  return applyUnary(x, Math.log2, undefined, 'float64') as NDArray<'float64'>;
}

/**
 * Base-10 logarithm element-wise
 */
export function log10<D extends DTypeName>(x: NDArray<D>): NDArray<'float64'> {
  return applyUnary(x, Math.log10, undefined, 'float64') as NDArray<'float64'>;
}

/**
 * Logarithm of the sum of exponentiations of the inputs
 */
export function logaddexp<D1 extends DTypeName, D2 extends DTypeName>(
  x1: NDArray<D1>,
  x2: NDArray<D2>
): NDArray<'float64'> {
  const { applyBinary } = require('./ufunc.js');
  return applyBinary(
    x1,
    x2,
    (a: number, b: number) => {
      const maxVal = Math.max(a, b);
      return maxVal + Math.log(Math.exp(a - maxVal) + Math.exp(b - maxVal));
    },
    undefined,
    'float64'
  ) as NDArray<'float64'>;
}

/**
 * Logarithm of the sum of exponentiations of the inputs in base-2
 */
export function logaddexp2<D1 extends DTypeName, D2 extends DTypeName>(
  x1: NDArray<D1>,
  x2: NDArray<D2>
): NDArray<'float64'> {
  const { applyBinary } = require('./ufunc.js');
  return applyBinary(
    x1,
    x2,
    (a: number, b: number) => {
      const maxVal = Math.max(a, b);
      return maxVal + Math.log2(Math.pow(2, a - maxVal) + Math.pow(2, b - maxVal));
    },
    undefined,
    'float64'
  ) as NDArray<'float64'>;
}

// ============================================================
// Special Functions
// ============================================================

/**
 * Compute the sigmoid function element-wise
 */
export function sigmoid<D extends DTypeName>(x: NDArray<D>): NDArray<'float64'> {
  return applyUnary(
    x,
    (val) => 1 / (1 + Math.exp(-val)),
    undefined,
    'float64'
  ) as NDArray<'float64'>;
}

/**
 * Convert angles from radians to degrees
 */
export function degrees<D extends DTypeName>(x: NDArray<D>): NDArray<'float64'> {
  return applyUnary(
    x,
    (val) => (val * 180) / Math.PI,
    undefined,
    'float64'
  ) as NDArray<'float64'>;
}

/**
 * Alias for degrees
 */
export const rad2deg = degrees;

/**
 * Convert angles from degrees to radians
 */
export function radians<D extends DTypeName>(x: NDArray<D>): NDArray<'float64'> {
  return applyUnary(
    x,
    (val) => (val * Math.PI) / 180,
    undefined,
    'float64'
  ) as NDArray<'float64'>;
}

/**
 * Alias for radians
 */
export const deg2rad = radians;

/**
 * Unwrap by changing deltas between values to 2*pi complement
 */
export function unwrap<D extends DTypeName>(
  p: NDArray<D>,
  discont?: number,
  axis: number = -1
): NDArray<'float64'> {
  const discontinuity = discont ?? Math.PI;
  const result = p.astype('float64');

  if (p.ndim !== 1) {
    throw new Error(`unwrap currently only supports 1D arrays (got axis=${axis})`);
  }

  for (let i = 1; i < result.size; i++) {
    const diff = Number(result.getFlat(i)) - Number(result.getFlat(i - 1));
    if (Math.abs(diff) > discontinuity) {
      const correction = Math.round(diff / (2 * Math.PI)) * 2 * Math.PI;
      for (let j = i; j < result.size; j++) {
        result.setFlat(j, Number(result.getFlat(j)) - correction);
      }
    }
  }

  return result;
}

// ============================================================
// Floating Point Routines
// ============================================================

/**
 * Test element-wise for finiteness
 */
export function isfinite<D extends DTypeName>(x: NDArray<D>): NDArray<'bool'> {
  const result = NDArray.zeros<'bool'>(x.shape, { dtype: 'bool' });

  let i = 0;
  for (const val of x) {
    result.setFlat(i++, Number.isFinite(Number(val)) ? 1 : 0);
  }

  return result;
}

/**
 * Test element-wise for positive or negative infinity
 */
export function isinf<D extends DTypeName>(x: NDArray<D>): NDArray<'bool'> {
  const result = NDArray.zeros<'bool'>(x.shape, { dtype: 'bool' });

  let i = 0;
  for (const val of x) {
    const num = Number(val);
    result.setFlat(i++, num === Infinity || num === -Infinity ? 1 : 0);
  }

  return result;
}

/**
 * Test element-wise for NaN
 */
export function isnan<D extends DTypeName>(x: NDArray<D>): NDArray<'bool'> {
  const result = NDArray.zeros<'bool'>(x.shape, { dtype: 'bool' });

  let i = 0;
  for (const val of x) {
    result.setFlat(i++, Number.isNaN(Number(val)) ? 1 : 0);
  }

  return result;
}

/**
 * Test element-wise for negative infinity
 */
export function isneginf<D extends DTypeName>(x: NDArray<D>): NDArray<'bool'> {
  const result = NDArray.zeros<'bool'>(x.shape, { dtype: 'bool' });

  let i = 0;
  for (const val of x) {
    result.setFlat(i++, Number(val) === -Infinity ? 1 : 0);
  }

  return result;
}

/**
 * Test element-wise for positive infinity
 */
export function isposinf<D extends DTypeName>(x: NDArray<D>): NDArray<'bool'> {
  const result = NDArray.zeros<'bool'>(x.shape, { dtype: 'bool' });

  let i = 0;
  for (const val of x) {
    result.setFlat(i++, Number(val) === Infinity ? 1 : 0);
  }

  return result;
}

/**
 * Replace NaN with zero and infinity with large finite numbers
 */
export function nanToNum<D extends DTypeName>(
  x: NDArray<D>,
  options: { nan?: number; posinf?: number; neginf?: number } = {}
): NDArray<'float64'> {
  const nan = options.nan ?? 0;
  const posinf = options.posinf ?? Number.MAX_VALUE;
  const neginf = options.neginf ?? -Number.MAX_VALUE;

  return applyUnary(
    x,
    (val) => {
      if (Number.isNaN(val)) return nan;
      if (val === Infinity) return posinf;
      if (val === -Infinity) return neginf;
      return val;
    },
    undefined,
    'float64'
  ) as NDArray<'float64'>;
}

// ============================================================
// Constants
// ============================================================

export const pi = Math.PI;
export const e = Math.E;
export const inf = Infinity;
export const nan = NaN;
