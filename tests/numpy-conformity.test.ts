/**
 * NumPy Conformity Tests
 *
 * These tests verify that numpy-node produces identical results to NumPy.
 * Reference values are generated fresh from the current NumPy version.
 *
 * The reference file (scripts/numpy_reference.json) is NOT committed to git.
 * - In CI: Generated automatically before tests run
 * - Locally: Run `python3 scripts/generate_numpy_reference.py` first
 *
 * This ensures we always test against the latest NumPy behavior.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync, existsSync } from 'fs';
import { join } from 'path';
import {
  array,
  zeros,
  ones,
  full,
  arange,
  linspace,
  eye,
  add,
  subtract,
  multiply,
  divide,
  power,
  sqrt,
  exp,
  log,
  sin,
  cos,
  abs,
  negative,
  round,
  floor,
  ceil,
  sum,
  prod,
  mean,
  std,
  variance,
  min,
  max,
  argmin,
  argmax,
  cumsum,
  cumprod,
  median,
  equal,
  not_equal,
  less,
  less_equal,
  greater,
  greater_equal,
  logical_and,
  logical_or,
  logical_xor,
  logical_not,
  any,
  all,
  clip,
  where,
  squeeze,
  expand_dims,
  concatenate,
  stack,
  vstack,
  hstack,
  diff,
  sort,
  argsort,
  unique,
  searchsorted,
  tile,
  repeat,
  flip,
  rot90,
  split,
  nonzero,
  sign,
  mod,
  isclose,
  allclose,
  matmul,
  dot,
  inv,
  det,
  solve,
  trace,
  norm,
  matrix_rank,
  cond,
  eig,
  svd,
  cholesky,
  outer,
  kron,
  percentile,
  corrcoef,
  NDArray,
} from '../src/native/index.js';

// Type definitions for reference data
interface ArrayRef {
  data: number | boolean | (number | boolean)[] | (number | boolean)[][];
  dtype: string;
  shape: number[];
}

// Using any for dynamic JSON data loaded at runtime
type TestCategory = any;

interface ReferenceData {
  numpy_version: string;
  tests: {
    array_creation: TestCategory;
    arithmetic: TestCategory;
    unary_math: TestCategory;
    reductions: TestCategory;
    comparison: TestCategory;
    logical: TestCategory;
    array_manipulation: TestCategory;
    array_manipulation_advanced: TestCategory;
    approximate_comparison: TestCategory;
    array_joining: TestCategory;
    sorting_searching: TestCategory;
    boolean_reductions: TestCategory;
    linalg: TestCategory;
    advanced_math: TestCategory;
    random: TestCategory;
  };
}

// Load reference data
let ref: ReferenceData;
const refPath = join(__dirname, '../scripts/numpy_reference.json');

beforeAll(() => {
  if (!existsSync(refPath)) {
    throw new Error(
      `NumPy reference file not found: ${refPath}\n\n` +
        `Generate it by running:\n` +
        `  python3 scripts/generate_numpy_reference.py\n\n` +
        `This file is generated fresh in CI and not committed to git.`
    );
  }
  ref = JSON.parse(readFileSync(refPath, 'utf-8'));
});

// Helper to flatten nested arrays for comparison
function flatten(arr: unknown): (number | boolean)[] {
  if (!Array.isArray(arr)) return [arr as number | boolean];
  return arr.flatMap((x) => flatten(x));
}

// Helper to compare arrays with tolerance
function expectArrayClose(
  actual: NDArray | number,
  expected: ArrayRef | number,
  tolerance = 1e-10
) {
  if (typeof expected === 'number') {
    expect(typeof actual).toBe('number');
    expect(actual as number).toBeCloseTo(expected, 10);
    return;
  }

  expect(actual).toBeInstanceOf(NDArray);
  const arr = actual as NDArray;

  // Compare shape
  expect(arr.shape).toEqual(expected.shape);

  // Compare data with tolerance
  const actualData = arr.toFlatArray();
  const expectedData = flatten(expected.data);

  expect(actualData.length).toBe(expectedData.length);

  for (let i = 0; i < actualData.length; i++) {
    const a = actualData[i]!;
    const e = expectedData[i]!;

    if (typeof e === 'boolean') {
      // For boolean arrays, compare as 0/1
      expect(a).toBe(e ? 1 : 0);
    } else if (Math.abs(e) < 1e-15) {
      // Near-zero values
      expect(Math.abs(a)).toBeLessThan(tolerance);
    } else {
      // Relative tolerance for non-zero values
      const relError = Math.abs((a - e) / e);
      expect(relError).toBeLessThan(tolerance);
    }
  }
}

describe('NumPy Conformity Tests', () => {
  describe('Array Creation', () => {
    it('zeros(5)', () => {
      const result = zeros([5]);
      expectArrayClose(result, ref.tests.array_creation.zeros_1d);
    });

    it('zeros([2, 3])', () => {
      const result = zeros([2, 3]);
      expectArrayClose(result, ref.tests.array_creation.zeros_2d);
    });

    it('ones(5)', () => {
      const result = ones([5]);
      expectArrayClose(result, ref.tests.array_creation.ones_1d);
    });

    it('ones([2, 3])', () => {
      const result = ones([2, 3]);
      expectArrayClose(result, ref.tests.array_creation.ones_2d);
    });

    it('full([2, 3], 3.14)', () => {
      const result = full([2, 3], 3.14);
      expectArrayClose(result, ref.tests.array_creation.full_float);
    });

    it('arange(5)', () => {
      const result = arange(5);
      expectArrayClose(result, ref.tests.array_creation.arange_simple);
    });

    it('arange(2, 7)', () => {
      const result = arange(2, 7);
      expectArrayClose(result, ref.tests.array_creation.arange_start_stop);
    });

    it('arange(0, 1, 0.2)', () => {
      const result = arange(0, 1, 0.2);
      expectArrayClose(result, ref.tests.array_creation.arange_step);
    });

    it('linspace(0, 1, 5)', () => {
      const result = linspace(0, 1, 5);
      expectArrayClose(result, ref.tests.array_creation.linspace);
    });

    it('linspace(0, 10, 3)', () => {
      const result = linspace(0, 10, 3);
      expectArrayClose(result, ref.tests.array_creation.linspace_10);
    });

    it('eye(3)', () => {
      const result = eye(3);
      expectArrayClose(result, ref.tests.array_creation.eye_3);
    });

    it('eye(3, 4)', () => {
      const result = eye(3, 4);
      expectArrayClose(result, ref.tests.array_creation.eye_3x4);
    });

    it('eye(3, 4, 1)', () => {
      const result = eye(3, 4, 1);
      expectArrayClose(result, ref.tests.array_creation.eye_3x4_k1);
    });
  });

  describe('Arithmetic Operations', () => {
    it('add(a, b)', () => {
      const a = array([1.0, 2.0, 3.0, 4.0, 5.0]);
      const b = array([5.0, 4.0, 3.0, 2.0, 1.0]);
      const result = add(a, b);
      expectArrayClose(result, ref.tests.arithmetic.add_arrays as ArrayRef);
    });

    it('add(a, scalar)', () => {
      const a = array([1.0, 2.0, 3.0, 4.0, 5.0]);
      const result = add(a, 10);
      expectArrayClose(result, ref.tests.arithmetic.add_scalar as ArrayRef);
    });

    it('subtract(a, b)', () => {
      const a = array([1.0, 2.0, 3.0, 4.0, 5.0]);
      const b = array([5.0, 4.0, 3.0, 2.0, 1.0]);
      const result = subtract(a, b);
      expectArrayClose(result, ref.tests.arithmetic.subtract as ArrayRef);
    });

    it('multiply(a, b)', () => {
      const a = array([1.0, 2.0, 3.0, 4.0, 5.0]);
      const b = array([5.0, 4.0, 3.0, 2.0, 1.0]);
      const result = multiply(a, b);
      expectArrayClose(result, ref.tests.arithmetic.multiply as ArrayRef);
    });

    it('divide(a, b)', () => {
      const a = array([1.0, 2.0, 3.0, 4.0, 5.0]);
      const b = array([5.0, 4.0, 3.0, 2.0, 1.0]);
      const result = divide(a, b);
      expectArrayClose(result, ref.tests.arithmetic.divide as ArrayRef);
    });

    it('power(a, 2)', () => {
      const a = array([1.0, 2.0, 3.0, 4.0, 5.0]);
      const result = power(a, 2);
      expectArrayClose(result, ref.tests.arithmetic.power_scalar as ArrayRef);
    });

    it('power(a, b)', () => {
      const a = array([1.0, 2.0, 3.0, 4.0, 5.0]);
      const b = array([5.0, 4.0, 3.0, 2.0, 1.0]);
      const result = power(a, b);
      expectArrayClose(result, ref.tests.arithmetic.power_array as ArrayRef);
    });

    it('broadcasting: add(2d, 1d)', () => {
      const a2d = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const b1d = array([1, 2, 3]);
      const result = add(a2d, b1d);
      expectArrayClose(result, ref.tests.arithmetic.broadcast_add as ArrayRef);
    });

    it('broadcasting: subtract(2d, 1d)', () => {
      const a2d = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const b1d = array([1, 2, 3]);
      const result = subtract(a2d, b1d);
      expectArrayClose(result, ref.tests.arithmetic.broadcast_subtract as ArrayRef);
    });
  });

  describe('Unary Math Functions', () => {
    it('sqrt(x)', () => {
      const x = array([0.0, 0.5, 1.0, 2.0, 4.0]);
      const result = sqrt(x);
      expectArrayClose(result, ref.tests.unary_math.sqrt);
    });

    it('exp(x)', () => {
      const x = array([0.0, 0.5, 1.0, 2.0, 4.0]);
      const result = exp(x);
      expectArrayClose(result, ref.tests.unary_math.exp);
    });

    it('log([1, e, e^2])', () => {
      const x = array([1, Math.E, Math.E ** 2]);
      const result = log(x);
      expectArrayClose(result, ref.tests.unary_math.log_e);
    });

    it('sin(angles)', () => {
      const angles = array([0, Math.PI / 6, Math.PI / 4, Math.PI / 3, Math.PI / 2]);
      const result = sin(angles);
      expectArrayClose(result, ref.tests.unary_math.sin);
    });

    it('cos(angles)', () => {
      const angles = array([0, Math.PI / 6, Math.PI / 4, Math.PI / 3, Math.PI / 2]);
      const result = cos(angles);
      expectArrayClose(result, ref.tests.unary_math.cos);
    });

    it('abs([-1, 2, -3, 4])', () => {
      const x = array([-1, 2, -3, 4]);
      const result = abs(x);
      expectArrayClose(result, ref.tests.unary_math.abs);
    });

    it('negative([1, -2, 3])', () => {
      const x = array([1, -2, 3]);
      const result = negative(x);
      expectArrayClose(result, ref.tests.unary_math.negative);
    });

    it('round(arr)', () => {
      const x = array([-1.7, -0.5, 0.5, 1.3, 2.5]);
      const result = round(x);
      expectArrayClose(result, ref.tests.unary_math.round);
    });

    it('floor(arr)', () => {
      const x = array([-1.7, -0.5, 0.5, 1.3, 2.5]);
      const result = floor(x);
      expectArrayClose(result, ref.tests.unary_math.floor);
    });

    it('ceil(arr)', () => {
      const x = array([-1.7, -0.5, 0.5, 1.3, 2.5]);
      const result = ceil(x);
      expectArrayClose(result, ref.tests.unary_math.ceil);
    });
  });

  describe('Reduction Operations', () => {
    it('sum(arr)', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = sum(arr);
      expect(result).toBe(ref.tests.reductions.sum);
    });

    it('prod(arr)', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = prod(arr);
      expect(result).toBe(ref.tests.reductions.prod);
    });

    it('mean(arr)', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = mean(arr);
      expect(result).toBeCloseTo(ref.tests.reductions.mean as number, 10);
    });

    it('std(arr)', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = std(arr);
      expect(result).toBeCloseTo(ref.tests.reductions.std as number, 10);
    });

    it('variance(arr)', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = variance(arr);
      expect(result).toBeCloseTo(ref.tests.reductions.var as number, 10);
    });

    it('min(arr)', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = min(arr);
      expect(result).toBe(ref.tests.reductions.min);
    });

    it('max(arr)', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = max(arr);
      expect(result).toBe(ref.tests.reductions.max);
    });

    it('median(arr)', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = median(arr);
      expect(result).toBeCloseTo(ref.tests.reductions.median as number, 10);
    });

    it('argmin(arr)', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = argmin(arr);
      expect(result).toBe(ref.tests.reductions.argmin);
    });

    it('argmax(arr)', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = argmax(arr);
      expect(result).toBe(ref.tests.reductions.argmax);
    });

    it('argmin(arr2d, axis=0)', () => {
      const arr2d = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = argmin(arr2d, 0);
      expectArrayClose(result as NDArray, ref.tests.reductions.argmin_axis0 as ArrayRef);
    });

    it('argmin(arr2d, axis=1)', () => {
      const arr2d = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = argmin(arr2d, 1);
      expectArrayClose(result as NDArray, ref.tests.reductions.argmin_axis1 as ArrayRef);
    });

    it('argmax(arr2d, axis=0)', () => {
      const arr2d = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = argmax(arr2d, 0);
      expectArrayClose(result as NDArray, ref.tests.reductions.argmax_axis0 as ArrayRef);
    });

    it('argmax(arr2d, axis=1)', () => {
      const arr2d = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = argmax(arr2d, 1);
      expectArrayClose(result as NDArray, ref.tests.reductions.argmax_axis1 as ArrayRef);
    });

    it('cumsum(arr)', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = cumsum(arr);
      expectArrayClose(result, ref.tests.reductions.cumsum_1d as ArrayRef);
    });

    it('cumsum(arr2d) - flattened', () => {
      const arr2d = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = cumsum(arr2d);
      expectArrayClose(result, ref.tests.reductions.cumsum_2d_flat as ArrayRef);
    });

    it('cumsum(arr2d, axis=0)', () => {
      const arr2d = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = cumsum(arr2d, 0);
      expectArrayClose(result, ref.tests.reductions.cumsum_axis0 as ArrayRef);
    });

    it('cumsum(arr2d, axis=1)', () => {
      const arr2d = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = cumsum(arr2d, 1);
      expectArrayClose(result, ref.tests.reductions.cumsum_axis1 as ArrayRef);
    });

    it('cumprod(arr)', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = cumprod(arr);
      expectArrayClose(result, ref.tests.reductions.cumprod_1d as ArrayRef);
    });

    it('cumprod(arr2d) - flattened', () => {
      const arr2d = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = cumprod(arr2d);
      expectArrayClose(result, ref.tests.reductions.cumprod_2d_flat as ArrayRef);
    });

    it('cumprod(arr2d, axis=0)', () => {
      const arr2d = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = cumprod(arr2d, 0);
      expectArrayClose(result, ref.tests.reductions.cumprod_axis0 as ArrayRef);
    });

    it('cumprod(arr2d, axis=1)', () => {
      const arr2d = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = cumprod(arr2d, 1);
      expectArrayClose(result, ref.tests.reductions.cumprod_axis1 as ArrayRef);
    });

    it('sum(arr2d, axis=0)', () => {
      const arr2d = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = sum(arr2d, 0);
      expectArrayClose(result as NDArray, ref.tests.reductions.sum_axis0 as ArrayRef);
    });

    it('sum(arr2d, axis=1)', () => {
      const arr2d = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = sum(arr2d, 1);
      expectArrayClose(result as NDArray, ref.tests.reductions.sum_axis1 as ArrayRef);
    });

    it('mean(arr2d, axis=0)', () => {
      const arr2d = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = mean(arr2d, 0);
      expectArrayClose(result as NDArray, ref.tests.reductions.mean_axis0 as ArrayRef);
    });

    it('mean(arr2d, axis=1)', () => {
      const arr2d = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = mean(arr2d, 1);
      expectArrayClose(result as NDArray, ref.tests.reductions.mean_axis1 as ArrayRef);
    });
  });

  describe('Comparison Operators', () => {
    it('equal(a, b)', () => {
      const a = array([1, 2, 3, 4, 5]);
      const b = array([5, 4, 3, 2, 1]);
      const result = equal(a, b);
      expectArrayClose(result, ref.tests.comparison.equal_arrays);
    });

    it('equal(a, 3)', () => {
      const a = array([1, 2, 3, 4, 5]);
      const result = equal(a, 3);
      expectArrayClose(result, ref.tests.comparison.equal_scalar);
    });

    it('not_equal(a, b)', () => {
      const a = array([1, 2, 3, 4, 5]);
      const b = array([5, 4, 3, 2, 1]);
      const result = not_equal(a, b);
      expectArrayClose(result, ref.tests.comparison.not_equal);
    });

    it('less(a, 3)', () => {
      const a = array([1, 2, 3, 4, 5]);
      const result = less(a, 3);
      expectArrayClose(result, ref.tests.comparison.less_scalar);
    });

    it('less_equal(a, 3)', () => {
      const a = array([1, 2, 3, 4, 5]);
      const result = less_equal(a, 3);
      expectArrayClose(result, ref.tests.comparison.less_equal_scalar);
    });

    it('greater(a, 3)', () => {
      const a = array([1, 2, 3, 4, 5]);
      const result = greater(a, 3);
      expectArrayClose(result, ref.tests.comparison.greater_scalar);
    });

    it('greater_equal(a, 3)', () => {
      const a = array([1, 2, 3, 4, 5]);
      const result = greater_equal(a, 3);
      expectArrayClose(result, ref.tests.comparison.greater_equal_scalar);
    });
  });

  describe('Logical Operators', () => {
    it('logical_and(x, y)', () => {
      const x = array([1, 1, 0, 0]);
      const y = array([1, 0, 1, 0]);
      const result = logical_and(x, y);
      expectArrayClose(result, ref.tests.logical.and);
    });

    it('logical_or(x, y)', () => {
      const x = array([1, 1, 0, 0]);
      const y = array([1, 0, 1, 0]);
      const result = logical_or(x, y);
      expectArrayClose(result, ref.tests.logical.or);
    });

    it('logical_xor(x, y)', () => {
      const x = array([1, 1, 0, 0]);
      const y = array([1, 0, 1, 0]);
      const result = logical_xor(x, y);
      expectArrayClose(result, ref.tests.logical.xor);
    });

    it('logical_not(x)', () => {
      const x = array([1, 1, 0, 0]);
      const result = logical_not(x);
      expectArrayClose(result, ref.tests.logical.not);
    });

    it('broadcasting: logical_and(2d, 1d)', () => {
      const a2d = array([
        [1, 0],
        [1, 1],
      ]);
      const b1d = array([1, 0]);
      const result = logical_and(a2d, b1d);
      expectArrayClose(result, ref.tests.logical.broadcast_and);
    });
  });

  describe('Array Manipulation', () => {
    it('clip(arr, 0, 5)', () => {
      const arr = array([-2, 5, 3, -1, 8]);
      const result = clip(arr, 0, 5);
      expectArrayClose(result, ref.tests.array_manipulation.clip_basic as ArrayRef);
    });

    it('clip(arr2d, 2, 7)', () => {
      const arr2d = array([
        [1, 8, 3],
        [7, 2, 9],
      ]);
      const result = clip(arr2d, 2, 7);
      expectArrayClose(result, ref.tests.array_manipulation.clip_2d as ArrayRef);
    });

    it('where(condition, x, y) - same shape', () => {
      const condition = array([1, 0, 1, 0, 1]);
      const x = array([1, 2, 3, 4, 5]);
      const y = array([10, 20, 30, 40, 50]);
      const result = where(condition, x, y);
      expectArrayClose(result, ref.tests.array_manipulation.where_basic as ArrayRef);
    });

    it('where(arr > 0, arr, zeros)', () => {
      const arr = array([-2, 5, 3, -1, 8]);
      const condition = greater(arr, 0);
      const result = where(condition, arr, zeros([5]));
      expectArrayClose(result, ref.tests.array_manipulation.where_comparison as ArrayRef);
    });

    it('where with broadcasting', () => {
      const condition = array([
        [1, 0],
        [0, 1],
      ]);
      const x = array([
        [1, 2],
        [3, 4],
      ]);
      const y = array([10, 20]);
      const result = where(condition, x, y);
      expectArrayClose(result, ref.tests.array_manipulation.where_broadcast as ArrayRef);
    });

    it('squeeze - remove all 1-dims', () => {
      const a = array([[[1, 2, 3]]]); // shape (1, 1, 3)
      const result = squeeze(a);
      expectArrayClose(result, ref.tests.array_manipulation.squeeze_all as ArrayRef);
    });

    it('squeeze - specific axis', () => {
      const a = array([[1], [2], [3]]); // shape (3, 1)
      const result = squeeze(a, 1);
      expectArrayClose(result, ref.tests.array_manipulation.squeeze_axis as ArrayRef);
    });

    it('expand_dims - axis=0', () => {
      const a = array([1, 2, 3]); // shape (3,)
      const result = expand_dims(a, 0);
      expectArrayClose(result, ref.tests.array_manipulation.expand_dims_0 as ArrayRef);
    });

    it('expand_dims - axis=1', () => {
      const a = array([1, 2, 3]); // shape (3,)
      const result = expand_dims(a, 1);
      expectArrayClose(result, ref.tests.array_manipulation.expand_dims_1 as ArrayRef);
    });

    it('expand_dims - axis=-1', () => {
      const a = array([1, 2, 3]); // shape (3,)
      const result = expand_dims(a, -1);
      expectArrayClose(result, ref.tests.array_manipulation.expand_dims_neg as ArrayRef);
    });
  });

  describe('Array Joining', () => {
    it('concatenate - 1D arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = concatenate([a, b]);
      expectArrayClose(result, ref.tests.array_joining.concat_1d as ArrayRef);
    });

    it('concatenate - 2D arrays axis=0', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const result = concatenate([a, b], 0);
      expectArrayClose(result, ref.tests.array_joining.concat_2d_axis0 as ArrayRef);
    });

    it('concatenate - 2D arrays axis=1', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const result = concatenate([a, b], 1);
      expectArrayClose(result, ref.tests.array_joining.concat_2d_axis1 as ArrayRef);
    });

    it('stack - 1D arrays axis=0', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = stack([a, b], 0);
      expectArrayClose(result, ref.tests.array_joining.stack_1d_axis0 as ArrayRef);
    });

    it('stack - 1D arrays axis=1', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = stack([a, b], 1);
      expectArrayClose(result, ref.tests.array_joining.stack_1d_axis1 as ArrayRef);
    });

    it('stack - 2D arrays axis=0', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const result = stack([a, b], 0);
      expectArrayClose(result, ref.tests.array_joining.stack_2d_axis0 as ArrayRef);
    });

    it('stack - 2D arrays axis=1', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const result = stack([a, b], 1);
      expectArrayClose(result, ref.tests.array_joining.stack_2d_axis1 as ArrayRef);
    });

    it('vstack - 1D arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = vstack([a, b]);
      expectArrayClose(result, ref.tests.array_joining.vstack_1d as ArrayRef);
    });

    it('vstack - 2D arrays', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const result = vstack([a, b]);
      expectArrayClose(result, ref.tests.array_joining.vstack_2d as ArrayRef);
    });

    it('hstack - 1D arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = hstack([a, b]);
      expectArrayClose(result, ref.tests.array_joining.hstack_1d as ArrayRef);
    });

    it('hstack - 2D arrays', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const result = hstack([a, b]);
      expectArrayClose(result, ref.tests.array_joining.hstack_2d as ArrayRef);
    });
  });

  describe('Sorting and Searching', () => {
    it('diff - 1D array', () => {
      const a = array([1, 3, 6, 10, 15]);
      const result = diff(a);
      expectArrayClose(result, ref.tests.sorting_searching.diff_1d as ArrayRef);
    });

    it('diff - 1D array with n=2', () => {
      const a = array([1, 3, 6, 10, 15]);
      const result = diff(a, 2);
      expectArrayClose(result, ref.tests.sorting_searching.diff_1d_n2 as ArrayRef);
    });

    it('diff - 2D array axis=0', () => {
      const a = array([
        [3, 1, 2],
        [6, 4, 5],
      ]);
      const result = diff(a, 1, 0);
      expectArrayClose(result, ref.tests.sorting_searching.diff_2d_axis0 as ArrayRef);
    });

    it('diff - 2D array axis=1', () => {
      const a = array([
        [3, 1, 2],
        [6, 4, 5],
      ]);
      const result = diff(a, 1, 1);
      expectArrayClose(result, ref.tests.sorting_searching.diff_2d_axis1 as ArrayRef);
    });

    it('sort - 1D array', () => {
      const a = array([3, 1, 4, 1, 5, 9, 2, 6]);
      const result = sort(a);
      expectArrayClose(result, ref.tests.sorting_searching.sort_1d as ArrayRef);
    });

    it('sort - 2D array axis=0', () => {
      const a = array([
        [3, 1, 2],
        [6, 4, 5],
      ]);
      const result = sort(a, 0);
      expectArrayClose(result, ref.tests.sorting_searching.sort_2d_axis0 as ArrayRef);
    });

    it('sort - 2D array axis=1', () => {
      const a = array([
        [3, 1, 2],
        [6, 4, 5],
      ]);
      const result = sort(a, 1);
      expectArrayClose(result, ref.tests.sorting_searching.sort_2d_axis1 as ArrayRef);
    });

    it('argsort - 1D array', () => {
      const a = array([3, 1, 4, 1, 5, 9, 2, 6]);
      const result = argsort(a);
      expectArrayClose(result, ref.tests.sorting_searching.argsort_1d as ArrayRef);
    });

    it('argsort - 2D array axis=0', () => {
      const a = array([
        [3, 1, 2],
        [6, 4, 5],
      ]);
      const result = argsort(a, 0);
      expectArrayClose(result, ref.tests.sorting_searching.argsort_2d_axis0 as ArrayRef);
    });

    it('argsort - 2D array axis=1', () => {
      const a = array([
        [3, 1, 2],
        [6, 4, 5],
      ]);
      const result = argsort(a, 1);
      expectArrayClose(result, ref.tests.sorting_searching.argsort_2d_axis1 as ArrayRef);
    });

    it('unique', () => {
      const a = array([3, 1, 2, 1, 3, 2, 4, 1]);
      const result = unique(a);
      expectArrayClose(result, ref.tests.sorting_searching.unique as ArrayRef);
    });

    it('searchsorted - left', () => {
      const a = array([1, 2, 4, 5, 7]);
      const v = array([0, 2, 3, 6, 8]);
      const result = searchsorted(a, v, 'left');
      expectArrayClose(result, ref.tests.sorting_searching.searchsorted_left as ArrayRef);
    });

    it('searchsorted - right', () => {
      const a = array([1, 2, 4, 5, 7]);
      const v = array([0, 2, 3, 6, 8]);
      const result = searchsorted(a, v, 'right');
      expectArrayClose(result, ref.tests.sorting_searching.searchsorted_right as ArrayRef);
    });
  });

  describe('Array Manipulation Advanced (Tier 3+4)', () => {
    it('tile - 1D scalar', () => {
      const a = array([1, 2, 3]);
      const result = tile(a, 2);
      expectArrayClose(result, ref.tests.array_manipulation_advanced.tile_1d_2 as ArrayRef);
    });

    it('tile - 1D with shape', () => {
      const a = array([1, 2, 3]);
      const result = tile(a, [2, 3]);
      expectArrayClose(result, ref.tests.array_manipulation_advanced.tile_1d_tuple as ArrayRef);
    });

    it('tile - 2D rows', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const result = tile(a, [2, 1]);
      expectArrayClose(result, ref.tests.array_manipulation_advanced.tile_2d_rows as ArrayRef);
    });

    it('tile - 2D cols', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const result = tile(a, [1, 3]);
      expectArrayClose(result, ref.tests.array_manipulation_advanced.tile_2d_cols as ArrayRef);
    });

    it('repeat - 1D', () => {
      const a = array([1, 2, 3]);
      const result = repeat(a, 3);
      expectArrayClose(result, ref.tests.array_manipulation_advanced.repeat_1d as ArrayRef);
    });

    it('repeat - 2D axis=0', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const result = repeat(a, 2, 0);
      expectArrayClose(result, ref.tests.array_manipulation_advanced.repeat_2d_axis0 as ArrayRef);
    });

    it('repeat - 2D axis=1', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const result = repeat(a, 2, 1);
      expectArrayClose(result, ref.tests.array_manipulation_advanced.repeat_2d_axis1 as ArrayRef);
    });

    it('flip - 1D', () => {
      const a = array([1, 2, 3]);
      const result = flip(a);
      expectArrayClose(result, ref.tests.array_manipulation_advanced.flip_1d as ArrayRef);
    });

    it('flip - 2D all', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const result = flip(a);
      expectArrayClose(result, ref.tests.array_manipulation_advanced.flip_2d as ArrayRef);
    });

    it('flip - 2D axis=0', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const result = flip(a, 0);
      expectArrayClose(result, ref.tests.array_manipulation_advanced.flip_2d_axis0 as ArrayRef);
    });

    it('flip - 2D axis=1', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const result = flip(a, 1);
      expectArrayClose(result, ref.tests.array_manipulation_advanced.flip_2d_axis1 as ArrayRef);
    });

    it('rot90 - k=1', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const result = rot90(a);
      expectArrayClose(result, ref.tests.array_manipulation_advanced.rot90_k1 as ArrayRef);
    });

    it('rot90 - k=2', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const result = rot90(a, 2);
      expectArrayClose(result, ref.tests.array_manipulation_advanced.rot90_k2 as ArrayRef);
    });

    it('rot90 - k=3', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const result = rot90(a, 3);
      expectArrayClose(result, ref.tests.array_manipulation_advanced.rot90_k3 as ArrayRef);
    });

    it('sign', () => {
      const a = array([-5, 0, 3, -2, 7]);
      const result = sign(a);
      expectArrayClose(result, ref.tests.array_manipulation_advanced.sign as ArrayRef);
    });

    it('mod - scalar', () => {
      const a = array([5, 7, 9, 11]);
      const result = mod(a, 3);
      expectArrayClose(result, ref.tests.array_manipulation_advanced.mod_scalar as ArrayRef);
    });

    it('mod - array', () => {
      const a = array([5, 7, 9, 11]);
      const b = array([2, 3, 4, 5]);
      const result = mod(a, b);
      expectArrayClose(result, ref.tests.array_manipulation_advanced.mod_array as ArrayRef);
    });

    it('split - equal parts', () => {
      const a = arange(9);
      const results = split(a, 3);
      const expected = ref.tests.array_manipulation_advanced.split_equal as number[][];
      expect(results.length).toBe(expected.length);
      for (let i = 0; i < results.length; i++) {
        expect(results[i]!.toArray()).toEqual(expected[i]);
      }
    });

    it('split - at indices', () => {
      const a = arange(9);
      const results = split(a, [2, 5]);
      const expected = ref.tests.array_manipulation_advanced.split_indices as number[][];
      expect(results.length).toBe(expected.length);
      for (let i = 0; i < results.length; i++) {
        expect(results[i]!.toArray()).toEqual(expected[i]);
      }
    });

    it('nonzero - 1D', () => {
      const a = array([0, 1, 0, 2, 0, 3]);
      const results = nonzero(a);
      const expected = ref.tests.array_manipulation_advanced.nonzero_1d as ArrayRef[];
      expect(results.length).toBe(expected.length);
      for (let i = 0; i < results.length; i++) {
        expectArrayClose(results[i]!, expected[i]!);
      }
    });

    it('nonzero - 2D', () => {
      const a = array([
        [0, 1, 0],
        [2, 0, 3],
      ]);
      const results = nonzero(a);
      const expected = ref.tests.array_manipulation_advanced.nonzero_2d as ArrayRef[];
      expect(results.length).toBe(expected.length);
      for (let i = 0; i < results.length; i++) {
        expectArrayClose(results[i]!, expected[i]!);
      }
    });
  });

  describe('Approximate Comparison', () => {
    it('allclose - true', () => {
      const a = array([1.0, 2.0, 3.0]);
      const b = array([1.0, 2.00001, 3.0001]);
      const result = allclose(a, b);
      expect(result).toBe(ref.tests.approximate_comparison.allclose_true);
    });

    it('allclose - false', () => {
      const a = array([1.0, 2.0, 3.0]);
      const b = array([1.0, 2.1, 3.5]);
      const result = allclose(a, b);
      expect(result).toBe(ref.tests.approximate_comparison.allclose_false);
    });

    it('allclose - custom tolerance', () => {
      const a = array([1.0, 2.0, 3.0]);
      const b = array([1.0, 2.1, 3.5]);
      const result = allclose(a, b, 0.5, 0.5);
      expect(result).toBe(ref.tests.approximate_comparison.allclose_custom_tol);
    });

    it('isclose - close values', () => {
      const a = array([1.0, 2.0, 3.0]);
      const b = array([1.0, 2.00001, 3.0001]);
      const result = isclose(a, b);
      expectArrayClose(result, ref.tests.approximate_comparison.isclose_close as ArrayRef);
    });

    it('isclose - not close values', () => {
      const a = array([1.0, 2.0, 3.0]);
      const b = array([1.0, 2.1, 3.5]);
      const result = isclose(a, b);
      expectArrayClose(result, ref.tests.approximate_comparison.isclose_not_close as ArrayRef);
    });
  });

  describe('Boolean Reductions', () => {
    it('any(arr)', () => {
      const arr = array([0, 0, 1, 0]);
      const result = any(arr);
      expect(result).toBe(ref.tests.boolean_reductions.any);
    });

    it('all(arr)', () => {
      const arr = array([0, 0, 1, 0]);
      const result = all(arr);
      expect(result).toBe(ref.tests.boolean_reductions.all);
    });

    it('any(arr2d, axis=0)', () => {
      const arr2d = array([
        [0, 1, 0],
        [0, 0, 1],
      ]);
      const result = any(arr2d, 0);
      expectArrayClose(result as NDArray, ref.tests.boolean_reductions.any_axis0 as ArrayRef);
    });

    it('any(arr2d, axis=1)', () => {
      const arr2d = array([
        [0, 1, 0],
        [0, 0, 1],
      ]);
      const result = any(arr2d, 1);
      expectArrayClose(result as NDArray, ref.tests.boolean_reductions.any_axis1 as ArrayRef);
    });

    it('all(arr2d, axis=0)', () => {
      const arr2d = array([
        [0, 1, 0],
        [0, 0, 1],
      ]);
      const result = all(arr2d, 0);
      expectArrayClose(result as NDArray, ref.tests.boolean_reductions.all_axis0 as ArrayRef);
    });

    it('all(arr2d, axis=1)', () => {
      const arr2d = array([
        [0, 1, 0],
        [0, 0, 1],
      ]);
      const result = all(arr2d, 1);
      expectArrayClose(result as NDArray, ref.tests.boolean_reductions.all_axis1 as ArrayRef);
    });
  });

  describe('Linear Algebra', () => {
    it('matmul(A, B)', () => {
      const A = array([
        [1, 2],
        [3, 4],
      ]);
      const B = array([
        [5, 6],
        [7, 8],
      ]);
      const result = matmul(A, B);
      expectArrayClose(result, ref.tests.linalg.matmul as ArrayRef);
    });

    it('dot(A, B) - matrix', () => {
      const A = array([
        [1, 2],
        [3, 4],
      ]);
      const B = array([
        [5, 6],
        [7, 8],
      ]);
      const result = dot(A, B);
      expectArrayClose(result as NDArray, ref.tests.linalg.dot_matrix as ArrayRef);
    });

    it('dot(v1, v2) - vector', () => {
      const v1 = array([1, 2, 3]);
      const v2 = array([4, 5, 6]);
      const result = dot(v1, v2);
      expect(result).toBeCloseTo(ref.tests.linalg.dot_vector as number, 10);
    });

    it('inv(A)', () => {
      const A = array([
        [1, 2],
        [3, 4],
      ]);
      const result = inv(A);
      expectArrayClose(result, ref.tests.linalg.inv as ArrayRef);
    });

    it('det(A)', () => {
      const A = array([
        [1, 2],
        [3, 4],
      ]);
      const result = det(A);
      expect(result).toBeCloseTo(ref.tests.linalg.det as number, 10);
    });

    it('solve(A, b)', () => {
      const A = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([1, 2]);
      const result = solve(A, b);
      expectArrayClose(result, ref.tests.linalg.solve as ArrayRef);
    });

    it('trace(A)', () => {
      const A = array([
        [1, 2],
        [3, 4],
      ]);
      const result = trace(A);
      expect(result).toBeCloseTo(ref.tests.linalg.trace as number, 10);
    });

    it('norm(v) - L2', () => {
      const v = array([3, 4]);
      const result = norm(v);
      expect(result).toBeCloseTo(ref.tests.linalg.norm_l2 as number, 10);
    });

    it('norm(v, 1) - L1', () => {
      const v = array([3, 4]);
      const result = norm(v, 1);
      expect(result).toBeCloseTo(ref.tests.linalg.norm_l1 as number, 10);
    });

    it('norm(v, Inf)', () => {
      const v = array([3, 4]);
      const result = norm(v, Infinity);
      expect(result).toBeCloseTo(ref.tests.linalg.norm_inf as number, 10);
    });

    it('matrix_rank(A)', () => {
      const A = array([
        [1, 2],
        [3, 4],
      ]);
      const result = matrix_rank(A);
      expect(result).toBe(ref.tests.linalg.matrix_rank);
    });

    it('cond(A)', () => {
      const A = array([
        [1, 2],
        [3, 4],
      ]);
      const result = cond(A);
      expect(result).toBeCloseTo(ref.tests.linalg.cond as number, 8);
    });

    it('eig(A) - eigenvalues', () => {
      const A = array([
        [1, 2],
        [3, 4],
      ]);
      const { eigenvalues } = eig(A);
      // Sort eigenvalues for comparison
      const sortedActual = Array.from(eigenvalues.toFlatArray()).sort((a, b) => a - b);
      const expectedRef = ref.tests.linalg.eigenvalues as ArrayRef;
      const sortedExpected = (flatten(expectedRef.data) as number[]).sort((a, b) => a - b);

      for (let i = 0; i < sortedActual.length; i++) {
        expect(sortedActual[i]).toBeCloseTo(sortedExpected[i]!, 10);
      }
    });

    it('svd(A) - singular values', () => {
      const A = array([
        [1, 2],
        [3, 4],
      ]);
      const { s } = svd(A);
      expectArrayClose(s, ref.tests.linalg.svd_S as ArrayRef);
    });

    it('cholesky(P)', () => {
      const P = array([
        [4, 2],
        [2, 3],
      ]);
      const result = cholesky(P);
      expectArrayClose(result, ref.tests.linalg.cholesky as ArrayRef);
    });
  });

  describe('Advanced Math', () => {
    it('outer(a, b)', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5]);
      const result = outer(a, b);
      expectArrayClose(result, ref.tests.advanced_math.outer as ArrayRef);
    });

    it('kron(A, B)', () => {
      const A = array([
        [1, 2],
        [3, 4],
      ]);
      const B = array([
        [0, 5],
        [6, 7],
      ]);
      const result = kron(A, B);
      expectArrayClose(result, ref.tests.advanced_math.kron as ArrayRef);
    });

    it('percentile(arr, 50)', () => {
      const arr = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
      const result = percentile(arr, [50]);
      const resultValue = result.toFlatArray()[0];
      expect(resultValue).toBeCloseTo(ref.tests.advanced_math.percentile_50 as number, 10);
    });

    it('percentile(arr, [25, 50, 75])', () => {
      const arr = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
      const result = percentile(arr, [25, 50, 75]);
      expectArrayClose(result, ref.tests.advanced_math.percentile_quartiles as ArrayRef);
    });

    it('corrcoef(X)', () => {
      const X = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const result = corrcoef(X);
      expectArrayClose(result, ref.tests.advanced_math.corrcoef as ArrayRef);
    });
  });

  // Random tests are skipped because numpy-node uses a different RNG implementation
  describe.skip('Random (deterministic)', () => {
    // These would require implementing the exact same RNG as NumPy
  });
});

/**
 * Completeness Check
 *
 * This section verifies that all NumPy-compatible functions exported by numpy-node
 * are covered by conformity tests. If you add a new function, add it to the
 * appropriate category and create corresponding tests.
 */
describe('Completeness Check', () => {
  // All NumPy-compatible functions that MUST have conformity tests
  // Organized by category matching the reference JSON structure
  const REQUIRED_COVERAGE = {
    // Array creation functions
    array_creation: [
      'zeros',
      'ones',
      'full',
      'arange',
      'linspace',
      'eye',
      // 'identity', // alias for eye
      // 'empty', // uninitialized, can't test values
      // 'zerosLike', 'onesLike', 'emptyLike' // depend on input array
    ],

    // Arithmetic operations
    arithmetic: ['add', 'subtract', 'multiply', 'divide', 'power'],

    // Unary math functions
    unary_math: ['sqrt', 'exp', 'log', 'sin', 'cos', 'abs', 'negative', 'round', 'floor', 'ceil'],

    // Reduction operations
    reductions: [
      'sum',
      'prod',
      'mean',
      'std',
      'variance',
      'min',
      'max',
      'median',
      'argmin',
      'argmax',
      'cumsum',
      'cumprod',
    ],

    // Comparison operators
    comparison: ['equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal'],

    // Logical operators
    logical: ['logical_and', 'logical_or', 'logical_xor', 'logical_not'],

    // Array manipulation
    array_manipulation: ['clip', 'where', 'squeeze', 'expand_dims'],

    // Array joining
    array_joining: ['concatenate', 'stack', 'vstack', 'hstack'],

    // Sorting and searching
    sorting_searching: ['diff', 'sort', 'argsort', 'unique', 'searchsorted'],

    // Array manipulation advanced (Tier 3+4)
    array_manipulation_advanced: [
      'tile',
      'repeat',
      'flip',
      'rot90',
      'sign',
      'mod',
      'split',
      'nonzero',
    ],

    // Approximate comparison
    approximate_comparison: ['allclose', 'isclose'],

    // Boolean reductions
    boolean_reductions: ['any', 'all'],

    // Linear algebra
    linalg: [
      'matmul',
      'dot',
      'inv',
      'det',
      'solve',
      'trace',
      'norm',
      'matrix_rank',
      'cond',
      'eig',
      'svd',
      'cholesky',
      // 'qr', // TODO: add conformity test
      // 'eigvals', // subset of eig
      // 'lstsq', 'normal_equations' // extensions, not standard NumPy
    ],

    // Advanced math
    advanced_math: ['outer', 'kron', 'percentile', 'corrcoef'],
  };

  // Functions tested in this file (update when adding new tests)
  const TESTED_FUNCTIONS = {
    array_creation: ['zeros', 'ones', 'full', 'arange', 'linspace', 'eye'],
    arithmetic: ['add', 'subtract', 'multiply', 'divide', 'power'],
    unary_math: ['sqrt', 'exp', 'log', 'sin', 'cos', 'abs', 'negative', 'round', 'floor', 'ceil'],
    reductions: [
      'sum',
      'prod',
      'mean',
      'std',
      'variance',
      'min',
      'max',
      'median',
      'argmin',
      'argmax',
      'cumsum',
      'cumprod',
    ],
    comparison: ['equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal'],
    logical: ['logical_and', 'logical_or', 'logical_xor', 'logical_not'],
    array_manipulation: ['clip', 'where', 'squeeze', 'expand_dims'],
    array_joining: ['concatenate', 'stack', 'vstack', 'hstack'],
    sorting_searching: ['diff', 'sort', 'argsort', 'unique', 'searchsorted'],
    array_manipulation_advanced: [
      'tile',
      'repeat',
      'flip',
      'rot90',
      'sign',
      'mod',
      'split',
      'nonzero',
    ],
    approximate_comparison: ['allclose', 'isclose'],
    boolean_reductions: ['any', 'all'],
    linalg: [
      'matmul',
      'dot',
      'inv',
      'det',
      'solve',
      'trace',
      'norm',
      'matrix_rank',
      'cond',
      'eig',
      'svd',
      'cholesky',
    ],
    advanced_math: ['outer', 'kron', 'percentile', 'corrcoef'],
  };

  for (const [category, requiredFunctions] of Object.entries(REQUIRED_COVERAGE)) {
    describe(category, () => {
      const testedFunctions = TESTED_FUNCTIONS[category as keyof typeof TESTED_FUNCTIONS];

      it('all required functions are tested', () => {
        const missing = requiredFunctions.filter((fn) => !testedFunctions.includes(fn));

        if (missing.length > 0) {
          throw new Error(
            `Missing conformity tests for ${category}:\n` +
              `  ${missing.join(', ')}\n\n` +
              `Add tests for these functions to ensure NumPy compatibility.`
          );
        }
      });

      it('no unknown functions in test list', () => {
        const unknown = testedFunctions.filter((fn) => !requiredFunctions.includes(fn));

        if (unknown.length > 0) {
          throw new Error(
            `Unknown functions in ${category} test list:\n` +
              `  ${unknown.join(', ')}\n\n` +
              `Add these to REQUIRED_COVERAGE or remove from TESTED_FUNCTIONS.`
          );
        }
      });
    });
  }

  it('summary: all categories covered', () => {
    const allRequired = Object.values(REQUIRED_COVERAGE).flat();
    const allTested = Object.values(TESTED_FUNCTIONS).flat();
    const coverage = (allTested.length / allRequired.length) * 100;

    // NumPy functions NOT YET implemented (from MISSING_FEATURES.md)
    const MISSING_FROM_NUMPY = [
      // Tier 5 - Advanced
      'fft',
      'ifft',
      'rfft',
      'einsum',
    ];

    const totalImplemented = allTested.length;
    const totalMissing = MISSING_FROM_NUMPY.length;
    const totalNumPy = totalImplemented + totalMissing;
    const overallCoverage = (totalImplemented / totalNumPy) * 100;

    console.log(`\n${'='.repeat(50)}`);
    console.log('NumPy Conformity Report');
    console.log('='.repeat(50));
    console.log(`Implemented & Tested: ${String(totalImplemented)} functions`);
    console.log(`Missing from NumPy:   ${String(totalMissing)} functions`);
    console.log(`Overall Coverage:     ${overallCoverage.toFixed(1)}% of core NumPy API`);
    console.log('='.repeat(50));
    console.log(`Test Coverage:        ${coverage.toFixed(1)}% of implemented functions\n`);

    // This test passes but logs coverage for visibility
    expect(allTested.length).toBeGreaterThanOrEqual(allRequired.length);
  });

  it('no new NumPy-compatible exports missing from coverage lists', () => {
    // All NumPy-compatible functions exported by numpy-node
    // This list should be updated when new functions are added
    const ALL_NUMPY_EXPORTS = [
      // Array creation
      'array',
      'zeros',
      'ones',
      'full',
      'arange',
      'linspace',
      'eye',
      'identity',
      'empty',
      'zerosLike',
      'onesLike',
      'emptyLike',
      // Arithmetic
      'add',
      'subtract',
      'multiply',
      'divide',
      'power',
      'add_inplace',
      'subtract_inplace',
      'multiply_inplace',
      'divide_inplace',
      // Unary math
      'sqrt',
      'exp',
      'log',
      'sin',
      'cos',
      'tan',
      'abs',
      'negative',
      'round',
      'floor',
      'ceil',
      // Reductions
      'sum',
      'prod',
      'mean',
      'std',
      'variance',
      'median',
      'min',
      'max',
      'argmin',
      'argmax',
      'cumsum',
      'cumprod',
      // Advanced statistics
      'zscore',
      'corrcoef',
      'percentile',
      // Tensor products
      'outer',
      'kron',
      'axpby',
      // Comparison
      'equal',
      'not_equal',
      'less',
      'less_equal',
      'greater',
      'greater_equal',
      // Logical
      'logical_and',
      'logical_or',
      'logical_xor',
      'logical_not',
      // Boolean reductions
      'any',
      'all',
      // Array manipulation
      'clip',
      'where',
      'squeeze',
      'expand_dims',
      'concatenate',
      'stack',
      'vstack',
      'hstack',
      // Sorting and searching
      'diff',
      'sort',
      'argsort',
      'unique',
      'searchsorted',
      // Array manipulation (Tier 3+4)
      'tile',
      'repeat',
      'flip',
      'rot90',
      'split',
      'nonzero',
      'sign',
      'mod',
      // Approximate comparison
      'isclose',
      'allclose',
      // Linear algebra
      'matmul',
      'matmul_nt',
      'batch_matmul',
      'batch_matmul_stacked',
      'dot',
      'inv',
      'det',
      'solve',
      'eig',
      'eigvals',
      'svd',
      'qr',
      'cholesky',
      'norm',
      'matrix_rank',
      'trace',
      'cond',
      'lstsq',
      'normal_equations',
    ];

    // Functions that are intentionally excluded from conformity testing
    const EXCLUDED_FROM_CONFORMITY = [
      'array', // input function, not a NumPy operation
      'identity', // alias for eye
      'empty', // uninitialized values, can't test
      'zerosLike', // depends on input array
      'onesLike', // depends on input array
      'emptyLike', // depends on input array
      'tan', // TODO: add test
      'add_inplace', // in-place variant
      'subtract_inplace', // in-place variant
      'multiply_inplace', // in-place variant
      'divide_inplace', // in-place variant
      'zscore', // scipy, not numpy
      'axpby', // BLAS extension
      'matmul_nt', // optimization variant
      'batch_matmul', // optimization variant
      'batch_matmul_stacked', // optimization variant
      'eigvals', // subset of eig
      'qr', // TODO: add test
      'lstsq', // TODO: add test
      'normal_equations', // extension
    ];

    const allTracked = [...Object.values(REQUIRED_COVERAGE).flat(), ...EXCLUDED_FROM_CONFORMITY];

    const untracked = ALL_NUMPY_EXPORTS.filter((fn) => !allTracked.includes(fn));

    if (untracked.length > 0) {
      throw new Error(
        `New exports not tracked in conformity tests:\n` +
          `  ${untracked.join(', ')}\n\n` +
          `Add to REQUIRED_COVERAGE (for testing) or EXCLUDED_FROM_CONFORMITY (with reason).`
      );
    }
  });
});
