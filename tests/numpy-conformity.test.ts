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
  sum,
  prod,
  mean,
  std,
  variance,
  min,
  max,
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
    unary_math: ['sqrt', 'exp', 'log', 'sin', 'cos', 'abs', 'negative'],

    // Reduction operations
    reductions: ['sum', 'prod', 'mean', 'std', 'variance', 'min', 'max', 'median'],

    // Comparison operators
    comparison: ['equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal'],

    // Logical operators
    logical: ['logical_and', 'logical_or', 'logical_xor', 'logical_not'],

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
    unary_math: ['sqrt', 'exp', 'log', 'sin', 'cos', 'abs', 'negative'],
    reductions: ['sum', 'prod', 'mean', 'std', 'variance', 'min', 'max', 'median'],
    comparison: ['equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal'],
    logical: ['logical_and', 'logical_or', 'logical_xor', 'logical_not'],
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

    console.log(
      `\nNumPy Conformity Coverage: ${String(allTested.length)}/${String(allRequired.length)} functions (${coverage.toFixed(1)}%)`
    );

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
      // Reductions
      'sum',
      'prod',
      'mean',
      'std',
      'variance',
      'median',
      'min',
      'max',
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
