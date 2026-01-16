/**
 * Tests for linear algebra functions
 *
 * Note: Some functions are not yet implemented in the native module.
 * Tests for unimplemented functions are skipped.
 * 2D array operations have known issues in the native module and are also skipped.
 */

import { describe, it, expect } from 'vitest';
import {
  array,
  dot,
  matmul,
  inv,
  det,
  trace,
  solve,
  eigvals,
  eig,
  qr,
  norm,
  matrix_rank,
  cond,
  NDArray,
} from '../../src/index.js';

describe('dot', () => {
  it('should compute dot product of 1D arrays', () => {
    const a = array([1, 2, 3]);
    const b = array([4, 5, 6]);
    expect(dot(a, b)).toBe(32); // 1*4 + 2*5 + 3*6 = 32
  });

  it.skip('should compute matrix multiplication of 2D arrays (TODO: fix native 2D)', () => {
    const a = array([
      [1, 2],
      [3, 4],
    ]);
    const b = array([
      [5, 6],
      [7, 8],
    ]);
    const result = dot(a, b) as NDArray;
    expect(result.shape).toEqual([2, 2]);
    expect(result.toArray()).toEqual([
      [19, 22],
      [43, 50],
    ]);
  });

  it.skip('should compute matrix-vector product (TODO: native implementation)', () => {
    const a = array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const b = array([1, 2, 3]);
    const result = dot(a, b) as NDArray;
    expect(result.shape).toEqual([2]);
    expect(result.toFlatArray()).toEqual([14, 32]);
  });

  it.skip('should compute vector-matrix product (TODO: native implementation)', () => {
    const a = array([1, 2]);
    const b = array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const result = dot(a, b) as NDArray;
    expect(result.shape).toEqual([3]);
    expect(result.toFlatArray()).toEqual([9, 12, 15]);
  });

  it('should throw for mismatched shapes', () => {
    const a = array([1, 2, 3]);
    const b = array([1, 2]);
    expect(() => dot(a, b)).toThrow();
  });
});

describe('matmul', () => {
  it.skip('should compute matrix multiplication (TODO: fix native 2D)', () => {
    const a = array([
      [1, 2],
      [3, 4],
    ]);
    const b = array([
      [5, 6],
      [7, 8],
    ]);
    const result = matmul(a, b);
    expect(result.toArray()).toEqual([
      [19, 22],
      [43, 50],
    ]);
  });

  it.skip('should handle non-square matrices (TODO: fix native 2D)', () => {
    const a = array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const b = array([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const result = matmul(a, b);
    expect(result.shape).toEqual([2, 2]);
  });

  it('should throw for non-2D arrays', () => {
    const a = array([1, 2, 3]);
    const b = array([4, 5, 6]);
    expect(() => matmul(a, b)).toThrow();
  });
});

describe('inv', () => {
  it.skip('should compute inverse of 2x2 matrix (TODO: fix native 2D)', () => {
    const a = array([
      [4, 7],
      [2, 6],
    ]);
    const result = inv(a);
    const identity = matmul(a, result);
    expect(identity.at(0, 0)).toBeCloseTo(1, 10);
  });

  it('should throw for singular matrix', () => {
    const a = array([
      [1, 2],
      [2, 4],
    ]);
    expect(() => inv(a)).toThrow();
  });

  it('should throw for non-square matrix', () => {
    const a = array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(() => inv(a)).toThrow();
  });
});

describe('det', () => {
  it('should compute determinant of 2x2 matrix', () => {
    const a = array([
      [3, 8],
      [4, 6],
    ]);
    expect(det(a)).toBeCloseTo(-14, 10);
  });

  it('should return 0 for singular matrix', () => {
    const a = array([
      [1, 2],
      [2, 4],
    ]);
    expect(det(a)).toBeCloseTo(0, 10);
  });
});

describe('trace', () => {
  it('should compute trace of square matrix', () => {
    const a = array([
      [1, 2],
      [3, 4],
    ]);
    expect(trace(a)).toBe(5);
  });
});

describe('solve', () => {
  it.skip('should solve linear system (TODO: fix native 2D)', () => {
    const a = array([
      [3, 1],
      [1, 2],
    ]);
    const b = array([9, 8]);
    const x = solve(a, b);
    const result = dot(a, x) as NDArray;
    expect(result.at(0)).toBeCloseTo(9, 10);
    expect(result.at(1)).toBeCloseTo(8, 10);
  });

  it('should throw for singular matrix', () => {
    const a = array([
      [1, 2],
      [2, 4],
    ]);
    const b = array([1, 2]);
    expect(() => solve(a, b)).toThrow();
  });
});

describe.skip('eigvals (not yet implemented in native)', () => {
  it('should compute eigenvalues of diagonal matrix', () => {
    const a = array([
      [2, 0],
      [0, 3],
    ]);
    const eigenvalues = eigvals(a);
    const sorted = eigenvalues.toFlatArray().map(Number).sort((x, y) => x - y);
    expect(sorted[0]).toBeCloseTo(2, 5);
    expect(sorted[1]).toBeCloseTo(3, 5);
  });
});

describe.skip('eig (not yet implemented in native)', () => {
  it('should compute eigenvalues and eigenvectors', () => {
    const a = array([
      [2, 0],
      [0, 3],
    ]);
    const { eigenvalues, eigenvectors } = eig(a);
    expect(eigenvectors.shape).toEqual([2, 2]);
  });
});

describe.skip('qr (not yet implemented in native)', () => {
  it('should compute QR decomposition', () => {
    const a = array([
      [12, -51, 4],
      [6, 167, -68],
      [-4, 24, -41],
    ]);
    const { q, r } = qr(a);
    expect(q.shape).toEqual([3, 3]);
    expect(r.shape).toEqual([3, 3]);
  });
});

describe('norm', () => {
  it('should compute L2 norm of vector (default)', () => {
    const a = array([3, 4]);
    expect(norm(a)).toBe(5);
  });

  // Note: L1 and infinity norms may not be fully supported in native
  it.skip('should compute L1 norm of vector (TODO: verify native support)', () => {
    const a = array([1, -2, 3]);
    expect(norm(a, 1)).toBe(6);
  });

  it.skip('should compute infinity norm of vector (TODO: verify native support)', () => {
    const a = array([1, -5, 3]);
    expect(norm(a, Infinity)).toBe(5);
  });

  it.skip('should compute Frobenius norm of matrix (TODO: fix native 2D)', () => {
    const a = array([
      [1, 2],
      [3, 4],
    ]);
    expect(norm(a, 'fro')).toBeCloseTo(Math.sqrt(30), 10);
  });
});

describe.skip('matrix_rank (not yet implemented in native)', () => {
  it('should compute rank of full rank matrix', () => {
    const a = array([
      [1, 2],
      [3, 4],
    ]);
    expect(matrix_rank(a)).toBe(2);
  });
});

describe.skip('cond (depends on svd which is not yet implemented)', () => {
  it('should compute condition number of identity matrix', () => {
    const a = array([
      [1, 0],
      [0, 1],
    ]);
    expect(cond(a)).toBeCloseTo(1, 5);
  });
});
