/**
 * Tests for linear algebra functions
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
} from '../../src/index.js';
import { NDArray } from '../../src/core/ndarray.js';

describe('dot', () => {
  it('should compute dot product of 1D arrays', () => {
    const a = array([1, 2, 3]);
    const b = array([4, 5, 6]);
    expect(dot(a, b)).toBe(32); // 1*4 + 2*5 + 3*6 = 32
  });

  it('should compute matrix multiplication of 2D arrays', () => {
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

  it('should compute matrix-vector product', () => {
    const a = array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const b = array([1, 2, 3]);
    const result = dot(a, b) as NDArray;
    expect(result.shape).toEqual([2]);
    expect(result.toFlatArray()).toEqual([14, 32]);
  });

  it('should compute vector-matrix product', () => {
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
  it('should compute matrix multiplication', () => {
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

  it('should handle non-square matrices', () => {
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
    expect(result.toArray()).toEqual([
      [22, 28],
      [49, 64],
    ]);
  });

  it('should throw for non-2D arrays', () => {
    const a = array([1, 2, 3]);
    const b = array([4, 5, 6]);
    expect(() => matmul(a, b)).toThrow();
  });
});

describe('inv', () => {
  it('should compute inverse of 2x2 matrix', () => {
    const a = array([
      [4, 7],
      [2, 6],
    ]);
    const result = inv(a);
    // Verify A * A^-1 = I
    const identity = matmul(a, result);
    expect(identity.at(0, 0)).toBeCloseTo(1, 10);
    expect(identity.at(0, 1)).toBeCloseTo(0, 10);
    expect(identity.at(1, 0)).toBeCloseTo(0, 10);
    expect(identity.at(1, 1)).toBeCloseTo(1, 10);
  });

  it('should compute inverse of 3x3 matrix', () => {
    const a = array([
      [1, 2, 3],
      [0, 1, 4],
      [5, 6, 0],
    ]);
    const result = inv(a);
    const identity = matmul(a, result);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(identity.at(i, j)).toBeCloseTo(i === j ? 1 : 0, 10);
      }
    }
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

  it('should compute determinant of 3x3 matrix', () => {
    const a = array([
      [6, 1, 1],
      [4, -2, 5],
      [2, 8, 7],
    ]);
    expect(det(a)).toBeCloseTo(-306, 8);
  });

  it('should return 0 for singular matrix', () => {
    const a = array([
      [1, 2],
      [2, 4],
    ]);
    expect(det(a)).toBeCloseTo(0, 10);
  });

  it('should compute determinant of identity matrix', () => {
    const a = array([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ]);
    expect(det(a)).toBeCloseTo(1, 10);
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

  it('should compute trace with offset', () => {
    const a = array([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);
    expect(trace(a, 0)).toBe(15); // 1 + 5 + 9
    expect(trace(a, 1)).toBe(8); // 2 + 6
    expect(trace(a, -1)).toBe(12); // 4 + 8
  });

  it('should work with non-square matrices', () => {
    const a = array([
      [1, 2, 3, 4],
      [5, 6, 7, 8],
    ]);
    expect(trace(a)).toBe(7); // 1 + 6
  });
});

describe('solve', () => {
  it('should solve linear system with 1D b', () => {
    const a = array([
      [3, 1],
      [1, 2],
    ]);
    const b = array([9, 8]);
    const x = solve(a, b);
    // Verify Ax = b
    const result = dot(a, x) as NDArray;
    expect(result.at(0)).toBeCloseTo(9, 10);
    expect(result.at(1)).toBeCloseTo(8, 10);
  });

  it('should solve linear system with 2D b', () => {
    const a = array([
      [1, 2],
      [3, 4],
    ]);
    const b = array([
      [5, 6],
      [7, 8],
    ]);
    const x = solve(a, b);
    const result = matmul(a, x);
    expect(result.at(0, 0)).toBeCloseTo(5, 10);
    expect(result.at(0, 1)).toBeCloseTo(6, 10);
    expect(result.at(1, 0)).toBeCloseTo(7, 10);
    expect(result.at(1, 1)).toBeCloseTo(8, 10);
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

describe('eigvals', () => {
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

  it('should compute eigenvalues of symmetric matrix', () => {
    const a = array([
      [4, -2],
      [-2, 1],
    ]);
    const eigenvalues = eigvals(a);
    // Eigenvalues of this matrix are 0 and 5
    const sorted = eigenvalues.toFlatArray().map(Number).sort((x, y) => x - y);
    expect(sorted[0]).toBeCloseTo(0, 5);
    expect(sorted[1]).toBeCloseTo(5, 5);
  });
});

describe('eig', () => {
  it('should compute eigenvalues and eigenvectors', () => {
    const a = array([
      [2, 0],
      [0, 3],
    ]);
    const { eigenvalues, eigenvectors } = eig(a);

    // Check eigenvalues
    const sortedEig = eigenvalues.toFlatArray().map(Number).sort((x, y) => x - y);
    expect(sortedEig[0]).toBeCloseTo(2, 5);
    expect(sortedEig[1]).toBeCloseTo(3, 5);

    // Eigenvectors should be orthonormal for symmetric matrix
    expect(eigenvectors.shape).toEqual([2, 2]);
  });
});

describe('qr', () => {
  it('should compute QR decomposition', () => {
    const a = array([
      [12, -51, 4],
      [6, 167, -68],
      [-4, 24, -41],
    ]);
    const { q, r } = qr(a);

    // Q should be orthogonal (Q^T * Q = I)
    const qtq = matmul(q.T as NDArray, q);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(qtq.at(i, j)).toBeCloseTo(i === j ? 1 : 0, 5);
      }
    }

    // R should be upper triangular
    for (let i = 1; i < 3; i++) {
      for (let j = 0; j < i; j++) {
        expect(r.at(i, j)).toBeCloseTo(0, 5);
      }
    }

    // Q * R should equal A
    const reconstructed = matmul(q, r);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(reconstructed.at(i, j)).toBeCloseTo(Number(a.at(i, j)), 5);
      }
    }
  });
});

describe('norm', () => {
  it('should compute L2 norm of vector (default)', () => {
    const a = array([3, 4]);
    expect(norm(a)).toBe(5);
  });

  it('should compute L1 norm of vector', () => {
    const a = array([1, -2, 3]);
    expect(norm(a, 1)).toBe(6);
  });

  it('should compute infinity norm of vector', () => {
    const a = array([1, -5, 3]);
    expect(norm(a, Infinity)).toBe(5);
  });

  it('should compute Frobenius norm of matrix', () => {
    const a = array([
      [1, 2],
      [3, 4],
    ]);
    expect(norm(a, 'fro')).toBeCloseTo(Math.sqrt(30), 10);
  });

  it('should compute 1-norm of matrix (max column sum)', () => {
    const a = array([
      [1, -2],
      [-3, 4],
    ]);
    expect(norm(a, 1)).toBe(6); // max(|1|+|-3|, |-2|+|4|) = max(4, 6) = 6
  });
});

describe('matrix_rank', () => {
  it('should compute rank of full rank matrix', () => {
    const a = array([
      [1, 2],
      [3, 4],
    ]);
    expect(matrix_rank(a)).toBe(2);
  });

  it('should compute rank of rank-deficient matrix', () => {
    const a = array([
      [1, 2, 3],
      [2, 4, 6],
      [1, 1, 1],
    ]);
    expect(matrix_rank(a)).toBe(2);
  });

  it('should compute rank of zero matrix', () => {
    const a = array([
      [0, 0],
      [0, 0],
    ]);
    expect(matrix_rank(a)).toBe(0);
  });
});

describe('cond', () => {
  it('should compute condition number of identity matrix', () => {
    const a = array([
      [1, 0],
      [0, 1],
    ]);
    expect(cond(a)).toBeCloseTo(1, 5);
  });

  it('should compute condition number of well-conditioned matrix', () => {
    const a = array([
      [1, 0],
      [0, 2],
    ]);
    const c = cond(a);
    expect(c).toBeCloseTo(2, 5);
  });

  it('should return high value for ill-conditioned matrix', () => {
    const a = array([
      [1, 1],
      [1, 1.0001],
    ]);
    const c = cond(a);
    expect(c).toBeGreaterThan(1000);
  });
});
