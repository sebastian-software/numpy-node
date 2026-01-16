/**
 * Linear algebra functions - thin wrapper around native linalg module
 */

import { native } from './loader.js';
import { NDArray } from './ndarray.js';

const { linalg } = native;

/**
 * Matrix multiplication
 */
export function matmul(a: NDArray, b: NDArray): NDArray {
  return new NDArray(linalg.matmul(a._native, b._native));
}

/**
 * Dot product / matrix multiplication
 */
export function dot(a: NDArray, b: NDArray): NDArray | number {
  const result = linalg.dot(a._native, b._native);
  if (typeof result === 'number') {
    return result;
  }
  return new NDArray(result);
}

/**
 * Matrix inverse
 */
export function inv(a: NDArray): NDArray {
  return new NDArray(linalg.inv(a._native));
}

/**
 * Matrix determinant
 */
export function det(a: NDArray): number {
  return linalg.det(a._native);
}

/**
 * Solve linear system Ax = b
 */
export function solve(a: NDArray, b: NDArray): NDArray {
  return new NDArray(linalg.solve(a._native, b._native));
}

/**
 * Eigenvalues and eigenvectors
 */
export function eig(a: NDArray): { eigenvalues: NDArray; eigenvectors: NDArray } {
  const result = linalg.eig(a._native);
  return {
    eigenvalues: new NDArray(result.eigenvalues),
    eigenvectors: new NDArray(result.eigenvectors),
  };
}

/**
 * Eigenvalues only
 */
export function eigvals(a: NDArray): NDArray {
  return new NDArray(linalg.eigvals(a._native));
}

/**
 * Singular value decomposition
 */
export function svd(a: NDArray): { u: NDArray; s: NDArray; vh: NDArray } {
  const result = linalg.svd(a._native);
  return {
    u: new NDArray(result.u),
    s: new NDArray(result.s),
    vh: new NDArray(result.vh),
  };
}

/**
 * QR decomposition
 */
export function qr(a: NDArray): { q: NDArray; r: NDArray } {
  const result = linalg.qr(a._native);
  return {
    q: new NDArray(result.q),
    r: new NDArray(result.r),
  };
}

/**
 * Cholesky decomposition
 */
export function cholesky(a: NDArray): NDArray {
  return new NDArray(linalg.cholesky(a._native));
}

/**
 * Matrix/vector norm
 */
export function norm(a: NDArray, ord?: number | 'fro'): number {
  // Convert JavaScript Infinity to string 'inf' for native handling
  const normOrd = ord === Infinity ? 'inf' : ord;
  return linalg.norm(a._native, normOrd);
}

/**
 * Matrix rank
 */
export function matrix_rank(a: NDArray): number {
  return linalg.matrix_rank(a._native);
}

/**
 * Matrix trace
 */
export function trace(a: NDArray): number {
  return linalg.trace(a._native);
}

/**
 * Condition number
 */
export function cond(a: NDArray): number {
  // Compute using SVD: cond = max(s) / min(s)
  const { s } = svd(a);
  const singularValues = s.toFlatArray();
  const maxS = Math.max(...singularValues);
  const minS = Math.min(...singularValues.filter(v => v > 1e-10));
  return minS > 0 ? maxS / minS : Infinity;
}
