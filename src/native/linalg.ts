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
 * Matrix multiplication with B transposed: A @ B.T
 * Avoids explicit transpose for better performance.
 * Common in attention mechanisms: Q @ K.T
 */
export function matmul_nt(a: NDArray, b: NDArray): NDArray {
  return new NDArray(linalg.matmul_nt(a._native, b._native));
}

/**
 * Batch matrix multiplication - performs multiple matmuls in a single native call.
 * Reduces N-API overhead for batched operations.
 * @param as Array of 2D matrices (left operands)
 * @param bs Array of 2D matrices (right operands)
 * @returns Array of result matrices
 */
export function batch_matmul(as: NDArray[], bs: NDArray[]): NDArray[] {
  const nativeResults = linalg.batch_matmul(
    as.map((a) => a._native),
    bs.map((b) => b._native)
  );
  return nativeResults.map((r) => new NDArray(r));
}

/**
 * Batch matrix multiplication with stacked 3D arrays.
 * Much more efficient than batch_matmul as it only creates one output array.
 *
 * @param a 3D array with shape [batch, m, k]
 * @param b 3D array with shape [batch, k, n]
 * @returns 3D array with shape [batch, m, n]
 */
export function batch_matmul_stacked(a: NDArray, b: NDArray): NDArray {
  return new NDArray(linalg.batch_matmul_stacked(a._native, b._native));
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
 * Least squares solution: min ||b - A*x||
 * Much faster than computing (A'A)^-1 * A'b manually.
 * Uses LAPACK dgels which is optimized for this.
 */
export function lstsq(a: NDArray, b: NDArray): NDArray {
  return new NDArray(linalg.lstsq(a._native, b._native));
}

/**
 * Solve normal equations: beta = (X'X)^(-1) X'y
 * Fused operation that's faster than computing X.T, matmul, and solve separately.
 */
export function normal_equations(X: NDArray, y: NDArray): NDArray {
  return new NDArray(linalg.normal_equations(X._native, y._native));
}

/**
 * Gram matrix: X'X
 * Computes the Gram matrix using optimized dsyrk BLAS operation.
 * For X with shape (m, n), returns (n, n) symmetric matrix.
 * Much faster than matmul(X.T, X) for tall matrices.
 */
export function gram(X: NDArray): NDArray {
  return new NDArray(linalg.gram(X._native));
}

/**
 * X'y - transposed matrix-vector multiplication.
 * Computes X.T @ y using optimized dgemv/dgemm BLAS operations.
 * Much faster than matmul(X.T, y) as it avoids explicit transpose.
 */
export function xty(X: NDArray, y: NDArray): NDArray {
  return new NDArray(linalg.xty(X._native, y._native));
}

/**
 * Condition number
 */
export function cond(a: NDArray): number {
  // Compute using SVD: cond = max(s) / min(s)
  const { s } = svd(a);
  const singularValues = s.toFlatArray();
  const maxS = Math.max(...singularValues);
  const minS = Math.min(...singularValues.filter((v) => v > 1e-10));
  return minS > 0 ? maxS / minS : Infinity;
}
