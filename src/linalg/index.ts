/**
 * Linear algebra functions for numpy-ts
 * Provides common linear algebra operations
 */

import { NDArray, type DTypeName, type Shape } from '../core/index.js';

/**
 * Compute the dot product of two arrays.
 * For 2-D arrays it is equivalent to matrix multiplication.
 * For 1-D arrays it is the inner product.
 */
export function dot<D1 extends DTypeName, D2 extends DTypeName>(
  a: NDArray<D1>,
  b: NDArray<D2>
): NDArray<'float64'> | number {
  // 1D dot product (inner product)
  if (a.ndim === 1 && b.ndim === 1) {
    if (a.size !== b.size) {
      throw new Error(`Shapes (${a.size},) and (${b.size},) not aligned`);
    }
    let sum = 0;
    for (let i = 0; i < a.size; i++) {
      sum += Number(a.at(i)) * Number(b.at(i));
    }
    return sum;
  }

  // 2D matrix multiplication
  if (a.ndim === 2 && b.ndim === 2) {
    const [m, k1] = a.shape;
    const [k2, n] = b.shape;
    if (k1 !== k2) {
      throw new Error(
        `Shapes (${m},${k1}) and (${k2},${n}) not aligned: ${k1} (dim 1) != ${k2} (dim 0)`
      );
    }
    return matmul2d(a, b);
  }

  // 2D @ 1D: Matrix-vector product
  if (a.ndim === 2 && b.ndim === 1) {
    const [m, k1] = a.shape;
    const k2 = b.size;
    if (k1 !== k2) {
      throw new Error(
        `Shapes (${m},${k1}) and (${k2},) not aligned: ${k1} (dim 1) != ${k2} (dim 0)`
      );
    }
    const result = NDArray.zeros<'float64'>([m], { dtype: 'float64' });
    for (let i = 0; i < m; i++) {
      let sum = 0;
      for (let j = 0; j < k1!; j++) {
        sum += Number(a.at(i, j)) * Number(b.at(j));
      }
      result.setFlat(i, sum);
    }
    return result;
  }

  // 1D @ 2D: Vector-matrix product
  if (a.ndim === 1 && b.ndim === 2) {
    const k1 = a.size;
    const [k2, n] = b.shape;
    if (k1 !== k2) {
      throw new Error(
        `Shapes (${k1},) and (${k2},${n}) not aligned: ${k1} (dim 0) != ${k2} (dim 0)`
      );
    }
    const result = NDArray.zeros<'float64'>([n!], { dtype: 'float64' });
    for (let j = 0; j < n!; j++) {
      let sum = 0;
      for (let i = 0; i < k1; i++) {
        sum += Number(a.at(i)) * Number(b.at(i, j));
      }
      result.setFlat(j, sum);
    }
    return result;
  }

  throw new Error(
    `dot not supported for arrays with ${a.ndim} and ${b.ndim} dimensions`
  );
}

/**
 * Matrix multiplication of two 2D arrays
 */
export function matmul<D1 extends DTypeName, D2 extends DTypeName>(
  a: NDArray<D1>,
  b: NDArray<D2>
): NDArray<'float64'> {
  if (a.ndim !== 2 || b.ndim !== 2) {
    throw new Error('matmul requires 2-D arrays');
  }
  return matmul2d(a, b);
}

/**
 * Internal 2D matrix multiplication
 */
function matmul2d<D1 extends DTypeName, D2 extends DTypeName>(
  a: NDArray<D1>,
  b: NDArray<D2>
): NDArray<'float64'> {
  const [m, k] = a.shape;
  const [_, n] = b.shape;

  const result = NDArray.zeros<'float64'>([m, n!], { dtype: 'float64' });

  // Simple O(n³) algorithm - good for small matrices
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n!; j++) {
      let sum = 0;
      for (let l = 0; l < k!; l++) {
        sum += Number(a.at(i, l)) * Number(b.at(l, j));
      }
      result.set([i, j], sum);
    }
  }

  return result;
}

/**
 * Compute the (multiplicative) inverse of a matrix
 */
export function inv<D extends DTypeName>(a: NDArray<D>): NDArray<'float64'> {
  if (a.ndim !== 2) {
    throw new Error('inv requires a 2-D array');
  }
  const [n, m] = a.shape;
  if (n !== m) {
    throw new Error('inv requires a square matrix');
  }

  // Use Gauss-Jordan elimination
  const size = n;
  const augmented = NDArray.zeros<'float64'>([size, size * 2], { dtype: 'float64' });

  // Copy a to left half and identity to right half
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      augmented.set([i, j], Number(a.at(i, j)));
    }
    augmented.set([i, i + size], 1);
  }

  // Forward elimination with partial pivoting
  for (let col = 0; col < size; col++) {
    // Find pivot
    let maxRow = col;
    let maxVal = Math.abs(Number(augmented.at(col, col)));
    for (let row = col + 1; row < size; row++) {
      const val = Math.abs(Number(augmented.at(row, col)));
      if (val > maxVal) {
        maxVal = val;
        maxRow = row;
      }
    }

    if (maxVal < 1e-10) {
      throw new Error('Matrix is singular');
    }

    // Swap rows
    if (maxRow !== col) {
      for (let j = 0; j < size * 2; j++) {
        const tmp = Number(augmented.at(col, j));
        augmented.set([col, j], Number(augmented.at(maxRow, j)));
        augmented.set([maxRow, j], tmp);
      }
    }

    // Scale pivot row
    const pivot = Number(augmented.at(col, col));
    for (let j = 0; j < size * 2; j++) {
      augmented.set([col, j], Number(augmented.at(col, j)) / pivot);
    }

    // Eliminate column
    for (let row = 0; row < size; row++) {
      if (row !== col) {
        const factor = Number(augmented.at(row, col));
        for (let j = 0; j < size * 2; j++) {
          augmented.set(
            [row, j],
            Number(augmented.at(row, j)) - factor * Number(augmented.at(col, j))
          );
        }
      }
    }
  }

  // Extract inverse from right half
  const result = NDArray.zeros<'float64'>([size, size], { dtype: 'float64' });
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      result.set([i, j], Number(augmented.at(i, j + size)));
    }
  }

  return result;
}

/**
 * Compute the determinant of a matrix
 */
export function det<D extends DTypeName>(a: NDArray<D>): number {
  if (a.ndim !== 2) {
    throw new Error('det requires a 2-D array');
  }
  const [n, m] = a.shape;
  if (n !== m) {
    throw new Error('det requires a square matrix');
  }

  const size = n;

  // Use LU decomposition approach
  const work = NDArray.zeros<'float64'>([size, size], { dtype: 'float64' });
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      work.set([i, j], Number(a.at(i, j)));
    }
  }

  let determinant = 1.0;

  for (let col = 0; col < size; col++) {
    // Find pivot
    let maxRow = col;
    let maxVal = Math.abs(Number(work.at(col, col)));
    for (let row = col + 1; row < size; row++) {
      const val = Math.abs(Number(work.at(row, col)));
      if (val > maxVal) {
        maxVal = val;
        maxRow = row;
      }
    }

    if (maxVal < 1e-10) {
      return 0; // Singular matrix
    }

    // Swap rows
    if (maxRow !== col) {
      for (let j = 0; j < size; j++) {
        const tmp = Number(work.at(col, j));
        work.set([col, j], Number(work.at(maxRow, j)));
        work.set([maxRow, j], tmp);
      }
      determinant *= -1; // Row swap changes sign
    }

    determinant *= Number(work.at(col, col));

    // Eliminate column
    for (let row = col + 1; row < size; row++) {
      const factor = Number(work.at(row, col)) / Number(work.at(col, col));
      for (let j = col; j < size; j++) {
        work.set(
          [row, j],
          Number(work.at(row, j)) - factor * Number(work.at(col, j))
        );
      }
    }
  }

  return determinant;
}

/**
 * Return the sum along diagonals of the array
 */
export function trace<D extends DTypeName>(
  a: NDArray<D>,
  offset: number = 0
): number {
  if (a.ndim !== 2) {
    throw new Error('trace requires a 2-D array');
  }
  const [n, m] = a.shape;

  let sum = 0;
  const startRow = offset >= 0 ? 0 : -offset;
  const startCol = offset >= 0 ? offset : 0;

  for (let i = 0; startRow + i < n && startCol + i < m!; i++) {
    sum += Number(a.at(startRow + i, startCol + i));
  }

  return sum;
}

/**
 * Solve a linear matrix equation Ax = b
 */
export function solve<D1 extends DTypeName, D2 extends DTypeName>(
  a: NDArray<D1>,
  b: NDArray<D2>
): NDArray<'float64'> {
  if (a.ndim !== 2) {
    throw new Error('solve requires a 2-D coefficient matrix');
  }
  const [n, m] = a.shape;
  if (n !== m) {
    throw new Error('solve requires a square coefficient matrix');
  }

  const size = n;

  // Handle 1D b
  if (b.ndim === 1) {
    if (b.size !== size) {
      throw new Error(`Shapes (${size},${size}) and (${b.size},) not aligned`);
    }

    // Augmented matrix [A|b]
    const augmented = NDArray.zeros<'float64'>([size, size + 1], { dtype: 'float64' });
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        augmented.set([i, j], Number(a.at(i, j)));
      }
      augmented.set([i, size], Number(b.at(i)));
    }

    // Gauss elimination with partial pivoting
    for (let col = 0; col < size; col++) {
      let maxRow = col;
      let maxVal = Math.abs(Number(augmented.at(col, col)));
      for (let row = col + 1; row < size; row++) {
        const val = Math.abs(Number(augmented.at(row, col)));
        if (val > maxVal) {
          maxVal = val;
          maxRow = row;
        }
      }

      if (maxVal < 1e-10) {
        throw new Error('Singular matrix');
      }

      if (maxRow !== col) {
        for (let j = 0; j <= size; j++) {
          const tmp = Number(augmented.at(col, j));
          augmented.set([col, j], Number(augmented.at(maxRow, j)));
          augmented.set([maxRow, j], tmp);
        }
      }

      for (let row = col + 1; row < size; row++) {
        const factor = Number(augmented.at(row, col)) / Number(augmented.at(col, col));
        for (let j = col; j <= size; j++) {
          augmented.set(
            [row, j],
            Number(augmented.at(row, j)) - factor * Number(augmented.at(col, j))
          );
        }
      }
    }

    // Back substitution
    const x = NDArray.zeros<'float64'>([size], { dtype: 'float64' });
    for (let i = size - 1; i >= 0; i--) {
      let sum = Number(augmented.at(i, size));
      for (let j = i + 1; j < size; j++) {
        sum -= Number(augmented.at(i, j)) * Number(x.at(j));
      }
      x.setFlat(i, sum / Number(augmented.at(i, i)));
    }

    return x;
  }

  // Handle 2D b
  if (b.ndim === 2) {
    const [bRows, bCols] = b.shape;
    if (bRows !== size) {
      throw new Error(`Shapes (${size},${size}) and (${bRows},${bCols}) not aligned`);
    }

    const result = NDArray.zeros<'float64'>([size, bCols!], { dtype: 'float64' });

    // Solve for each column of b
    for (let col = 0; col < bCols!; col++) {
      const bCol = NDArray.zeros<'float64'>([size], { dtype: 'float64' });
      for (let i = 0; i < size; i++) {
        bCol.setFlat(i, Number(b.at(i, col)));
      }
      const xCol = solve(a, bCol);
      for (let i = 0; i < size; i++) {
        result.set([i, col], Number(xCol.at(i)));
      }
    }

    return result;
  }

  throw new Error('b must be 1-D or 2-D');
}

/**
 * Compute the eigenvalues of a square matrix.
 * Uses QR algorithm for real symmetric matrices.
 * For non-symmetric matrices, returns approximate eigenvalues.
 */
export function eigvals<D extends DTypeName>(a: NDArray<D>): NDArray<'float64'> {
  if (a.ndim !== 2) {
    throw new Error('eigvals requires a 2-D array');
  }
  const [n, m] = a.shape;
  if (n !== m) {
    throw new Error('eigvals requires a square matrix');
  }

  const size = n;

  // Copy matrix
  let work = NDArray.zeros<'float64'>([size, size], { dtype: 'float64' });
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      work.set([i, j], Number(a.at(i, j)));
    }
  }

  // QR iteration
  const maxIter = 100 * size;
  for (let iter = 0; iter < maxIter; iter++) {
    const { q, r } = qr(work);
    work = matmul(r, q);

    // Check for convergence (off-diagonal elements small)
    let offDiag = 0;
    for (let i = 1; i < size; i++) {
      for (let j = 0; j < i; j++) {
        offDiag += Math.abs(Number(work.at(i, j)));
      }
    }
    if (offDiag < 1e-10 * size * size) {
      break;
    }
  }

  // Extract eigenvalues from diagonal
  const eigenvalues = NDArray.zeros<'float64'>([size], { dtype: 'float64' });
  for (let i = 0; i < size; i++) {
    eigenvalues.setFlat(i, Number(work.at(i, i)));
  }

  return eigenvalues;
}

/**
 * Compute the eigenvalues and eigenvectors of a square matrix.
 */
export function eig<D extends DTypeName>(
  a: NDArray<D>
): { eigenvalues: NDArray<'float64'>; eigenvectors: NDArray<'float64'> } {
  if (a.ndim !== 2) {
    throw new Error('eig requires a 2-D array');
  }
  const [n, m] = a.shape;
  if (n !== m) {
    throw new Error('eig requires a square matrix');
  }

  const size = n;

  // Copy matrix
  let work = NDArray.zeros<'float64'>([size, size], { dtype: 'float64' });
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      work.set([i, j], Number(a.at(i, j)));
    }
  }

  // Accumulate transformations
  let v = eye(size);

  // QR iteration
  const maxIter = 100 * size;
  for (let iter = 0; iter < maxIter; iter++) {
    const { q, r } = qr(work);
    work = matmul(r, q);
    v = matmul(v, q);

    let offDiag = 0;
    for (let i = 1; i < size; i++) {
      for (let j = 0; j < i; j++) {
        offDiag += Math.abs(Number(work.at(i, j)));
      }
    }
    if (offDiag < 1e-10 * size * size) {
      break;
    }
  }

  const eigenvalues = NDArray.zeros<'float64'>([size], { dtype: 'float64' });
  for (let i = 0; i < size; i++) {
    eigenvalues.setFlat(i, Number(work.at(i, i)));
  }

  return { eigenvalues, eigenvectors: v };
}

/**
 * QR decomposition using Gram-Schmidt
 */
export function qr<D extends DTypeName>(
  a: NDArray<D>
): { q: NDArray<'float64'>; r: NDArray<'float64'> } {
  if (a.ndim !== 2) {
    throw new Error('qr requires a 2-D array');
  }

  const [m, n] = a.shape;
  const q = NDArray.zeros<'float64'>([m, n!], { dtype: 'float64' });
  const r = NDArray.zeros<'float64'>([n!, n!], { dtype: 'float64' });

  for (let j = 0; j < n!; j++) {
    // Copy column j of a to v
    const v: number[] = [];
    for (let i = 0; i < m; i++) {
      v.push(Number(a.at(i, j)));
    }

    // Subtract projections onto previous q columns
    for (let i = 0; i < j; i++) {
      let dotProduct = 0;
      for (let k = 0; k < m; k++) {
        dotProduct += Number(q.at(k, i)) * v[k]!;
      }
      r.set([i, j], dotProduct);
      for (let k = 0; k < m; k++) {
        v[k]! -= dotProduct * Number(q.at(k, i));
      }
    }

    // Normalize
    let norm = 0;
    for (let k = 0; k < m; k++) {
      norm += v[k]! * v[k]!;
    }
    norm = Math.sqrt(norm);

    r.set([j, j], norm);
    if (norm > 1e-10) {
      for (let k = 0; k < m; k++) {
        q.set([k, j], v[k]! / norm);
      }
    }
  }

  return { q, r };
}

/**
 * Compute the matrix norm
 */
export function norm<D extends DTypeName>(
  a: NDArray<D>,
  ord?: number | 'fro'
): number {
  if (a.ndim === 1) {
    // Vector norm
    if (ord === undefined || ord === 2) {
      let sum = 0;
      for (const val of a) {
        sum += Number(val) * Number(val);
      }
      return Math.sqrt(sum);
    }
    if (ord === 1) {
      let sum = 0;
      for (const val of a) {
        sum += Math.abs(Number(val));
      }
      return sum;
    }
    if (ord === Infinity) {
      let max = 0;
      for (const val of a) {
        max = Math.max(max, Math.abs(Number(val)));
      }
      return max;
    }
    if (typeof ord === 'number') {
      let sum = 0;
      for (const val of a) {
        sum += Math.pow(Math.abs(Number(val)), ord);
      }
      return Math.pow(sum, 1 / ord);
    }
  }

  if (a.ndim === 2) {
    if (ord === 'fro' || ord === undefined) {
      // Frobenius norm
      let sum = 0;
      for (const val of a) {
        sum += Number(val) * Number(val);
      }
      return Math.sqrt(sum);
    }
    if (ord === 1) {
      // Maximum column sum
      const [_, n] = a.shape;
      let max = 0;
      for (let j = 0; j < n!; j++) {
        let sum = 0;
        for (let i = 0; i < a.shape[0]!; i++) {
          sum += Math.abs(Number(a.at(i, j)));
        }
        max = Math.max(max, sum);
      }
      return max;
    }
    if (ord === Infinity) {
      // Maximum row sum
      const [m, _] = a.shape;
      let max = 0;
      for (let i = 0; i < m; i++) {
        let sum = 0;
        for (let j = 0; j < a.shape[1]!; j++) {
          sum += Math.abs(Number(a.at(i, j)));
        }
        max = Math.max(max, sum);
      }
      return max;
    }
  }

  throw new Error(`Unsupported norm order: ${ord}`);
}

/**
 * Create identity matrix
 */
function eye(n: number): NDArray<'float64'> {
  const result = NDArray.zeros<'float64'>([n, n], { dtype: 'float64' });
  for (let i = 0; i < n; i++) {
    result.set([i, i], 1);
  }
  return result;
}

/**
 * Compute the rank of a matrix
 */
export function matrix_rank<D extends DTypeName>(
  a: NDArray<D>,
  tol?: number
): number {
  if (a.ndim !== 2) {
    throw new Error('matrix_rank requires a 2-D array');
  }

  const { r } = qr(a);
  const minDim = Math.min(a.shape[0]!, a.shape[1]!);

  // Count non-zero diagonal elements
  const tolerance = tol ?? 1e-10 * Math.max(a.shape[0]!, a.shape[1]!);
  let rank = 0;
  for (let i = 0; i < minDim; i++) {
    if (Math.abs(Number(r.at(i, i))) > tolerance) {
      rank++;
    }
  }

  return rank;
}

/**
 * Compute the condition number of a matrix
 */
export function cond<D extends DTypeName>(a: NDArray<D>): number {
  if (a.ndim !== 2) {
    throw new Error('cond requires a 2-D array');
  }

  // For a simple implementation, use the ratio of largest to smallest eigenvalue
  // of A^T A (which gives singular values squared)
  const ata = matmul(a.T as NDArray<D>, a);
  const eigenvalues = eigvals(ata);

  let maxEig = -Infinity;
  let minEig = Infinity;
  for (const val of eigenvalues) {
    const v = Math.abs(Number(val));
    if (v > maxEig) maxEig = v;
    if (v > 1e-10 && v < minEig) minEig = v;
  }

  if (minEig === Infinity || minEig < 1e-10) {
    return Infinity;
  }

  return Math.sqrt(maxEig / minEig);
}
