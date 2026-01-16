/**
 * Array creation routines for numpy-node
 * Provides NumPy-compatible functions for creating arrays
 */

import { NDArray, type Shape, type DTypeName, type ArrayInput } from '../core/index.js';

/**
 * Create an array from nested JavaScript arrays or TypedArrays
 */
export function array<D extends DTypeName = 'float64'>(
  data: ArrayInput,
  options: { dtype?: D } = {}
): NDArray<D> {
  return NDArray.from(data, options);
}

/**
 * Create an array filled with zeros
 */
export function zeros<D extends DTypeName = 'float64'>(
  shape: Shape | number,
  options: { dtype?: D } = {}
): NDArray<D> {
  const normalizedShape = typeof shape === 'number' ? [shape] : shape;
  return NDArray.zeros(normalizedShape, options);
}

/**
 * Create an array of zeros with the same shape and dtype as a given array
 */
export function zerosLike<D extends DTypeName = 'float64'>(
  a: NDArray,
  options: { dtype?: D } = {}
): NDArray<D> {
  const dtype = options.dtype ?? (a.dtype as D);
  return NDArray.zeros(a.shape, { dtype });
}

/**
 * Create an array filled with ones
 */
export function ones<D extends DTypeName = 'float64'>(
  shape: Shape | number,
  options: { dtype?: D } = {}
): NDArray<D> {
  const normalizedShape = typeof shape === 'number' ? [shape] : shape;
  return NDArray.ones(normalizedShape, options);
}

/**
 * Create an array of ones with the same shape and dtype as a given array
 */
export function onesLike<D extends DTypeName = 'float64'>(
  a: NDArray,
  options: { dtype?: D } = {}
): NDArray<D> {
  const dtype = options.dtype ?? (a.dtype as D);
  return NDArray.ones(a.shape, { dtype });
}

/**
 * Create an array filled with a given value
 */
export function full<D extends DTypeName = 'float64'>(
  shape: Shape | number,
  fillValue: number | bigint,
  options: { dtype?: D } = {}
): NDArray<D> {
  const normalizedShape = typeof shape === 'number' ? [shape] : shape;
  return NDArray.full(normalizedShape, fillValue, options);
}

/**
 * Create a full array with the same shape and dtype as a given array
 */
export function fullLike<D extends DTypeName = 'float64'>(
  a: NDArray,
  fillValue: number | bigint,
  options: { dtype?: D } = {}
): NDArray<D> {
  const dtype = options.dtype ?? (a.dtype as D);
  return NDArray.full(a.shape, fillValue, { dtype });
}

/**
 * Create an uninitialized array
 */
export function empty<D extends DTypeName = 'float64'>(
  shape: Shape | number,
  options: { dtype?: D } = {}
): NDArray<D> {
  const normalizedShape = typeof shape === 'number' ? [shape] : shape;
  return NDArray.empty(normalizedShape, options);
}

/**
 * Create an empty array with the same shape and dtype as a given array
 */
export function emptyLike<D extends DTypeName = 'float64'>(
  a: NDArray,
  options: { dtype?: D } = {}
): NDArray<D> {
  const dtype = options.dtype ?? (a.dtype as D);
  return NDArray.empty(a.shape, { dtype });
}

/**
 * Create an array with evenly spaced values within a given interval
 */
export function arange<D extends DTypeName = 'float64'>(
  startOrStop: number,
  stop?: number,
  step?: number,
  options: { dtype?: D } = {}
): NDArray<D> {
  return NDArray.arange(startOrStop, stop, step ?? 1, options);
}

/**
 * Create an array with evenly spaced values over a specified interval
 */
export function linspace<D extends DTypeName = 'float64'>(
  start: number,
  stop: number,
  num: number = 50,
  options: { dtype?: D; endpoint?: boolean } = {}
): NDArray<D> {
  return NDArray.linspace(start, stop, num, options);
}

/**
 * Create an array with values spaced evenly on a log scale
 */
export function logspace<D extends DTypeName = 'float64'>(
  start: number,
  stop: number,
  num: number = 50,
  options: { dtype?: D; endpoint?: boolean; base?: number } = {}
): NDArray<D> {
  const base = options.base ?? 10;
  const endpoint = options.endpoint ?? true;
  const linear = linspace(start, stop, num, { endpoint });
  const result = NDArray.zeros<D>([num], { dtype: options.dtype ?? ('float64' as D) });

  let i = 0;
  for (const val of linear) {
    result.setFlat(i++, Math.pow(base, Number(val)));
  }

  return result;
}

/**
 * Create an array with values spaced evenly on a log scale (in linear space)
 */
export function geomspace<D extends DTypeName = 'float64'>(
  start: number,
  stop: number,
  num: number = 50,
  options: { dtype?: D; endpoint?: boolean } = {}
): NDArray<D> {
  if (start === 0 || stop === 0) {
    throw new Error('Geometric sequence cannot include zero');
  }
  if ((start < 0) !== (stop < 0)) {
    throw new Error('Geometric sequence endpoints must have the same sign');
  }

  const logStart = Math.log10(Math.abs(start));
  const logStop = Math.log10(Math.abs(stop));
  const result = logspace(logStart, logStop, num, { ...options, base: 10 });

  if (start < 0) {
    for (let i = 0; i < result.size; i++) {
      result.setFlat(i, -Number(result.getFlat(i)));
    }
  }

  return result;
}

/**
 * Create an identity matrix
 */
export function eye<D extends DTypeName = 'float64'>(
  n: number,
  m?: number,
  k: number = 0,
  options: { dtype?: D } = {}
): NDArray<D> {
  return NDArray.eye(n, m, k, options);
}

/**
 * Create a 2D array with ones on the diagonal and zeros elsewhere
 */
export function identity<D extends DTypeName = 'float64'>(
  n: number,
  options: { dtype?: D } = {}
): NDArray<D> {
  return eye(n, n, 0, options);
}

/**
 * Extract a diagonal or construct a diagonal array
 */
export function diag<D extends DTypeName = 'float64'>(
  v: NDArray<D>,
  k: number = 0
): NDArray<D> {
  if (v.ndim === 1) {
    // Construct diagonal matrix from 1D array
    const n = v.size + Math.abs(k);
    const result = NDArray.zeros<D>([n, n], { dtype: v.dtype as D });

    for (let i = 0; i < v.size; i++) {
      const row = k >= 0 ? i : i - k;
      const col = k >= 0 ? i + k : i;
      result.set([row, col], v.at(i) as number);
    }

    return result;
  } else if (v.ndim === 2) {
    // Extract diagonal from 2D array
    const [rows, cols] = v.shape;
    if (rows === undefined || cols === undefined) {
      throw new Error('Invalid 2D array shape');
    }

    const diagLength = k >= 0
      ? Math.min(rows, cols - k)
      : Math.min(rows + k, cols);

    if (diagLength <= 0) {
      return NDArray.zeros<D>([0], { dtype: v.dtype as D });
    }

    const result = NDArray.zeros<D>([diagLength], { dtype: v.dtype as D });

    for (let i = 0; i < diagLength; i++) {
      const row = k >= 0 ? i : i - k;
      const col = k >= 0 ? i + k : i;
      result.setFlat(i, v.at(row, col) as number);
    }

    return result;
  } else {
    throw new Error('Input must be 1-D or 2-D');
  }
}

/**
 * Return a 2D array with ones at and below the given diagonal
 */
export function tri<D extends DTypeName = 'float64'>(
  n: number,
  m?: number,
  k: number = 0,
  options: { dtype?: D } = {}
): NDArray<D> {
  const cols = m ?? n;
  const result = NDArray.zeros<D>([n, cols], options);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= Math.min(cols - 1, i + k); j++) {
      result.set([i, j], 1);
    }
  }

  return result;
}

/**
 * Lower triangle of an array
 */
export function tril<D extends DTypeName>(v: NDArray<D>, k: number = 0): NDArray<D> {
  if (v.ndim !== 2) {
    throw new Error('Input must be 2-D');
  }

  const [rows, cols] = v.shape;
  if (rows === undefined || cols === undefined) {
    throw new Error('Invalid 2D array shape');
  }

  const result = v.copy();

  for (let i = 0; i < rows; i++) {
    for (let j = i + k + 1; j < cols; j++) {
      result.set([i, j], 0);
    }
  }

  return result;
}

/**
 * Upper triangle of an array
 */
export function triu<D extends DTypeName>(v: NDArray<D>, k: number = 0): NDArray<D> {
  if (v.ndim !== 2) {
    throw new Error('Input must be 2-D');
  }

  const [rows, cols] = v.shape;
  if (rows === undefined || cols === undefined) {
    throw new Error('Invalid 2D array shape');
  }

  const result = v.copy();

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < Math.min(cols, i + k); j++) {
      result.set([i, j], 0);
    }
  }

  return result;
}

/**
 * Create a Vandermonde matrix
 */
export function vander<D extends DTypeName = 'float64'>(
  x: NDArray,
  n?: number,
  increasing: boolean = false
): NDArray<D> {
  if (x.ndim !== 1) {
    throw new Error('Input must be 1-D');
  }

  const numRows = x.size;
  const numCols = n ?? numRows;
  const result = NDArray.zeros<D>([numRows, numCols], { dtype: 'float64' as D });

  for (let i = 0; i < numRows; i++) {
    const val = Number(x.at(i));
    for (let j = 0; j < numCols; j++) {
      const exp = increasing ? j : numCols - 1 - j;
      result.set([i, j], Math.pow(val, exp));
    }
  }

  return result;
}
