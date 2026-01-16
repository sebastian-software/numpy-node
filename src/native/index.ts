/**
 * numpy-node - NumPy for Node.js with native bindings
 *
 * This module provides NumPy-like array operations backed by
 * native C++ code with BLAS/LAPACK acceleration.
 */

// Core NDArray class and types
export {
  NDArray,
  type DTypeName,
  type Shape,
  type TypedArray,
  type ArrayInput,
  // Creation functions
  array,
  astype,
  zeros,
  ones,
  full,
  arange,
  linspace,
  eye,
  identity,
  empty,
  zerosLike,
  onesLike,
  emptyLike,
} from './ndarray.js';

// Math operations
export {
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
  tan,
  sum,
  mean,
  std,
  variance,
  median,
  min,
  max,
  prod,
  abs,
  negative,
} from './math.js';

// Linear algebra (namespace and top-level exports)
export * as linalg from './linalg.js';
export {
  matmul,
  dot,
  inv,
  det,
  solve,
  eig,
  eigvals,
  svd,
  qr,
  cholesky,
  norm,
  matrix_rank,
  trace,
  cond,
} from './linalg.js';

// Random number generation
export * as random from './random.js';

// Native module info
export { backend, version } from './loader.js';
