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
  // In-place operations
  add_inplace,
  subtract_inplace,
  multiply_inplace,
  divide_inplace,
  sqrt,
  exp,
  log,
  sin,
  cos,
  tan,
  tanh,
  sum,
  mean,
  std,
  variance,
  median,
  min,
  max,
  argmin,
  argmax,
  prod,
  abs,
  negative,
  round,
  floor,
  ceil,
  cumsum,
  cumprod,
  concatenate,
  stack,
  vstack,
  hstack,
  diff,
  sort,
  argsort,
  unique,
  zscore,
  corrcoef,
  percentile,
  kron,
  outer,
  axpby,
  // Comparison operators
  equal,
  not_equal,
  less,
  less_equal,
  greater,
  greater_equal,
  // Logical operators
  logical_and,
  logical_or,
  logical_xor,
  logical_not,
  // Boolean reductions
  any,
  all,
  // Array manipulation
  clip,
  where,
  squeeze,
  expand_dims,
  tile,
  repeat,
  flip,
  rot90,
  split,
  slice,
  type SliceSpec,
  gradient_2d,
  heat_step_2d,
  nonzero,
  // Sorting and searching
  searchsorted,
  // Element-wise math (additional)
  sign,
  mod,
  // Approximate comparison
  isclose,
  allclose,
  // Einstein summation
  einsum,
  // Fused operations (reduce N-API overhead)
  normalize,
  affine,
  muladd,
  softmax,
  minmax_scale,
  batch_norm,
  // Optimizer operations
  adam_step,
  power_iter,
  // Attention operations
  batched_attention,
  // Financial operations
  black_scholes,
  // Text/NLP operations
  tfidf,
  // Neural network operations
  dropout,
  cross_entropy,
  // Fused reductions
  sumsq,
  layer_norm,
  softmax_cross_entropy,
} from './math.js';

// Linear algebra (namespace and top-level exports)
export * as linalg from './linalg.js';
export {
  matmul,
  matmul_nt,
  batch_matmul,
  batch_matmul_stacked,
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
  lstsq,
  normal_equations,
  gram,
  xty,
} from './linalg.js';

// Random number generation
export * as random from './random.js';

// FFT (Fast Fourier Transform)
export * as fft from './fft.js';
export {
  fft as fftTransform,
  ifft,
  rfft,
  irfft,
  fftfreq,
  rfftfreq,
  type ComplexArray,
} from './fft.js';

// Native module info
export { backend, version } from './loader.js';
