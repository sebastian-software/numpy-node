/**
 * np-ts - A high-performance TypeScript implementation of NumPy
 *
 * @packageDocumentation
 */

// Core exports
export {
  // Types
  type DTypeName,
  type TypedArray,
  type TypedArrayConstructor,
  type DTypeToTypedArray,
  type DTypeToElement,
  type NumericDTypeName,
  type DTypeDescriptor,
  type Shape,
  type Strides,
  type Order,
  type SliceSpec,
  type IndexSpec,
  type NDArrayOptions,
  type ArrayInput,
  // Classes
  DType,
  NDArray,
  // DType utilities
  dtypes,
  getDTypeDescriptor,
  isValidDTypeName,
  inferDType,
  inferDTypeFromArray,
  promoteDTypes,
  canCast,
  DEFAULT_DTYPE,
  // Shape utilities
  validateShape,
  shapeToSize,
  shapeToNdim,
  shapesEqual,
  normalizeShape,
  inferShape,
  normalizeAxis,
  // Stride utilities
  calculateStrides,
  // Broadcasting
  broadcastShapes,
  areBroadcastable,
  // Slicing
  slice,
} from './core/index.js';

// Array creation
export {
  array,
  zeros,
  zerosLike,
  ones,
  onesLike,
  full,
  fullLike,
  empty,
  emptyLike,
  arange,
  linspace,
  logspace,
  geomspace,
  eye,
  identity,
  diag,
  tri,
  tril,
  triu,
  vander,
} from './creation/index.js';

// Operations
export {
  // Arithmetic
  add,
  subtract,
  multiply,
  divide,
  trueDivide,
  floorDivide,
  mod,
  remainder,
  fmod,
  divmod,
  power,
  sqrt,
  cbrt,
  square,
  abs,
  absolute,
  fabs,
  sign,
  copysign,
  negative,
  positive,
  reciprocal,
  maximum,
  minimum,
  fmax,
  fmin,
  // Reductions
  sum,
  prod,
  cumsum,
  cumprod,
  max,
  amax,
  min,
  amin,
  ptp,
  clip,
  // Trigonometric
  sin,
  cos,
  tan,
  arcsin,
  asin,
  arccos,
  acos,
  arctan,
  atan,
  arctan2,
  atan2,
  hypot,
  // Hyperbolic
  sinh,
  cosh,
  tanh,
  arcsinh,
  asinh,
  arccosh,
  acosh,
  arctanh,
  atanh,
  // Rounding
  round,
  around,
  rint,
  floor,
  ceil,
  trunc,
  fix,
  // Exponential and Logarithmic
  exp,
  expm1,
  exp2,
  log,
  log1p,
  log2,
  log10,
  logaddexp,
  logaddexp2,
  // Special
  sigmoid,
  degrees,
  rad2deg,
  radians,
  deg2rad,
  unwrap,
  // Floating point
  isfinite,
  isinf,
  isnan,
  isneginf,
  isposinf,
  nanToNum,
  // Comparison
  equal,
  notEqual,
  not_equal,
  less,
  lessEqual,
  less_equal,
  greater,
  greaterEqual,
  greater_equal,
  // Array comparison
  allclose,
  arrayEqual,
  array_equal,
  arrayEquiv,
  array_equiv,
  // Logical reductions
  all,
  any,
  // Logical element-wise
  logicalAnd,
  logical_and,
  logicalOr,
  logical_or,
  logicalXor,
  logical_xor,
  logicalNot,
  logical_not,
  // Where and select
  where,
  nonzero,
  flatnonzero,
  argmax,
  argmin,
  countNonzero,
  count_nonzero,
  // Constants
  pi,
  e,
  inf,
  nan,
} from './ops/index.js';

// Statistics
export {
  mean,
  average,
  variance,
  var_ as var,
  std,
  median,
  percentile,
  quantile,
  histogram,
  histogram2d,
  cov,
  corrcoef,
} from './statistics/index.js';

// Linear Algebra
export {
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
} from './linalg/index.js';

// Default export as np-like namespace
import { NDArray, slice } from './core/index.js';
import * as creation from './creation/index.js';
import * as ops from './ops/index.js';
import * as statistics from './statistics/index.js';
import * as linalg from './linalg/index.js';

/**
 * NumPy-like namespace object
 * Use: import np from 'np-ts'
 */
const np = {
  // Core
  NDArray,
  slice,
  // Creation functions
  ...creation,
  // Operations
  ...ops,
  // Statistics
  ...statistics,
  // Linear Algebra
  linalg,
  ...linalg,
};

export default np;
