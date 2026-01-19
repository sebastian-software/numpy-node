/**
 * Native module loader for numpy-node
 *
 * Loads the native binary from platform-specific npm packages (production)
 * or from local build directory (development).
 */

import { createRequire } from 'module';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const require = createRequire(import.meta.url);
const __dirname = dirname(fileURLToPath(import.meta.url));

/**
 * Get the platform-specific package name
 */
function getPlatformPackage(): string {
  const platform = process.platform;
  const arch = process.arch;
  return `@numpy-node/${platform}-${arch}`;
}

/**
 * Load the native module
 */
function loadNativeModule(): NativeModule {
  const errors: string[] = [];

  // 1. Try platform-specific npm package first (production)
  const platformPackage = getPlatformPackage();
  try {
    return require(platformPackage) as NativeModule;
  } catch (e) {
    errors.push(`${platformPackage}: ${(e as Error).message}`);
  }

  // 2. Try local development builds
  const possiblePaths = [
    // Development build (from src/)
    join(__dirname, '../../build/Release/numpy_node_native.node'),
    // Development build (from dist/)
    join(__dirname, '../../../build/Release/numpy_node_native.node'),
    // Platform package in monorepo (for local testing)
    join(__dirname, `../../packages/${process.platform}-${process.arch}/numpy_node_native.node`),
    // Direct path from project root
    join(process.cwd(), 'build/Release/numpy_node_native.node'),
  ];

  for (const modulePath of possiblePaths) {
    try {
      return require(modulePath) as NativeModule;
    } catch (e) {
      errors.push(`${modulePath}: ${(e as Error).message}`);
      continue;
    }
  }

  throw new Error(
    `Failed to load numpy-node native module.\n\n` +
      `Platform: ${process.platform}-${process.arch}\n` +
      `Expected package: ${platformPackage}\n\n` +
      `If you're developing locally, run: pnpm build:native\n` +
      `If you installed from npm, your platform may not be supported.\n\n` +
      `Tried:\n` +
      errors.map((e) => `  - ${e}`).join('\n')
  );
}

/**
 * Native NDArray interface
 */
export interface NativeNDArray {
  readonly shape: number[];
  readonly strides: number[];
  readonly dtype: string;
  readonly ndim: number;
  readonly size: number;
  readonly data: ArrayBuffer;

  copy(): NativeNDArray;
  reshape(shape: number[]): NativeNDArray;
  transpose(axes?: number[]): NativeNDArray;
  asContiguous(): NativeNDArray;
  set(indices: number[], value: number): void;
  fill(value: number): NativeNDArray;
}

/**
 * Native NDArray constructor type
 */
export interface NativeNDArrayConstructor {
  new (shape: number[], dtype: string): NativeNDArray;
  new (data: ArrayBufferView, dtype?: string): NativeNDArray;
}

/**
 * Linalg module interface
 */
export interface LinalgModule {
  matmul(a: NativeNDArray, b: NativeNDArray): NativeNDArray;
  matmul_nt(a: NativeNDArray, b: NativeNDArray): NativeNDArray;
  batch_matmul(as: NativeNDArray[], bs: NativeNDArray[]): NativeNDArray[];
  batch_matmul_stacked(a: NativeNDArray, b: NativeNDArray): NativeNDArray;
  dot(a: NativeNDArray, b: NativeNDArray): NativeNDArray | number;
  inv(a: NativeNDArray): NativeNDArray;
  det(a: NativeNDArray): number;
  solve(a: NativeNDArray, b: NativeNDArray): NativeNDArray;
  eig(a: NativeNDArray): { eigenvalues: NativeNDArray; eigenvectors: NativeNDArray };
  eigvals(a: NativeNDArray): NativeNDArray;
  svd(a: NativeNDArray): { u: NativeNDArray; s: NativeNDArray; vh: NativeNDArray };
  qr(a: NativeNDArray): { q: NativeNDArray; r: NativeNDArray };
  cholesky(a: NativeNDArray): NativeNDArray;
  norm(a: NativeNDArray, ord?: number | string): number;
  matrix_rank(a: NativeNDArray): number;
  trace(a: NativeNDArray): number;
  lstsq(a: NativeNDArray, b: NativeNDArray): NativeNDArray;
  normal_equations(X: NativeNDArray, y: NativeNDArray): NativeNDArray;
}

/**
 * Math module interface
 */
export interface MathModule {
  add(a: NativeNDArray, b: NativeNDArray | number): NativeNDArray;
  subtract(a: NativeNDArray, b: NativeNDArray | number): NativeNDArray;
  multiply(a: NativeNDArray, b: NativeNDArray | number): NativeNDArray;
  divide(a: NativeNDArray, b: NativeNDArray | number): NativeNDArray;
  power(a: NativeNDArray, b: NativeNDArray | number): NativeNDArray;
  // In-place operations (modifies first argument, returns same array)
  add_inplace(a: NativeNDArray, b: NativeNDArray | number): NativeNDArray;
  subtract_inplace(a: NativeNDArray, b: NativeNDArray | number): NativeNDArray;
  multiply_inplace(a: NativeNDArray, b: NativeNDArray | number): NativeNDArray;
  divide_inplace(a: NativeNDArray, b: NativeNDArray | number): NativeNDArray;
  sqrt(a: NativeNDArray): NativeNDArray;
  exp(a: NativeNDArray): NativeNDArray;
  log(a: NativeNDArray): NativeNDArray;
  sin(a: NativeNDArray): NativeNDArray;
  cos(a: NativeNDArray): NativeNDArray;
  tan(a: NativeNDArray): NativeNDArray;
  sum(a: NativeNDArray, axis?: number): NativeNDArray | number;
  prod(a: NativeNDArray, axis?: number): NativeNDArray | number;
  mean(a: NativeNDArray, axis?: number): NativeNDArray | number;
  std(a: NativeNDArray, axis?: number): NativeNDArray | number;
  var(a: NativeNDArray, axis?: number): NativeNDArray | number;
  median(a: NativeNDArray): number;
  min(a: NativeNDArray, axis?: number): NativeNDArray | number;
  max(a: NativeNDArray, axis?: number): NativeNDArray | number;
  argmin(a: NativeNDArray, axis?: number): NativeNDArray | number;
  argmax(a: NativeNDArray, axis?: number): NativeNDArray | number;
  cumsum(a: NativeNDArray, axis?: number): NativeNDArray;
  cumprod(a: NativeNDArray, axis?: number): NativeNDArray;
  concatenate(arrays: NativeNDArray[], axis?: number): NativeNDArray;
  stack(arrays: NativeNDArray[], axis?: number): NativeNDArray;
  diff(a: NativeNDArray, n?: number, axis?: number): NativeNDArray;
  sort(a: NativeNDArray, axis?: number): NativeNDArray;
  argsort(a: NativeNDArray, axis?: number): NativeNDArray;
  unique(a: NativeNDArray): NativeNDArray;
  searchsorted(a: NativeNDArray, v: NativeNDArray, side?: string): NativeNDArray;
  // Array manipulation (Tier 3+4)
  tile(a: NativeNDArray, reps: number[]): NativeNDArray;
  repeat(a: NativeNDArray, repeats: number, axis?: number): NativeNDArray;
  flip(a: NativeNDArray, axis?: number | number[]): NativeNDArray;
  rot90(a: NativeNDArray, k?: number, axes?: number[]): NativeNDArray;
  split(a: NativeNDArray, indices_or_sections: number | number[], axis?: number): NativeNDArray[];
  nonzero(a: NativeNDArray): NativeNDArray[];
  // Element-wise math (additional)
  sign(a: NativeNDArray): NativeNDArray;
  mod(a: NativeNDArray, b: NativeNDArray | number): NativeNDArray;
  // Approximate comparison
  isclose(a: NativeNDArray, b: NativeNDArray, rtol?: number, atol?: number): NativeNDArray;
  allclose(a: NativeNDArray, b: NativeNDArray, rtol?: number, atol?: number): boolean;
  abs(a: NativeNDArray): NativeNDArray;
  round(a: NativeNDArray): NativeNDArray;
  floor(a: NativeNDArray): NativeNDArray;
  ceil(a: NativeNDArray): NativeNDArray;
  // Array manipulation
  clip(a: NativeNDArray, min: number, max: number): NativeNDArray;
  where(condition: NativeNDArray, x: NativeNDArray, y: NativeNDArray): NativeNDArray;
  squeeze(a: NativeNDArray, axis?: number): NativeNDArray;
  expand_dims(a: NativeNDArray, axis: number): NativeNDArray;
  zscore(a: NativeNDArray, axis?: number): NativeNDArray;
  corrcoef(a: NativeNDArray): NativeNDArray;
  percentile(a: NativeNDArray, q: number[], axis?: number): NativeNDArray;
  kron(a: NativeNDArray, b: NativeNDArray): NativeNDArray;
  outer(a: NativeNDArray, b: NativeNDArray): NativeNDArray;
  axpby(alpha: number, x: NativeNDArray, beta?: number, y?: NativeNDArray): NativeNDArray;
  // Comparison operators
  equal(a: NativeNDArray, b: NativeNDArray | number): NativeNDArray;
  not_equal(a: NativeNDArray, b: NativeNDArray | number): NativeNDArray;
  less(a: NativeNDArray, b: NativeNDArray | number): NativeNDArray;
  less_equal(a: NativeNDArray, b: NativeNDArray | number): NativeNDArray;
  greater(a: NativeNDArray, b: NativeNDArray | number): NativeNDArray;
  greater_equal(a: NativeNDArray, b: NativeNDArray | number): NativeNDArray;
  // Logical operators
  logical_and(a: NativeNDArray, b: NativeNDArray): NativeNDArray;
  logical_or(a: NativeNDArray, b: NativeNDArray): NativeNDArray;
  logical_xor(a: NativeNDArray, b: NativeNDArray): NativeNDArray;
  logical_not(a: NativeNDArray): NativeNDArray;
  // Boolean reductions
  any(a: NativeNDArray, axis?: number): NativeNDArray | boolean;
  all(a: NativeNDArray, axis?: number): NativeNDArray | boolean;
}

/**
 * Random module interface
 */
export interface RandomModule {
  seed(value: number): void;
  random(shape: number[]): NativeNDArray;
  uniform(low: number, high: number, shape: number[]): NativeNDArray;
  normal(mean: number, std: number, shape: number[]): NativeNDArray;
  randn(...shape: number[]): NativeNDArray;
  randint(low: number, high: number, shape: number[]): NativeNDArray;
}

/**
 * Native module interface
 */
export interface NativeModule {
  version: string;
  backend: 'accelerate' | 'openblas' | 'pure';

  NativeNDArray: NativeNDArrayConstructor;

  // Creation functions
  zeros(shape: number[], dtype?: string): NativeNDArray;
  ones(shape: number[], dtype?: string): NativeNDArray;
  full(shape: number[], value: number, dtype?: string): NativeNDArray;
  fromTypedArray(data: ArrayBufferView, shape?: number[]): NativeNDArray;
  arange(start: number, stop?: number, step?: number, dtype?: string): NativeNDArray;
  linspace(start: number, stop: number, num?: number, dtype?: string): NativeNDArray;
  eye(n: number, m?: number, k?: number, dtype?: string): NativeNDArray;

  // Submodules
  linalg: LinalgModule;
  math: MathModule;
  random: RandomModule;
}

/**
 * The loaded native module instance
 */
export const native: NativeModule = loadNativeModule();

/**
 * Backend information
 */
export const backend = native.backend;
export const version = native.version;
