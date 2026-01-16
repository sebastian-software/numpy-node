/**
 * Native module loader for numpy-node
 */

import { createRequire } from 'module';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const require = createRequire(import.meta.url);
const __dirname = dirname(fileURLToPath(import.meta.url));

/**
 * Load the native module
 */
function loadNativeModule(): NativeModule {
  const possiblePaths = [
    // Development build (from src/)
    join(__dirname, '../../build/Release/numpy_node_native.node'),
    // Development build (from dist/)
    join(__dirname, '../../../build/Release/numpy_node_native.node'),
    // Installed prebuilds (from src/)
    join(__dirname, '../../prebuilds/numpy_node_native.node'),
    // Installed prebuilds (from dist/)
    join(__dirname, '../../../prebuilds/numpy_node_native.node'),
    // Direct path from project root
    join(process.cwd(), 'build/Release/numpy_node_native.node'),
  ];

  const errors: string[] = [];
  for (const modulePath of possiblePaths) {
    try {
      return require(modulePath) as NativeModule;
    } catch (e) {
      errors.push(`${modulePath}: ${(e as Error).message}`);
      continue;
    }
  }

  throw new Error(
    'Failed to load numpy-node native module. ' +
      'Make sure to run `pnpm build:native` first.\n' +
      'Tried paths:\n' +
      errors.join('\n')
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
  abs(a: NativeNDArray): NativeNDArray;
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
