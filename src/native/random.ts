/**
 * Random number generation - thin wrapper around native random module
 */

import { native } from './loader.js';
import { NDArray } from './ndarray.js';

const nativeRandom = native.random;

/**
 * Set the random seed
 */
export function seed(value: number): void {
  nativeRandom.seed(value);
}

/**
 * Random values in [0, 1) with variadic shape
 */
export function rand(...shape: number[]): NDArray {
  return new NDArray(nativeRandom.rand(...shape));
}

/**
 * Random values in [0, 1) with array shape
 */
export function random(shape: number[]): NDArray {
  return new NDArray(nativeRandom.random(shape));
}

/**
 * Uniform distribution
 */
export function uniform(low: number, high: number, shape: number[]): NDArray {
  return new NDArray(nativeRandom.uniform(low, high, shape));
}

/**
 * Normal (Gaussian) distribution
 */
export function normal(mean: number, std: number, shape: number[]): NDArray {
  return new NDArray(nativeRandom.normal(mean, std, shape));
}

/**
 * Standard normal distribution
 */
export function randn(...shape: number[]): NDArray {
  return new NDArray(nativeRandom.randn(...shape));
}

/**
 * Random integers
 */
export function randint(low: number, high: number, shape: number[]): NDArray {
  return new NDArray(nativeRandom.randint(low, high, shape));
}
