/**
 * Random number generation - thin wrapper around native random module
 */

import { native } from './loader.js';
import { NDArray } from './ndarray.js';

const { random } = native;

/**
 * Set the random seed
 */
export function seed(value: number): void {
  random.seed(value);
}

/**
 * Random values in [0, 1)
 */
export function rand(...shape: number[]): NDArray {
  return new NDArray(random.random(shape));
}

/**
 * Uniform distribution
 */
export function uniform(low: number, high: number, shape: number[]): NDArray {
  return new NDArray(random.uniform(low, high, shape));
}

/**
 * Normal (Gaussian) distribution
 */
export function normal(mean: number, std: number, shape: number[]): NDArray {
  return new NDArray(random.normal(mean, std, shape));
}

/**
 * Standard normal distribution
 */
export function randn(...shape: number[]): NDArray {
  return new NDArray(random.randn(...shape));
}

/**
 * Random integers
 */
export function randint(low: number, high: number, shape: number[]): NDArray {
  return new NDArray(random.randint(low, high, shape));
}
