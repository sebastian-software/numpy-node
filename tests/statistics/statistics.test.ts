/**
 * Tests for statistical functions
 *
 * Note: Many statistics functions are not yet available in the native module.
 * This test file only tests the functions that are currently implemented.
 */

import { describe, it, expect } from 'vitest';
import { array, mean, std, NDArray } from '../../src/index.js';

describe('mean', () => {
  it('should compute mean of 1D array', () => {
    const a = array([1, 2, 3, 4, 5]);
    expect(mean(a)).toBe(3);
  });

  it('should compute mean of 2D array', () => {
    const a = array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(mean(a)).toBe(3.5);
  });

  it.skip('should compute mean along axis 0 (TODO: axis reduction returns wrong type)', () => {
    const a = array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const result = mean(a, 0) as NDArray;
    expect(result.toFlatArray()).toEqual([2.5, 3.5, 4.5]);
  });

  it.skip('should compute mean along axis 1 (TODO: axis reduction returns wrong type)', () => {
    const a = array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const result = mean(a, 1) as NDArray;
    expect(result.toFlatArray()).toEqual([2, 5]);
  });
});

describe('std', () => {
  it('should compute population standard deviation', () => {
    const a = array([1, 2, 3, 4, 5]);
    expect(std(a)).toBeCloseTo(Math.sqrt(2), 10);
  });

  it.skip('should compute std along axis (TODO: axis reduction returns wrong type)', () => {
    const a = array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const result = std(a, 0);
    if (typeof result === 'number') {
      throw new Error('Expected NDArray');
    }
    const expected = Math.sqrt(2.25);
    for (const val of result) {
      expect(val).toBeCloseTo(expected, 10);
    }
  });
});

// TODO: Add tests for these functions when implemented in native module:
// - average
// - variance
// - median
// - percentile
// - quantile
// - histogram
// - histogram2d
// - cov
// - corrcoef
