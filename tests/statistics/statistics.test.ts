/**
 * Tests for statistical functions
 */

import { describe, it, expect } from 'vitest';
import {
  array,
  mean,
  average,
  variance,
  std,
  median,
  percentile,
  quantile,
  histogram,
  histogram2d,
  cov,
  corrcoef,
} from '../../src/index.js';
import { NDArray } from '../../src/core/ndarray.js';

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

  it('should compute mean along axis 0', () => {
    const a = array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const result = mean(a, 0) as NDArray;
    expect(result.toFlatArray()).toEqual([2.5, 3.5, 4.5]);
  });

  it('should compute mean along axis 1', () => {
    const a = array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const result = mean(a, 1) as NDArray;
    expect(result.toFlatArray()).toEqual([2, 5]);
  });

  it('should handle negative axis', () => {
    const a = array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const result = mean(a, -1) as NDArray;
    expect(result.toFlatArray()).toEqual([2, 5]);
  });
});

describe('average', () => {
  it('should compute simple average', () => {
    const a = array([1, 2, 3, 4, 5]);
    expect(average(a)).toBe(3);
  });

  it('should compute weighted average', () => {
    const a = array([1, 2, 3, 4, 5]);
    const weights = array([5, 4, 3, 2, 1]);
    const result = average(a, { weights });
    expect(result).toBeCloseTo(2.333333, 5);
  });

  it('should return weights sum when requested', () => {
    const a = array([1, 2, 3]);
    const [avg, wsum] = average(a, { returned: true }) as [number, number];
    expect(avg).toBe(2);
    expect(wsum).toBe(3);
  });
});

describe('variance', () => {
  it('should compute population variance', () => {
    const a = array([1, 2, 3, 4, 5]);
    expect(variance(a)).toBe(2);
  });

  it('should compute sample variance with ddof=1', () => {
    const a = array([1, 2, 3, 4, 5]);
    expect(variance(a, undefined, 1)).toBe(2.5);
  });

  it('should compute variance along axis', () => {
    const a = array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const result = variance(a, 0) as NDArray;
    expect(result.toFlatArray()).toEqual([2.25, 2.25, 2.25]);
  });

  it('should handle keepdims', () => {
    const a = array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const result = variance(a, 1, 0, true) as NDArray;
    expect(result.shape).toEqual([2, 1]);
  });
});

describe('std', () => {
  it('should compute population standard deviation', () => {
    const a = array([1, 2, 3, 4, 5]);
    expect(std(a)).toBeCloseTo(Math.sqrt(2), 10);
  });

  it('should compute sample standard deviation', () => {
    const a = array([1, 2, 3, 4, 5]);
    expect(std(a, undefined, 1)).toBeCloseTo(Math.sqrt(2.5), 10);
  });

  it('should compute std along axis', () => {
    const a = array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const result = std(a, 0) as NDArray;
    const expected = Math.sqrt(2.25);
    for (const val of result) {
      expect(val).toBeCloseTo(expected, 10);
    }
  });
});

describe('median', () => {
  it('should compute median of odd-length array', () => {
    const a = array([1, 3, 2, 5, 4]);
    expect(median(a)).toBe(3);
  });

  it('should compute median of even-length array', () => {
    const a = array([1, 2, 3, 4]);
    expect(median(a)).toBe(2.5);
  });

  it('should compute median along axis', () => {
    const a = array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const result = median(a, 0) as NDArray;
    expect(result.toFlatArray()).toEqual([2.5, 3.5, 4.5]);
  });

  it('should handle already sorted array', () => {
    const a = array([1, 2, 3, 4, 5]);
    expect(median(a)).toBe(3);
  });
});

describe('percentile', () => {
  it('should compute 50th percentile (median)', () => {
    const a = array([1, 2, 3, 4, 5]);
    expect(percentile(a, 50)).toBe(3);
  });

  it('should compute 0th percentile (min)', () => {
    const a = array([1, 2, 3, 4, 5]);
    expect(percentile(a, 0)).toBe(1);
  });

  it('should compute 100th percentile (max)', () => {
    const a = array([1, 2, 3, 4, 5]);
    expect(percentile(a, 100)).toBe(5);
  });

  it('should compute 25th percentile', () => {
    const a = array([1, 2, 3, 4, 5]);
    expect(percentile(a, 25)).toBe(2);
  });

  it('should compute multiple percentiles', () => {
    const a = array([1, 2, 3, 4, 5]);
    const result = percentile(a, [25, 50, 75]) as NDArray;
    expect(result.toFlatArray()).toEqual([2, 3, 4]);
  });

  it('should throw for invalid percentile', () => {
    const a = array([1, 2, 3]);
    expect(() => percentile(a, 101)).toThrow();
    expect(() => percentile(a, -1)).toThrow();
  });
});

describe('quantile', () => {
  it('should compute 0.5 quantile (median)', () => {
    const a = array([1, 2, 3, 4, 5]);
    expect(quantile(a, 0.5)).toBe(3);
  });

  it('should compute multiple quantiles', () => {
    const a = array([1, 2, 3, 4, 5]);
    const result = quantile(a, [0.25, 0.5, 0.75]) as NDArray;
    expect(result.toFlatArray()).toEqual([2, 3, 4]);
  });
});

describe('histogram', () => {
  it('should compute histogram with default bins', () => {
    const a = array([1, 2, 2, 3, 3, 3, 4, 4, 5]);
    const { hist, binEdges } = histogram(a);
    expect(hist.size).toBe(10);
    expect(binEdges.size).toBe(11);
  });

  it('should compute histogram with specified bins', () => {
    const a = array([1, 2, 2, 3, 3, 3, 4, 4, 5]);
    const { hist, binEdges } = histogram(a, 5);
    expect(hist.size).toBe(5);
    expect(binEdges.size).toBe(6);
  });

  it('should compute histogram with custom range', () => {
    const a = array([1, 2, 3, 4, 5]);
    const { hist, binEdges } = histogram(a, 5, [0, 10]);
    expect(binEdges.at(0)).toBe(0);
    expect(binEdges.at(-1)).toBe(10);
  });

  it('should count values correctly', () => {
    const a = array([0, 0, 1, 1, 1, 2]);
    const { hist } = histogram(a, 3, [0, 3]);
    expect(hist.toFlatArray()).toEqual([2, 3, 1]);
  });
});

describe('histogram2d', () => {
  it('should compute 2D histogram', () => {
    const x = array([1, 2, 3, 4, 5]);
    const y = array([1, 2, 3, 4, 5]);
    const { hist, xEdges, yEdges } = histogram2d(x, y, 5);
    expect(hist.shape).toEqual([5, 5]);
    expect(xEdges.size).toBe(6);
    expect(yEdges.size).toBe(6);
  });

  it('should throw for mismatched sizes', () => {
    const x = array([1, 2, 3]);
    const y = array([1, 2]);
    expect(() => histogram2d(x, y)).toThrow();
  });

  it('should handle different bin sizes for x and y', () => {
    const x = array([1, 2, 3, 4, 5]);
    const y = array([1, 2, 3, 4, 5]);
    const { hist } = histogram2d(x, y, [3, 5]);
    expect(hist.shape).toEqual([3, 5]);
  });
});

describe('cov', () => {
  it('should compute covariance of 1D array', () => {
    const a = array([1, 2, 3, 4, 5]);
    const result = cov(a);
    expect(result.shape).toEqual([1, 1]);
    expect(result.at(0, 0)).toBeCloseTo(2.5, 10);
  });

  it('should compute covariance matrix of 2D array', () => {
    const a = array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const result = cov(a);
    expect(result.shape).toEqual([2, 2]);
    expect(result.at(0, 0)).toBeCloseTo(1, 10);
    expect(result.at(1, 1)).toBeCloseTo(1, 10);
    expect(result.at(0, 1)).toBeCloseTo(1, 10);
  });

  it('should handle rowvar=false', () => {
    const a = array([
      [1, 4],
      [2, 5],
      [3, 6],
    ]);
    const result = cov(a, false);
    expect(result.shape).toEqual([2, 2]);
  });

  it('should handle ddof parameter', () => {
    const a = array([1, 2, 3, 4, 5]);
    const result0 = cov(a, true, 0);
    const result1 = cov(a, true, 1);
    expect(result0.at(0, 0)).toBeCloseTo(2, 10);
    expect(result1.at(0, 0)).toBeCloseTo(2.5, 10);
  });
});

describe('corrcoef', () => {
  it('should compute correlation coefficient of 1D array', () => {
    const a = array([1, 2, 3, 4, 5]);
    const result = corrcoef(a);
    expect(result.shape).toEqual([1, 1]);
    expect(result.at(0, 0)).toBeCloseTo(1, 10);
  });

  it('should compute perfect positive correlation', () => {
    const a = array([
      [1, 2, 3],
      [2, 4, 6],
    ]);
    const result = corrcoef(a);
    expect(result.at(0, 1)).toBeCloseTo(1, 10);
  });

  it('should compute perfect negative correlation', () => {
    const a = array([
      [1, 2, 3],
      [6, 4, 2],
    ]);
    const result = corrcoef(a);
    expect(result.at(0, 1)).toBeCloseTo(-1, 10);
  });

  it('should have 1s on diagonal', () => {
    const a = array([
      [1, 2, 3, 4],
      [4, 5, 6, 7],
      [7, 8, 9, 10],
    ]);
    const result = corrcoef(a);
    expect(result.at(0, 0)).toBeCloseTo(1, 10);
    expect(result.at(1, 1)).toBeCloseTo(1, 10);
    expect(result.at(2, 2)).toBeCloseTo(1, 10);
  });
});
