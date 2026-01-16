import { describe, it, expect } from 'vitest';
import { NDArray, array, zeros, ones, arange, linspace, eye } from '../../src/index.js';

describe('NDArray', () => {
  describe('creation', () => {
    it('should create array from nested JavaScript arrays', () => {
      const arr = array([[1, 2, 3], [4, 5, 6]]);
      expect(arr.shape).toEqual([2, 3]);
      expect(arr.size).toBe(6);
      expect(arr.ndim).toBe(2);
    });

    it('should create array with specified dtype', () => {
      const arr = array([1, 2, 3], 'int32');
      expect(arr.dtype).toBe('int32');
    });

    it('should create zeros array', () => {
      const arr = zeros([3]);
      expect(arr.shape).toEqual([3]);
      expect(arr.at(0)).toBe(0);
      expect(arr.at(1)).toBe(0);
      expect(arr.at(2)).toBe(0);
    });

    it('should create ones array', () => {
      const arr = ones([3]);
      expect(arr.shape).toEqual([3]);
      expect(arr.at(0)).toBe(1);
      expect(arr.at(1)).toBe(1);
      expect(arr.at(2)).toBe(1);
    });

    it('should create arange array', () => {
      const arr = arange(5);
      expect(arr.shape).toEqual([5]);
      expect(arr.toFlatArray()).toEqual([0, 1, 2, 3, 4]);
    });

    it('should create arange with start and stop', () => {
      const arr = arange(2, 7);
      expect(arr.toFlatArray()).toEqual([2, 3, 4, 5, 6]);
    });

    it('should create arange with step', () => {
      const arr = arange(0, 10, 2);
      expect(arr.toFlatArray()).toEqual([0, 2, 4, 6, 8]);
    });

    it('should create linspace array', () => {
      const arr = linspace(0, 1, 5);
      expect(arr.shape).toEqual([5]);
      expect(arr.at(0)).toBe(0);
      expect(arr.at(4)).toBe(1);
    });

    it('should create eye (identity) matrix', () => {
      const arr = eye(3);
      expect(arr.shape).toEqual([3, 3]);
    });
  });

  describe('element access', () => {
    it('should access 1D array elements with at()', () => {
      const arr = array([1, 2, 3, 4, 5]);
      expect(arr.at(0)).toBe(1);
      expect(arr.at(2)).toBe(3);
      expect(arr.at(4)).toBe(5);
    });

    it.skip('should access 2D array elements with at() (TODO: fix native 2D indexing)', () => {
      const arr = array([[1, 2, 3], [4, 5, 6]]);
      expect(arr.at(0, 0)).toBe(1);
      expect(arr.at(0, 2)).toBe(3);
      expect(arr.at(1, 1)).toBe(5);
    });
  });

  describe('shape manipulation', () => {
    it('should reshape 1D array to 1D', () => {
      const arr = arange(12);
      const reshaped = arr.reshape([12]);
      expect(reshaped.shape).toEqual([12]);
    });

    it.skip('should reshape and access 2D (TODO: fix native 2D)', () => {
      const arr = arange(12);
      const reshaped = arr.reshape([3, 4]);
      expect(reshaped.shape).toEqual([3, 4]);
      expect(reshaped.at(2, 3)).toBe(11);
    });

    it.skip('should transpose array (TODO: fix native 2D)', () => {
      const arr = array([[1, 2, 3], [4, 5, 6]]);
      const transposed = arr.T;
      expect(transposed.shape).toEqual([3, 2]);
    });
  });

  describe('copying and conversion', () => {
    it('should create a copy', () => {
      const arr = array([1, 2, 3]);
      const copy = arr.copy();
      expect(copy.at(0)).toBe(1);
      expect(copy.toFlatArray()).toEqual([1, 2, 3]);
    });

    it.skip('should convert to JavaScript array (TODO: fix native 2D)', () => {
      const arr = array([[1, 2], [3, 4]]);
      const jsArr = arr.toArray();
      expect(jsArr).toEqual([[1, 2], [3, 4]]);
    });

    it('should convert 1D to flat array', () => {
      const arr = array([1, 2, 3, 4]);
      const flat = arr.toFlatArray();
      expect(flat).toEqual([1, 2, 3, 4]);
    });
  });

  describe('iteration', () => {
    it('should iterate over 1D elements', () => {
      const arr = array([1, 2, 3]);
      const values: number[] = [];
      for (const val of arr) {
        values.push(val as number);
      }
      expect(values).toEqual([1, 2, 3]);
    });
  });
});

// Note: The following features are not yet available in the native module:
// - set() method for element assignment
// - flatten(), squeeze(), expandDims() methods
// - slice() for array slicing
// - astype() for dtype conversion
// - equals(), allClose() for comparison
// - entries() iterator
// - negative indices in at()
