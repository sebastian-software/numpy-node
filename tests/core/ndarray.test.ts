import { describe, it, expect } from 'vitest';
import { NDArray, array, zeros, ones, arange, linspace, eye, slice } from '../../src/index.js';

describe('NDArray', () => {
  describe('creation', () => {
    it('should create array from nested JavaScript arrays', () => {
      const arr = array([[1, 2, 3], [4, 5, 6]]);
      expect(arr.shape).toEqual([2, 3]);
      expect(arr.size).toBe(6);
      expect(arr.ndim).toBe(2);
    });

    it('should create array with specified dtype', () => {
      const arr = array([1, 2, 3], { dtype: 'int32' });
      expect(arr.dtype).toBe('int32');
    });

    it('should create zeros array', () => {
      const arr = zeros([2, 3]);
      expect(arr.shape).toEqual([2, 3]);
      expect(arr.at(0, 0)).toBe(0);
      expect(arr.at(1, 2)).toBe(0);
    });

    it('should create ones array', () => {
      const arr = ones([3, 2]);
      expect(arr.shape).toEqual([3, 2]);
      expect(arr.at(0, 0)).toBe(1);
      expect(arr.at(2, 1)).toBe(1);
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
      expect(arr.at(0, 0)).toBe(1);
      expect(arr.at(1, 1)).toBe(1);
      expect(arr.at(2, 2)).toBe(1);
      expect(arr.at(0, 1)).toBe(0);
    });
  });

  describe('element access', () => {
    it('should access elements with at()', () => {
      const arr = array([[1, 2, 3], [4, 5, 6]]);
      expect(arr.at(0, 0)).toBe(1);
      expect(arr.at(0, 2)).toBe(3);
      expect(arr.at(1, 1)).toBe(5);
    });

    it('should support negative indices', () => {
      const arr = array([1, 2, 3, 4, 5]);
      expect(arr.at(-1)).toBe(5);
      expect(arr.at(-2)).toBe(4);
    });

    it('should set elements with set()', () => {
      const arr = zeros([2, 2]);
      arr.set([0, 1], 42);
      expect(arr.at(0, 1)).toBe(42);
    });
  });

  describe('shape manipulation', () => {
    it('should reshape array', () => {
      const arr = arange(12);
      const reshaped = arr.reshape([3, 4]);
      expect(reshaped.shape).toEqual([3, 4]);
      expect(reshaped.at(2, 3)).toBe(11);
    });

    it('should reshape with inferred dimension', () => {
      const arr = arange(12);
      const reshaped = arr.reshape([3, -1]);
      expect(reshaped.shape).toEqual([3, 4]);
    });

    it('should transpose array', () => {
      const arr = array([[1, 2, 3], [4, 5, 6]]);
      const transposed = arr.T;
      expect(transposed.shape).toEqual([3, 2]);
      expect(transposed.at(0, 1)).toBe(4);
    });

    it('should flatten array', () => {
      const arr = array([[1, 2], [3, 4]]);
      const flat = arr.flatten();
      expect(flat.shape).toEqual([4]);
      expect(flat.toFlatArray()).toEqual([1, 2, 3, 4]);
    });

    it('should squeeze dimensions', () => {
      const arr = zeros([1, 3, 1, 4]);
      const squeezed = arr.squeeze();
      expect(squeezed.shape).toEqual([3, 4]);
    });

    it('should expand dimensions', () => {
      const arr = zeros([3, 4]);
      const expanded = arr.expandDims(0);
      expect(expanded.shape).toEqual([1, 3, 4]);
    });
  });

  describe('slicing', () => {
    it('should slice single row', () => {
      const arr = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
      const row = arr.slice([1]);
      expect(row.shape).toEqual([3]);
      expect(row.toFlatArray()).toEqual([4, 5, 6]);
    });

    it('should slice with range', () => {
      const arr = arange(10);
      const sliced = arr.slice([slice(2, 7)]);
      expect(sliced.toFlatArray()).toEqual([2, 3, 4, 5, 6]);
    });

    it('should slice with step', () => {
      const arr = arange(10);
      const sliced = arr.slice([slice(0, 10, 2)]);
      expect(sliced.toFlatArray()).toEqual([0, 2, 4, 6, 8]);
    });

    it('should slice 2D array', () => {
      const arr = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
      const sliced = arr.slice([slice(0, 2), slice(1, 3)]);
      expect(sliced.shape).toEqual([2, 2]);
      expect(sliced.at(0, 0)).toBe(2);
      expect(sliced.at(1, 1)).toBe(6);
    });
  });

  describe('copying and conversion', () => {
    it('should create a copy', () => {
      const arr = array([1, 2, 3]);
      const copy = arr.copy();
      copy.set([0], 99);
      expect(arr.at(0)).toBe(1);
      expect(copy.at(0)).toBe(99);
    });

    it('should convert dtype with astype', () => {
      const arr = array([1.5, 2.7, 3.1]);
      const intArr = arr.astype('int32');
      expect(intArr.dtype).toBe('int32');
    });

    it('should convert to JavaScript array', () => {
      const arr = array([[1, 2], [3, 4]]);
      const jsArr = arr.toArray();
      expect(jsArr).toEqual([[1, 2], [3, 4]]);
    });

    it('should convert to flat array', () => {
      const arr = array([[1, 2], [3, 4]]);
      const flat = arr.toFlatArray();
      expect(flat).toEqual([1, 2, 3, 4]);
    });
  });

  describe('comparison', () => {
    it('should check equality', () => {
      const a = array([1, 2, 3]);
      const b = array([1, 2, 3]);
      const c = array([1, 2, 4]);
      expect(a.equals(b)).toBe(true);
      expect(a.equals(c)).toBe(false);
    });

    it('should check approximate equality', () => {
      const a = array([1.0, 2.0, 3.0]);
      const b = array([1.0000001, 2.0000001, 3.0000001]);
      expect(a.allClose(b)).toBe(true);
    });
  });

  describe('iteration', () => {
    it('should iterate over elements', () => {
      const arr = array([1, 2, 3]);
      const values: number[] = [];
      for (const val of arr) {
        values.push(val as number);
      }
      expect(values).toEqual([1, 2, 3]);
    });

    it('should iterate with entries', () => {
      const arr = array([[1, 2], [3, 4]]);
      const entries: [readonly number[], number][] = [];
      for (const [idx, val] of arr.entries()) {
        entries.push([idx, val as number]);
      }
      expect(entries).toHaveLength(4);
      expect(entries[0]).toEqual([[0, 0], 1]);
    });
  });
});
