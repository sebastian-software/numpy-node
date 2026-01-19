import { describe, it, expect } from 'vitest';
import {
  array,
  add,
  subtract,
  multiply,
  divide,
  power,
  sqrt,
  abs,
  sum,
  prod,
  max,
  min,
  mean,
  negative,
  add_inplace,
  subtract_inplace,
  multiply_inplace,
  divide_inplace,
  NDArray,
} from '../../src/index.js';

describe('Arithmetic Operations', () => {
  describe('basic arithmetic', () => {
    it('should add two arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = add(a, b);
      expect(result.toFlatArray()).toEqual([5, 7, 9]);
    });

    it('should add scalar to array', () => {
      const a = array([1, 2, 3]);
      const result = add(a, 10);
      expect(result.toFlatArray()).toEqual([11, 12, 13]);
    });

    it('should subtract arrays', () => {
      const a = array([5, 7, 9]);
      const b = array([1, 2, 3]);
      const result = subtract(a, b);
      expect(result.toFlatArray()).toEqual([4, 5, 6]);
    });

    it('should multiply arrays', () => {
      const a = array([2, 3, 4]);
      const b = array([5, 6, 7]);
      const result = multiply(a, b);
      expect(result.toFlatArray()).toEqual([10, 18, 28]);
    });

    it('should divide arrays', () => {
      const a = array([10, 20, 30]);
      const b = array([2, 4, 5]);
      const result = divide(a, b);
      expect(result.toFlatArray()).toEqual([5, 5, 6]);
    });

    it('should compute power', () => {
      const a = array([2, 3, 4]);
      const result = power(a, 2);
      expect(result.toFlatArray()).toEqual([4, 9, 16]);
    });

    it('should compute sqrt', () => {
      const a = array([4, 9, 16]);
      const result = sqrt(a);
      expect(result.toFlatArray()).toEqual([2, 3, 4]);
    });

    it('should compute abs', () => {
      const a = array([-1, 2, -3, 4]);
      const result = abs(a);
      expect(result.toFlatArray()).toEqual([1, 2, 3, 4]);
    });

    it('should negate array', () => {
      const a = array([1, -2, 3]);
      const result = negative(a);
      expect(result.toFlatArray()).toEqual([-1, 2, -3]);
    });
  });

  describe('broadcasting', () => {
    it('should broadcast scalar to 2D', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const result = add(a, 10);
      expect(result.toArray()).toEqual([
        [11, 12],
        [13, 14],
      ]);
    });

    it('should broadcast 1D to 2D', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const b = array([10, 20, 30]);
      const result = add(a, b);
      expect(result.toArray()).toEqual([
        [11, 22, 33],
        [14, 25, 36],
      ]);
    });

    it('should broadcast column vector', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const b = array([[10], [20]]);
      const result = add(a, b);
      expect(result.toArray()).toEqual([
        [11, 12, 13],
        [24, 25, 26],
      ]);
    });
  });

  describe('in-place operations', () => {
    it('should add in-place with scalar', () => {
      const a = array([1, 2, 3]);
      const result = add_inplace(a, 10);
      expect(a.toFlatArray()).toEqual([11, 12, 13]);
      expect(result).toBe(a); // Returns same array
    });

    it('should add in-place with array', () => {
      const a = array([1, 2, 3]);
      const b = array([10, 20, 30]);
      add_inplace(a, b);
      expect(a.toFlatArray()).toEqual([11, 22, 33]);
    });

    it('should subtract in-place', () => {
      const a = array([10, 20, 30]);
      subtract_inplace(a, 5);
      expect(a.toFlatArray()).toEqual([5, 15, 25]);
    });

    it('should multiply in-place', () => {
      const a = array([1, 2, 3]);
      multiply_inplace(a, 2);
      expect(a.toFlatArray()).toEqual([2, 4, 6]);
    });

    it('should divide in-place', () => {
      const a = array([10, 20, 30]);
      divide_inplace(a, 2);
      expect(a.toFlatArray()).toEqual([5, 10, 15]);
    });

    it('should support NDArray methods for in-place ops', () => {
      const a = array([2, 4, 6]);
      a.imul(2);
      expect(a.toFlatArray()).toEqual([4, 8, 12]);

      a.iadd(1);
      expect(a.toFlatArray()).toEqual([5, 9, 13]);

      a.isub(1);
      expect(a.toFlatArray()).toEqual([4, 8, 12]);

      a.idiv(2);
      expect(a.toFlatArray()).toEqual([2, 4, 6]);
    });

    it('should support chaining of in-place methods', () => {
      const a = array([5, 10, 15]);
      a.imul(2).iadd(10).idiv(5);
      expect(a.toFlatArray()).toEqual([4, 6, 8]);
    });
  });

  describe('reductions', () => {
    it('should compute sum', () => {
      const a = array([1, 2, 3, 4, 5]);
      expect(sum(a)).toBe(15);
    });

    it('should compute sum along axis 0', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = sum(a, 0);
      expect((result as NDArray).toFlatArray()).toEqual([5, 7, 9]);
    });

    it('should compute sum along axis 1', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = sum(a, 1);
      expect((result as NDArray).toFlatArray()).toEqual([6, 15]);
    });

    it('should compute product', () => {
      const a = array([1, 2, 3, 4]);
      expect(prod(a)).toBe(24);
    });

    it('should compute max', () => {
      const a = array([3, 1, 4, 1, 5, 9, 2, 6]);
      expect(max(a)).toBe(9);
    });

    it('should compute min', () => {
      const a = array([3, 1, 4, 1, 5, 9, 2, 6]);
      expect(min(a)).toBe(1);
    });

    it('should compute mean', () => {
      const a = array([1, 2, 3, 4, 5]);
      expect(mean(a)).toBe(3);
    });
  });
});
