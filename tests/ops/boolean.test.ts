/**
 * Boolean Array Operations Tests
 *
 * All tests are verified against NumPy 2.4.1 behavior.
 * Each test includes the NumPy verification command as a comment.
 */

import { describe, it, expect } from 'vitest';
import {
  array,
  zeros,
  ones,
  equal,
  not_equal,
  less,
  less_equal,
  greater,
  greater_equal,
  logical_and,
  logical_or,
  logical_xor,
  logical_not,
  any,
  all,
  NDArray,
} from '../../src/index.js';

describe('Boolean Array Operations', () => {
  describe('comparison operators', () => {
    describe('equal', () => {
      // >>> np.equal(np.array([1, 2, 3, 4, 5]), np.array([5, 4, 3, 2, 1]))
      // array([False, False,  True, False, False])
      it('should compare arrays element-wise', () => {
        const a = array([1, 2, 3, 4, 5]);
        const b = array([5, 4, 3, 2, 1]);
        const result = equal(a, b);
        expect(result.dtype).toBe('bool');
        expect(Array.from(result.toFlatArray())).toEqual([0, 0, 1, 0, 0]);
      });

      // >>> np.equal(np.array([1, 2, 3, 4, 5]), 3)
      // array([False, False,  True, False, False])
      it('should compare array with scalar', () => {
        const a = array([1, 2, 3, 4, 5]);
        const result = equal(a, 3);
        expect(result.dtype).toBe('bool');
        expect(Array.from(result.toFlatArray())).toEqual([0, 0, 1, 0, 0]);
      });

      // >>> np.equal(np.array([[1, 2], [3, 4]]), np.array([[1, 0], [3, 0]]))
      // array([[ True, False], [ True, False]])
      it('should work with 2D arrays', () => {
        const a = array([
          [1, 2],
          [3, 4],
        ]);
        const b = array([
          [1, 0],
          [3, 0],
        ]);
        const result = equal(a, b);
        expect(result.shape).toEqual([2, 2]);
        expect(result.dtype).toBe('bool');
        expect(Array.from(result.toFlatArray())).toEqual([1, 0, 1, 0]);
      });
    });

    describe('not_equal', () => {
      // >>> np.not_equal(np.array([1, 2, 3, 4, 5]), np.array([5, 4, 3, 2, 1]))
      // array([ True,  True, False,  True,  True])
      it('should compare arrays element-wise', () => {
        const a = array([1, 2, 3, 4, 5]);
        const b = array([5, 4, 3, 2, 1]);
        const result = not_equal(a, b);
        expect(result.dtype).toBe('bool');
        expect(Array.from(result.toFlatArray())).toEqual([1, 1, 0, 1, 1]);
      });

      // >>> np.not_equal(np.array([1, 2, 3, 2, 1]), 2)
      // array([ True, False,  True, False,  True])
      it('should compare array with scalar', () => {
        const a = array([1, 2, 3, 2, 1]);
        const result = not_equal(a, 2);
        expect(Array.from(result.toFlatArray())).toEqual([1, 0, 1, 0, 1]);
      });
    });

    describe('less', () => {
      // >>> np.less(np.array([1, 2, 3, 4, 5]), 3)
      // array([ True,  True, False, False, False])
      it('should compare array with scalar', () => {
        const a = array([1, 2, 3, 4, 5]);
        const result = less(a, 3);
        expect(result.dtype).toBe('bool');
        expect(Array.from(result.toFlatArray())).toEqual([1, 1, 0, 0, 0]);
      });

      // >>> np.less(np.array([1, 2, 3, 4]), np.array([2, 2, 2, 2]))
      // array([ True, False, False, False])
      it('should compare arrays element-wise', () => {
        const a = array([1, 2, 3, 4]);
        const b = array([2, 2, 2, 2]);
        const result = less(a, b);
        expect(Array.from(result.toFlatArray())).toEqual([1, 0, 0, 0]);
      });
    });

    describe('less_equal', () => {
      // >>> np.less_equal(np.array([1, 2, 3, 4, 5]), 3)
      // array([ True,  True,  True, False, False])
      it('should compare array with scalar', () => {
        const a = array([1, 2, 3, 4, 5]);
        const result = less_equal(a, 3);
        expect(Array.from(result.toFlatArray())).toEqual([1, 1, 1, 0, 0]);
      });

      // >>> np.less_equal(np.array([1, 2, 3, 4]), np.array([2, 2, 2, 2]))
      // array([ True,  True, False, False])
      it('should compare arrays element-wise', () => {
        const a = array([1, 2, 3, 4]);
        const b = array([2, 2, 2, 2]);
        const result = less_equal(a, b);
        expect(Array.from(result.toFlatArray())).toEqual([1, 1, 0, 0]);
      });
    });

    describe('greater', () => {
      // >>> np.greater(np.array([1, 2, 3, 4, 5]), 3)
      // array([False, False, False,  True,  True])
      it('should compare array with scalar', () => {
        const a = array([1, 2, 3, 4, 5]);
        const result = greater(a, 3);
        expect(result.dtype).toBe('bool');
        expect(Array.from(result.toFlatArray())).toEqual([0, 0, 0, 1, 1]);
      });

      // >>> np.greater(np.array([1, 2, 3, 4]), np.array([2, 2, 2, 2]))
      // array([False, False,  True,  True])
      it('should compare arrays element-wise', () => {
        const a = array([1, 2, 3, 4]);
        const b = array([2, 2, 2, 2]);
        const result = greater(a, b);
        expect(Array.from(result.toFlatArray())).toEqual([0, 0, 1, 1]);
      });
    });

    describe('greater_equal', () => {
      // >>> np.greater_equal(np.array([1, 2, 3, 4, 5]), 3)
      // array([False, False,  True,  True,  True])
      it('should compare array with scalar', () => {
        const a = array([1, 2, 3, 4, 5]);
        const result = greater_equal(a, 3);
        expect(Array.from(result.toFlatArray())).toEqual([0, 0, 1, 1, 1]);
      });

      // >>> np.greater_equal(np.array([1, 2, 3, 4]), np.array([2, 2, 2, 2]))
      // array([False,  True,  True,  True])
      it('should compare arrays element-wise', () => {
        const a = array([1, 2, 3, 4]);
        const b = array([2, 2, 2, 2]);
        const result = greater_equal(a, b);
        expect(Array.from(result.toFlatArray())).toEqual([0, 1, 1, 1]);
      });
    });

    describe('broadcasting', () => {
      // >>> np.greater(np.array([[1, 2, 3], [4, 5, 6]]), 3)
      // array([[False, False, False], [ True,  True,  True]])
      it('should broadcast scalar to 2D array', () => {
        const a = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const result = greater(a, 3);
        expect(result.shape).toEqual([2, 3]);
        expect(Array.from(result.toFlatArray())).toEqual([0, 0, 0, 1, 1, 1]);
      });

      // >>> np.greater(np.array([[1, 2, 3], [4, 5, 6]]), np.array([2, 2, 2]))
      // array([[False, False,  True], [ True,  True,  True]])
      it('should broadcast 1D to 2D array', () => {
        const a = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const b = array([2, 2, 2]);
        const result = greater(a, b);
        expect(result.shape).toEqual([2, 3]);
        expect(Array.from(result.toFlatArray())).toEqual([0, 0, 1, 1, 1, 1]);
      });
    });
  });

  describe('logical operators', () => {
    describe('logical_and', () => {
      // >>> np.logical_and(np.array([True, True, False, False]), np.array([True, False, True, False]))
      // array([ True, False, False, False])
      it('should compute element-wise AND', () => {
        const a = array([1, 1, 0, 0]);
        const b = array([1, 0, 1, 0]);
        const result = logical_and(a, b);
        expect(result.dtype).toBe('bool');
        expect(Array.from(result.toFlatArray())).toEqual([1, 0, 0, 0]);
      });

      // >>> x = np.array([1, 2, 3, 4, 5])
      // >>> np.logical_and(np.greater(x, 1), np.less(x, 5))
      // array([False,  True,  True,  True, False])
      it('should work with boolean results from comparisons', () => {
        const x = array([1, 2, 3, 4, 5]);
        const mask1 = greater(x, 1);
        const mask2 = less(x, 5);
        const result = logical_and(mask1, mask2);
        expect(Array.from(result.toFlatArray())).toEqual([0, 1, 1, 1, 0]);
      });

      // >>> np.logical_and(np.array([[1, 0], [1, 1]]), np.array([1, 0]))
      // array([[ True, False], [ True, False]])
      it('should broadcast 1D to 2D array', () => {
        const a2d = array([
          [1, 0],
          [1, 1],
        ]);
        const b1d = array([1, 0]);
        const result = logical_and(a2d, b1d);
        expect(result.shape).toEqual([2, 2]);
        expect(Array.from(result.toFlatArray())).toEqual([1, 0, 1, 0]);
      });
    });

    describe('logical_or', () => {
      // >>> np.logical_or(np.array([True, True, False, False]), np.array([True, False, True, False]))
      // array([ True,  True,  True, False])
      it('should compute element-wise OR', () => {
        const a = array([1, 1, 0, 0]);
        const b = array([1, 0, 1, 0]);
        const result = logical_or(a, b);
        expect(result.dtype).toBe('bool');
        expect(Array.from(result.toFlatArray())).toEqual([1, 1, 1, 0]);
      });
    });

    describe('logical_xor', () => {
      // >>> np.logical_xor(np.array([True, True, False, False]), np.array([True, False, True, False]))
      // array([False,  True,  True, False])
      it('should compute element-wise XOR', () => {
        const a = array([1, 1, 0, 0]);
        const b = array([1, 0, 1, 0]);
        const result = logical_xor(a, b);
        expect(result.dtype).toBe('bool');
        expect(Array.from(result.toFlatArray())).toEqual([0, 1, 1, 0]);
      });
    });

    describe('logical_not', () => {
      // >>> np.logical_not(np.array([True, False, True, False]))
      // array([False,  True, False,  True])
      it('should compute element-wise NOT', () => {
        const a = array([1, 0, 1, 0]);
        const result = logical_not(a);
        expect(result.dtype).toBe('bool');
        expect(Array.from(result.toFlatArray())).toEqual([0, 1, 0, 1]);
      });

      // >>> np.logical_not(np.array([0.0, 1.5, 0.0, -2.0]))
      // array([ True, False,  True, False])
      it('should work with float arrays (0 is falsy)', () => {
        const a = array([0.0, 1.5, 0.0, -2.0]);
        const result = logical_not(a);
        expect(Array.from(result.toFlatArray())).toEqual([1, 0, 1, 0]);
      });
    });
  });

  describe('boolean reductions', () => {
    describe('any', () => {
      // >>> np.any(np.array([0, 0, 1, 0]))
      // True
      it('should return true if any element is truthy', () => {
        const a = array([0, 0, 1, 0]);
        expect(any(a)).toBe(true);
      });

      // >>> np.any(np.array([0, 0, 0, 0]))
      // False
      it('should return false if all elements are falsy', () => {
        const a = array([0, 0, 0, 0]);
        expect(any(a)).toBe(false);
      });

      // >>> x = np.array([1, 2, 3, 4, 5])
      // >>> np.any(np.greater(x, 3))
      // True
      it('should work with boolean mask', () => {
        const x = array([1, 2, 3, 4, 5]);
        const mask = greater(x, 3);
        expect(any(mask)).toBe(true);
      });

      // >>> np.any(np.array([[0, 1, 0], [0, 0, 1]]), axis=0)
      // array([False,  True,  True])
      it('should reduce along axis 0', () => {
        const a = array([
          [0, 1, 0],
          [0, 0, 1],
        ]);
        const result = any(a, 0);
        expect(result).toBeInstanceOf(NDArray);
        expect((result as NDArray).dtype).toBe('bool');
        expect(Array.from((result as NDArray).toFlatArray())).toEqual([0, 1, 1]);
      });

      // >>> np.any(np.array([[0, 1, 0], [0, 0, 1]]), axis=1)
      // array([ True,  True])
      it('should reduce along axis 1', () => {
        const a = array([
          [0, 1, 0],
          [0, 0, 1],
        ]);
        const result = any(a, 1);
        expect(result).toBeInstanceOf(NDArray);
        expect(Array.from((result as NDArray).toFlatArray())).toEqual([1, 1]);
      });

      // >>> np.any(np.array([[0, 0, 0], [0, 0, 0]]), axis=1)
      // array([False, False])
      it('should return all false when no elements are truthy along axis', () => {
        const a = array([
          [0, 0, 0],
          [0, 0, 0],
        ]);
        const result = any(a, 1);
        expect(Array.from((result as NDArray).toFlatArray())).toEqual([0, 0]);
      });
    });

    describe('all', () => {
      // >>> np.all(np.array([1, 1, 1, 1]))
      // True
      it('should return true if all elements are truthy', () => {
        const a = array([1, 1, 1, 1]);
        expect(all(a)).toBe(true);
      });

      // >>> np.all(np.array([1, 1, 0, 1]))
      // False
      it('should return false if any element is falsy', () => {
        const a = array([1, 1, 0, 1]);
        expect(all(a)).toBe(false);
      });

      // >>> x = np.array([1, 2, 3, 4, 5])
      // >>> np.all(np.greater(x, 0))
      // True
      it('should work with boolean mask', () => {
        const x = array([1, 2, 3, 4, 5]);
        const mask = greater(x, 0);
        expect(all(mask)).toBe(true);
      });

      // >>> np.all(np.array([[1, 1, 0], [1, 0, 0]]), axis=0)
      // array([ True, False, False])
      it('should reduce along axis 0', () => {
        const a = array([
          [1, 1, 0],
          [1, 0, 0],
        ]);
        const result = all(a, 0);
        expect(result).toBeInstanceOf(NDArray);
        expect((result as NDArray).dtype).toBe('bool');
        expect(Array.from((result as NDArray).toFlatArray())).toEqual([1, 0, 0]);
      });

      // >>> np.all(np.array([[1, 1, 1], [1, 0, 1]]), axis=1)
      // array([ True, False])
      it('should reduce along axis 1', () => {
        const a = array([
          [1, 1, 1],
          [1, 0, 1],
        ]);
        const result = all(a, 1);
        expect(result).toBeInstanceOf(NDArray);
        expect(Array.from((result as NDArray).toFlatArray())).toEqual([1, 0]);
      });
    });
  });

  describe('bool dtype creation', () => {
    // >>> np.zeros(3, dtype=bool)
    // array([False, False, False])
    it('should create bool array with zeros', () => {
      const a = zeros([3], 'bool');
      expect(a.dtype).toBe('bool');
      expect(Array.from(a.toFlatArray())).toEqual([0, 0, 0]);
    });

    // >>> np.ones(3, dtype=bool)
    // array([ True,  True,  True])
    it('should create bool array with ones', () => {
      const a = ones([3], 'bool');
      expect(a.dtype).toBe('bool');
      expect(Array.from(a.toFlatArray())).toEqual([1, 1, 1]);
    });
  });

  describe('integration tests', () => {
    // Real-world usage patterns verified against NumPy

    // >>> a = np.array([1, 2, 3, 4, 5])
    // >>> mask = np.greater(a, 3)
    // >>> np.any(mask), np.all(mask)
    // (True, False)
    it('should filter values using comparison and reductions', () => {
      const a = array([1, 2, 3, 4, 5]);
      const mask = greater(a, 3);
      expect(any(mask)).toBe(true);
      expect(all(mask)).toBe(false);
    });

    // >>> a = np.array([1, 2, 3, 4, 5])
    // >>> np.logical_and(np.greater(a, 1), np.less(a, 5))
    // array([False,  True,  True,  True, False])
    it('should combine multiple conditions', () => {
      const a = array([1, 2, 3, 4, 5]);
      const inRange = logical_and(greater(a, 1), less(a, 5));
      expect(Array.from(inRange.toFlatArray())).toEqual([0, 1, 1, 1, 0]);
    });

    // >>> x = np.array([1, 2, 3, 4, 5])
    // >>> y = np.array([5, 4, 3, 2, 1])
    // >>> np.logical_or(np.greater(x, 2), np.less(y, 4))
    // array([False, False,  True,  True,  True])
    it('should chain logical operations', () => {
      const x = array([1, 2, 3, 4, 5]);
      const y = array([5, 4, 3, 2, 1]);
      const cond1 = greater(x, 2);
      const cond2 = less(y, 4);
      const combined = logical_or(cond1, cond2);
      expect(Array.from(combined.toFlatArray())).toEqual([0, 0, 1, 1, 1]);
    });
  });
});
