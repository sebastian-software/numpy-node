import { describe, it, expect } from 'vitest';
import {
  array,
  zeros,
  ones,
  arange,
  linspace,
  eye,
  identity,
  full,
  empty,
  zerosLike,
  onesLike,
  emptyLike,
  astype,
} from '../../src/index.js';

describe('NDArray', () => {
  describe('creation', () => {
    it('should create array from nested JavaScript arrays', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
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

    it('should create identity matrix', () => {
      const arr = identity(3);
      expect(arr.shape).toEqual([3, 3]);
      expect(arr.at(0, 0)).toBe(1);
      expect(arr.at(1, 1)).toBe(1);
      expect(arr.at(0, 1)).toBe(0);
    });

    it('should create full array', () => {
      const arr = full([2, 3], 7);
      expect(arr.shape).toEqual([2, 3]);
      expect(arr.at(0, 0)).toBe(7);
      expect(arr.at(1, 2)).toBe(7);
    });

    it('should create empty array (uninitialized)', () => {
      const arr = empty([2, 2]);
      expect(arr.shape).toEqual([2, 2]);
      expect(arr.dtype).toBe('float64');
    });

    it('should create zerosLike', () => {
      const a = array([1, 2, 3], 'float32');
      const b = zerosLike(a);
      expect(b.shape).toEqual([3]);
      expect(b.dtype).toBe('float32');
      expect(b.toFlatArray()).toEqual([0, 0, 0]);
    });

    it('should create onesLike', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = onesLike(a);
      expect(b.shape).toEqual([2, 2]);
      expect(b.toFlatArray()).toEqual([1, 1, 1, 1]);
    });

    it('should create emptyLike', () => {
      const a = array([1, 2, 3]);
      const b = emptyLike(a);
      expect(b.shape).toEqual([3]);
      expect(b.dtype).toBe('float64');
    });

    it('should create array from NDArray (copy)', () => {
      const a = array([1, 2, 3]);
      const b = array(a);
      expect(b.toFlatArray()).toEqual([1, 2, 3]);
      // Verify it's a copy
      a.set([0], 99);
      expect(b.at(0)).toBe(1);
    });

    it('should create array from TypedArray via reshape', () => {
      // TypedArrays need to go through array() which flattens and creates shape
      const a = array([1, 2, 3, 4]);
      const reshaped = a.reshape([2, 2]);
      expect(reshaped.shape).toEqual([2, 2]);
      expect(reshaped.toFlatArray()).toEqual([1, 2, 3, 4]);
    });
  });

  describe('element access', () => {
    it('should access 1D array elements with at()', () => {
      const arr = array([1, 2, 3, 4, 5]);
      expect(arr.at(0)).toBe(1);
      expect(arr.at(2)).toBe(3);
      expect(arr.at(4)).toBe(5);
    });

    it('should access 2D array elements with at()', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
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

    it('should reshape and access 2D', () => {
      const arr = arange(12);
      const reshaped = arr.reshape([3, 4]);
      expect(reshaped.shape).toEqual([3, 4]);
      expect(reshaped.at(2, 3)).toBe(11);
    });

    it('should transpose array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
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

    it('should convert to JavaScript array', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const jsArr = arr.toArray();
      expect(jsArr).toEqual([
        [1, 2],
        [3, 4],
      ]);
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
        values.push(val);
      }
      expect(values).toEqual([1, 2, 3]);
    });
  });

  describe('element assignment', () => {
    it('should set 1D array element with set()', () => {
      const arr = array([1, 2, 3, 4, 5]);
      arr.set([2], 99);
      expect(arr.at(2)).toBe(99);
    });

    it('should set 2D array element with set()', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      arr.set([1, 1], 99);
      expect(arr.at(1, 1)).toBe(99);
    });
  });

  describe('flatten and ravel', () => {
    it('should flatten 2D array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const flat = arr.flatten();
      expect(flat.shape).toEqual([6]);
      expect(flat.toFlatArray()).toEqual([1, 2, 3, 4, 5, 6]);
    });

    it('should ravel 2D array (alias for flatten)', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const flat = arr.ravel();
      expect(flat.shape).toEqual([4]);
      expect(flat.toFlatArray()).toEqual([1, 2, 3, 4]);
    });
  });

  describe('fill', () => {
    it('should fill array with value', () => {
      const arr = zeros([3]);
      arr.fill(7);
      expect(arr.toFlatArray()).toEqual([7, 7, 7]);
    });
  });

  describe('squeeze', () => {
    it('should remove axes with size 1', () => {
      const arr = array([[[1], [2], [3]]]);
      expect(arr.shape).toEqual([1, 3, 1]);
      const squeezed = arr.squeeze();
      expect(squeezed.shape).toEqual([3]);
    });

    it('should remove specific axis', () => {
      const arr = array([[[1, 2, 3]]]);
      expect(arr.shape).toEqual([1, 1, 3]);
      const squeezed = arr.squeeze(0);
      expect(squeezed.shape).toEqual([1, 3]);
    });
  });
});

describe('dtype conversions', () => {
  it('should return copy when astype same dtype', () => {
    const a = array([1, 2, 3]);
    const b = astype(a, 'float64');
    expect(b.dtype).toBe('float64');
    a.set([0], 99);
    expect(b.at(0)).toBe(1);
  });

  it('should create array with different dtype', () => {
    const a = array([1, 2, 3]);
    const b = astype(a, 'int32');
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([3]);
  });

  it('should handle int8 dtype', () => {
    const arr = zeros([3], 'int8');
    expect(arr.dtype).toBe('int8');
    const ta = arr.toTypedArray();
    expect(ta).toBeInstanceOf(Int8Array);
  });

  it('should handle int16 dtype', () => {
    const arr = zeros([3], 'int16');
    expect(arr.dtype).toBe('int16');
    const ta = arr.toTypedArray();
    expect(ta).toBeInstanceOf(Int16Array);
  });

  it('should handle uint8 dtype', () => {
    const arr = zeros([3], 'uint8');
    expect(arr.dtype).toBe('uint8');
    const ta = arr.toTypedArray();
    expect(ta).toBeInstanceOf(Uint8Array);
  });

  it('should handle uint16 dtype', () => {
    const arr = zeros([3], 'uint16');
    expect(arr.dtype).toBe('uint16');
    const ta = arr.toTypedArray();
    expect(ta).toBeInstanceOf(Uint16Array);
  });

  it('should handle uint32 dtype', () => {
    const arr = zeros([3], 'uint32');
    expect(arr.dtype).toBe('uint32');
    const ta = arr.toTypedArray();
    expect(ta).toBeInstanceOf(Uint32Array);
  });

  it('should handle float32 dtype', () => {
    const arr = zeros([3], 'float32');
    expect(arr.dtype).toBe('float32');
    const ta = arr.toTypedArray();
    expect(ta).toBeInstanceOf(Float32Array);
  });
});

describe('toString', () => {
  it('should return string representation', () => {
    const arr = array([1, 2, 3]);
    const str = arr.toString();
    expect(str).toContain('NDArray');
    expect(str).toContain('[1,2,3]');
    expect(str).toContain('float64');
  });
});

describe('asContiguous', () => {
  it('should return contiguous array', () => {
    const arr = array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const transposed = arr.T;
    const contiguous = transposed.asContiguous();
    expect(contiguous.shape).toEqual([3, 2]);
  });
});

// Note: The following features could be added in the future:
// - slice() for array slicing
// - equals(), allClose() for comparison
// - entries() iterator
// - negative indices in at()
