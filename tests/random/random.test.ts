import { describe, it, expect } from 'vitest';
import * as np from '../../src/index.js';

describe('random', () => {
  describe('randn', () => {
    it('should generate array with correct shape', () => {
      const arr = np.random.randn(2, 5);
      expect(arr.shape).toEqual([2, 5]);
    });

    it('should generate values', () => {
      const arr = np.random.randn(10);
      expect(arr.size).toBe(10);
      // Values should be numbers (not NaN or undefined)
      const values = arr.toFlatArray();
      expect(values.every((v) => !isNaN(v))).toBe(true);
    });
  });

  describe('randint', () => {
    it('should generate integers in [low, high)', () => {
      const arr = np.random.randint(0, 10, [100]);
      const values = arr.toFlatArray();
      expect(values.every((v) => v >= 0 && v < 10 && Number.isInteger(v))).toBe(true);
    });

    it('should generate array with correct shape', () => {
      const arr = np.random.randint(0, 100, [4, 5]);
      expect(arr.shape).toEqual([4, 5]);
    });
  });
});
