import { describe, it, expect } from 'vitest';
import {
  broadcastShapes,
  areBroadcastable,
  getBroadcastStrides,
} from '../../src/core/broadcasting.js';

describe('Broadcasting', () => {
  describe('broadcastShapes', () => {
    it('should broadcast same shapes', () => {
      expect(broadcastShapes([3, 4], [3, 4])).toEqual([3, 4]);
    });

    it('should broadcast scalar', () => {
      expect(broadcastShapes([3, 4], [])).toEqual([3, 4]);
    });

    it('should broadcast 1D to 2D', () => {
      expect(broadcastShapes([3, 4], [4])).toEqual([3, 4]);
    });

    it('should broadcast with dimension 1', () => {
      expect(broadcastShapes([3, 1], [1, 4])).toEqual([3, 4]);
    });

    it('should broadcast multiple dimensions', () => {
      expect(broadcastShapes([8, 1, 6, 1], [7, 1, 5])).toEqual([8, 7, 6, 5]);
    });

    it('should throw for incompatible shapes', () => {
      expect(() => broadcastShapes([3, 4], [5, 4])).toThrow();
    });
  });

  describe('areBroadcastable', () => {
    it('should return true for compatible shapes', () => {
      expect(areBroadcastable([3, 4], [3, 4])).toBe(true);
      expect(areBroadcastable([3, 4], [4])).toBe(true);
      expect(areBroadcastable([3, 4], [1])).toBe(true);
      expect(areBroadcastable([3, 4], [])).toBe(true);
    });

    it('should return false for incompatible shapes', () => {
      expect(areBroadcastable([3, 4], [5, 4])).toBe(false);
      expect(areBroadcastable([3, 4], [3, 5])).toBe(false);
    });
  });

  describe('getBroadcastStrides', () => {
    it('should compute strides for broadcast', () => {
      const strides = getBroadcastStrides([4], [4], [3, 4]);
      expect(strides).toEqual([0, 4]);
    });

    it('should set stride 0 for broadcast dimension', () => {
      const strides = getBroadcastStrides([1, 4], [4, 4], [3, 4]);
      expect(strides).toEqual([0, 4]);
    });
  });
});
