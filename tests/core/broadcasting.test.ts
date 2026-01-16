import { describe, it, expect } from 'vitest';

// Note: Broadcasting is now handled internally by the native C++ module.
// These tests are skipped as the pure TypeScript broadcasting utilities have been removed.
// The native module's broadcasting behavior is tested indirectly through the arithmetic tests.

describe.skip('Broadcasting (pure TS - removed)', () => {
  describe('broadcastShapes', () => {
    it('should broadcast same shapes', () => {
      // Previously tested pure TS broadcasting utilities
    });
  });
});
