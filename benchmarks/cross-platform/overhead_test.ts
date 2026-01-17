#!/usr/bin/env npx tsx
/**
 * Test to measure N-API call overhead vs actual computation.
 */
import { array, mean, std, subtract, divide, zeros } from '../../src/index.js';

// Helper to create random matrix
function randomMatrix(rows: number, cols: number): number[][] {
  const data: number[][] = [];
  for (let i = 0; i < rows; i++) {
    const row: number[] = [];
    for (let j = 0; j < cols; j++) {
      row.push(Math.random() * 2 - 1);
    }
    data.push(row);
  }
  return data;
}

// Create test data
const data = array(randomMatrix(10000, 100));

function measureOverhead() {
  const results: Record<string, number | string> = {};

  // 1. Just the function call overhead (tiny array)
  const tiny = array([1.0]);

  let start = performance.now();
  for (let i = 0; i < 100000; i++) {
    mean(tiny);
  }
  let end = performance.now();
  results['mean(tiny) per call'] = ((end - start) / 100000) * 1000; // microseconds

  // 2. Mean on large array
  start = performance.now();
  for (let i = 0; i < 1000; i++) {
    mean(data);
  }
  end = performance.now();
  results['mean(10k x 100) per call'] = ((end - start) / 1000) * 1000;

  // 3. Z-score as separate operations
  start = performance.now();
  for (let i = 0; i < 100; i++) {
    const m = mean(data, 0);
    const s = std(data, 0);
    const centered = subtract(data, m);
    const result = divide(centered, s);
  }
  end = performance.now();
  results['zscore separate ops'] = (end - start) / 100; // ms

  // 4. Count the operations
  results['zscore op count'] = '4 ops: mean, std, subtract, divide';

  return results;
}

const results = measureOverhead();

console.log('Node.js/numpy-node Call Overhead Analysis');
console.log('='.repeat(50));
for (const [k, v] of Object.entries(results)) {
  if (typeof v === 'number') {
    console.log(`${k}: ${v < 1000 ? v.toFixed(2) + ' µs' : (v / 1000).toFixed(2) + ' ms'}`);
  } else {
    console.log(`${k}: ${v}`);
  }
}

const tinyOverhead = results['mean(tiny) per call'] as number;
const largeTime = results['mean(10k x 100) per call'] as number;

console.log('\nKey insight:');
console.log(`  - Tiny array mean: ~${tinyOverhead.toFixed(1)} µs (mostly N-API overhead)`);
console.log(`  - Large array mean: ~${largeTime.toFixed(0)} µs`);
console.log(`  - Overhead ratio: ${(largeTime / tinyOverhead).toFixed(1)}x more work`);
