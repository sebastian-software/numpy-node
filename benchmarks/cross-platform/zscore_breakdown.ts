#!/usr/bin/env npx tsx
/**
 * Break down where time is spent in zscore calculation.
 */
import { array, mean, std, subtract, divide, NDArray } from '../../src/index.js';

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

const data = array(randomMatrix(10000, 100));

console.log('Z-Score Operation Breakdown (10k x 100 array)');
console.log('='.repeat(50));

const iterations = 50;

// Measure each operation separately
let start: number, end: number;

// 1. mean(data, axis=0)
start = performance.now();
let m: NDArray | number = 0;
for (let i = 0; i < iterations; i++) {
  m = mean(data, 0);
}
end = performance.now();
console.log(`mean(data, 0):     ${((end - start) / iterations).toFixed(2)} ms`);

// 2. std(data, axis=0)
start = performance.now();
let s: NDArray | number = 0;
for (let i = 0; i < iterations; i++) {
  s = std(data, 0);
}
end = performance.now();
console.log(`std(data, 0):      ${((end - start) / iterations).toFixed(2)} ms`);

// Get actual m and s for next operations
m = mean(data, 0) as NDArray;
s = std(data, 0) as NDArray;

// 3. subtract(data, m) - this involves broadcasting!
start = performance.now();
let centered: NDArray = data;
for (let i = 0; i < iterations; i++) {
  centered = subtract(data, m);
}
end = performance.now();
console.log(
  `subtract(data, m): ${((end - start) / iterations).toFixed(2)} ms  <-- Broadcasting 10k x 100 - 100`
);

// 4. divide(centered, s) - also broadcasting
start = performance.now();
for (let i = 0; i < iterations; i++) {
  divide(centered, s);
}
end = performance.now();
console.log(
  `divide(c, s):      ${((end - start) / iterations).toFixed(2)} ms  <-- Broadcasting 10k x 100 / 100`
);

console.log('\n' + '='.repeat(50));

// Now measure all together
start = performance.now();
for (let i = 0; i < iterations; i++) {
  const m2 = mean(data, 0) as NDArray;
  const s2 = std(data, 0) as NDArray;
  const c2 = subtract(data, m2);
  const r2 = divide(c2, s2);
}
end = performance.now();
console.log(`Total zscore:      ${((end - start) / iterations).toFixed(2)} ms`);

// Compare with same-shape operations (no broadcasting)
const data2 = array(randomMatrix(10000, 100));
start = performance.now();
for (let i = 0; i < iterations; i++) {
  subtract(data, data2);
}
end = performance.now();
console.log(
  `\nsubtract same shape: ${((end - start) / iterations).toFixed(2)} ms  <-- No broadcasting`
);
