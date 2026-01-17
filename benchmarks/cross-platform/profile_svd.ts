#!/usr/bin/env npx tsx
/**
 * Profile SVD operation breakdown.
 */
import { array, mean, subtract, svd, matmul, NDArray } from '../../src/index.js';

function randomMatrix(rows: number, cols: number): NDArray {
  const data: number[][] = [];
  for (let i = 0; i < rows; i++) {
    const row: number[] = [];
    for (let j = 0; j < cols; j++) {
      row.push(Math.random() * 2 - 1);
    }
    data.push(row);
  }
  return array(data);
}

console.log('=== PCA via SVD (2k x 100) Breakdown ===\n');

const data = randomMatrix(2000, 100);
const iterations = 10;

// Step 1: Center the data
let start = performance.now();
for (let i = 0; i < iterations; i++) {
  const m = mean(data, 0) as NDArray;
  subtract(data, m);
}
let t = (performance.now() - start) / iterations;
console.log(`Center (mean + subtract): ${t.toFixed(2)} ms`);

const m = mean(data, 0) as NDArray;
const centered = subtract(data, m);

// Step 2: SVD only
start = performance.now();
for (let i = 0; i < iterations; i++) {
  svd(centered);
}
t = (performance.now() - start) / iterations;
console.log(`SVD only:                 ${t.toFixed(2)} ms  <-- This is the bottleneck`);

// Step 3: Get V and transpose
const { vh } = svd(centered);
start = performance.now();
for (let i = 0; i < iterations; i++) {
  vh.T;
}
t = (performance.now() - start) / iterations;
console.log(`vh.T (transpose):         ${t.toFixed(2)} ms`);

// Step 4: Project
const V = vh.T;
start = performance.now();
for (let i = 0; i < iterations; i++) {
  matmul(centered, V);
}
t = (performance.now() - start) / iterations;
console.log(`matmul(centered, V):      ${t.toFixed(2)} ms`);

console.log('\n--- Full PCA ---');
start = performance.now();
for (let i = 0; i < iterations; i++) {
  const m2 = mean(data, 0) as NDArray;
  const c2 = subtract(data, m2);
  const { vh: vh2 } = svd(c2);
  const V2 = vh2.T;
  matmul(c2, V2);
}
t = (performance.now() - start) / iterations;
console.log(`Full PCA:                 ${t.toFixed(2)} ms`);
