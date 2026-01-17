#!/usr/bin/env npx tsx
/**
 * Profile Linear Regression breakdown.
 */
import { array, matmul, solve, NDArray } from '../../src/index.js';

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

console.log('=== Linear Regression (5k x 50) Breakdown ===\n');

const X = randomMatrix(5000, 50);
const y = randomMatrix(5000, 1);
const iterations = 100;

// Step 1: Transpose X
let start = performance.now();
for (let i = 0; i < iterations; i++) {
  X.T;
}
let t = (performance.now() - start) / iterations;
console.log(`X.T:                ${t.toFixed(3)} ms`);

const Xt = X.T;

// Step 2: Xt @ X
start = performance.now();
for (let i = 0; i < iterations; i++) {
  matmul(Xt, X);
}
t = (performance.now() - start) / iterations;
console.log(`matmul(Xt, X):      ${t.toFixed(3)} ms  [50x5000 @ 5000x50 = 50x50]`);

const XtX = matmul(Xt, X);

// Step 3: Xt @ y
start = performance.now();
for (let i = 0; i < iterations; i++) {
  matmul(Xt, y);
}
t = (performance.now() - start) / iterations;
console.log(`matmul(Xt, y):      ${t.toFixed(3)} ms  [50x5000 @ 5000x1 = 50x1]`);

const Xty = matmul(Xt, y);

// Step 4: solve(XtX, Xty)
start = performance.now();
for (let i = 0; i < iterations; i++) {
  solve(XtX, Xty);
}
t = (performance.now() - start) / iterations;
console.log(`solve(XtX, Xty):    ${t.toFixed(3)} ms  [50x50, 50x1]`);

console.log('\n--- Full Linear Regression ---');
start = performance.now();
for (let i = 0; i < iterations; i++) {
  const Xt2 = X.T;
  const XtX2 = matmul(Xt2, X);
  const Xty2 = matmul(Xt2, y);
  solve(XtX2, Xty2);
}
t = (performance.now() - start) / iterations;
console.log(`Full LR:            ${t.toFixed(3)} ms`);
console.log(`\nNumPy does this in ~0.17ms, we do it in ~${t.toFixed(2)}ms`);
