#!/usr/bin/env npx tsx
/**
 * Profile individual scenarios to find optimization opportunities.
 */
import { array, mean, std, subtract, divide, matmul, exp, sum, NDArray } from '../../src/index.js';

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

function profileDataNormalization() {
  console.log('\n=== Data Normalization (10k x 100) ===');
  const data = randomMatrix(10000, 100);
  const iterations = 50;

  // Warmup
  for (let i = 0; i < 3; i++) {
    const m = mean(data, 0) as NDArray;
    const s = std(data, 0) as NDArray;
    const c = subtract(data, m);
    divide(c, s);
  }

  // Profile each step
  let start: number,
    total = 0;

  start = performance.now();
  for (let i = 0; i < iterations; i++) {
    mean(data, 0);
  }
  let meanTime = (performance.now() - start) / iterations;
  total += meanTime;
  console.log(`  mean(data, 0):     ${meanTime.toFixed(3)} ms`);

  start = performance.now();
  for (let i = 0; i < iterations; i++) {
    std(data, 0);
  }
  let stdTime = (performance.now() - start) / iterations;
  total += stdTime;
  console.log(`  std(data, 0):      ${stdTime.toFixed(3)} ms`);

  const m = mean(data, 0) as NDArray;
  const s = std(data, 0) as NDArray;

  start = performance.now();
  for (let i = 0; i < iterations; i++) {
    subtract(data, m);
  }
  let subTime = (performance.now() - start) / iterations;
  total += subTime;
  console.log(`  subtract(data, m): ${subTime.toFixed(3)} ms`);

  const centered = subtract(data, m);

  start = performance.now();
  for (let i = 0; i < iterations; i++) {
    divide(centered, s);
  }
  let divTime = (performance.now() - start) / iterations;
  total += divTime;
  console.log(`  divide(c, s):      ${divTime.toFixed(3)} ms`);

  console.log(`  --------------------------`);
  console.log(`  Sum of parts:      ${total.toFixed(3)} ms`);

  // Now measure full operation
  start = performance.now();
  for (let i = 0; i < iterations; i++) {
    const m2 = mean(data, 0) as NDArray;
    const s2 = std(data, 0) as NDArray;
    const c2 = subtract(data, m2);
    divide(c2, s2);
  }
  let fullTime = (performance.now() - start) / iterations;
  console.log(`  Full operation:    ${fullTime.toFixed(3)} ms`);
  console.log(`  Overhead:          ${(fullTime - total).toFixed(3)} ms`);
}

function profileCorrelationMatrix() {
  console.log('\n=== Correlation Matrix (5k x 50) ===');
  const data = randomMatrix(5000, 50);
  const n = 5000;
  const iterations = 50;

  // Warmup
  for (let i = 0; i < 3; i++) {
    const m = mean(data, 0) as NDArray;
    const s = std(data, 0) as NDArray;
    const c = subtract(data, m);
    const st = divide(c, s);
    const Xt = st.T;
    const XtX = matmul(Xt, st);
    divide(XtX, n - 1);
  }

  let start: number,
    total = 0;

  start = performance.now();
  for (let i = 0; i < iterations; i++) {
    mean(data, 0);
  }
  let t = (performance.now() - start) / iterations;
  total += t;
  console.log(`  mean(data, 0):     ${t.toFixed(3)} ms`);

  start = performance.now();
  for (let i = 0; i < iterations; i++) {
    std(data, 0);
  }
  t = (performance.now() - start) / iterations;
  total += t;
  console.log(`  std(data, 0):      ${t.toFixed(3)} ms`);

  const m = mean(data, 0) as NDArray;
  const s = std(data, 0) as NDArray;

  start = performance.now();
  for (let i = 0; i < iterations; i++) {
    subtract(data, m);
  }
  t = (performance.now() - start) / iterations;
  total += t;
  console.log(`  subtract(data, m): ${t.toFixed(3)} ms`);

  const centered = subtract(data, m);

  start = performance.now();
  for (let i = 0; i < iterations; i++) {
    divide(centered, s);
  }
  t = (performance.now() - start) / iterations;
  total += t;
  console.log(`  divide(c, s):      ${t.toFixed(3)} ms`);

  const standardized = divide(centered, s);

  start = performance.now();
  for (let i = 0; i < iterations; i++) {
    standardized.T;
  }
  t = (performance.now() - start) / iterations;
  total += t;
  console.log(`  .T (transpose):    ${t.toFixed(3)} ms`);

  const Xt = standardized.T;

  start = performance.now();
  for (let i = 0; i < iterations; i++) {
    matmul(Xt, standardized);
  }
  t = (performance.now() - start) / iterations;
  total += t;
  console.log(`  matmul(Xt, X):     ${t.toFixed(3)} ms`);

  const XtX = matmul(Xt, standardized);

  start = performance.now();
  for (let i = 0; i < iterations; i++) {
    divide(XtX, n - 1);
  }
  t = (performance.now() - start) / iterations;
  total += t;
  console.log(`  divide(XtX, n-1):  ${t.toFixed(3)} ms`);

  console.log(`  --------------------------`);
  console.log(`  Sum of parts:      ${total.toFixed(3)} ms`);
}

function profileNeuralNetForward() {
  console.log('\n=== Neural Net Forward (1k x 256->128->64->10) ===');
  const X = randomMatrix(1000, 256);
  const W1 = randomMatrix(256, 128);
  const W2 = randomMatrix(128, 64);
  const W3 = randomMatrix(64, 10);
  const iterations = 100;

  // Warmup
  for (let i = 0; i < 3; i++) {
    const h1 = matmul(X, W1);
    const h2 = matmul(h1, W2);
    const out = matmul(h2, W3);
    const expOut = exp(out);
    const sumExp = sum(expOut);
    divide(expOut, sumExp as number);
  }

  let start: number,
    total = 0;

  start = performance.now();
  for (let i = 0; i < iterations; i++) {
    matmul(X, W1);
  }
  let t = (performance.now() - start) / iterations;
  total += t;
  console.log(`  matmul(X, W1):     ${t.toFixed(3)} ms`);

  const h1 = matmul(X, W1);

  start = performance.now();
  for (let i = 0; i < iterations; i++) {
    matmul(h1, W2);
  }
  t = (performance.now() - start) / iterations;
  total += t;
  console.log(`  matmul(h1, W2):    ${t.toFixed(3)} ms`);

  const h2 = matmul(h1, W2);

  start = performance.now();
  for (let i = 0; i < iterations; i++) {
    matmul(h2, W3);
  }
  t = (performance.now() - start) / iterations;
  total += t;
  console.log(`  matmul(h2, W3):    ${t.toFixed(3)} ms`);

  const out = matmul(h2, W3);

  start = performance.now();
  for (let i = 0; i < iterations; i++) {
    exp(out);
  }
  t = (performance.now() - start) / iterations;
  total += t;
  console.log(`  exp(out):          ${t.toFixed(3)} ms`);

  const expOut = exp(out);

  start = performance.now();
  for (let i = 0; i < iterations; i++) {
    sum(expOut);
  }
  t = (performance.now() - start) / iterations;
  total += t;
  console.log(`  sum(expOut):       ${t.toFixed(3)} ms`);

  const sumExp = sum(expOut) as number;

  start = performance.now();
  for (let i = 0; i < iterations; i++) {
    divide(expOut, sumExp);
  }
  t = (performance.now() - start) / iterations;
  total += t;
  console.log(`  divide(exp, sum):  ${t.toFixed(3)} ms`);

  console.log(`  --------------------------`);
  console.log(`  Sum of parts:      ${total.toFixed(3)} ms`);
}

profileDataNormalization();
profileCorrelationMatrix();
profileNeuralNetForward();
