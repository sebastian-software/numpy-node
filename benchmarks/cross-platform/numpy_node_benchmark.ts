#!/usr/bin/env npx tsx
/**
 * numpy-node benchmark suite for comparison with NumPy (Python).
 * Run with: npx tsx numpy_node_benchmark.ts
 */

import np, {
  zeros,
  ones,
  arange,
  linspace,
  add,
  multiply,
  sqrt,
  exp,
  sum,
  mean,
  std,
  min,
  max,
  matmul,
  dot,
  inv,
  det,
  svd,
  qr,
  eig,
  solve,
  array,
} from '../../src/index.js';

interface BenchmarkResult {
  name: string;
  mean: number;
  min: number;
  max: number;
  median: number;
  p95: number;
  iterations: number;
}

function benchmark(name: string, fn: () => void, warmup = 3, iterations = 100): BenchmarkResult {
  // Warmup
  for (let i = 0; i < warmup; i++) {
    fn();
  }

  // Timed runs
  const times: number[] = [];
  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    fn();
    const end = performance.now();
    times.push(end - start);
  }

  times.sort((a, b) => a - b);

  return {
    name,
    mean: times.reduce((a, b) => a + b, 0) / times.length,
    min: times[0],
    max: times[times.length - 1],
    median: times[Math.floor(times.length / 2)],
    p95: times[Math.floor(times.length * 0.95)],
    iterations,
  };
}

// Simple random array generator (since np.random might not be fully implemented)
function randomArray(rows: number, cols: number): number[][] {
  const result: number[][] = [];
  for (let i = 0; i < rows; i++) {
    const row: number[] = [];
    for (let j = 0; j < cols; j++) {
      row.push(Math.random());
    }
    result.push(row);
  }
  return result;
}

function runBenchmarks(): BenchmarkResult[] {
  const results: BenchmarkResult[] = [];

  // ============================================
  // Array Creation
  // ============================================

  results.push(benchmark('zeros(1000x1000)', () => zeros([1000, 1000])));

  results.push(benchmark('ones(1000x1000)', () => ones([1000, 1000])));

  results.push(benchmark('arange(100000)', () => arange(0, 100000)));

  results.push(benchmark('linspace(0, 1, 100000)', () => linspace(0, 1, 100000)));

  // ============================================
  // Element-wise Operations
  // ============================================

  const a = array(randomArray(1000, 1000));
  const b = array(randomArray(1000, 1000));

  results.push(benchmark('add(1000x1000)', () => add(a, b)));

  results.push(benchmark('multiply(1000x1000)', () => multiply(a, b)));

  results.push(benchmark('sqrt(1000x1000)', () => sqrt(a)));

  results.push(benchmark('exp(1000x1000)', () => exp(a)));

  // ============================================
  // Reductions
  // ============================================

  results.push(benchmark('sum(1000x1000)', () => sum(a)));

  results.push(benchmark('mean(1000x1000)', () => mean(a)));

  results.push(benchmark('std(1000x1000)', () => std(a)));

  results.push(benchmark('min(1000x1000)', () => min(a)));

  results.push(benchmark('max(1000x1000)', () => max(a)));

  // ============================================
  // Linear Algebra (BLAS/LAPACK)
  // ============================================

  const m1 = array(randomArray(500, 500));
  const m2 = array(randomArray(500, 500));

  results.push(benchmark('matmul(500x500)', () => matmul(m1, m2), 3, 50));

  results.push(benchmark('dot(500x500)', () => dot(m1, m2), 3, 50));

  const mSmall = array(randomArray(100, 100));

  results.push(benchmark('inv(100x100)', () => inv(mSmall), 3, 50));

  results.push(benchmark('det(100x100)', () => det(mSmall), 3, 50));

  results.push(benchmark('svd(100x100)', () => svd(mSmall), 3, 50));

  results.push(benchmark('qr(100x100)', () => qr(mSmall), 3, 50));

  results.push(benchmark('eig(100x100)', () => eig(mSmall), 3, 50));

  // Solve linear system
  const A = array(randomArray(100, 100));
  const bVec = array(Array.from({ length: 100 }, () => Math.random()));

  results.push(benchmark('solve(100x100)', () => solve(A, bVec), 3, 50));

  // ============================================
  // Many Small Operations (Loop Overhead)
  // ============================================

  function manySmallOps() {
    for (let i = 0; i < 1000; i++) {
      const x = array([1, 2, 3, 4, 5]);
      const y = array([5, 4, 3, 2, 1]);
      add(x, y);
    }
  }

  results.push(benchmark('1000x small array ops', manySmallOps, 3, 20));

  return results;
}

// Main
console.error('numpy-node benchmark');
console.error('Running benchmarks...');

const results = runBenchmarks();

const output = {
  runtime: 'node',
  version: process.version,
  results,
};

console.log(JSON.stringify(output, null, 2));
