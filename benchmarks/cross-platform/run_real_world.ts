#!/usr/bin/env npx tsx
/**
 * Run and compare real-world benchmarks between NumPy and numpy-node.
 */

import { execSync } from 'child_process';
import { existsSync, writeFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

interface BenchmarkResult {
  name: string;
  mean: number;
  min: number;
  max: number;
  median: number;
  p95: number;
  iterations: number;
}

interface BenchmarkOutput {
  runtime: string;
  version: string;
  type: string;
  results: BenchmarkResult[];
}

function runPythonBenchmark(): BenchmarkOutput | null {
  try {
    console.error('Running NumPy real-world benchmarks...');
    const venvPython = join(__dirname, '.venv', 'bin', 'python3');
    const pythonCmd = existsSync(venvPython) ? venvPython : 'python3';
    const output = execSync(`${pythonCmd} ${join(__dirname, 'real_world_numpy.py')}`, {
      encoding: 'utf-8',
      stdio: ['pipe', 'pipe', 'inherit'],
      timeout: 300000, // 5 minutes
    });
    return JSON.parse(output);
  } catch (e) {
    console.error('Failed to run Python benchmark.');
    console.error(e);
    return null;
  }
}

function runNodeBenchmark(): BenchmarkOutput | null {
  try {
    console.error('Running numpy-node real-world benchmarks...');
    const output = execSync(`npx tsx ${join(__dirname, 'real_world_numpy_node.ts')}`, {
      encoding: 'utf-8',
      stdio: ['pipe', 'pipe', 'inherit'],
      cwd: join(__dirname, '../..'),
      timeout: 300000, // 5 minutes
    });
    return JSON.parse(output);
  } catch (e) {
    console.error('Failed to run Node.js benchmark.');
    console.error(e);
    return null;
  }
}

function formatMs(ms: number): string {
  if (ms < 1) return `${(ms * 1000).toFixed(1)}µs`;
  if (ms < 1000) return `${ms.toFixed(2)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

function formatSpeedup(ratio: number): string {
  if (ratio > 1) {
    return `\x1b[32m${ratio.toFixed(2)}x faster\x1b[0m`;
  } else if (ratio < 1) {
    return `\x1b[31m${(1 / ratio).toFixed(2)}x slower\x1b[0m`;
  }
  return 'same';
}

function generateMarkdownReport(python: BenchmarkOutput, node: BenchmarkOutput): string {
  const lines: string[] = [];

  lines.push('# Real-World Benchmark Results\n');
  lines.push('These benchmarks represent actual use cases rather than isolated method calls.\n');
  lines.push(`- **NumPy**: Python ${python.version}`);
  lines.push(`- **numpy-node**: Node.js ${node.version}`);
  lines.push(`- **Date**: ${new Date().toISOString().split('T')[0]}`);
  lines.push(`- **Platform**: ${process.platform} ${process.arch}\n`);

  lines.push('## Scenarios\n');
  lines.push('| Scenario | NumPy | numpy-node | Speedup |');
  lines.push('|----------|-------|------------|---------|');

  const pythonMap = new Map(python.results.map((r) => [r.name, r]));
  let wins = 0;
  let total = 0;

  for (const nodeResult of node.results) {
    const pythonResult = pythonMap.get(nodeResult.name);
    if (pythonResult) {
      total++;
      const speedup = pythonResult.median / nodeResult.median;
      if (speedup >= 1) wins++;
      const speedupStr =
        speedup >= 1 ? `**${speedup.toFixed(2)}x faster**` : `${(1 / speedup).toFixed(2)}x slower`;

      lines.push(
        `| ${nodeResult.name} | ${formatMs(pythonResult.median)} | ${formatMs(nodeResult.median)} | ${speedupStr} |`
      );
    }
  }

  lines.push(
    `\n**Summary: numpy-node wins ${wins}/${total} scenarios (${((wins / total) * 100).toFixed(0)}%)**\n`
  );

  lines.push('## Interpretation\n');
  lines.push('- These scenarios combine multiple operations, amortizing N-API overhead');
  lines.push(
    '- Compute-heavy operations (SVD, eigendecomposition) benefit from same BLAS/LAPACK backend'
  );
  lines.push(
    '- Real applications should see performance closer to these results than micro-benchmarks\n'
  );

  return lines.join('\n');
}

function printComparison(python: BenchmarkOutput, node: BenchmarkOutput) {
  console.log('\n' + '='.repeat(90));
  console.log('REAL-WORLD BENCHMARK COMPARISON: NumPy (Python) vs numpy-node (Node.js)');
  console.log('='.repeat(90));
  console.log(`NumPy version: ${python.version}`);
  console.log(`Node.js version: ${node.version}`);
  console.log('='.repeat(90) + '\n');

  const pythonMap = new Map(python.results.map((r) => [r.name, r]));

  console.log(
    'Scenario'.padEnd(45) + 'NumPy'.padStart(12) + 'numpy-node'.padStart(12) + '  Speedup'
  );
  console.log('-'.repeat(90));

  let wins = 0;
  let total = 0;

  for (const nodeResult of node.results) {
    const pythonResult = pythonMap.get(nodeResult.name);
    if (pythonResult) {
      total++;
      const speedup = pythonResult.median / nodeResult.median;
      if (speedup >= 1) wins++;
      console.log(
        nodeResult.name.padEnd(45) +
          formatMs(pythonResult.median).padStart(12) +
          formatMs(nodeResult.median).padStart(12) +
          '  ' +
          formatSpeedup(speedup)
      );
    }
  }

  console.log('\n' + '='.repeat(90));
  console.log(
    `\nSummary: numpy-node wins ${wins}/${total} scenarios (${((wins / total) * 100).toFixed(0)}%)`
  );
  console.log('='.repeat(90));
}

// Main
async function main() {
  console.log('Real-World NumPy vs numpy-node Benchmark\n');

  const pythonResults = runPythonBenchmark();
  const nodeResults = runNodeBenchmark();

  if (!pythonResults || !nodeResults) {
    console.error('\nBenchmark comparison failed.');
    process.exit(1);
  }

  printComparison(pythonResults, nodeResults);

  // Generate markdown report
  const markdown = generateMarkdownReport(pythonResults, nodeResults);
  const reportPath = join(__dirname, 'REAL_WORLD_RESULTS.md');
  writeFileSync(reportPath, markdown);
  console.log(`\nMarkdown report saved to: ${reportPath}`);
}

main();
