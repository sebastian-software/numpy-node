#!/usr/bin/env npx tsx
/**
 * Runs both NumPy (Python) and numpy-node benchmarks and compares results.
 *
 * Usage: npx tsx run_comparison.ts
 *
 * Requirements:
 * - Python 3 with NumPy installed
 * - Node.js with numpy-node built
 */

import { execSync } from 'child_process';
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
  results: BenchmarkResult[];
}

function runPythonBenchmark(): BenchmarkOutput | null {
  try {
    console.error('Running NumPy (Python) benchmarks...');
    const output = execSync(`python3 ${join(__dirname, 'numpy_benchmark.py')}`, {
      encoding: 'utf-8',
      stdio: ['pipe', 'pipe', 'inherit'],
    });
    return JSON.parse(output);
  } catch {
    console.error('Failed to run Python benchmark. Is NumPy installed?');
    console.error('Install with: pip install numpy');
    return null;
  }
}

function runNodeBenchmark(): BenchmarkOutput | null {
  try {
    console.error('Running numpy-node benchmarks...');
    const output = execSync(`npx tsx ${join(__dirname, 'numpy_node_benchmark.ts')}`, {
      encoding: 'utf-8',
      stdio: ['pipe', 'pipe', 'inherit'],
      cwd: join(__dirname, '../..'),
    });
    return JSON.parse(output);
  } catch (e) {
    console.error('Failed to run Node.js benchmark.');
    console.error(e);
    return null;
  }
}

function formatMs(ms: number): string {
  if (ms < 0.01) return `${(ms * 1000).toFixed(2)}µs`;
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

  lines.push('# NumPy vs numpy-node Benchmark Results\n');
  lines.push(`- **NumPy**: Python ${python.version}`);
  lines.push(`- **numpy-node**: Node.js ${node.version}`);
  lines.push(`- **Date**: ${new Date().toISOString().split('T')[0]}`);
  lines.push(`- **Platform**: ${process.platform} ${process.arch}\n`);

  lines.push('## Results\n');
  lines.push('| Benchmark | NumPy (ms) | numpy-node (ms) | Speedup |');
  lines.push('|-----------|------------|-----------------|---------|');

  const pythonMap = new Map(python.results.map((r) => [r.name, r]));

  for (const nodeResult of node.results) {
    const pythonResult = pythonMap.get(nodeResult.name);
    if (pythonResult) {
      const speedup = pythonResult.median / nodeResult.median;
      const speedupStr =
        speedup > 1 ? `**${speedup.toFixed(2)}x faster**` : `${(1 / speedup).toFixed(2)}x slower`;

      lines.push(
        `| ${nodeResult.name} | ${pythonResult.median.toFixed(3)} | ${nodeResult.median.toFixed(3)} | ${speedupStr} |`
      );
    }
  }

  lines.push('\n## Interpretation\n');
  lines.push('- **Speedup > 1**: numpy-node is faster');
  lines.push('- **Speedup < 1**: NumPy (Python) is faster');
  lines.push(
    '- Large matrix operations (BLAS/LAPACK) should be similar as both use the same underlying libraries'
  );
  lines.push("- Small operations and loops should favor Node.js due to V8's JIT compilation\n");

  return lines.join('\n');
}

function printComparison(python: BenchmarkOutput, node: BenchmarkOutput) {
  console.log('\n' + '='.repeat(80));
  console.log('BENCHMARK COMPARISON: NumPy (Python) vs numpy-node (Node.js)');
  console.log('='.repeat(80));
  console.log(`NumPy version: ${python.version}`);
  console.log(`Node.js version: ${node.version}`);
  console.log('='.repeat(80) + '\n');

  const pythonMap = new Map(python.results.map((r) => [r.name, r]));

  console.log(
    'Benchmark'.padEnd(25) + 'NumPy'.padStart(12) + 'numpy-node'.padStart(12) + '  Speedup'
  );
  console.log('-'.repeat(70));

  for (const nodeResult of node.results) {
    const pythonResult = pythonMap.get(nodeResult.name);
    if (pythonResult) {
      const speedup = pythonResult.median / nodeResult.median;
      console.log(
        nodeResult.name.padEnd(25) +
          formatMs(pythonResult.median).padStart(12) +
          formatMs(nodeResult.median).padStart(12) +
          '  ' +
          formatSpeedup(speedup)
      );
    }
  }

  console.log('\n' + '='.repeat(80));
}

// Main
async function main() {
  console.log('NumPy vs numpy-node Benchmark\n');

  const pythonResults = runPythonBenchmark();
  const nodeResults = runNodeBenchmark();

  if (!pythonResults || !nodeResults) {
    console.error('\nBenchmark comparison failed.');
    process.exit(1);
  }

  printComparison(pythonResults, nodeResults);

  // Generate markdown report
  const markdown = generateMarkdownReport(pythonResults, nodeResults);
  const reportPath = join(__dirname, 'RESULTS.md');
  await import('fs').then((fs) => fs.writeFileSync(reportPath, markdown));
  console.log(`\nMarkdown report saved to: ${reportPath}`);
}

main();
