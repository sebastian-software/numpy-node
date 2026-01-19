#!/usr/bin/env node
/**
 * Generates NumPy reference values for conformity tests.
 * Fails gracefully if Python or NumPy is not available.
 */

import { execSync, spawnSync } from 'child_process';
import { existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const scriptPath = join(__dirname, 'generate_numpy_reference.py');
const outputPath = join(__dirname, 'numpy_reference.json');

// Skip if reference file already exists
if (existsSync(outputPath)) {
  console.log('NumPy reference file already exists, skipping generation.');
  process.exit(0);
}

// Check if Python3 is available
const pythonCheck = spawnSync('python3', ['--version'], { stdio: 'pipe' });
if (pythonCheck.status !== 0) {
  console.log('\x1b[33m%s\x1b[0m', 'Python3 not found - skipping NumPy reference generation.');
  console.log('To run conformity tests, install Python3 and NumPy, then run:');
  console.log('  python3 scripts/generate_numpy_reference.py\n');
  process.exit(0);
}

// Check if NumPy is available
const numpyCheck = spawnSync('python3', ['-c', 'import numpy'], { stdio: 'pipe' });
if (numpyCheck.status !== 0) {
  console.log('\x1b[33m%s\x1b[0m', 'NumPy not installed - skipping reference generation.');
  console.log('To run conformity tests, install NumPy and run:');
  console.log('  pip install numpy');
  console.log('  python3 scripts/generate_numpy_reference.py\n');
  process.exit(0);
}

// Generate reference values
console.log('Generating NumPy reference values...');
try {
  execSync(`python3 "${scriptPath}"`, { stdio: 'inherit' });
  console.log('NumPy reference values generated successfully.\n');
} catch (error) {
  console.error('\x1b[33m%s\x1b[0m', 'Failed to generate NumPy reference values.');
  console.error('You can generate them manually with:');
  console.error('  python3 scripts/generate_numpy_reference.py\n');
  process.exit(0); // Don't fail the install
}
