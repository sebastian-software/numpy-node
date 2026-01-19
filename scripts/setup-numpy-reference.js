#!/usr/bin/env node
/**
 * Generates NumPy reference values for conformity tests.
 * Uses a local .venv to install NumPy without affecting the system.
 */

import { execSync, spawnSync } from 'child_process';
import { existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(__dirname, '..');
const venvPath = join(projectRoot, '.venv');
const scriptPath = join(__dirname, 'generate_numpy_reference.py');
const outputPath = join(__dirname, 'numpy_reference.json');

// Platform-specific paths
const isWindows = process.platform === 'win32';
const venvPython = isWindows
  ? join(venvPath, 'Scripts', 'python.exe')
  : join(venvPath, 'bin', 'python3');
const venvPip = isWindows
  ? join(venvPath, 'Scripts', 'pip.exe')
  : join(venvPath, 'bin', 'pip3');

function log(msg) {
  console.log(`[numpy-ref] ${msg}`);
}

function warn(msg) {
  console.log('\x1b[33m%s\x1b[0m', `[numpy-ref] ${msg}`);
}

// Skip if reference file already exists
if (existsSync(outputPath)) {
  log('Reference file already exists, skipping generation.');
  process.exit(0);
}

// Check if Python3 is available
const pythonCheck = spawnSync('python3', ['--version'], { stdio: 'pipe' });
if (pythonCheck.status !== 0) {
  warn('Python3 not found - skipping NumPy reference generation.');
  console.log('To run conformity tests, install Python3 first.\n');
  process.exit(0);
}

// Create venv if it doesn't exist
if (!existsSync(venvPython)) {
  log('Creating Python virtual environment...');
  try {
    execSync(`python3 -m venv "${venvPath}"`, { stdio: 'inherit' });
  } catch (error) {
    warn('Failed to create virtual environment.');
    console.log('You can create it manually with: python3 -m venv .venv\n');
    process.exit(0);
  }
}

// Install/upgrade NumPy in venv
log('Installing latest NumPy in virtual environment...');
try {
  execSync(`"${venvPip}" install --upgrade numpy`, { stdio: 'inherit' });
} catch (error) {
  warn('Failed to install NumPy.');
  console.log('You can install it manually with: .venv/bin/pip install numpy\n');
  process.exit(0);
}

// Generate reference values using venv Python
log('Generating NumPy reference values...');
try {
  execSync(`"${venvPython}" "${scriptPath}"`, { stdio: 'inherit' });
  log('NumPy reference values generated successfully.\n');
} catch (error) {
  warn('Failed to generate NumPy reference values.');
  console.log('You can generate them manually with:');
  console.log(`  "${venvPython}" scripts/generate_numpy_reference.py\n`);
  process.exit(0);
}
