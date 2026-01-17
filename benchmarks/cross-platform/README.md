# NumPy vs numpy-node Benchmark

This directory contains benchmarks comparing NumPy (Python) with numpy-node (Node.js/TypeScript).

## Prerequisites

### Python

```bash
pip install numpy
```

### Node.js

```bash
# From project root
pnpm install
pnpm build:native
pnpm build
```

## Running the Benchmark

### Full Comparison

```bash
npx tsx benchmarks/cross-platform/run_comparison.ts
```

This will:

1. Run the NumPy (Python) benchmark
2. Run the numpy-node benchmark
3. Print a comparison table
4. Generate a `RESULTS.md` file

### Individual Benchmarks

```bash
# Python only
python3 benchmarks/cross-platform/numpy_benchmark.py

# Node.js only
npx tsx benchmarks/cross-platform/numpy_node_benchmark.ts
```

## Benchmark Categories

| Category              | Description                                      |
| --------------------- | ------------------------------------------------ |
| Array Creation        | `zeros`, `ones`, `arange`, `linspace`            |
| Element-wise Ops      | `add`, `multiply`, `sqrt`, `exp`                 |
| Reductions            | `sum`, `mean`, `std`, `min`, `max`               |
| Linear Algebra (BLAS) | `matmul`, `dot`, `inv`, `det`, `svd`, `qr`, etc. |
| Loop Overhead         | Many small operations in a loop                  |

## What to Expect

- **BLAS/LAPACK operations** (matmul, svd, etc.): Similar performance, as both use the same underlying libraries (Accelerate on macOS, OpenBLAS on Linux/Windows)

- **Small operations & loops**: numpy-node should be faster due to V8's JIT compilation vs. Python's interpreter

- **Array creation**: May vary depending on memory allocation patterns

## Sample Output

```
================================================================================
BENCHMARK COMPARISON: NumPy (Python) vs numpy-node (Node.js)
================================================================================
NumPy version: 1.26.4
Node.js version: v22.0.0
================================================================================

Benchmark                       NumPy    numpy-node  Speedup
----------------------------------------------------------------------
zeros(1000x1000)              0.42ms       0.38ms  1.11x faster
matmul(500x500)              12.34ms      12.51ms  same
1000x small array ops        45.23ms       8.12ms  5.57x faster
```
