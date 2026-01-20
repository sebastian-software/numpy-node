# numpy-node

<p align="center">
  <a href="https://numpy.org/"><img src="https://numpy.org/images/logo.svg" alt="NumPy" height="60"></a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <span style="font-size: 2em">+</span>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://nodejs.org/"><img src="https://nodejs.org/static/logos/nodejsLight.svg" alt="Node.js" height="60"></a>
</p>

> [NumPy](https://numpy.org/) for [Node.js](https://nodejs.org/) - Fast, type-safe n-dimensional arrays with native BLAS/LAPACK acceleration

[![npm version](https://img.shields.io/npm/v/numpy-node.svg)](https://www.npmjs.com/package/numpy-node)
[![CI](https://github.com/sebastian-software/numpy-node/actions/workflows/ci.yml/badge.svg)](https://github.com/sebastian-software/numpy-node/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.7-blue.svg)](https://www.typescriptlang.org/)
[![Node.js](https://img.shields.io/badge/Node.js-%3E%3D22-green.svg)](https://nodejs.org/)

**numpy-node** brings the power of [NumPy](https://numpy.org/) to the Node.js ecosystem. Write scientific computing code in TypeScript with a familiar API, backed by native C++ for maximum performance.

## Why numpy-node?

- **Familiar API** - If you know NumPy, you know numpy-node
- **Native Performance** - BLAS/LAPACK acceleration via platform-optimized libraries
- **Type-Safe** - Full TypeScript support with intelligent autocompletion
- **Zero Config** - Prebuilt binaries for all major platforms, no compiler needed
- **Lightweight** - Only installs the binary for your platform (~280KB)

## NumPy Compatibility Guarantee

**All numpy-node functions are verified to be 1:1 compatible with NumPy.**

This means:

- Same function behavior and output values
- Same broadcasting semantics
- Same edge case handling
- Same default parameter values

Every function includes tests verified against actual NumPy output. See [ADR-0005](docs/adr/0005-numpy-compatibility-policy.md) for our compatibility policy.

```typescript
// NumPy: np.greater(np.array([1, 2, 3, 4, 5]), 3) → [False, False, False, True, True]
// numpy-node: identical behavior
const mask = greater(array([1, 2, 3, 4, 5]), 3); // [0, 0, 0, 1, 1] (bool array)
```

## Installation

```bash
npm install numpy-node
# or
pnpm add numpy-node
```

## Quick Start

```typescript
import np, { array, zeros, ones, arange } from 'numpy-node';

// Create arrays
const a = array([
  [1, 2, 3],
  [4, 5, 6],
]);
const b = zeros([3, 3]);
const c = ones([2, 2]);
const d = arange(0, 10, 2); // [0, 2, 4, 6, 8]

// Arithmetic operations (with broadcasting)
const sum = np.add(a, 10);
const product = np.multiply(a, array([1, 2, 3]));

// Linear algebra
import { dot, matmul, inv, solve, svd, qr, eig } from 'numpy-node';

const x = matmul(a, b);
const inverse = inv(
  array([
    [1, 2],
    [3, 4],
  ])
);
const solution = solve(
  array([
    [3, 1],
    [1, 2],
  ]),
  array([9, 8])
);

// Decompositions
const { u, s, vh } = svd(a);
const { q, r } = qr(a);
const { eigenvalues, eigenvectors } = eig(
  array([
    [1, 0],
    [0, 2],
  ])
);

// Statistics
import { mean, std, variance, median, sum, min, max } from 'numpy-node';

const avg = mean(a);
const stdDev = std(a, 0); // along axis 0
const med = median(a);
```

## API Reference

### Array Creation

| Function                      | Description                     |
| ----------------------------- | ------------------------------- |
| `array(data, dtype?)`         | Create array from nested arrays |
| `zeros(shape, dtype?)`        | Array filled with zeros         |
| `ones(shape, dtype?)`         | Array filled with ones          |
| `full(shape, value, dtype?)`  | Array filled with value         |
| `arange(start, stop, step?)`  | Evenly spaced values            |
| `linspace(start, stop, num?)` | Evenly spaced over interval     |
| `eye(n, m?, k?)`              | Identity matrix                 |

### NDArray Methods

| Method                | Description                |
| --------------------- | -------------------------- |
| `reshape(shape)`      | Return reshaped array      |
| `transpose()` / `.T`  | Transpose array            |
| `flatten()`           | Return flattened 1D array  |
| `squeeze(axis?)`      | Remove axes with size 1    |
| `copy()`              | Return copy of array       |
| `at(...indices)`      | Get element at indices     |
| `set(indices, value)` | Set element at indices     |
| `fill(value)`         | Fill array with value      |
| `toArray()`           | Convert to nested JS array |

### Math Operations

| Function                                | Description              |
| --------------------------------------- | ------------------------ |
| `add`, `subtract`, `multiply`, `divide` | Element-wise arithmetic  |
| `power`, `sqrt`, `abs`, `negative`      | Element-wise operations  |
| `exp`, `log`, `sin`, `cos`, `tan`       | Transcendental functions |
| `sum`, `prod`, `min`, `max`             | Reductions               |
| `mean`, `std`, `variance`, `median`     | Statistics               |
| `percentile`, `corrcoef`                | Advanced statistics      |
| `outer`, `kron`                         | Tensor products          |

### Comparison & Logical

| Function                                         | Description              |
| ------------------------------------------------ | ------------------------ |
| `equal`, `not_equal`                             | Element-wise equality    |
| `less`, `less_equal`, `greater`, `greater_equal` | Element-wise comparison  |
| `logical_and`, `logical_or`, `logical_xor`       | Element-wise logical ops |
| `logical_not`                                    | Element-wise negation    |
| `any`, `all`                                     | Boolean reductions       |

### Linear Algebra

| Function         | Description                                 |
| ---------------- | ------------------------------------------- |
| `dot(a, b)`      | Dot product / matrix multiplication         |
| `matmul(a, b)`   | Matrix multiplication                       |
| `inv(a)`         | Matrix inverse                              |
| `det(a)`         | Determinant                                 |
| `solve(a, b)`    | Solve linear system Ax = b                  |
| `eig(a)`         | Eigenvalues and eigenvectors                |
| `eigvals(a)`     | Eigenvalues only                            |
| `svd(a)`         | Singular value decomposition                |
| `qr(a)`          | QR decomposition                            |
| `cholesky(a)`    | Cholesky decomposition                      |
| `norm(a, ord?)`  | Vector/matrix norm (L1, L2, Inf, Frobenius) |
| `matrix_rank(a)` | Matrix rank                                 |
| `cond(a)`        | Condition number                            |
| `trace(a)`       | Matrix trace                                |

## Broadcasting

numpy-node supports NumPy-style broadcasting for element-wise operations:

```typescript
const a = array([
  [1, 2, 3],
  [4, 5, 6],
]); // shape: [2, 3]
const b = array([10, 20, 30]); // shape: [3]
const c = add(a, b); // broadcasts b to match a
// [[11, 22, 33], [14, 25, 36]]

const d = array([[10], [20]]); // shape: [2, 1]
const e = add(a, d); // broadcasts d to match a
// [[11, 12, 13], [24, 25, 26]]
```

## Supported Platforms

| Platform | Architecture | Native Backend   |
| -------- | ------------ | ---------------- |
| macOS    | ARM64 (M1+)  | Apple Accelerate |
| Linux    | x64          | OpenBLAS         |
| Linux    | ARM64        | OpenBLAS         |
| Windows  | x64          | OpenBLAS         |

Prebuilt binaries are included - no compiler required for end users.

## Development

### Prerequisites

- Node.js >= 22.0.0
- pnpm
- Python 3 (for NumPy conformity tests)
- CMake >= 3.15
- C++ compiler (clang, gcc, or MSVC)
- BLAS/LAPACK development libraries

### Setup

```bash
git clone https://github.com/sebastian-software/numpy-node.git
cd numpy-node
pnpm install      # Also creates .venv and installs NumPy for conformity tests
pnpm build:native
pnpm build
pnpm test
```

The `pnpm install` command automatically:

1. Installs Node.js dependencies
2. Creates a Python virtual environment (`.venv`)
3. Installs the latest NumPy in the venv
4. Generates reference values for conformity tests

### Scripts

| Script               | Description                 |
| -------------------- | --------------------------- |
| `pnpm build`         | Build TypeScript            |
| `pnpm build:native`  | Build native C++ module     |
| `pnpm build:all`     | Build everything            |
| `pnpm test`          | Run tests                   |
| `pnpm test:coverage` | Run tests with coverage     |
| `pnpm lint`          | Run ESLint                  |
| `pnpm typecheck`     | Run TypeScript type checker |

### NumPy Conformity Tests

We verify compatibility against the latest NumPy version:

- **CI**: Generates fresh reference values from current NumPy on every run
- **Local**: `pnpm install` sets up a `.venv` with NumPy automatically

To regenerate reference values manually:

```bash
.venv/bin/python3 scripts/generate_numpy_reference.py
```

## Architecture

numpy-node uses a hybrid approach:

- **TypeScript layer** - API, array creation, shape manipulation, broadcasting logic
- **Native C++ layer** - Performance-critical operations via [N-API](https://nodejs.org/api/n-api.html)
- **Platform-optimized BLAS/LAPACK** - [Accelerate](https://developer.apple.com/accelerate/) (macOS), [OpenBLAS](https://www.openblas.net/) (Linux/Windows)

The native binaries are distributed as platform-specific npm packages (`@numpy-node/darwin-arm64`, etc.) and automatically selected via `optionalDependencies`.

For detailed design decisions, see the [Architecture Decision Records](docs/adr/).

## Performance Optimizations

### NumPy (Python) vs numpy-node Comparison

Both libraries use similar low-level optimizations. The table below shows which techniques each library employs:

| Optimization                  | NumPy (Python) | numpy-node | Notes                             |
| ----------------------------- | :------------: | :--------: | --------------------------------- |
| **BLAS/LAPACK Backend**       |                |            |                                   |
| ├─ Apple Accelerate (macOS)   |       ✅       |     ✅     | vecLib, vDSP, LAPACK              |
| ├─ OpenBLAS (Linux/Windows)   |       ✅       |     ✅     | Multi-threaded BLAS               |
| └─ Intel MKL (optional)       |       ✅       |     ❌     | Not yet supported                 |
| **Matrix Operations**         |                |            |                                   |
| ├─ `dgemm` (matmul)           |       ✅       |     ✅     | Level 3 BLAS                      |
| ├─ `dgemv` (matrix-vector)    |       ✅       |     ✅     | Level 2 BLAS                      |
| ├─ `dsyrk` (gram matrix X'X)  |       ✅       |     ✅     | Symmetric rank-k                  |
| ├─ `dger` (rank-1 update)     |       ✅       |     ❌     | Uses vsmul loop                   |
| └─ Batch matmul               |       ✅       |     ✅     | Loop over dgemm                   |
| **Decompositions**            |                |            |                                   |
| ├─ `dgetrf/dgetrs` (LU solve) |       ✅       |     ✅     | LAPACK                            |
| ├─ `dgesvd` (SVD)             |       ✅       |     ✅     | LAPACK                            |
| ├─ `dgeqrf` (QR)              |       ✅       |     ✅     | LAPACK                            |
| ├─ `dpotrf` (Cholesky)        |       ✅       |     ✅     | LAPACK                            |
| └─ `dgeev` (Eigendecomp)      |       ✅       |     ✅     | LAPACK                            |
| **SIMD Vectorization**        |                |            |                                   |
| ├─ vDSP (macOS)               |       ✅       |     ✅     | sum, mean, sqrt, etc.             |
| ├─ AVX/SSE intrinsics         |       ✅       |     ❌     | Via OpenBLAS only                 |
| └─ NEON (ARM)                 |       ✅       |     ✅     | Via Accelerate                    |
| **Fused Operations**          |                |            | numpy-node reduces N-API overhead |
| ├─ `normalize` (z-score)      |       ❌       |     ✅     | (x - μ) / σ                       |
| ├─ `layer_norm`               |       ❌       |     ✅     | With γ, β params                  |
| ├─ `batch_norm`               |       ❌       |     ✅     | Over batch dim                    |
| ├─ `softmax`                  |       ❌       |     ✅     | Numerically stable                |
| ├─ `softmax_cross_entropy`    |       ❌       |     ✅     | Log-sum-exp trick                 |
| ├─ `minmax_scale`             |       ❌       |     ✅     | [0,1] scaling                     |
| ├─ `adam_step`                |       ❌       |     ✅     | Optimizer update                  |
| ├─ `sumsq`                    |       ❌       |     ✅     | Σx² (vDSP_svesqD)                 |
| ├─ `axpby`                    |    partial     |     ✅     | αx + βy                           |
| └─ `muladd`                   |    partial     |     ✅     | a\*b + c                          |
| **Domain-Specific**           |                |            |                                   |
| ├─ `black_scholes`            |       ❌       |     ✅     | Option pricing                    |
| ├─ `tfidf`                    |       ❌       |     ✅     | Text processing                   |
| ├─ `dropout`                  |       ❌       |     ✅     | Neural networks                   |
| ├─ `cross_entropy`            |       ❌       |     ✅     | Loss function                     |
| ├─ `power_iter`               |       ❌       |     ✅     | Eigenvalue approx                 |
| ├─ `batched_attention`        |       ❌       |     ✅     | Transformer attention             |
| └─ `heat_step_2d`             |       ❌       |     ✅     | PDE solver                        |
| **Memory**                    |                |            |                                   |
| ├─ Contiguous arrays          |       ✅       |     ✅     | Row-major (C order)               |
| ├─ View/stride support        |       ✅       |     ✅     | Zero-copy reshape                 |
| ├─ In-place operations        |       ✅       |     ✅     | `add_inplace`, etc.               |
| └─ Memory pooling             |       ✅       |     ❌     | Not implemented                   |

### Benchmark Results

In real-world benchmarks (56 scenarios), numpy-node wins **65-70%** of scenarios:

| Category                  | numpy-node Advantage | Reason                                    |
| ------------------------- | -------------------- | ----------------------------------------- |
| **Fused ML ops**          | 2-3x faster          | Single native call vs multiple Python ops |
| **Loop-heavy algorithms** | 30-300x faster       | Native C++ vs Python interpreter          |
| **Statistics**            | 1.5-2.5x faster      | vDSP vectorization                        |
| **Large reductions**      | 1.5-2x faster        | vDSP_sveD, vDSP_meanvD                    |

| Category                     | NumPy Advantage | Reason                            |
| ---------------------------- | --------------- | --------------------------------- |
| **Small array creation**     | 2-5x faster     | Lower per-operation overhead      |
| **Outer/Kronecker products** | 1.5-2x faster   | Highly optimized C implementation |
| **Pure BLAS ops**            | 1-1.5x faster   | Tighter C integration             |

See `benchmarks/cross-platform/REAL_WORLD_RESULTS.md` for detailed results.

## Security

This package uses [npm Trusted Publishing](https://docs.npmjs.com/generating-provenance-statements) with cryptographic provenance attestation. Every release is built in GitHub Actions and signed, providing a verifiable link between the published package and its source code.

## License

MIT License - see [LICENSE](LICENSE) for details.

Copyright (c) 2025-present [Sebastian Software GmbH](https://sebastian-software.de)

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/amazing-feature`)
3. Use [conventional commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, etc.)
4. Run tests and linting (`pnpm test && pnpm lint`)
5. Submit a pull request

## Related Projects

- [NumPy](https://numpy.org/) - The original Python library
- [ndarray](https://github.com/scijs/ndarray) - Modular n-dimensional arrays for JavaScript
- [tensorflow.js](https://www.tensorflow.org/js) - ML library with tensor operations
