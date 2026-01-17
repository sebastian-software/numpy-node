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
- CMake >= 3.15
- C++ compiler (clang, gcc, or MSVC)
- BLAS/LAPACK development libraries

### Setup

```bash
git clone https://github.com/sebastian-software/numpy-node.git
cd numpy-node
pnpm install
pnpm build:native
pnpm build
pnpm test
```

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

## Architecture

numpy-node uses a hybrid approach:

- **TypeScript layer** - API, array creation, shape manipulation, broadcasting logic
- **Native C++ layer** - Performance-critical operations via [N-API](https://nodejs.org/api/n-api.html)
- **Platform-optimized BLAS/LAPACK** - [Accelerate](https://developer.apple.com/accelerate/) (macOS), [OpenBLAS](https://www.openblas.net/) (Linux/Windows)

The native binaries are distributed as platform-specific npm packages (`@numpy-node/darwin-arm64`, etc.) and automatically selected via `optionalDependencies`.

For detailed design decisions, see the [Architecture Decision Records](docs/adr/).

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
