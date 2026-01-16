# numpy-node

A high-performance TypeScript implementation of NumPy with native C++ backend using BLAS/LAPACK.

[![CI](https://github.com/YOUR_USERNAME/numpy-node/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/numpy-node/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/YOUR_USERNAME/numpy-node/branch/main/graph/badge.svg)](https://codecov.io/gh/YOUR_USERNAME/numpy-node)
[![npm version](https://badge.fury.io/js/numpy-node.svg)](https://www.npmjs.com/package/numpy-node)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **NumPy-compatible API** - Familiar interface for Python developers
- **Native performance** - C++ backend with BLAS/LAPACK acceleration
- **Full TypeScript support** - Complete type definitions
- **Cross-platform** - macOS (Accelerate), Linux (OpenBLAS), Windows (OpenBLAS)

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

## Development

### Prerequisites

- Node.js >= 20.0.0
- pnpm
- CMake >= 3.15
- C++ compiler (clang, gcc, or MSVC)

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/numpy-node.git
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

numpy-node uses a native C++ backend for performance-critical operations:

- **macOS**: Apple Accelerate framework (vecLib/BLAS/LAPACK)
- **Linux**: OpenBLAS
- **Windows**: OpenBLAS

See the [Architecture Decision Records](docs/adr/) for design decisions.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request
