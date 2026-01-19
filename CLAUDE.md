# Claude Code Context

## Project Overview

numpy-node is a TypeScript NumPy implementation with optional native C++ bindings (N-API).

## Release Workflow

This project uses **Release Please** with **npm Trusted Publishing (OIDC)** for fully automated releases.

### How It Works

1. **Conventional Commits** trigger version bumps:
   - `feat:` → minor bump (0.1.0 → 0.2.0)
   - `fix:` → patch bump (0.1.0 → 0.1.1)
   - `feat!:` or `BREAKING CHANGE:` → major bump

2. **Release Please** (on push to main):
   - Analyzes commits since last release
   - Creates/updates a release PR with changelog
   - When PR is merged → creates GitHub Release

3. **Publish Workflow** (on release published):
   - Builds native modules for all platforms
   - Publishes to npm using OIDC (no tokens needed)

### Key Files

- `release-please-config.json` - Release Please configuration
- `.release-please-manifest.json` - Current version tracking
- `.github/workflows/release-please.yml` - Creates release PRs
- `.github/workflows/publish.yml` - Builds and publishes to npm

## Important Learnings

### npm Trusted Publishing (OIDC)

1. **Requires npm v11+** - Node.js 22 ships with npm v10 which does NOT work. Use Node.js 24 for publish jobs.

2. **Configure on npm for EACH package separately**:
   - Go to npmjs.com → Package → Settings → Trusted Publisher
   - Repository: `owner/repo`
   - Workflow: `publish.yml` (filename only, not full path)

3. **package.json must have `repository` field** for provenance validation:

   ```json
   "repository": {
     "type": "git",
     "url": "https://github.com/owner/repo.git"
   }
   ```

4. **Workflow permissions required**:

   ```yaml
   permissions:
     contents: read
     id-token: write
   ```

5. **Don't use `workflow_call`** for publish workflows - OIDC tokens may contain claims from the calling workflow that don't match npm's expectations. Trigger via `release: [published]` event instead.

### Release Please

1. **Use a PAT (Classic Personal Access Token)** with `repo` scope, not GITHUB_TOKEN. Store as `RELEASE_PLEASE_TOKEN` secret.

2. **If release was created manually**, update the PR label from `autorelease: pending` to `autorelease: tagged` to tell Release Please to move on.

3. **Workflow should NOT call publish.yml** via `workflow_call`. Let the release event trigger publish.yml directly.

## Platform-Specific Packages (for native modules)

Uses esbuild-style distribution with optional dependencies:

- `@numpy-node/darwin-arm64`
- `@numpy-node/linux-x64`
- `@numpy-node/linux-arm64`
- `@numpy-node/win32-x64`

Main package has these as `optionalDependencies` - npm installs only the matching platform.

## CI/CD Notes

- **macOS**: Use `macos-15` (Intel runners retired)
- **Windows LAPACK**: Install `openblas:x64-windows` AND `lapack-reference:x64-windows` separately via vcpkg
- **Windows M_PI**: Add `#define _USE_MATH_DEFINES` before ANY includes

## Commands

```bash
pnpm build          # Build TypeScript
pnpm build:native   # Build native module
pnpm test           # Run tests
pnpm lint           # Lint code
```

## NumPy Compatibility Requirements

**CRITICAL**: All implementations MUST be 1:1 compatible with NumPy. This is the core value proposition of this library.

### Verification Process

1. **Before implementing any NumPy function**: Check the NumPy documentation for exact behavior
2. **Create a Python verification script** to test NumPy's actual output:
   ```python
   import numpy as np
   # Test the function with various inputs
   # Document exact return types, shapes, and edge cases
   ```
3. **Port NumPy's behavior exactly**, including:
   - Broadcasting semantics
   - Return types (dtype)
   - Edge cases (empty arrays, special values like NaN/Inf)
   - Parameter names and defaults

### Key Requirements

- **Function signatures**: Must match NumPy (e.g., `any(a, axis=None, keepdims=False)`)
- **Broadcasting**: All element-wise operations must support NumPy broadcasting rules
- **Return types**: Boolean ops return `dtype='bool'`, reductions without axis return scalars
- **Axis parameter**: Support negative indices, handle all dimensions
- **Edge cases**: Empty arrays, single elements, NaN/Inf handling must match NumPy

### Testing Guidelines

Tests should:

1. Compare output directly against NumPy results
2. Test broadcasting scenarios (scalar, 1D, 2D, N-D)
3. Verify return types and shapes match exactly
4. Include edge cases from NumPy documentation

Example test pattern:

```typescript
// Verified against NumPy:
// >>> np.greater(np.array([1, 2, 3, 4, 5]), 3)
// array([False, False, False,  True,  True])
it('should match NumPy greater()', () => {
  const a = array([1, 2, 3, 4, 5]);
  const result = greater(a, 3);
  expect(result.dtype).toBe('bool');
  expect(Array.from(result.toFlatArray())).toEqual([0, 0, 0, 1, 1]);
});
```

### NumPy Conformity CI

The CI runs automated NumPy conformity checks on every PR:

1. **`scripts/generate_numpy_reference.py`** - Generates `numpy_reference.json` with expected values from actual NumPy
2. **`tests/numpy-conformity.test.ts`** - Compares numpy-node output against reference values
3. **CI Job** - Regenerates reference values from latest NumPy and runs conformity tests

To add new conformity tests:

1. Add test case to `generate_numpy_reference.py`
2. Regenerate: `python3 scripts/generate_numpy_reference.py`
3. Add corresponding test in `numpy-conformity.test.ts`
4. Run tests: `pnpm test -- tests/numpy-conformity.test.ts`

### Architecture Decision Record

See [ADR-0005: NumPy Compatibility Policy](docs/adr/0005-numpy-compatibility-policy.md) for the formal policy.

### Currently Implemented NumPy Functions

#### Array Creation

- `array(data, dtype?)` - Create from nested arrays
- `zeros(shape, dtype?)` - Zeros array
- `ones(shape, dtype?)` - Ones array
- `full(shape, value, dtype?)` - Fill with value
- `arange(start, stop?, step?)` - Evenly spaced values
- `linspace(start, stop, num?)` - Values over interval
- `eye(n, m?, k?)` - Identity/diagonal matrix
- `identity(n, dtype?)` - Identity matrix
- `empty(shape, dtype?)` - Uninitialized array
- `zerosLike(a)`, `onesLike(a)`, `emptyLike(a)` - Like existing array

#### Arithmetic Operations

- `add`, `subtract`, `multiply`, `divide`, `power` - Element-wise with broadcasting
- `add_inplace`, `subtract_inplace`, `multiply_inplace`, `divide_inplace` - In-place operations

#### Unary Math Functions

- `sqrt`, `exp`, `log` - Element-wise math
- `sin`, `cos`, `tan` - Trigonometric
- `abs`, `negative` - Element-wise operations

#### Reduction Operations

- `sum(a, axis?)` - Sum elements
- `prod(a, axis?)` - Product of elements
- `mean(a, axis?)` - Mean value
- `std(a, axis?)` - Standard deviation
- `variance(a, axis?)` - Variance
- `median(a)` - Median value
- `min(a, axis?)`, `max(a, axis?)` - Extrema

#### Advanced Statistics

- `zscore(a, axis?)` - Z-score normalization
- `corrcoef(a)` - Correlation coefficient matrix
- `percentile(a, q, axis?)` - Percentile values

#### Tensor Products

- `outer(a, b)` - Outer product
- `kron(a, b)` - Kronecker product
- `axpby(alpha, x, beta?, y?)` - BLAS-style scalar multiply-add

#### Comparison Operators (return bool arrays)

- `equal`, `not_equal` - Equality
- `less`, `less_equal`, `greater`, `greater_equal` - Comparison

#### Logical Operators (return bool arrays)

- `logical_and`, `logical_or`, `logical_xor` - Binary logical (with broadcasting)
- `logical_not` - Unary logical negation

#### Boolean Reductions

- `any(a, axis?)` - True if any element truthy
- `all(a, axis?)` - True if all elements truthy

#### Linear Algebra

- `matmul(a, b)` - Matrix multiplication
- `matmul_nt(a, b)` - Matmul with B transposed (optimization)
- `batch_matmul(a, b, n)` - Batch matrix multiplication
- `batch_matmul_stacked(a, b)` - Stacked batch matmul
- `dot(a, b)` - Dot product / matrix multiplication
- `inv(a)` - Matrix inverse
- `det(a)` - Determinant
- `solve(a, b)` - Solve linear system Ax = b
- `eig(a)` - Eigenvalues and eigenvectors
- `eigvals(a)` - Eigenvalues only
- `svd(a)` - Singular value decomposition
- `qr(a)` - QR decomposition
- `cholesky(a)` - Cholesky decomposition
- `norm(a, ord?)` - Vector/matrix norm
- `matrix_rank(a)` - Matrix rank
- `trace(a)` - Matrix trace
- `cond(a)` - Condition number
- `lstsq(a, b)` - Least squares solution
- `normal_equations(a, b)` - Normal equations solver

#### Random Number Generation (np.random)

- `random.random(shape?)` - Uniform [0, 1)
- `random.uniform(low, high, shape?)` - Uniform [low, high)
- `random.normal(mean, std, shape?)` - Normal distribution
- `random.randint(low, high, shape?)` - Random integers
- `random.seed(seed)` - Set random seed

### TODO for Full NumPy Compatibility

- Add `keepdims` parameter to `any`/`all`
- Add `axis` parameter to `median`
- Add `where` parameter to reduction functions
- Add `out` parameter for in-place output

### Known Issues

None currently.
