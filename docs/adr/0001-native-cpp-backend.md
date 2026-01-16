# ADR-0001: Use Native C++ Backend with N-API

## Status

Accepted

## Context

We need to provide high-performance numerical operations for TypeScript/JavaScript. Pure JavaScript implementations of linear algebra operations are significantly slower than native implementations using optimized BLAS/LAPACK libraries.

Options considered:

1. Pure TypeScript implementation
2. WebAssembly (WASM) compilation
3. Native C++ with Node.js N-API
4. Native C++ with node-gyp bindings

## Decision

Use native C++ backend with Node.js N-API (via node-addon-api).

Key factors:

- **Performance**: Direct access to hardware-optimized BLAS/LAPACK
- **Stability**: N-API provides ABI stability across Node.js versions
- **Ecosystem**: node-addon-api provides clean C++ abstractions
- **Build tooling**: cmake-js simplifies cross-platform builds

## Consequences

### Positive

- Near-native performance for numerical operations
- Access to mature, battle-tested BLAS/LAPACK implementations
- ABI stability means prebuilt binaries work across Node.js versions
- Can leverage platform-specific optimizations (Apple Accelerate, Intel MKL)

### Negative

- Requires C++ compiler for building from source
- Cross-platform builds are more complex
- Debugging native code is harder than pure JavaScript
- Binary distribution requires prebuilds for each platform/architecture

## References

- [Node.js N-API Documentation](https://nodejs.org/api/n-api.html)
- [node-addon-api](https://github.com/nodejs/node-addon-api)
- [cmake-js](https://github.com/nicknisi/cmake-js)
