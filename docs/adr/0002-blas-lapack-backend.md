# ADR-0002: Platform-Specific BLAS/LAPACK Backends

## Status

Accepted

## Context

BLAS (Basic Linear Algebra Subprograms) and LAPACK (Linear Algebra Package) are the industry standard for numerical linear algebra. Multiple implementations exist with different trade-offs.

Options considered:

1. OpenBLAS everywhere
2. Platform-specific backends
3. Intel MKL
4. Custom implementations

## Decision

Use platform-specific BLAS/LAPACK backends:

- **macOS**: Apple Accelerate framework
- **Linux**: OpenBLAS
- **Windows**: OpenBLAS

Detection is done at compile time via CMake.

## Consequences

### Positive

- **macOS**: Accelerate is pre-installed, optimized for Apple Silicon and Intel
- **Performance**: Each platform uses its best-optimized implementation
- **No runtime dependencies**: Accelerate is part of macOS system
- **Automatic updates**: System libraries updated with OS

### Negative

- Platform-specific code paths in build system
- Testing must cover all platforms
- OpenBLAS must be installed on Linux/Windows (or bundled)
- Different numerical precision characteristics across platforms

## References

- [Apple Accelerate](https://developer.apple.com/documentation/accelerate)
- [OpenBLAS](https://www.openblas.net/)
- [LAPACK](https://www.netlib.org/lapack/)
