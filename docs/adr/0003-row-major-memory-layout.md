# ADR-0003: Row-Major Memory Layout

## Status

Accepted

## Context

Multi-dimensional arrays can be stored in memory using different layouts:

- **Row-major (C-style)**: Elements in the same row are contiguous
- **Column-major (Fortran-style)**: Elements in the same column are contiguous

NumPy uses row-major by default. BLAS/LAPACK traditionally use column-major (Fortran heritage).

## Decision

Use row-major (C-style) memory layout internally, matching NumPy's default behavior.

For BLAS/LAPACK calls:

- Use CBLAS interface with `CblasRowMajor` flag where available
- For LAPACK routines, transpose data before/after calls when necessary
- Store strides in bytes (not elements) in the native layer

## Consequences

### Positive

- Matches NumPy's default behavior for easier porting
- Cache-friendly for row-wise iteration (common in many algorithms)
- CBLAS provides row-major support for most operations
- JavaScript TypedArrays are row-major compatible

### Negative

- LAPACK routines expect column-major; requires transposition
- Some operations need temporary copies for layout conversion
- Mixed column-major/row-major code can be error-prone

## References

- [NumPy Memory Layout](https://numpy.org/doc/stable/reference/arrays.ndarray.html#internal-memory-layout-of-an-ndarray)
- [CBLAS Row-Major Support](https://www.netlib.org/blas/blast-forum/cinterface.pdf)
