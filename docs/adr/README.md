# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for numpy-node.

## What is an ADR?

An ADR is a document that captures an important architectural decision made along with its context and consequences.

## Index

| ADR                                     | Title                                  | Status   |
| --------------------------------------- | -------------------------------------- | -------- |
| [0000](0000-template.md)                | ADR Template                           | -        |
| [0001](0001-native-cpp-backend.md)      | Use Native C++ Backend with N-API      | Accepted |
| [0002](0002-blas-lapack-backend.md)     | Platform-Specific BLAS/LAPACK Backends | Accepted |
| [0003](0003-row-major-memory-layout.md) | Row-Major Memory Layout                | Accepted |
| [0004](0004-cpu-based-computation.md)   | CPU-Based Computation (No GPU)         | Accepted |

## Creating a New ADR

1. Copy `0000-template.md` to `XXXX-title.md` (next sequential number)
2. Fill in the template sections
3. Update this README with the new ADR
4. Submit a PR for review

## References

- [ADR GitHub Organization](https://adr.github.io/)
- [Michael Nygard's article](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
