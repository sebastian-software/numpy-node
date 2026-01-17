# ADR-0004: CPU-Based Computation (No GPU/Neural Engine)

## Status

Accepted

## Context

Modern Apple Silicon chips offer multiple compute units:

- **CPU** - General purpose, with NEON SIMD and AMX (Apple Matrix coprocessor)
- **GPU** - Massively parallel, accessible via Metal/MPS
- **Neural Engine** - Specialized for ML inference, accessible via Core ML/MLX

The question arose whether numpy-node should leverage GPU or Neural Engine for better performance.

### Options Considered

1. **CPU only via Accelerate** (current approach)
2. **GPU via Metal Performance Shaders (MPS)**
3. **GPU via MLX** (Apple's ML framework)
4. **Hybrid CPU/GPU** with automatic selection

## Decision

Use **CPU-based computation only** via platform-optimized BLAS/LAPACK libraries (Accelerate on macOS, OpenBLAS on Linux/Windows).

## Rationale

### 1. NumPy's Design Philosophy

NumPy is fundamentally a **synchronous, CPU-based** library. Its API assumes:

- Immediate results (no futures/promises)
- Direct memory access to array data
- Predictable, deterministic execution

GPU computation would require async patterns that break API compatibility.

### 2. Data Transfer Overhead

GPU acceleration only benefits large operations:

| Array Size        | CPU        | GPU (including transfer)    |
| ----------------- | ---------- | --------------------------- |
| < 10K elements    | Faster     | Slower (transfer dominates) |
| 10K - 1M elements | Comparable | Comparable                  |
| > 1M elements     | Slower     | Faster                      |

Typical NumPy usage involves many small-to-medium operations where CPU wins.

### 3. Accelerate is Already Highly Optimized

Apple's Accelerate framework on Apple Silicon:

- Uses **AMX** (Apple Matrix coprocessor) for matrix operations
- Uses **NEON SIMD** for vectorized operations
- Is tuned specifically for M1/M2/M3/M4 cache hierarchies
- Requires zero configuration or installation

For CPU-bound linear algebra, Accelerate approaches theoretical peak performance.

### 4. GPU Alternatives Exist

Users needing GPU acceleration have mature options:

- **TensorFlow.js** - GPU-accelerated tensors via WebGL/WebGPU
- **ONNX Runtime** - Cross-platform ML inference with GPU support
- **Custom Metal/CUDA** - For specialized workloads

These serve different use cases (ML, large-scale compute) with appropriate APIs.

### 5. Implementation Complexity

GPU support would require:

- Async API or worker threads
- Memory management (GPU buffers)
- Fallback logic for unsupported operations
- Platform-specific Metal/CUDA/OpenCL code
- Significant testing matrix expansion

This complexity contradicts our goal of a simple, NumPy-compatible API.

## Consequences

### Positive

- Simple, synchronous API matching NumPy
- No GPU driver dependencies
- Predictable performance characteristics
- Works in all Node.js environments (including containers)
- Smaller binary size

### Negative

- Not optimal for very large array operations (>1M elements)
- Users needing GPU must use different libraries
- Cannot leverage Neural Engine for supported operations

## Alternatives for GPU Users

| Use Case                | Recommended Library         |
| ----------------------- | --------------------------- |
| ML inference            | TensorFlow.js, ONNX Runtime |
| Large matrix operations | TensorFlow.js               |
| Custom GPU compute      | Direct Metal/WebGPU         |
| Apple Neural Engine     | Core ML (Swift/ObjC only)   |

## References

- [Apple Accelerate Framework](https://developer.apple.com/documentation/accelerate)
- [Apple AMX Coprocessor](https://github.com/corsix/amx)
- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)
- [MLX by Apple](https://github.com/ml-explore/mlx)
- [TensorFlow.js](https://www.tensorflow.org/js)
