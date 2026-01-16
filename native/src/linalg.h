#ifndef NUMPY_TS_LINALG_H
#define NUMPY_TS_LINALG_H

#include <napi.h>
#include "ndarray.h"

namespace numpy_node {
namespace linalg {

/**
 * Matrix multiplication (matmul)
 * Supports 2D matrices and batched operations
 */
Napi::Value Matmul(const Napi::CallbackInfo& info);

/**
 * Dot product
 * - For 1D arrays: inner product
 * - For 2D arrays: matrix multiplication
 * - For higher dimensions: sum product over last axes
 */
Napi::Value Dot(const Napi::CallbackInfo& info);

/**
 * Compute the determinant of a matrix
 */
Napi::Value Det(const Napi::CallbackInfo& info);

/**
 * Compute the inverse of a matrix
 */
Napi::Value Inv(const Napi::CallbackInfo& info);

/**
 * Solve a linear system Ax = b
 */
Napi::Value Solve(const Napi::CallbackInfo& info);

/**
 * Compute eigenvalues and eigenvectors
 */
Napi::Value Eig(const Napi::CallbackInfo& info);

/**
 * Compute eigenvalues only
 */
Napi::Value Eigvals(const Napi::CallbackInfo& info);

/**
 * Singular Value Decomposition
 */
Napi::Value Svd(const Napi::CallbackInfo& info);

/**
 * QR decomposition
 */
Napi::Value Qr(const Napi::CallbackInfo& info);

/**
 * Cholesky decomposition
 */
Napi::Value Cholesky(const Napi::CallbackInfo& info);

/**
 * Compute the norm of a vector or matrix
 */
Napi::Value Norm(const Napi::CallbackInfo& info);

/**
 * Compute the matrix rank
 */
Napi::Value MatrixRank(const Napi::CallbackInfo& info);

/**
 * Compute the trace of a matrix
 */
Napi::Value Trace(const Napi::CallbackInfo& info);

/**
 * Initialize linalg module
 */
Napi::Object Init(Napi::Env env, Napi::Object exports);

} // namespace linalg
} // namespace numpy_node

#endif // NUMPY_TS_LINALG_H
