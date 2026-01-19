#ifndef NUMPY_TS_EINSUM_H
#define NUMPY_TS_EINSUM_H

#include <napi.h>
#include "ndarray.h"

namespace numpy_node {
namespace einsum {

/**
 * Einstein summation convention.
 *
 * Evaluates the Einstein summation convention on the operands.
 *
 * Examples:
 *   einsum("ij,jk->ik", a, b)  // Matrix multiplication
 *   einsum("ii->i", a)         // Diagonal
 *   einsum("ii", a)            // Trace
 *   einsum("i,i", a, b)        // Inner product
 *   einsum("i,j->ij", a, b)    // Outer product
 */
Napi::Value Einsum(const Napi::CallbackInfo& info);

/**
 * Initialize einsum module
 */
Napi::Object Init(Napi::Env env, Napi::Object exports);

} // namespace einsum
} // namespace numpy_node

#endif // NUMPY_TS_EINSUM_H
