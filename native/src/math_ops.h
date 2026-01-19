#ifndef NUMPY_TS_MATH_OPS_H
#define NUMPY_TS_MATH_OPS_H

#include <napi.h>
#include "ndarray.h"

namespace numpy_node {
namespace math {

// Element-wise arithmetic
Napi::Value Add(const Napi::CallbackInfo& info);
Napi::Value Subtract(const Napi::CallbackInfo& info);
Napi::Value Multiply(const Napi::CallbackInfo& info);
Napi::Value Divide(const Napi::CallbackInfo& info);
Napi::Value Power(const Napi::CallbackInfo& info);

// In-place arithmetic (modifies first argument)
Napi::Value AddInplace(const Napi::CallbackInfo& info);
Napi::Value SubtractInplace(const Napi::CallbackInfo& info);
Napi::Value MultiplyInplace(const Napi::CallbackInfo& info);
Napi::Value DivideInplace(const Napi::CallbackInfo& info);

// Element-wise math functions
Napi::Value Sqrt(const Napi::CallbackInfo& info);
Napi::Value Exp(const Napi::CallbackInfo& info);
Napi::Value Log(const Napi::CallbackInfo& info);
Napi::Value Sin(const Napi::CallbackInfo& info);
Napi::Value Cos(const Napi::CallbackInfo& info);
Napi::Value Tan(const Napi::CallbackInfo& info);
Napi::Value Abs(const Napi::CallbackInfo& info);
Napi::Value Round(const Napi::CallbackInfo& info);
Napi::Value Floor(const Napi::CallbackInfo& info);
Napi::Value Ceil(const Napi::CallbackInfo& info);

// Reductions
Napi::Value Sum(const Napi::CallbackInfo& info);
Napi::Value Prod(const Napi::CallbackInfo& info);
Napi::Value Max(const Napi::CallbackInfo& info);
Napi::Value Min(const Napi::CallbackInfo& info);
Napi::Value Argmax(const Napi::CallbackInfo& info);
Napi::Value Argmin(const Napi::CallbackInfo& info);
Napi::Value Mean(const Napi::CallbackInfo& info);
Napi::Value Std(const Napi::CallbackInfo& info);

// Cumulative operations
Napi::Value Cumsum(const Napi::CallbackInfo& info);
Napi::Value Cumprod(const Napi::CallbackInfo& info);

// Array joining
Napi::Value Concatenate(const Napi::CallbackInfo& info);
Napi::Value Stack(const Napi::CallbackInfo& info);

// Sorting and searching
Napi::Value Diff(const Napi::CallbackInfo& info);
Napi::Value Sort(const Napi::CallbackInfo& info);
Napi::Value Argsort(const Napi::CallbackInfo& info);
Napi::Value Unique(const Napi::CallbackInfo& info);
Napi::Value Searchsorted(const Napi::CallbackInfo& info);

// Array manipulation
Napi::Value Tile(const Napi::CallbackInfo& info);
Napi::Value Repeat(const Napi::CallbackInfo& info);
Napi::Value Flip(const Napi::CallbackInfo& info);
Napi::Value Rot90(const Napi::CallbackInfo& info);
Napi::Value Split(const Napi::CallbackInfo& info);
Napi::Value Nonzero(const Napi::CallbackInfo& info);

// Element-wise math (additional)
Napi::Value Sign(const Napi::CallbackInfo& info);
Napi::Value Mod(const Napi::CallbackInfo& info);

// Approximate comparison
Napi::Value Isclose(const Napi::CallbackInfo& info);
Napi::Value Allclose(const Napi::CallbackInfo& info);

// Array manipulation (existing)
Napi::Value Clip(const Napi::CallbackInfo& info);
Napi::Value Where(const Napi::CallbackInfo& info);
Napi::Value Squeeze(const Napi::CallbackInfo& info);
Napi::Value ExpandDims(const Napi::CallbackInfo& info);

// Comparison operators
Napi::Value Equal(const Napi::CallbackInfo& info);
Napi::Value NotEqual(const Napi::CallbackInfo& info);
Napi::Value Less(const Napi::CallbackInfo& info);
Napi::Value LessEqual(const Napi::CallbackInfo& info);
Napi::Value Greater(const Napi::CallbackInfo& info);
Napi::Value GreaterEqual(const Napi::CallbackInfo& info);

// Logical operators
Napi::Value LogicalAnd(const Napi::CallbackInfo& info);
Napi::Value LogicalOr(const Napi::CallbackInfo& info);
Napi::Value LogicalXor(const Napi::CallbackInfo& info);
Napi::Value LogicalNot(const Napi::CallbackInfo& info);

// Boolean reductions
Napi::Value Any(const Napi::CallbackInfo& info);
Napi::Value All(const Napi::CallbackInfo& info);

// Fused operations (reduce N-API overhead)
Napi::Value Normalize(const Napi::CallbackInfo& info);  // (x - mean) / std
Napi::Value Affine(const Napi::CallbackInfo& info);     // x * scale + bias
Napi::Value MulAdd(const Napi::CallbackInfo& info);     // a * b + c
Napi::Value Softmax(const Napi::CallbackInfo& info);    // numerically stable softmax

/**
 * Initialize math module
 */
Napi::Object Init(Napi::Env env, Napi::Object exports);

} // namespace math
} // namespace numpy_node

#endif // NUMPY_TS_MATH_OPS_H
