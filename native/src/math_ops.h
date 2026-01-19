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

// Reductions
Napi::Value Sum(const Napi::CallbackInfo& info);
Napi::Value Prod(const Napi::CallbackInfo& info);
Napi::Value Max(const Napi::CallbackInfo& info);
Napi::Value Min(const Napi::CallbackInfo& info);
Napi::Value Mean(const Napi::CallbackInfo& info);
Napi::Value Std(const Napi::CallbackInfo& info);

/**
 * Initialize math module
 */
Napi::Object Init(Napi::Env env, Napi::Object exports);

} // namespace math
} // namespace numpy_node

#endif // NUMPY_TS_MATH_OPS_H
