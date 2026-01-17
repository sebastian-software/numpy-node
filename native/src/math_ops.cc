#include "math_ops.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>

#if defined(USE_ACCELERATE)
    #include <Accelerate/Accelerate.h>
#endif

namespace numpy_node {
namespace math {

// Compute broadcast shape from two input shapes
// Returns empty vector if shapes are not broadcastable
static std::vector<int64_t> computeBroadcastShape(
    const std::vector<int64_t>& shapeA,
    const std::vector<int64_t>& shapeB
) {
    size_t ndimA = shapeA.size();
    size_t ndimB = shapeB.size();
    size_t ndimResult = std::max(ndimA, ndimB);

    std::vector<int64_t> result(ndimResult);

    for (size_t i = 0; i < ndimResult; i++) {
        // Get dimensions from right (end) of each shape
        int64_t dimA = (i < ndimA) ? shapeA[ndimA - 1 - i] : 1;
        int64_t dimB = (i < ndimB) ? shapeB[ndimB - 1 - i] : 1;

        if (dimA == dimB) {
            result[ndimResult - 1 - i] = dimA;
        } else if (dimA == 1) {
            result[ndimResult - 1 - i] = dimB;
        } else if (dimB == 1) {
            result[ndimResult - 1 - i] = dimA;
        } else {
            // Not broadcastable
            return {};
        }
    }

    return result;
}

// Compute flat index for a given multi-dimensional index with broadcasting
static int64_t computeBroadcastIndex(
    const std::vector<int64_t>& resultIndices,
    const std::vector<int64_t>& shape
) {
    size_t ndim = shape.size();
    size_t resultNdim = resultIndices.size();

    int64_t flatIndex = 0;
    int64_t multiplier = 1;

    for (size_t i = 0; i < ndim; i++) {
        size_t dimIdx = ndim - 1 - i;
        size_t resultDimIdx = resultNdim - 1 - i;

        // If this dimension is 1 (broadcast), use index 0
        int64_t idx = (shape[dimIdx] == 1) ? 0 : resultIndices[resultDimIdx];
        flatIndex += idx * multiplier;
        multiplier *= shape[dimIdx];
    }

    return flatIndex;
}

// Helper template for element-wise binary operations with broadcasting
template<typename Op>
Napi::Value BinaryOp(const Napi::CallbackInfo& info, Op op) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected two arguments").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    bool bIsScalar = info[1].IsNumber();

    double* dataA = static_cast<double*>(a->data());
    const auto& shapeA = a->shape();

    if (bIsScalar) {
        // Scalar operation - no broadcasting needed
        Napi::Array jsShape = Napi::Array::New(env, shapeA.size());
        for (size_t i = 0; i < shapeA.size(); i++) {
            jsShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(shapeA[i])));
        }

        // Use skipInit=true since we'll overwrite all values
        Napi::Object result = NativeNDArray::constructor.New({
            jsShape,
            Napi::String::New(env, "float64"),
            Napi::Boolean::New(env, true)  // skipInit
        });
        NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
        double* dataC = static_cast<double*>(c->data());

        double scalar = info[1].As<Napi::Number>().DoubleValue();
        for (int64_t i = 0; i < a->size(); i++) {
            dataC[i] = op(dataA[i], scalar);
        }
        return result;
    }

    NativeNDArray* b = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());
    double* dataB = static_cast<double*>(b->data());
    const auto& shapeB = b->shape();

    // Fast path: same shape, no broadcasting needed
    if (shapeA == shapeB) {
        Napi::Array jsShape = Napi::Array::New(env, shapeA.size());
        for (size_t i = 0; i < shapeA.size(); i++) {
            jsShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(shapeA[i])));
        }

        // Use skipInit=true since we'll overwrite all values
        Napi::Object result = NativeNDArray::constructor.New({
            jsShape,
            Napi::String::New(env, "float64"),
            Napi::Boolean::New(env, true)  // skipInit
        });
        NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
        double* dataC = static_cast<double*>(c->data());

        for (int64_t i = 0; i < a->size(); i++) {
            dataC[i] = op(dataA[i], dataB[i]);
        }
        return result;
    }

    // Fast path: 2D array with 1D row vector [m, n] op [n]
    if (shapeA.size() == 2 && shapeB.size() == 1 && shapeA[1] == shapeB[0]) {
        int64_t m = shapeA[0];
        int64_t n = shapeA[1];

        Napi::Array jsShape = Napi::Array::New(env, 2);
        jsShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(m)));
        jsShape.Set(uint32_t(1), Napi::Number::New(env, static_cast<double>(n)));

        Napi::Object result = NativeNDArray::constructor.New({
            jsShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
        });
        NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
        double* dataC = static_cast<double*>(c->data());

        // Apply row vector to each row
        for (int64_t i = 0; i < m; i++) {
            for (int64_t j = 0; j < n; j++) {
                dataC[i * n + j] = op(dataA[i * n + j], dataB[j]);
            }
        }
        return result;
    }

    // Fast path: 2D array with column vector [m, n] op [m, 1]
    if (shapeA.size() == 2 && shapeB.size() == 2 &&
        shapeB[0] == shapeA[0] && shapeB[1] == 1) {
        int64_t m = shapeA[0];
        int64_t n = shapeA[1];

        Napi::Array jsShape = Napi::Array::New(env, 2);
        jsShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(m)));
        jsShape.Set(uint32_t(1), Napi::Number::New(env, static_cast<double>(n)));

        Napi::Object result = NativeNDArray::constructor.New({
            jsShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
        });
        NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
        double* dataC = static_cast<double*>(c->data());

        // Apply column vector to each column
        for (int64_t i = 0; i < m; i++) {
            double bVal = dataB[i];
            for (int64_t j = 0; j < n; j++) {
                dataC[i * n + j] = op(dataA[i * n + j], bVal);
            }
        }
        return result;
    }

    // Fast path: 1D vector with 2D array [n] op [m, n]
    if (shapeA.size() == 1 && shapeB.size() == 2 && shapeA[0] == shapeB[1]) {
        int64_t m = shapeB[0];
        int64_t n = shapeB[1];

        Napi::Array jsShape = Napi::Array::New(env, 2);
        jsShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(m)));
        jsShape.Set(uint32_t(1), Napi::Number::New(env, static_cast<double>(n)));

        Napi::Object result = NativeNDArray::constructor.New({
            jsShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
        });
        NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
        double* dataC = static_cast<double*>(c->data());

        // Apply row vector to each row
        for (int64_t i = 0; i < m; i++) {
            for (int64_t j = 0; j < n; j++) {
                dataC[i * n + j] = op(dataA[j], dataB[i * n + j]);
            }
        }
        return result;
    }

    // Generic broadcasting path (slower, but handles all cases)
    std::vector<int64_t> resultShape = computeBroadcastShape(shapeA, shapeB);

    if (resultShape.empty()) {
        Napi::Error::New(env, "Shapes are not broadcastable").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Create result array with skipInit
    Napi::Array jsResultShape = Napi::Array::New(env, resultShape.size());
    for (size_t i = 0; i < resultShape.size(); i++) {
        jsResultShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(resultShape[i])));
    }

    Napi::Object result = NativeNDArray::constructor.New({
        jsResultShape,
        Napi::String::New(env, "float64"),
        Napi::Boolean::New(env, true)  // skipInit
    });
    NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* dataC = static_cast<double*>(c->data());

    // Compute total size
    int64_t totalSize = 1;
    for (int64_t dim : resultShape) {
        totalSize *= dim;
    }

    // Pad shapes with leading 1s to match result dimensions
    std::vector<int64_t> paddedShapeA(resultShape.size(), 1);
    std::vector<int64_t> paddedShapeB(resultShape.size(), 1);

    for (size_t i = 0; i < shapeA.size(); i++) {
        paddedShapeA[resultShape.size() - shapeA.size() + i] = shapeA[i];
    }
    for (size_t i = 0; i < shapeB.size(); i++) {
        paddedShapeB[resultShape.size() - shapeB.size() + i] = shapeB[i];
    }

    // Iterate over all result elements
    std::vector<int64_t> indices(resultShape.size(), 0);

    for (int64_t flatIdx = 0; flatIdx < totalSize; flatIdx++) {
        // Compute multi-dimensional indices
        int64_t temp = flatIdx;
        for (int i = static_cast<int>(resultShape.size()) - 1; i >= 0; i--) {
            indices[i] = temp % resultShape[i];
            temp /= resultShape[i];
        }

        // Get indices in original arrays (with broadcasting)
        int64_t idxA = computeBroadcastIndex(indices, paddedShapeA);
        int64_t idxB = computeBroadcastIndex(indices, paddedShapeB);

        dataC[flatIdx] = op(dataA[idxA], dataB[idxB]);
    }

    return result;
}

// Helper template for element-wise unary operations
template<typename Op>
Napi::Value UnaryOp(const Napi::CallbackInfo& info, Op op) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    double* dataA = static_cast<double*>(a->data());
    int64_t size = a->size();

    Napi::Array shape = Napi::Array::New(env, a->shape().size());
    for (size_t i = 0; i < a->shape().size(); i++) {
        shape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(a->shape()[i])));
    }

    // Use skipInit=true since we'll overwrite all values
    Napi::Object result = NativeNDArray::constructor.New({
        shape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
    });
    NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* dataC = static_cast<double*>(c->data());

    for (int64_t i = 0; i < size; i++) {
        dataC[i] = op(dataA[i]);
    }

    return result;
}

Napi::Value Add(const Napi::CallbackInfo& info) {
#if defined(USE_ACCELERATE)
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected two arguments").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());

    // Scalar - use vDSP
    if (info[1].IsNumber()) {
        double scalar = info[1].As<Napi::Number>().DoubleValue();

        Napi::Array shape = Napi::Array::New(env, a->shape().size());
        for (size_t i = 0; i < a->shape().size(); i++) {
            shape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(a->shape()[i])));
        }

        Napi::Object result = NativeNDArray::constructor.New({
            shape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
        });
        NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);

        vDSP_vsaddD(static_cast<double*>(a->data()), 1, &scalar,
                    static_cast<double*>(c->data()), 1, a->size());
        return result;
    }

    NativeNDArray* b = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());

    // Same shape - use vDSP
    if (a->shape() == b->shape()) {
        Napi::Array shape = Napi::Array::New(env, a->shape().size());
        for (size_t i = 0; i < a->shape().size(); i++) {
            shape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(a->shape()[i])));
        }

        Napi::Object result = NativeNDArray::constructor.New({
            shape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
        });
        NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);

        vDSP_vaddD(static_cast<double*>(a->data()), 1,
                   static_cast<double*>(b->data()), 1,
                   static_cast<double*>(c->data()), 1, a->size());
        return result;
    }

    // Different shapes - fall back to broadcasting implementation
    return BinaryOp(info, [](double x, double y) { return x + y; });
#else
    return BinaryOp(info, [](double a, double b) { return a + b; });
#endif
}

Napi::Value Subtract(const Napi::CallbackInfo& info) {
#if defined(USE_ACCELERATE)
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected two arguments").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());

    // Scalar - use vDSP
    if (info[1].IsNumber()) {
        Napi::Array shape = Napi::Array::New(env, a->shape().size());
        for (size_t i = 0; i < a->shape().size(); i++) {
            shape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(a->shape()[i])));
        }

        Napi::Object result = NativeNDArray::constructor.New({
            shape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
        });
        NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);

        double scalar = -info[1].As<Napi::Number>().DoubleValue();
        vDSP_vsaddD(static_cast<double*>(a->data()), 1, &scalar,
                    static_cast<double*>(c->data()), 1, a->size());
        return result;
    }

    NativeNDArray* b = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());
    const auto& shapeA = a->shape();
    const auto& shapeB = b->shape();

    // Same shape - use vDSP
    if (shapeA == shapeB) {
        Napi::Array shape = Napi::Array::New(env, shapeA.size());
        for (size_t i = 0; i < shapeA.size(); i++) {
            shape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(shapeA[i])));
        }

        Napi::Object result = NativeNDArray::constructor.New({
            shape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
        });
        NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);

        vDSP_vsubD(static_cast<double*>(b->data()), 1,
                   static_cast<double*>(a->data()), 1,
                   static_cast<double*>(c->data()), 1, a->size());
        return result;
    }

    // Fast path: 2D - 1D broadcasting (common in ML: subtract mean from each row)
    // Shape (M, N) - (N,) -> broadcast along rows
    if (shapeA.size() == 2 && shapeB.size() == 1 && shapeA[1] == shapeB[0]) {
        int64_t rows = shapeA[0];
        int64_t cols = shapeA[1];
        double* dataA = static_cast<double*>(a->data());
        double* dataB = static_cast<double*>(b->data());

        Napi::Array jsShape = Napi::Array::New(env, 2);
        jsShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(rows)));
        jsShape.Set(uint32_t(1), Napi::Number::New(env, static_cast<double>(cols)));

        Napi::Object result = NativeNDArray::constructor.New({
            jsShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
        });
        NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
        double* dataC = static_cast<double*>(c->data());

        // Subtract the 1D vector from each row
        for (int64_t row = 0; row < rows; row++) {
            vDSP_vsubD(dataB, 1, dataA + row * cols, 1, dataC + row * cols, 1, cols);
        }
        return result;
    }

    // Different shapes - fall back to broadcasting implementation
    return BinaryOp(info, [](double x, double y) { return x - y; });
#else
    return BinaryOp(info, [](double a, double b) { return a - b; });
#endif
}

Napi::Value Multiply(const Napi::CallbackInfo& info) {
#if defined(USE_ACCELERATE)
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected two arguments").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());

    // Scalar - use vDSP
    if (info[1].IsNumber()) {
        Napi::Array shape = Napi::Array::New(env, a->shape().size());
        for (size_t i = 0; i < a->shape().size(); i++) {
            shape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(a->shape()[i])));
        }

        Napi::Object result = NativeNDArray::constructor.New({
            shape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
        });
        NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);

        double scalar = info[1].As<Napi::Number>().DoubleValue();
        vDSP_vsmulD(static_cast<double*>(a->data()), 1, &scalar,
                    static_cast<double*>(c->data()), 1, a->size());
        return result;
    }

    NativeNDArray* b = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());

    // Same shape - use vDSP
    if (a->shape() == b->shape()) {
        Napi::Array shape = Napi::Array::New(env, a->shape().size());
        for (size_t i = 0; i < a->shape().size(); i++) {
            shape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(a->shape()[i])));
        }

        Napi::Object result = NativeNDArray::constructor.New({
            shape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
        });
        NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);

        vDSP_vmulD(static_cast<double*>(a->data()), 1,
                   static_cast<double*>(b->data()), 1,
                   static_cast<double*>(c->data()), 1, a->size());
        return result;
    }

    // Different shapes - fall back to broadcasting implementation
    return BinaryOp(info, [](double x, double y) { return x * y; });
#else
    return BinaryOp(info, [](double a, double b) { return a * b; });
#endif
}

Napi::Value Divide(const Napi::CallbackInfo& info) {
#if defined(USE_ACCELERATE)
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected two arguments").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());

    // Scalar - use vDSP
    if (info[1].IsNumber()) {
        Napi::Array shape = Napi::Array::New(env, a->shape().size());
        for (size_t i = 0; i < a->shape().size(); i++) {
            shape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(a->shape()[i])));
        }

        Napi::Object result = NativeNDArray::constructor.New({
            shape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
        });
        NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);

        double scalar = info[1].As<Napi::Number>().DoubleValue();
        vDSP_vsdivD(static_cast<double*>(a->data()), 1, &scalar,
                    static_cast<double*>(c->data()), 1, a->size());
        return result;
    }

    NativeNDArray* b = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());
    const auto& shapeA = a->shape();
    const auto& shapeB = b->shape();

    // Same shape - use vDSP
    if (shapeA == shapeB) {
        Napi::Array shape = Napi::Array::New(env, shapeA.size());
        for (size_t i = 0; i < shapeA.size(); i++) {
            shape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(shapeA[i])));
        }

        Napi::Object result = NativeNDArray::constructor.New({
            shape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
        });
        NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);

        vDSP_vdivD(static_cast<double*>(b->data()), 1,
                   static_cast<double*>(a->data()), 1,
                   static_cast<double*>(c->data()), 1, a->size());
        return result;
    }

    // Fast path: 2D / 1D broadcasting (common in ML: divide by std for each row)
    // Shape (M, N) / (N,) -> broadcast along rows
    if (shapeA.size() == 2 && shapeB.size() == 1 && shapeA[1] == shapeB[0]) {
        int64_t rows = shapeA[0];
        int64_t cols = shapeA[1];
        double* dataA = static_cast<double*>(a->data());
        double* dataB = static_cast<double*>(b->data());

        Napi::Array jsShape = Napi::Array::New(env, 2);
        jsShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(rows)));
        jsShape.Set(uint32_t(1), Napi::Number::New(env, static_cast<double>(cols)));

        Napi::Object result = NativeNDArray::constructor.New({
            jsShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
        });
        NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
        double* dataC = static_cast<double*>(c->data());

        // Divide each row by the 1D vector
        for (int64_t row = 0; row < rows; row++) {
            vDSP_vdivD(dataB, 1, dataA + row * cols, 1, dataC + row * cols, 1, cols);
        }
        return result;
    }

    // Different shapes - fall back to broadcasting implementation
    return BinaryOp(info, [](double x, double y) { return x / y; });
#else
    return BinaryOp(info, [](double a, double b) { return a / b; });
#endif
}

Napi::Value Power(const Napi::CallbackInfo& info) {
    return BinaryOp(info, [](double a, double b) { return std::pow(a, b); });
}

// Vectorized unary operations using Accelerate/vecLib on macOS
#if defined(USE_ACCELERATE)
// Helper for vecLib unary operations
template<typename VecLibFn>
Napi::Value VecLibUnaryOp(const Napi::CallbackInfo& info, VecLibFn vecFn) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    double* dataA = static_cast<double*>(a->data());
    int n = static_cast<int>(a->size());

    Napi::Array shape = Napi::Array::New(env, a->shape().size());
    for (size_t i = 0; i < a->shape().size(); i++) {
        shape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(a->shape()[i])));
    }

    Napi::Object result = NativeNDArray::constructor.New({
        shape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
    });
    NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* dataC = static_cast<double*>(c->data());

    vecFn(dataC, dataA, &n);

    return result;
}
#endif

Napi::Value Sqrt(const Napi::CallbackInfo& info) {
#if defined(USE_ACCELERATE)
    return VecLibUnaryOp(info, vvsqrt);
#else
    return UnaryOp(info, [](double a) { return std::sqrt(a); });
#endif
}

Napi::Value Exp(const Napi::CallbackInfo& info) {
#if defined(USE_ACCELERATE)
    return VecLibUnaryOp(info, vvexp);
#else
    return UnaryOp(info, [](double a) { return std::exp(a); });
#endif
}

Napi::Value Log(const Napi::CallbackInfo& info) {
#if defined(USE_ACCELERATE)
    return VecLibUnaryOp(info, vvlog);
#else
    return UnaryOp(info, [](double a) { return std::log(a); });
#endif
}

Napi::Value Sin(const Napi::CallbackInfo& info) {
#if defined(USE_ACCELERATE)
    return VecLibUnaryOp(info, vvsin);
#else
    return UnaryOp(info, [](double a) { return std::sin(a); });
#endif
}

Napi::Value Cos(const Napi::CallbackInfo& info) {
#if defined(USE_ACCELERATE)
    return VecLibUnaryOp(info, vvcos);
#else
    return UnaryOp(info, [](double a) { return std::cos(a); });
#endif
}

Napi::Value Tan(const Napi::CallbackInfo& info) {
#if defined(USE_ACCELERATE)
    return VecLibUnaryOp(info, vvtan);
#else
    return UnaryOp(info, [](double a) { return std::tan(a); });
#endif
}

Napi::Value Abs(const Napi::CallbackInfo& info) {
    return UnaryOp(info, [](double a) { return std::abs(a); });
}

// Helper function to compute flat index from multi-dimensional indices
static int64_t computeFlatIndex(const std::vector<int64_t>& indices, const std::vector<int64_t>& shape) {
    int64_t flatIndex = 0;
    int64_t multiplier = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; i--) {
        flatIndex += indices[i] * multiplier;
        multiplier *= shape[i];
    }
    return flatIndex;
}

// Reductions
Napi::Value Sum(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    double* data = static_cast<double*>(a->data());
    int64_t size = a->size();

    // Check for axis parameter
    if (info.Length() >= 2 && !info[1].IsUndefined()) {
        int axis = info[1].As<Napi::Number>().Int32Value();
        const auto& shape = a->shape();
        int ndim = static_cast<int>(shape.size());

        if (axis < 0 || axis >= ndim) {
            Napi::Error::New(env, "Axis out of bounds").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        // Fast path for 2D arrays (most common case in ML/data science)
        if (ndim == 2) {
            int64_t rows = shape[0];
            int64_t cols = shape[1];

            if (axis == 0) {
                // Sum along rows -> result has shape [cols]
                Napi::Array jsResultShape = Napi::Array::New(env, 1);
                jsResultShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(cols)));

                Napi::Object result = NativeNDArray::constructor.New({
                    jsResultShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
                });
                NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
                double* resultData = static_cast<double*>(resultArr->data());

                // Initialize and accumulate using direct indexing
                std::memset(resultData, 0, cols * sizeof(double));
                for (int64_t row = 0; row < rows; row++) {
                    double* rowPtr = data + row * cols;
#if defined(USE_ACCELERATE)
                    vDSP_vaddD(resultData, 1, rowPtr, 1, resultData, 1, cols);
#else
                    for (int64_t col = 0; col < cols; col++) {
                        resultData[col] += rowPtr[col];
                    }
#endif
                }
                return result;

            } else { // axis == 1
                // Sum along columns -> result has shape [rows]
                Napi::Array jsResultShape = Napi::Array::New(env, 1);
                jsResultShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(rows)));

                Napi::Object result = NativeNDArray::constructor.New({
                    jsResultShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
                });
                NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
                double* resultData = static_cast<double*>(resultArr->data());

                // Sum each row using vDSP
                for (int64_t row = 0; row < rows; row++) {
                    double* rowPtr = data + row * cols;
#if defined(USE_ACCELERATE)
                    vDSP_sveD(rowPtr, 1, &resultData[row], cols);
#else
                    double sum = 0.0;
                    for (int64_t col = 0; col < cols; col++) {
                        sum += rowPtr[col];
                    }
                    resultData[row] = sum;
#endif
                }
                return result;
            }
        }

        // Generic N-dimensional case (fallback)
        // Compute result shape (remove the axis dimension)
        std::vector<int64_t> resultShape;
        for (int i = 0; i < ndim; i++) {
            if (i != axis) {
                resultShape.push_back(shape[i]);
            }
        }

        // Create result array
        Napi::Array jsResultShape = Napi::Array::New(env, resultShape.size());
        for (size_t i = 0; i < resultShape.size(); i++) {
            jsResultShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(resultShape[i])));
        }

        Napi::Object result = NativeNDArray::constructor.New({jsResultShape, Napi::String::New(env, "float64")});
        NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
        double* resultData = static_cast<double*>(resultArr->data());

        // Initialize result to zero
        std::fill(resultData, resultData + resultArr->size(), 0.0);

        // Iterate over all elements and accumulate
        std::vector<int64_t> indices(ndim, 0);
        for (int64_t flatIdx = 0; flatIdx < size; flatIdx++) {
            // Compute multi-dimensional indices from flat index
            int64_t temp = flatIdx;
            for (int i = ndim - 1; i >= 0; i--) {
                indices[i] = temp % shape[i];
                temp /= shape[i];
            }

            // Compute result index (indices without the axis dimension)
            std::vector<int64_t> resultIndices;
            for (int i = 0; i < ndim; i++) {
                if (i != axis) {
                    resultIndices.push_back(indices[i]);
                }
            }

            int64_t resultFlatIdx = computeFlatIndex(resultIndices, resultShape);
            resultData[resultFlatIdx] += data[flatIdx];
        }

        return result;
    }

    // No axis - reduce over all elements
    double sum = 0.0;

#if defined(USE_ACCELERATE)
    vDSP_sveD(data, 1, &sum, size);
#else
    for (int64_t i = 0; i < size; i++) {
        sum += data[i];
    }
#endif

    return Napi::Number::New(env, sum);
}

Napi::Value Prod(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    double* data = static_cast<double*>(a->data());
    int64_t size = a->size();

    double prod = 1.0;
    for (int64_t i = 0; i < size; i++) {
        prod *= data[i];
    }

    return Napi::Number::New(env, prod);
}

Napi::Value Max(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    double* data = static_cast<double*>(a->data());
    int64_t size = a->size();

    // Check for axis parameter
    if (info.Length() >= 2 && !info[1].IsUndefined()) {
        int axis = info[1].As<Napi::Number>().Int32Value();
        const auto& shape = a->shape();
        int ndim = static_cast<int>(shape.size());

        if (axis < 0 || axis >= ndim) {
            Napi::Error::New(env, "Axis out of bounds").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        // Fast path for 2D arrays
        if (ndim == 2) {
            int64_t rows = shape[0];
            int64_t cols = shape[1];

            if (axis == 0) {
                // Max along rows -> result has shape [cols]
                Napi::Array jsResultShape = Napi::Array::New(env, 1);
                jsResultShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(cols)));

                Napi::Object result = NativeNDArray::constructor.New({
                    jsResultShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
                });
                NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
                double* resultData = static_cast<double*>(resultArr->data());

                // Initialize with first row
                std::memcpy(resultData, data, cols * sizeof(double));

                // Find max across rows
                for (int64_t row = 1; row < rows; row++) {
                    double* rowPtr = data + row * cols;
                    for (int64_t col = 0; col < cols; col++) {
                        if (rowPtr[col] > resultData[col]) {
                            resultData[col] = rowPtr[col];
                        }
                    }
                }
                return result;

            } else { // axis == 1
                // Max along columns -> result has shape [rows]
                Napi::Array jsResultShape = Napi::Array::New(env, 1);
                jsResultShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(rows)));

                Napi::Object result = NativeNDArray::constructor.New({
                    jsResultShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
                });
                NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
                double* resultData = static_cast<double*>(resultArr->data());

                for (int64_t row = 0; row < rows; row++) {
                    double* rowPtr = data + row * cols;
#if defined(USE_ACCELERATE)
                    vDSP_maxvD(rowPtr, 1, &resultData[row], cols);
#else
                    double maxVal = rowPtr[0];
                    for (int64_t col = 1; col < cols; col++) {
                        if (rowPtr[col] > maxVal) maxVal = rowPtr[col];
                    }
                    resultData[row] = maxVal;
#endif
                }
                return result;
            }
        }

        // Generic N-dim case: fall back to global max for now
        Napi::Error::New(env, "Axis reduction only supported for 2D arrays").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // No axis - global max
    double maxVal = -std::numeric_limits<double>::infinity();

#if defined(USE_ACCELERATE)
    vDSP_maxvD(data, 1, &maxVal, size);
#else
    for (int64_t i = 0; i < size; i++) {
        if (data[i] > maxVal) maxVal = data[i];
    }
#endif

    return Napi::Number::New(env, maxVal);
}

Napi::Value Min(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    double* data = static_cast<double*>(a->data());
    int64_t size = a->size();

    // Check for axis parameter
    if (info.Length() >= 2 && !info[1].IsUndefined()) {
        int axis = info[1].As<Napi::Number>().Int32Value();
        const auto& shape = a->shape();
        int ndim = static_cast<int>(shape.size());

        if (axis < 0 || axis >= ndim) {
            Napi::Error::New(env, "Axis out of bounds").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        // Fast path for 2D arrays
        if (ndim == 2) {
            int64_t rows = shape[0];
            int64_t cols = shape[1];

            if (axis == 0) {
                // Min along rows -> result has shape [cols]
                Napi::Array jsResultShape = Napi::Array::New(env, 1);
                jsResultShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(cols)));

                Napi::Object result = NativeNDArray::constructor.New({
                    jsResultShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
                });
                NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
                double* resultData = static_cast<double*>(resultArr->data());

                // Initialize with first row
                std::memcpy(resultData, data, cols * sizeof(double));

                // Find min across rows
                for (int64_t row = 1; row < rows; row++) {
                    double* rowPtr = data + row * cols;
                    for (int64_t col = 0; col < cols; col++) {
                        if (rowPtr[col] < resultData[col]) {
                            resultData[col] = rowPtr[col];
                        }
                    }
                }
                return result;

            } else { // axis == 1
                // Min along columns -> result has shape [rows]
                Napi::Array jsResultShape = Napi::Array::New(env, 1);
                jsResultShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(rows)));

                Napi::Object result = NativeNDArray::constructor.New({
                    jsResultShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
                });
                NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
                double* resultData = static_cast<double*>(resultArr->data());

                for (int64_t row = 0; row < rows; row++) {
                    double* rowPtr = data + row * cols;
#if defined(USE_ACCELERATE)
                    vDSP_minvD(rowPtr, 1, &resultData[row], cols);
#else
                    double minVal = rowPtr[0];
                    for (int64_t col = 1; col < cols; col++) {
                        if (rowPtr[col] < minVal) minVal = rowPtr[col];
                    }
                    resultData[row] = minVal;
#endif
                }
                return result;
            }
        }

        // Generic N-dim case: fall back to global min for now
        Napi::Error::New(env, "Axis reduction only supported for 2D arrays").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // No axis - global min
    double minVal = std::numeric_limits<double>::infinity();

#if defined(USE_ACCELERATE)
    vDSP_minvD(data, 1, &minVal, size);
#else
    for (int64_t i = 0; i < size; i++) {
        if (data[i] < minVal) minVal = data[i];
    }
#endif

    return Napi::Number::New(env, minVal);
}

Napi::Value Mean(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    double* data = static_cast<double*>(a->data());
    int64_t size = a->size();

    // Check for axis parameter
    if (info.Length() >= 2 && !info[1].IsUndefined()) {
        int axis = info[1].As<Napi::Number>().Int32Value();
        const auto& shape = a->shape();
        int ndim = static_cast<int>(shape.size());

        if (axis < 0 || axis >= ndim) {
            Napi::Error::New(env, "Axis out of bounds").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        int64_t axisSize = shape[axis];

        // Fast path for 2D arrays (most common case in ML/data science)
        if (ndim == 2) {
            int64_t rows = shape[0];
            int64_t cols = shape[1];

            if (axis == 0) {
                // Mean along rows -> result has shape [cols]
                Napi::Array jsResultShape = Napi::Array::New(env, 1);
                jsResultShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(cols)));

                Napi::Object result = NativeNDArray::constructor.New({
                    jsResultShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
                });
                NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
                double* resultData = static_cast<double*>(resultArr->data());

                // Initialize and accumulate using direct indexing
                std::memset(resultData, 0, cols * sizeof(double));
                for (int64_t row = 0; row < rows; row++) {
                    double* rowPtr = data + row * cols;
#if defined(USE_ACCELERATE)
                    vDSP_vaddD(resultData, 1, rowPtr, 1, resultData, 1, cols);
#else
                    for (int64_t col = 0; col < cols; col++) {
                        resultData[col] += rowPtr[col];
                    }
#endif
                }

                // Divide by row count to get mean
                double divisor = static_cast<double>(rows);
#if defined(USE_ACCELERATE)
                vDSP_vsdivD(resultData, 1, &divisor, resultData, 1, cols);
#else
                for (int64_t col = 0; col < cols; col++) {
                    resultData[col] /= divisor;
                }
#endif
                return result;

            } else { // axis == 1
                // Mean along columns -> result has shape [rows]
                Napi::Array jsResultShape = Napi::Array::New(env, 1);
                jsResultShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(rows)));

                Napi::Object result = NativeNDArray::constructor.New({
                    jsResultShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
                });
                NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
                double* resultData = static_cast<double*>(resultArr->data());

                // Mean of each row using vDSP
                for (int64_t row = 0; row < rows; row++) {
                    double* rowPtr = data + row * cols;
#if defined(USE_ACCELERATE)
                    vDSP_meanvD(rowPtr, 1, &resultData[row], cols);
#else
                    double sum = 0.0;
                    for (int64_t col = 0; col < cols; col++) {
                        sum += rowPtr[col];
                    }
                    resultData[row] = sum / cols;
#endif
                }
                return result;
            }
        }

        // Generic N-dimensional case (fallback)
        // Compute result shape (remove the axis dimension)
        std::vector<int64_t> resultShape;
        for (int i = 0; i < ndim; i++) {
            if (i != axis) {
                resultShape.push_back(shape[i]);
            }
        }

        // Create result array
        Napi::Array jsResultShape = Napi::Array::New(env, resultShape.size());
        for (size_t i = 0; i < resultShape.size(); i++) {
            jsResultShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(resultShape[i])));
        }

        Napi::Object result = NativeNDArray::constructor.New({jsResultShape, Napi::String::New(env, "float64")});
        NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
        double* resultData = static_cast<double*>(resultArr->data());

        // Initialize result to zero
        std::fill(resultData, resultData + resultArr->size(), 0.0);

        // Iterate over all elements and accumulate
        std::vector<int64_t> indices(ndim, 0);
        for (int64_t flatIdx = 0; flatIdx < size; flatIdx++) {
            // Compute multi-dimensional indices from flat index
            int64_t temp = flatIdx;
            for (int i = ndim - 1; i >= 0; i--) {
                indices[i] = temp % shape[i];
                temp /= shape[i];
            }

            // Compute result index (indices without the axis dimension)
            std::vector<int64_t> resultIndices;
            for (int i = 0; i < ndim; i++) {
                if (i != axis) {
                    resultIndices.push_back(indices[i]);
                }
            }

            int64_t resultFlatIdx = computeFlatIndex(resultIndices, resultShape);
            resultData[resultFlatIdx] += data[flatIdx];
        }

        // Divide by axis size to get mean
        for (int64_t i = 0; i < resultArr->size(); i++) {
            resultData[i] /= axisSize;
        }

        return result;
    }

    // No axis - reduce over all elements
    double mean = 0.0;

#if defined(USE_ACCELERATE)
    vDSP_meanvD(data, 1, &mean, size);
#else
    for (int64_t i = 0; i < size; i++) {
        mean += data[i];
    }
    mean /= size;
#endif

    return Napi::Number::New(env, mean);
}

Napi::Value Std(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    double* data = static_cast<double*>(a->data());
    int64_t size = a->size();

    // Check for axis parameter
    if (info.Length() >= 2 && !info[1].IsUndefined()) {
        int axis = info[1].As<Napi::Number>().Int32Value();
        const auto& shape = a->shape();
        int ndim = static_cast<int>(shape.size());

        if (axis < 0 || axis >= ndim) {
            Napi::Error::New(env, "Axis out of bounds").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        int64_t axisSize = shape[axis];

        // Fast path for 2D arrays (most common case in ML/data science)
        if (ndim == 2) {
            int64_t rows = shape[0];
            int64_t cols = shape[1];

            if (axis == 0) {
                // Std along rows -> result has shape [cols]
                Napi::Array jsResultShape = Napi::Array::New(env, 1);
                jsResultShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(cols)));

                Napi::Object result = NativeNDArray::constructor.New({
                    jsResultShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
                });
                NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
                double* resultData = static_cast<double*>(resultArr->data());

                // First compute means using direct indexing
                std::vector<double> meanData(cols, 0.0);
                for (int64_t row = 0; row < rows; row++) {
                    double* rowPtr = data + row * cols;
#if defined(USE_ACCELERATE)
                    vDSP_vaddD(meanData.data(), 1, rowPtr, 1, meanData.data(), 1, cols);
#else
                    for (int64_t col = 0; col < cols; col++) {
                        meanData[col] += rowPtr[col];
                    }
#endif
                }

                // Divide by row count to get mean
                double divisor = static_cast<double>(rows);
#if defined(USE_ACCELERATE)
                vDSP_vsdivD(meanData.data(), 1, &divisor, meanData.data(), 1, cols);
#else
                for (int64_t col = 0; col < cols; col++) {
                    meanData[col] /= divisor;
                }
#endif

                // Now compute variance using direct indexing
                std::memset(resultData, 0, cols * sizeof(double));
                for (int64_t row = 0; row < rows; row++) {
                    double* rowPtr = data + row * cols;
                    for (int64_t col = 0; col < cols; col++) {
                        double diff = rowPtr[col] - meanData[col];
                        resultData[col] += diff * diff;
                    }
                }

                // Divide by rows and take sqrt
                for (int64_t col = 0; col < cols; col++) {
                    resultData[col] = std::sqrt(resultData[col] / rows);
                }

                return result;

            } else { // axis == 1
                // Std along columns -> result has shape [rows]
                Napi::Array jsResultShape = Napi::Array::New(env, 1);
                jsResultShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(rows)));

                Napi::Object result = NativeNDArray::constructor.New({
                    jsResultShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
                });
                NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
                double* resultData = static_cast<double*>(resultArr->data());

                // Compute std of each row
                for (int64_t row = 0; row < rows; row++) {
                    double* rowPtr = data + row * cols;

                    // Compute mean of this row
                    double rowMean = 0.0;
#if defined(USE_ACCELERATE)
                    vDSP_meanvD(rowPtr, 1, &rowMean, cols);
#else
                    for (int64_t col = 0; col < cols; col++) {
                        rowMean += rowPtr[col];
                    }
                    rowMean /= cols;
#endif

                    // Compute variance of this row
                    double rowVar = 0.0;
                    for (int64_t col = 0; col < cols; col++) {
                        double diff = rowPtr[col] - rowMean;
                        rowVar += diff * diff;
                    }
                    resultData[row] = std::sqrt(rowVar / cols);
                }

                return result;
            }
        }

        // Generic N-dimensional case (fallback)
        // Compute result shape (remove the axis dimension)
        std::vector<int64_t> resultShape;
        for (int i = 0; i < ndim; i++) {
            if (i != axis) {
                resultShape.push_back(shape[i]);
            }
        }

        // Create result array
        Napi::Array jsResultShape = Napi::Array::New(env, resultShape.size());
        for (size_t i = 0; i < resultShape.size(); i++) {
            jsResultShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(resultShape[i])));
        }

        Napi::Object result = NativeNDArray::constructor.New({jsResultShape, Napi::String::New(env, "float64")});
        NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
        double* resultData = static_cast<double*>(resultArr->data());

        // First compute mean along axis
        std::vector<double> meanData(resultArr->size(), 0.0);

        std::vector<int64_t> indices(ndim, 0);
        for (int64_t flatIdx = 0; flatIdx < size; flatIdx++) {
            // Compute multi-dimensional indices from flat index
            int64_t temp = flatIdx;
            for (int i = ndim - 1; i >= 0; i--) {
                indices[i] = temp % shape[i];
                temp /= shape[i];
            }

            // Compute result index (indices without the axis dimension)
            std::vector<int64_t> resultIndices;
            for (int i = 0; i < ndim; i++) {
                if (i != axis) {
                    resultIndices.push_back(indices[i]);
                }
            }

            int64_t resultFlatIdx = computeFlatIndex(resultIndices, resultShape);
            meanData[resultFlatIdx] += data[flatIdx];
        }

        // Divide by axis size to get mean
        for (size_t i = 0; i < meanData.size(); i++) {
            meanData[i] /= axisSize;
        }

        // Now compute variance
        std::fill(resultData, resultData + resultArr->size(), 0.0);

        for (int64_t flatIdx = 0; flatIdx < size; flatIdx++) {
            // Compute multi-dimensional indices from flat index
            int64_t temp = flatIdx;
            for (int i = ndim - 1; i >= 0; i--) {
                indices[i] = temp % shape[i];
                temp /= shape[i];
            }

            // Compute result index (indices without the axis dimension)
            std::vector<int64_t> resultIndices;
            for (int i = 0; i < ndim; i++) {
                if (i != axis) {
                    resultIndices.push_back(indices[i]);
                }
            }

            int64_t resultFlatIdx = computeFlatIndex(resultIndices, resultShape);
            double diff = data[flatIdx] - meanData[resultFlatIdx];
            resultData[resultFlatIdx] += diff * diff;
        }

        // Divide by axis size and take sqrt
        for (int64_t i = 0; i < resultArr->size(); i++) {
            resultData[i] = std::sqrt(resultData[i] / axisSize);
        }

        return result;
    }

    // No axis - reduce over all elements
    // Compute mean
    double mean = 0.0;
    for (int64_t i = 0; i < size; i++) {
        mean += data[i];
    }
    mean /= size;

    // Compute variance
    double var = 0.0;
    for (int64_t i = 0; i < size; i++) {
        double diff = data[i] - mean;
        var += diff * diff;
    }
    var /= size;

    return Napi::Number::New(env, std::sqrt(var));
}

Napi::Value Var(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    double* data = static_cast<double*>(a->data());
    int64_t size = a->size();

    // Check for axis parameter
    if (info.Length() >= 2 && !info[1].IsUndefined()) {
        int axis = info[1].As<Napi::Number>().Int32Value();
        const auto& shape = a->shape();
        int ndim = static_cast<int>(shape.size());

        if (axis < 0 || axis >= ndim) {
            Napi::Error::New(env, "Axis out of bounds").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        int64_t axisSize = shape[axis];

        std::vector<int64_t> resultShape;
        for (int i = 0; i < ndim; i++) {
            if (i != axis) {
                resultShape.push_back(shape[i]);
            }
        }

        Napi::Array jsResultShape = Napi::Array::New(env, resultShape.size());
        for (size_t i = 0; i < resultShape.size(); i++) {
            jsResultShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(resultShape[i])));
        }

        Napi::Object result = NativeNDArray::constructor.New({jsResultShape, Napi::String::New(env, "float64")});
        NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
        double* resultData = static_cast<double*>(resultArr->data());

        // First compute mean along axis
        std::vector<double> meanData(resultArr->size(), 0.0);

        std::vector<int64_t> indices(ndim, 0);
        for (int64_t flatIdx = 0; flatIdx < size; flatIdx++) {
            int64_t temp = flatIdx;
            for (int i = ndim - 1; i >= 0; i--) {
                indices[i] = temp % shape[i];
                temp /= shape[i];
            }

            std::vector<int64_t> resultIndices;
            for (int i = 0; i < ndim; i++) {
                if (i != axis) {
                    resultIndices.push_back(indices[i]);
                }
            }

            int64_t resultFlatIdx = computeFlatIndex(resultIndices, resultShape);
            meanData[resultFlatIdx] += data[flatIdx];
        }

        for (size_t i = 0; i < meanData.size(); i++) {
            meanData[i] /= axisSize;
        }

        // Now compute variance
        std::fill(resultData, resultData + resultArr->size(), 0.0);

        for (int64_t flatIdx = 0; flatIdx < size; flatIdx++) {
            int64_t temp = flatIdx;
            for (int i = ndim - 1; i >= 0; i--) {
                indices[i] = temp % shape[i];
                temp /= shape[i];
            }

            std::vector<int64_t> resultIndices;
            for (int i = 0; i < ndim; i++) {
                if (i != axis) {
                    resultIndices.push_back(indices[i]);
                }
            }

            int64_t resultFlatIdx = computeFlatIndex(resultIndices, resultShape);
            double diff = data[flatIdx] - meanData[resultFlatIdx];
            resultData[resultFlatIdx] += diff * diff;
        }

        // Divide by axis size (no sqrt for variance)
        for (int64_t i = 0; i < resultArr->size(); i++) {
            resultData[i] = resultData[i] / axisSize;
        }

        return result;
    }

    // No axis - reduce over all elements
    double mean = 0.0;
    for (int64_t i = 0; i < size; i++) {
        mean += data[i];
    }
    mean /= size;

    double var = 0.0;
    for (int64_t i = 0; i < size; i++) {
        double diff = data[i] - mean;
        var += diff * diff;
    }
    var /= size;

    return Napi::Number::New(env, var);
}

Napi::Value Median(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    double* data = static_cast<double*>(a->data());
    int64_t size = a->size();

    // Copy data and sort
    std::vector<double> sorted(data, data + size);
    std::sort(sorted.begin(), sorted.end());

    // Compute median
    double median;
    if (size % 2 == 0) {
        median = (sorted[size / 2 - 1] + sorted[size / 2]) / 2.0;
    } else {
        median = sorted[size / 2];
    }

    return Napi::Number::New(env, median);
}

/**
 * Fused z-score normalization: (X - mean) / std along axis
 * Computes mean, std, and normalizes in a single native call.
 */
Napi::Value Zscore(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    double* data = static_cast<double*>(a->data());
    const auto& shape = a->shape();
    int ndim = static_cast<int>(shape.size());

    // Default axis = 0 (normalize columns)
    int axis = 0;
    if (info.Length() > 1 && info[1].IsNumber()) {
        axis = info[1].As<Napi::Number>().Int32Value();
        if (axis < 0) axis += ndim;
    }

    if (ndim != 2) {
        Napi::Error::New(env, "zscore currently only supports 2D arrays").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int64_t rows = shape[0];
    int64_t cols = shape[1];

    // Create result array with same shape
    Napi::Array jsShape = Napi::Array::New(env, 2);
    jsShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(rows)));
    jsShape.Set(uint32_t(1), Napi::Number::New(env, static_cast<double>(cols)));

    Napi::Object result = NativeNDArray::constructor.New({
        jsShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
    });
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* resultData = static_cast<double*>(resultArr->data());

    if (axis == 0) {
        // Normalize columns: each column has mean 0, std 1
        // Cache-friendly: iterate row-by-row for all passes

        // Pre-allocate per-column statistics
        std::vector<double> colMean(cols, 0.0);
        std::vector<double> colVar(cols, 0.0);
        std::vector<double> colInvStd(cols);

        // Pass 1: compute column sums (row-by-row for cache efficiency)
        for (int64_t i = 0; i < rows; i++) {
            for (int64_t j = 0; j < cols; j++) {
                colMean[j] += data[i * cols + j];
            }
        }
        // Convert to means
        for (int64_t j = 0; j < cols; j++) {
            colMean[j] /= rows;
        }

        // Pass 2: compute variance sums (row-by-row)
        for (int64_t i = 0; i < rows; i++) {
            for (int64_t j = 0; j < cols; j++) {
                double diff = data[i * cols + j] - colMean[j];
                colVar[j] += diff * diff;
            }
        }
        // Convert to inverse std
        for (int64_t j = 0; j < cols; j++) {
            double std = std::sqrt(colVar[j] / rows);
            colInvStd[j] = (std > 1e-10) ? 1.0 / std : 0.0;
        }

        // Pass 3: normalize (row-by-row)
        for (int64_t i = 0; i < rows; i++) {
            for (int64_t j = 0; j < cols; j++) {
                resultData[i * cols + j] = (data[i * cols + j] - colMean[j]) * colInvStd[j];
            }
        }
    } else {
        // Normalize rows: each row has mean 0, std 1
        for (int64_t i = 0; i < rows; i++) {
            double sum = 0.0;
            for (int64_t j = 0; j < cols; j++) {
                sum += data[i * cols + j];
            }
            double mean = sum / cols;

            double varSum = 0.0;
            for (int64_t j = 0; j < cols; j++) {
                double diff = data[i * cols + j] - mean;
                varSum += diff * diff;
            }
            double std = std::sqrt(varSum / cols);
            double invStd = (std > 1e-10) ? 1.0 / std : 0.0;

            for (int64_t j = 0; j < cols; j++) {
                resultData[i * cols + j] = (data[i * cols + j] - mean) * invStd;
            }
        }
    }

    return result;
}

/**
 * Fused correlation matrix: standardize and compute X'X / (n-1) in one call
 * Uses dsyrk for efficient symmetric matrix computation.
 */
Napi::Value Corrcoef(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    double* data = static_cast<double*>(a->data());
    const auto& shape = a->shape();

    if (shape.size() != 2) {
        Napi::Error::New(env, "corrcoef requires 2D array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int m = static_cast<int>(shape[0]);  // samples
    int n = static_cast<int>(shape[1]);  // variables

    // Step 1: Standardize data (mean=0, std=1 per column)
    std::vector<double> standardized(m * n);

    for (int j = 0; j < n; j++) {
        // Compute mean
        double sum = 0.0;
        for (int i = 0; i < m; i++) {
            sum += data[i * n + j];
        }
        double mean = sum / m;

        // Compute sample std (divide by m-1 for consistency with correlation formula)
        double varSum = 0.0;
        for (int i = 0; i < m; i++) {
            double diff = data[i * n + j] - mean;
            varSum += diff * diff;
        }
        double std = std::sqrt(varSum / (m - 1));
        double invStd = (std > 1e-10) ? 1.0 / std : 0.0;

        // Standardize
        for (int i = 0; i < m; i++) {
            standardized[i * n + j] = (data[i * n + j] - mean) * invStd;
        }
    }

    // Step 2: Compute correlation matrix = X'X / (m-1) using dsyrk
    Napi::Array jsShape = Napi::Array::New(env, 2);
    jsShape.Set(uint32_t(0), Napi::Number::New(env, n));
    jsShape.Set(uint32_t(1), Napi::Number::New(env, n));

    Napi::Object result = NativeNDArray::constructor.New({
        jsShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
    });
    NativeNDArray* corrArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* corrData = static_cast<double*>(corrArr->data());

#if defined(USE_ACCELERATE)
    // Use dsyrk: C = alpha * A' * A + beta * C
    double alpha = 1.0 / (m - 1);
    double beta = 0.0;

    cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                n, m, alpha, standardized.data(), n, beta, corrData, n);

    // Fill lower triangle
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            corrData[i * n + j] = corrData[j * n + i];
        }
    }
#else
    // Pure C++ fallback
    std::memset(corrData, 0, n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < m; k++) {
                sum += standardized[k * n + i] * standardized[k * n + j];
            }
            corrData[i * n + j] = sum / (m - 1);
            corrData[j * n + i] = corrData[i * n + j];
        }
    }
#endif

    return result;
}

/**
 * Gram matrix: X @ X.T
 * Uses dsyrk for efficient symmetric matrix computation.
 */
Napi::Value GramMatrix(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    double* data = static_cast<double*>(a->data());
    const auto& shape = a->shape();

    if (shape.size() != 2) {
        Napi::Error::New(env, "gram_matrix requires 2D array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int m = static_cast<int>(shape[0]);  // rows
    int n = static_cast<int>(shape[1]);  // cols

    // Result is m x m (X @ X.T)
    Napi::Array jsShape = Napi::Array::New(env, 2);
    jsShape.Set(uint32_t(0), Napi::Number::New(env, m));
    jsShape.Set(uint32_t(1), Napi::Number::New(env, m));

    Napi::Object result = NativeNDArray::constructor.New({
        jsShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
    });
    NativeNDArray* gramArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* gramData = static_cast<double*>(gramArr->data());

#if defined(USE_ACCELERATE)
    // dsyrk: C = alpha * A * A' + beta * C (when CblasNoTrans)
    // For row-major X (m x n), A * A' gives m x m
    cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans,
                m, n, 1.0, data, n, 0.0, gramData, m);

    // Fill lower triangle
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < i; j++) {
            gramData[i * m + j] = gramData[j * m + i];
        }
    }
#else
    // Pure C++ fallback: X @ X.T
    for (int i = 0; i < m; i++) {
        for (int j = i; j < m; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += data[i * n + k] * data[j * n + k];
            }
            gramData[i * m + j] = sum;
            gramData[j * m + i] = sum;
        }
    }
#endif

    return result;
}

/**
 * Fused softmax: exp(x) / sum(exp(x))
 * Computes softmax in a single native call with numerical stability.
 */
Napi::Value Softmax(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    double* data = static_cast<double*>(a->data());
    const auto& shape = a->shape();
    int64_t size = a->size();

    // Create result array with same shape
    Napi::Array jsShape = Napi::Array::New(env, shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
        jsShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(shape[i])));
    }

    Napi::Object result = NativeNDArray::constructor.New({
        jsShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
    });
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* resultData = static_cast<double*>(resultArr->data());

    // Find max for numerical stability (subtract max before exp)
    double maxVal = data[0];
    for (int64_t i = 1; i < size; i++) {
        if (data[i] > maxVal) maxVal = data[i];
    }

    // Compute exp(x - max) and sum
    double sumExp = 0.0;
    for (int64_t i = 0; i < size; i++) {
        resultData[i] = std::exp(data[i] - maxVal);
        sumExp += resultData[i];
    }

    // Normalize
    double invSum = 1.0 / sumExp;
    for (int64_t i = 0; i < size; i++) {
        resultData[i] *= invSum;
    }

    return result;
}

/**
 * Pairwise squared Euclidean distances: D_ij = ||x_i - x_j||^2
 * Fuses row norms, gram matrix, and combination in one call.
 * Uses: ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i · x_j
 */
Napi::Value PdistSq(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    double* data = static_cast<double*>(a->data());
    const auto& shape = a->shape();

    if (shape.size() != 2) {
        Napi::Error::New(env, "pdist_sq requires 2D array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int m = static_cast<int>(shape[0]);  // points
    int n = static_cast<int>(shape[1]);  // dimensions

    // Step 1: Compute row norms ||x_i||^2
    std::vector<double> rowNorms(m);
#if defined(USE_ACCELERATE)
    for (int i = 0; i < m; i++) {
        vDSP_dotprD(data + i * n, 1, data + i * n, 1, &rowNorms[i], n);
    }
#else
    for (int i = 0; i < m; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            double val = data[i * n + j];
            sum += val * val;
        }
        rowNorms[i] = sum;
    }
#endif

    // Step 2: Compute gram matrix (dot products) and combine
    // Result D_ij = ||x_i||^2 + ||x_j||^2 - 2 * x_i · x_j
    Napi::Array jsShape = Napi::Array::New(env, 2);
    jsShape.Set(uint32_t(0), Napi::Number::New(env, m));
    jsShape.Set(uint32_t(1), Napi::Number::New(env, m));

    Napi::Object result = NativeNDArray::constructor.New({
        jsShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
    });
    NativeNDArray* distArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* distData = static_cast<double*>(distArr->data());

#if defined(USE_ACCELERATE)
    // First compute gram matrix using dsyrk: G = X @ X.T
    cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans,
                m, n, 1.0, data, n, 0.0, distData, m);

    // Now combine: D_ij = rowNorms[i] + rowNorms[j] - 2 * G_ij
    // Only upper triangle was computed by dsyrk
    for (int i = 0; i < m; i++) {
        for (int j = i; j < m; j++) {
            double d = rowNorms[i] + rowNorms[j] - 2.0 * distData[i * m + j];
            // Clamp to 0 to handle numerical precision issues
            distData[i * m + j] = (d > 0.0) ? d : 0.0;
            distData[j * m + i] = distData[i * m + j];
        }
    }
#else
    // Pure C++ fallback
    for (int i = 0; i < m; i++) {
        for (int j = i; j < m; j++) {
            double dotprod = 0.0;
            for (int k = 0; k < n; k++) {
                dotprod += data[i * n + k] * data[j * n + k];
            }
            double d = rowNorms[i] + rowNorms[j] - 2.0 * dotprod;
            distData[i * m + j] = (d > 0.0) ? d : 0.0;
            distData[j * m + i] = distData[i * m + j];
        }
    }
#endif

    return result;
}

/**
 * Affine transform: gamma * x + beta (fused multiply-add with broadcasting)
 * Common in layer normalization and batch normalization.
 * Input: x [m, n], gamma [n], beta [n]
 * Output: [m, n] where result[i, j] = gamma[j] * x[i, j] + beta[j]
 */
Napi::Value Affine(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 3) {
        Napi::TypeError::New(env, "Expected three arrays (x, gamma, beta)").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* x = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    NativeNDArray* gamma = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());
    NativeNDArray* beta = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[2].As<Napi::Object>());

    const auto& shapeX = x->shape();
    const auto& shapeG = gamma->shape();
    const auto& shapeB = beta->shape();

    if (shapeX.size() != 2 || shapeG.size() != 1 || shapeB.size() != 1) {
        Napi::Error::New(env, "affine expects x [m,n], gamma [n], beta [n]").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int64_t m = shapeX[0];
    int64_t n = shapeX[1];

    if (shapeG[0] != n || shapeB[0] != n) {
        Napi::Error::New(env, "gamma and beta must have same size as x's columns").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    double* dataX = static_cast<double*>(x->data());
    double* dataG = static_cast<double*>(gamma->data());
    double* dataB = static_cast<double*>(beta->data());

    Napi::Array jsShape = Napi::Array::New(env, 2);
    jsShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(m)));
    jsShape.Set(uint32_t(1), Napi::Number::New(env, static_cast<double>(n)));

    Napi::Object result = NativeNDArray::constructor.New({
        jsShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
    });
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* dataR = static_cast<double*>(resultArr->data());

    // Fused multiply-add: result[i,j] = gamma[j] * x[i,j] + beta[j]
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            dataR[i * n + j] = dataG[j] * dataX[i * n + j] + dataB[j];
        }
    }

    return result;
}

/**
 * X.T @ X without explicit transpose.
 * Uses BLAS dsyrk for efficient symmetric matrix computation.
 * Input: X [m, n]
 * Output: [n, n] symmetric matrix = X.T @ X
 */
Napi::Value Xtx(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* x = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    const auto& shape = x->shape();

    if (shape.size() != 2) {
        Napi::Error::New(env, "xtx expects 2D array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int m = static_cast<int>(shape[0]);
    int n = static_cast<int>(shape[1]);
    double* dataX = static_cast<double*>(x->data());

    Napi::Array jsShape = Napi::Array::New(env, 2);
    jsShape.Set(uint32_t(0), Napi::Number::New(env, n));
    jsShape.Set(uint32_t(1), Napi::Number::New(env, n));

    Napi::Object result = NativeNDArray::constructor.New({
        jsShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
    });
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* dataR = static_cast<double*>(resultArr->data());

#if defined(USE_ACCELERATE)
    // dsyrk: C = alpha * A.T @ A + beta * C (CblasTrans means transpose A)
    // We want X.T @ X, so we use CblasTrans
    cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                n, m, 1.0, dataX, n, 0.0, dataR, n);

    // Fill lower triangle (dsyrk only fills upper)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            dataR[i * n + j] = dataR[j * n + i];
        }
    }
#else
    // Pure C++ fallback: X.T @ X
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < m; k++) {
                sum += dataX[k * n + i] * dataX[k * n + j];
            }
            dataR[i * n + j] = sum;
            dataR[j * n + i] = sum;
        }
    }
#endif

    return result;
}

/**
 * X.T @ y without explicit transpose.
 * Uses BLAS dgemv for efficient matrix-vector multiplication.
 * Input: X [m, n], y [m] or [m, 1]
 * Output: [n] or [n, 1]
 */
Napi::Value Xty(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected two arrays (X, y)").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* x = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    NativeNDArray* y = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());

    const auto& shapeX = x->shape();
    const auto& shapeY = y->shape();

    if (shapeX.size() != 2) {
        Napi::Error::New(env, "xty expects X to be 2D").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int m = static_cast<int>(shapeX[0]);
    int n = static_cast<int>(shapeX[1]);

    // y can be [m] or [m, 1]
    bool yIs2D = (shapeY.size() == 2);
    int yLen = yIs2D ? static_cast<int>(shapeY[0]) : static_cast<int>(shapeY[0]);

    if (yLen != m) {
        Napi::Error::New(env, "X rows must match y length").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    double* dataX = static_cast<double*>(x->data());
    double* dataY = static_cast<double*>(y->data());

    // Create result with same dimensionality as y
    Napi::Array jsShape;
    if (yIs2D) {
        jsShape = Napi::Array::New(env, 2);
        jsShape.Set(uint32_t(0), Napi::Number::New(env, n));
        jsShape.Set(uint32_t(1), Napi::Number::New(env, 1));
    } else {
        jsShape = Napi::Array::New(env, 1);
        jsShape.Set(uint32_t(0), Napi::Number::New(env, n));
    }

    Napi::Object result = NativeNDArray::constructor.New({
        jsShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
    });
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* dataR = static_cast<double*>(resultArr->data());

#if defined(USE_ACCELERATE)
    // dgemv: y = alpha * A.T @ x + beta * y (CblasTrans means transpose A)
    cblas_dgemv(CblasRowMajor, CblasTrans,
                m, n, 1.0, dataX, n, dataY, 1, 0.0, dataR, 1);
#else
    // Pure C++ fallback: X.T @ y
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int k = 0; k < m; k++) {
            sum += dataX[k * n + i] * dataY[k];
        }
        dataR[i] = sum;
    }
#endif

    return result;
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    Napi::Object math = Napi::Object::New(env);

    math.Set("add", Napi::Function::New(env, Add));
    math.Set("subtract", Napi::Function::New(env, Subtract));
    math.Set("multiply", Napi::Function::New(env, Multiply));
    math.Set("divide", Napi::Function::New(env, Divide));
    math.Set("power", Napi::Function::New(env, Power));
    math.Set("sqrt", Napi::Function::New(env, Sqrt));
    math.Set("exp", Napi::Function::New(env, Exp));
    math.Set("log", Napi::Function::New(env, Log));
    math.Set("sin", Napi::Function::New(env, Sin));
    math.Set("cos", Napi::Function::New(env, Cos));
    math.Set("tan", Napi::Function::New(env, Tan));
    math.Set("abs", Napi::Function::New(env, Abs));
    math.Set("sum", Napi::Function::New(env, Sum));
    math.Set("prod", Napi::Function::New(env, Prod));
    math.Set("max", Napi::Function::New(env, Max));
    math.Set("min", Napi::Function::New(env, Min));
    math.Set("mean", Napi::Function::New(env, Mean));
    math.Set("std", Napi::Function::New(env, Std));
    math.Set("var", Napi::Function::New(env, Var));
    math.Set("median", Napi::Function::New(env, Median));
    math.Set("zscore", Napi::Function::New(env, Zscore));
    math.Set("corrcoef", Napi::Function::New(env, Corrcoef));
    math.Set("gram_matrix", Napi::Function::New(env, GramMatrix));
    math.Set("softmax", Napi::Function::New(env, Softmax));
    math.Set("pdist_sq", Napi::Function::New(env, PdistSq));
    math.Set("affine", Napi::Function::New(env, Affine));
    math.Set("xtx", Napi::Function::New(env, Xtx));
    math.Set("xty", Napi::Function::New(env, Xty));

    exports.Set("math", math);
    return exports;
}

} // namespace math
} // namespace numpy_node
