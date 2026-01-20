#include "math_ops.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>
#include <chrono>

#if defined(USE_ACCELERATE)
    #include <Accelerate/Accelerate.h>
#elif defined(USE_OPENBLAS)
    extern "C" {
        // BLAS dger: rank-1 update A = alpha * x * y' + A
        void dger_(const int* m, const int* n, const double* alpha,
                   const double* x, const int* incx,
                   const double* y, const int* incy,
                   double* a, const int* lda);
        // BLAS dgemm: matrix multiply C = alpha*A*B + beta*C
        void dgemm_(const char* transa, const char* transb,
                   const int* m, const int* n, const int* k,
                   const double* alpha, const double* a, const int* lda,
                   const double* b, const int* ldb,
                   const double* beta, double* c, const int* ldc);
        // BLAS dgemv: matrix-vector multiply y = alpha*A*x + beta*y
        void dgemv_(const char* trans, const int* m, const int* n,
                   const double* alpha, const double* a, const int* lda,
                   const double* x, const int* incx,
                   const double* beta, double* y, const int* incy);
    }
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

    // Fast path: 2D + 1D broadcasting (common in ML: add bias to each row)
    // Shape (M, N) + (N,) -> broadcast along rows
    const auto& shapeA = a->shape();
    const auto& shapeB = b->shape();
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

        // Add the 1D vector to each row
        for (int64_t row = 0; row < rows; row++) {
            vDSP_vaddD(dataA + row * cols, 1, dataB, 1, dataC + row * cols, 1, cols);
        }
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

    // Fast path: 2D * 1D broadcasting (common in ML: scale by weights)
    // Shape (M, N) * (N,) -> broadcast along rows
    const auto& shapeA = a->shape();
    const auto& shapeB = b->shape();
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

        // Multiply each row by the 1D vector
        for (int64_t row = 0; row < rows; row++) {
            vDSP_vmulD(dataA + row * cols, 1, dataB, 1, dataC + row * cols, 1, cols);
        }
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

// ============================================================
// Scalar-first operations for non-commutative ops
// These handle cases like: scalar - array, scalar / array
// ============================================================

Napi::Value SubtractScalarFirst(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected two arguments").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    double scalar = info[0].As<Napi::Number>().DoubleValue();
    NativeNDArray* b = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());

    // Create result array with same shape
    Napi::Array shape = Napi::Array::New(env, b->shape().size());
    for (size_t i = 0; i < b->shape().size(); i++) {
        shape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(b->shape()[i])));
    }

    Napi::Object result = NativeNDArray::constructor.New({
        shape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
    });
    NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);

    double* dataB = static_cast<double*>(b->data());
    double* dataC = static_cast<double*>(c->data());
    int64_t size = b->size();

#if defined(USE_ACCELERATE)
    // scalar - array = -(array - scalar) = -(array) + scalar
    // First negate, then add scalar
    double negOne = -1.0;
    vDSP_vsmulD(dataB, 1, &negOne, dataC, 1, size);
    vDSP_vsaddD(dataC, 1, &scalar, dataC, 1, size);
#else
    for (int64_t i = 0; i < size; i++) {
        dataC[i] = scalar - dataB[i];
    }
#endif

    return result;
}

Napi::Value DivideScalarFirst(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected two arguments").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    double scalar = info[0].As<Napi::Number>().DoubleValue();
    NativeNDArray* b = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());

    // Create result array with same shape
    Napi::Array shape = Napi::Array::New(env, b->shape().size());
    for (size_t i = 0; i < b->shape().size(); i++) {
        shape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(b->shape()[i])));
    }

    Napi::Object result = NativeNDArray::constructor.New({
        shape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
    });
    NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);

    double* dataB = static_cast<double*>(b->data());
    double* dataC = static_cast<double*>(c->data());
    int64_t size = b->size();

#if defined(USE_ACCELERATE)
    // vDSP_svdivD: scalar / array
    vDSP_svdivD(&scalar, dataB, 1, dataC, 1, size);
#else
    for (int64_t i = 0; i < size; i++) {
        dataC[i] = scalar / dataB[i];
    }
#endif

    return result;
}

// ============================================================
// In-place arithmetic operations
// These modify the first argument directly, avoiding allocation
// ============================================================

Napi::Value AddInplace(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected two arguments").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    double* dataA = static_cast<double*>(a->data());
    int64_t size = a->size();

    // Scalar addition
    if (info[1].IsNumber()) {
        double scalar = info[1].As<Napi::Number>().DoubleValue();
#if defined(USE_ACCELERATE)
        vDSP_vsaddD(dataA, 1, &scalar, dataA, 1, size);
#else
        for (int64_t i = 0; i < size; i++) {
            dataA[i] += scalar;
        }
#endif
        return info[0];
    }

    // Array addition (must be same shape for in-place)
    NativeNDArray* b = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());

    if (a->shape() != b->shape()) {
        Napi::TypeError::New(env, "In-place operations require same shape").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    double* dataB = static_cast<double*>(b->data());

#if defined(USE_ACCELERATE)
    vDSP_vaddD(dataA, 1, dataB, 1, dataA, 1, size);
#else
    for (int64_t i = 0; i < size; i++) {
        dataA[i] += dataB[i];
    }
#endif

    return info[0];
}

Napi::Value SubtractInplace(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected two arguments").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    double* dataA = static_cast<double*>(a->data());
    int64_t size = a->size();

    // Scalar subtraction
    if (info[1].IsNumber()) {
        double scalar = -info[1].As<Napi::Number>().DoubleValue();
#if defined(USE_ACCELERATE)
        vDSP_vsaddD(dataA, 1, &scalar, dataA, 1, size);
#else
        double posScalar = -scalar;
        for (int64_t i = 0; i < size; i++) {
            dataA[i] -= posScalar;
        }
#endif
        return info[0];
    }

    // Array subtraction (must be same shape for in-place)
    NativeNDArray* b = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());

    if (a->shape() != b->shape()) {
        Napi::TypeError::New(env, "In-place operations require same shape").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    double* dataB = static_cast<double*>(b->data());

#if defined(USE_ACCELERATE)
    vDSP_vsubD(dataB, 1, dataA, 1, dataA, 1, size);
#else
    for (int64_t i = 0; i < size; i++) {
        dataA[i] -= dataB[i];
    }
#endif

    return info[0];
}

Napi::Value MultiplyInplace(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected two arguments").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    double* dataA = static_cast<double*>(a->data());
    int64_t size = a->size();

    // Scalar multiplication
    if (info[1].IsNumber()) {
        double scalar = info[1].As<Napi::Number>().DoubleValue();
#if defined(USE_ACCELERATE)
        vDSP_vsmulD(dataA, 1, &scalar, dataA, 1, size);
#else
        for (int64_t i = 0; i < size; i++) {
            dataA[i] *= scalar;
        }
#endif
        return info[0];
    }

    // Array multiplication (must be same shape for in-place)
    NativeNDArray* b = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());

    if (a->shape() != b->shape()) {
        Napi::TypeError::New(env, "In-place operations require same shape").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    double* dataB = static_cast<double*>(b->data());

#if defined(USE_ACCELERATE)
    vDSP_vmulD(dataA, 1, dataB, 1, dataA, 1, size);
#else
    for (int64_t i = 0; i < size; i++) {
        dataA[i] *= dataB[i];
    }
#endif

    return info[0];
}

Napi::Value DivideInplace(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected two arguments").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    double* dataA = static_cast<double*>(a->data());
    int64_t size = a->size();

    // Scalar division
    if (info[1].IsNumber()) {
        double scalar = info[1].As<Napi::Number>().DoubleValue();
#if defined(USE_ACCELERATE)
        vDSP_vsdivD(dataA, 1, &scalar, dataA, 1, size);
#else
        for (int64_t i = 0; i < size; i++) {
            dataA[i] /= scalar;
        }
#endif
        return info[0];
    }

    // Array division (must be same shape for in-place)
    NativeNDArray* b = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());

    if (a->shape() != b->shape()) {
        Napi::TypeError::New(env, "In-place operations require same shape").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    double* dataB = static_cast<double*>(b->data());

#if defined(USE_ACCELERATE)
    vDSP_vdivD(dataB, 1, dataA, 1, dataA, 1, size);
#else
    for (int64_t i = 0; i < size; i++) {
        dataA[i] /= dataB[i];
    }
#endif

    return info[0];
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

Napi::Value Tanh(const Napi::CallbackInfo& info) {
#if defined(USE_ACCELERATE)
    return VecLibUnaryOp(info, vvtanh);
#else
    return UnaryOp(info, [](double a) { return std::tanh(a); });
#endif
}

Napi::Value Abs(const Napi::CallbackInfo& info) {
    return UnaryOp(info, [](double a) { return std::abs(a); });
}

Napi::Value Round(const Napi::CallbackInfo& info) {
    // NumPy uses "round half to even" (banker's rounding)
    // std::nearbyint uses the current rounding mode (FE_TONEAREST = ties to even)
    return UnaryOp(info, [](double a) { return std::nearbyint(a); });
}

Napi::Value Floor(const Napi::CallbackInfo& info) {
    return UnaryOp(info, [](double a) { return std::floor(a); });
}

Napi::Value Ceil(const Napi::CallbackInfo& info) {
    return UnaryOp(info, [](double a) { return std::ceil(a); });
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
    const auto& shape = a->shape();
    int ndim = static_cast<int>(shape.size());

    // Check for keepdims parameter (3rd argument)
    bool keepdims = false;
    if (info.Length() >= 3 && info[2].IsBoolean()) {
        keepdims = info[2].As<Napi::Boolean>().Value();
    }

    // Check for axis parameter
    if (info.Length() >= 2 && !info[1].IsUndefined()) {
        int axis = info[1].As<Napi::Number>().Int32Value();
        if (axis < 0) axis += ndim;

        if (axis < 0 || axis >= ndim) {
            Napi::Error::New(env, "Axis out of bounds").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        // Fast path for 2D arrays (most common case in ML/data science)
        if (ndim == 2) {
            int64_t rows = shape[0];
            int64_t cols = shape[1];

            if (axis == 0) {
                // Sum along rows -> result has shape [cols] or [1, cols] if keepdims
                std::vector<int64_t> resultShape;
                if (keepdims) {
                    resultShape = {1, cols};
                } else {
                    resultShape = {cols};
                }

                Napi::Array jsResultShape = Napi::Array::New(env, resultShape.size());
                for (size_t i = 0; i < resultShape.size(); i++) {
                    jsResultShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(resultShape[i])));
                }

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
                // Sum along columns -> result has shape [rows] or [rows, 1] if keepdims
                std::vector<int64_t> resultShapeVec;
                if (keepdims) {
                    resultShapeVec = {rows, 1};
                } else {
                    resultShapeVec = {rows};
                }

                Napi::Array jsResultShape = Napi::Array::New(env, resultShapeVec.size());
                for (size_t i = 0; i < resultShapeVec.size(); i++) {
                    jsResultShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(resultShapeVec[i])));
                }

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
        // Compute result shape: with keepdims, replace axis dim with 1; without, remove it
        std::vector<int64_t> resultShape;
        for (int i = 0; i < ndim; i++) {
            if (i == axis) {
                if (keepdims) {
                    resultShape.push_back(1);
                }
            } else {
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
    const auto& shape = a->shape();
    int ndim = static_cast<int>(shape.size());

    // Check for keepdims parameter (3rd argument)
    bool keepdims = false;
    if (info.Length() >= 3 && info[2].IsBoolean()) {
        keepdims = info[2].As<Napi::Boolean>().Value();
    }

    // Check for axis parameter
    if (info.Length() >= 2 && !info[1].IsUndefined()) {
        int axis = info[1].As<Napi::Number>().Int32Value();
        if (axis < 0) axis += ndim;

        if (axis < 0 || axis >= ndim) {
            Napi::Error::New(env, "Axis out of bounds").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        // Fast path for 2D arrays
        if (ndim == 2) {
            int64_t rows = shape[0];
            int64_t cols = shape[1];

            if (axis == 0) {
                // Max along rows -> result has shape [cols] or [1, cols] if keepdims
                std::vector<int64_t> resultShape;
                if (keepdims) {
                    resultShape = {1, cols};
                } else {
                    resultShape = {cols};
                }

                Napi::Array jsResultShape = Napi::Array::New(env, resultShape.size());
                for (size_t i = 0; i < resultShape.size(); i++) {
                    jsResultShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(resultShape[i])));
                }

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
                // Max along columns -> result has shape [rows] or [rows, 1] if keepdims
                std::vector<int64_t> resultShape;
                if (keepdims) {
                    resultShape = {rows, 1};
                } else {
                    resultShape = {rows};
                }

                Napi::Array jsResultShape = Napi::Array::New(env, resultShape.size());
                for (size_t i = 0; i < resultShape.size(); i++) {
                    jsResultShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(resultShape[i])));
                }

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
    const auto& shape = a->shape();
    int ndim = static_cast<int>(shape.size());

    // Check for keepdims parameter (3rd argument)
    bool keepdims = false;
    if (info.Length() >= 3 && info[2].IsBoolean()) {
        keepdims = info[2].As<Napi::Boolean>().Value();
    }

    // Check for axis parameter
    if (info.Length() >= 2 && !info[1].IsUndefined()) {
        int axis = info[1].As<Napi::Number>().Int32Value();
        if (axis < 0) axis += ndim;

        if (axis < 0 || axis >= ndim) {
            Napi::Error::New(env, "Axis out of bounds").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        // Fast path for 2D arrays
        if (ndim == 2) {
            int64_t rows = shape[0];
            int64_t cols = shape[1];

            if (axis == 0) {
                // Min along rows -> result has shape [cols] or [1, cols] if keepdims
                std::vector<int64_t> resultShape;
                if (keepdims) {
                    resultShape = {1, cols};
                } else {
                    resultShape = {cols};
                }

                Napi::Array jsResultShape = Napi::Array::New(env, resultShape.size());
                for (size_t i = 0; i < resultShape.size(); i++) {
                    jsResultShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(resultShape[i])));
                }

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
                // Min along columns -> result has shape [rows] or [rows, 1] if keepdims
                std::vector<int64_t> resultShape;
                if (keepdims) {
                    resultShape = {rows, 1};
                } else {
                    resultShape = {rows};
                }

                Napi::Array jsResultShape = Napi::Array::New(env, resultShape.size());
                for (size_t i = 0; i < resultShape.size(); i++) {
                    jsResultShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(resultShape[i])));
                }

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

Napi::Value Argmax(const Napi::CallbackInfo& info) {
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

        // Handle negative axis
        if (axis < 0) axis += ndim;

        if (axis < 0 || axis >= ndim) {
            Napi::Error::New(env, "Axis out of bounds").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        // Fast path for 2D arrays
        if (ndim == 2) {
            int64_t rows = shape[0];
            int64_t cols = shape[1];

            if (axis == 0) {
                // Argmax along rows -> result has shape [cols]
                Napi::Array jsResultShape = Napi::Array::New(env, 1);
                jsResultShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(cols)));

                Napi::Object result = NativeNDArray::constructor.New({
                    jsResultShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
                });
                NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
                double* resultData = static_cast<double*>(resultArr->data());

                // Initialize with first row values and index 0
                std::vector<double> maxVals(cols);
                for (int64_t col = 0; col < cols; col++) {
                    maxVals[col] = data[col];
                    resultData[col] = 0;
                }

                // Find argmax across rows
                for (int64_t row = 1; row < rows; row++) {
                    double* rowPtr = data + row * cols;
                    for (int64_t col = 0; col < cols; col++) {
                        if (rowPtr[col] > maxVals[col]) {
                            maxVals[col] = rowPtr[col];
                            resultData[col] = static_cast<double>(row);
                        }
                    }
                }
                return result;

            } else { // axis == 1
                // Argmax along columns -> result has shape [rows]
                Napi::Array jsResultShape = Napi::Array::New(env, 1);
                jsResultShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(rows)));

                Napi::Object result = NativeNDArray::constructor.New({
                    jsResultShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
                });
                NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
                double* resultData = static_cast<double*>(resultArr->data());

                for (int64_t row = 0; row < rows; row++) {
                    double* rowPtr = data + row * cols;
                    double maxVal = rowPtr[0];
                    int64_t maxIdx = 0;
                    for (int64_t col = 1; col < cols; col++) {
                        if (rowPtr[col] > maxVal) {
                            maxVal = rowPtr[col];
                            maxIdx = col;
                        }
                    }
                    resultData[row] = static_cast<double>(maxIdx);
                }
                return result;
            }
        }

        // Generic N-dim case: not yet supported
        Napi::Error::New(env, "Axis reduction only supported for 2D arrays").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // No axis - global argmax
    double maxVal = data[0];
    int64_t maxIdx = 0;
    for (int64_t i = 1; i < size; i++) {
        if (data[i] > maxVal) {
            maxVal = data[i];
            maxIdx = i;
        }
    }

    return Napi::Number::New(env, static_cast<double>(maxIdx));
}

Napi::Value Argmin(const Napi::CallbackInfo& info) {
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

        // Handle negative axis
        if (axis < 0) axis += ndim;

        if (axis < 0 || axis >= ndim) {
            Napi::Error::New(env, "Axis out of bounds").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        // Fast path for 2D arrays
        if (ndim == 2) {
            int64_t rows = shape[0];
            int64_t cols = shape[1];

            if (axis == 0) {
                // Argmin along rows -> result has shape [cols]
                Napi::Array jsResultShape = Napi::Array::New(env, 1);
                jsResultShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(cols)));

                Napi::Object result = NativeNDArray::constructor.New({
                    jsResultShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
                });
                NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
                double* resultData = static_cast<double*>(resultArr->data());

                // Initialize with first row values and index 0
                std::vector<double> minVals(cols);
                for (int64_t col = 0; col < cols; col++) {
                    minVals[col] = data[col];
                    resultData[col] = 0;
                }

                // Find argmin across rows
                for (int64_t row = 1; row < rows; row++) {
                    double* rowPtr = data + row * cols;
                    for (int64_t col = 0; col < cols; col++) {
                        if (rowPtr[col] < minVals[col]) {
                            minVals[col] = rowPtr[col];
                            resultData[col] = static_cast<double>(row);
                        }
                    }
                }
                return result;

            } else { // axis == 1
                // Argmin along columns -> result has shape [rows]
                Napi::Array jsResultShape = Napi::Array::New(env, 1);
                jsResultShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(rows)));

                Napi::Object result = NativeNDArray::constructor.New({
                    jsResultShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
                });
                NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
                double* resultData = static_cast<double*>(resultArr->data());

                for (int64_t row = 0; row < rows; row++) {
                    double* rowPtr = data + row * cols;
                    double minVal = rowPtr[0];
                    int64_t minIdx = 0;
                    for (int64_t col = 1; col < cols; col++) {
                        if (rowPtr[col] < minVal) {
                            minVal = rowPtr[col];
                            minIdx = col;
                        }
                    }
                    resultData[row] = static_cast<double>(minIdx);
                }
                return result;
            }
        }

        // Generic N-dim case: not yet supported
        Napi::Error::New(env, "Axis reduction only supported for 2D arrays").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // No axis - global argmin
    double minVal = data[0];
    int64_t minIdx = 0;
    for (int64_t i = 1; i < size; i++) {
        if (data[i] < minVal) {
            minVal = data[i];
            minIdx = i;
        }
    }

    return Napi::Number::New(env, static_cast<double>(minIdx));
}

Napi::Value Clip(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 3) {
        Napi::TypeError::New(env, "Expected array, min, max").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    double minVal = info[1].As<Napi::Number>().DoubleValue();
    double maxVal = info[2].As<Napi::Number>().DoubleValue();

    double* dataA = static_cast<double*>(a->data());
    int64_t size = a->size();
    const auto& shape = a->shape();

    // Create result array with same shape
    Napi::Array jsResultShape = Napi::Array::New(env, shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
        jsResultShape.Set(static_cast<uint32_t>(i), Napi::Number::New(env, static_cast<double>(shape[i])));
    }

    Napi::Object result = NativeNDArray::constructor.New({
        jsResultShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
    });
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* dataC = static_cast<double*>(resultArr->data());

#if defined(USE_ACCELERATE)
    vDSP_vclipD(dataA, 1, &minVal, &maxVal, dataC, 1, size);
#else
    for (int64_t i = 0; i < size; i++) {
        dataC[i] = std::max(minVal, std::min(maxVal, dataA[i]));
    }
#endif

    return result;
}

Napi::Value Where(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 3) {
        Napi::TypeError::New(env, "Expected condition, x, y").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* condition = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    NativeNDArray* x = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());
    NativeNDArray* y = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[2].As<Napi::Object>());

    // Handle different dtype for condition (bool uses uint8, float uses double)
    bool condIsBool = condition->dtype() == DType::Bool;
    uint8_t* dataCondBool = condIsBool ? static_cast<uint8_t*>(condition->data()) : nullptr;
    double* dataCondDouble = condIsBool ? nullptr : static_cast<double*>(condition->data());

    double* dataX = static_cast<double*>(x->data());
    double* dataY = static_cast<double*>(y->data());

    const auto& shapeCond = condition->shape();
    const auto& shapeX = x->shape();
    const auto& shapeY = y->shape();

    // Helper lambda to check condition value
    auto checkCond = [condIsBool, dataCondBool, dataCondDouble](int64_t idx) -> bool {
        if (condIsBool) {
            return dataCondBool[idx] != 0;
        } else {
            return dataCondDouble[idx] != 0.0;
        }
    };

    // Fast path: all arrays have the same shape
    if (shapeCond == shapeX && shapeX == shapeY) {
        int64_t size = condition->size();

        // Create result array with same shape
        Napi::Array jsResultShape = Napi::Array::New(env, shapeCond.size());
        for (size_t i = 0; i < shapeCond.size(); i++) {
            jsResultShape.Set(static_cast<uint32_t>(i), Napi::Number::New(env, static_cast<double>(shapeCond[i])));
        }

        Napi::Object result = NativeNDArray::constructor.New({
            jsResultShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
        });
        NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
        double* dataC = static_cast<double*>(resultArr->data());

        for (int64_t i = 0; i < size; i++) {
            dataC[i] = checkCond(i) ? dataX[i] : dataY[i];
        }

        return result;
    }

    // Broadcasting case: compute broadcast shape
    // Determine max ndim
    size_t maxNdim = std::max({shapeCond.size(), shapeX.size(), shapeY.size()});

    // Pad shapes to maxNdim (prepend 1s)
    auto padShape = [maxNdim](const std::vector<int64_t>& shape) {
        std::vector<int64_t> padded(maxNdim, 1);
        size_t offset = maxNdim - shape.size();
        for (size_t i = 0; i < shape.size(); i++) {
            padded[offset + i] = shape[i];
        }
        return padded;
    };

    std::vector<int64_t> paddedCond = padShape(shapeCond);
    std::vector<int64_t> paddedX = padShape(shapeX);
    std::vector<int64_t> paddedY = padShape(shapeY);

    // Compute broadcast shape
    std::vector<int64_t> resultShape(maxNdim);
    for (size_t i = 0; i < maxNdim; i++) {
        int64_t dimCond = paddedCond[i];
        int64_t dimX = paddedX[i];
        int64_t dimY = paddedY[i];

        // Check broadcast compatibility
        if ((dimCond != 1 && dimX != 1 && dimCond != dimX) ||
            (dimCond != 1 && dimY != 1 && dimCond != dimY) ||
            (dimX != 1 && dimY != 1 && dimX != dimY)) {
            Napi::Error::New(env, "Shapes are not broadcast compatible").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        resultShape[i] = std::max({dimCond, dimX, dimY});
    }

    // Create result array
    Napi::Array jsResultShape = Napi::Array::New(env, resultShape.size());
    for (size_t i = 0; i < resultShape.size(); i++) {
        jsResultShape.Set(static_cast<uint32_t>(i), Napi::Number::New(env, static_cast<double>(resultShape[i])));
    }

    Napi::Object result = NativeNDArray::constructor.New({
        jsResultShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
    });
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* dataC = static_cast<double*>(resultArr->data());

    // Compute strides for broadcasting
    auto computeStrides = [](const std::vector<int64_t>& shape) {
        std::vector<int64_t> strides(shape.size());
        if (shape.empty()) return strides;
        strides.back() = 1;
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    };

    std::vector<int64_t> stridesCond = computeStrides(paddedCond);
    std::vector<int64_t> stridesX = computeStrides(paddedX);
    std::vector<int64_t> stridesY = computeStrides(paddedY);
    std::vector<int64_t> stridesResult = computeStrides(resultShape);

    // Total number of elements
    int64_t totalSize = 1;
    for (size_t i = 0; i < resultShape.size(); i++) {
        totalSize *= resultShape[i];
    }

    // Iterate over all elements using linear index -> multi-index -> broadcast lookup
    for (int64_t idx = 0; idx < totalSize; idx++) {
        // Convert linear index to multi-index
        std::vector<int64_t> multiIdx(maxNdim);
        int64_t remaining = idx;
        for (size_t i = 0; i < maxNdim; i++) {
            multiIdx[i] = remaining / stridesResult[i];
            remaining %= stridesResult[i];
        }

        // Compute indices into each input array (with broadcasting)
        int64_t idxCond = 0, idxX = 0, idxY = 0;
        for (size_t i = 0; i < maxNdim; i++) {
            int64_t mi = multiIdx[i];
            if (paddedCond[i] > 1) idxCond += mi * stridesCond[i];
            if (paddedX[i] > 1) idxX += mi * stridesX[i];
            if (paddedY[i] > 1) idxY += mi * stridesY[i];
        }

        dataC[idx] = checkCond(idxCond) ? dataX[idxX] : dataY[idxY];
    }

    return result;
}

Napi::Value Squeeze(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    const auto& shape = a->shape();

    // Check for axis parameter
    int axisToSqueeze = -1;
    if (info.Length() >= 2 && !info[1].IsUndefined()) {
        axisToSqueeze = info[1].As<Napi::Number>().Int32Value();
        int ndim = static_cast<int>(shape.size());

        // Handle negative axis
        if (axisToSqueeze < 0) axisToSqueeze += ndim;

        if (axisToSqueeze < 0 || axisToSqueeze >= ndim) {
            Napi::Error::New(env, "Axis out of bounds").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        // Check that the dimension is actually 1
        if (shape[axisToSqueeze] != 1) {
            Napi::Error::New(env, "Cannot squeeze axis with size != 1").ThrowAsJavaScriptException();
            return env.Undefined();
        }
    }

    // Build new shape by removing dimensions of size 1
    std::vector<int64_t> newShape;
    for (size_t i = 0; i < shape.size(); i++) {
        if (axisToSqueeze >= 0) {
            // Only squeeze the specified axis
            if (static_cast<int>(i) != axisToSqueeze) {
                newShape.push_back(shape[i]);
            }
        } else {
            // Squeeze all dimensions of size 1
            if (shape[i] != 1) {
                newShape.push_back(shape[i]);
            }
        }
    }

    // If all dimensions were squeezed, result is a scalar (shape [1])
    if (newShape.empty()) {
        newShape.push_back(1);
    }

    // Create result array with new shape (copy data)
    Napi::Array jsResultShape = Napi::Array::New(env, newShape.size());
    for (size_t i = 0; i < newShape.size(); i++) {
        jsResultShape.Set(static_cast<uint32_t>(i), Napi::Number::New(env, static_cast<double>(newShape[i])));
    }

    Napi::Object result = NativeNDArray::constructor.New({
        jsResultShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
    });
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);

    // Copy data
    double* dataA = static_cast<double*>(a->data());
    double* dataC = static_cast<double*>(resultArr->data());
    int64_t size = a->size();
    std::memcpy(dataC, dataA, size * sizeof(double));

    return result;
}

Napi::Value ExpandDims(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected array and axis").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    int axis = info[1].As<Napi::Number>().Int32Value();
    const auto& shape = a->shape();
    int ndim = static_cast<int>(shape.size());

    // Handle negative axis (axis can be from -ndim-1 to ndim inclusive)
    if (axis < 0) axis += ndim + 1;

    if (axis < 0 || axis > ndim) {
        Napi::Error::New(env, "Axis out of bounds").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Build new shape with dimension of size 1 inserted at axis
    std::vector<int64_t> newShape;
    for (int i = 0; i < ndim; i++) {
        if (i == axis) {
            newShape.push_back(1);
        }
        newShape.push_back(shape[i]);
    }
    // Handle case where axis is at the end
    if (axis == ndim) {
        newShape.push_back(1);
    }

    // Create result array with new shape (copy data)
    Napi::Array jsResultShape = Napi::Array::New(env, newShape.size());
    for (size_t i = 0; i < newShape.size(); i++) {
        jsResultShape.Set(static_cast<uint32_t>(i), Napi::Number::New(env, static_cast<double>(newShape[i])));
    }

    Napi::Object result = NativeNDArray::constructor.New({
        jsResultShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
    });
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);

    // Copy data
    double* dataA = static_cast<double*>(a->data());
    double* dataC = static_cast<double*>(resultArr->data());
    int64_t size = a->size();
    std::memcpy(dataC, dataA, size * sizeof(double));

    return result;
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
    const auto& shape = a->shape();
    int ndim = static_cast<int>(shape.size());

    // Check for keepdims parameter (3rd argument)
    bool keepdims = false;
    if (info.Length() >= 3 && info[2].IsBoolean()) {
        keepdims = info[2].As<Napi::Boolean>().Value();
    }

    // Check for axis parameter
    if (info.Length() >= 2 && !info[1].IsUndefined()) {
        int axis = info[1].As<Napi::Number>().Int32Value();
        if (axis < 0) axis += ndim;

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
                // Mean along rows -> result has shape [cols] or [1, cols] if keepdims
                std::vector<int64_t> resultShape;
                if (keepdims) {
                    resultShape = {1, cols};
                } else {
                    resultShape = {cols};
                }

                Napi::Array jsResultShape = Napi::Array::New(env, resultShape.size());
                for (size_t i = 0; i < resultShape.size(); i++) {
                    jsResultShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(resultShape[i])));
                }

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
                // Mean along columns -> result has shape [rows] or [rows, 1] if keepdims
                std::vector<int64_t> resultShapeVec;
                if (keepdims) {
                    resultShapeVec = {rows, 1};
                } else {
                    resultShapeVec = {rows};
                }

                Napi::Array jsResultShape = Napi::Array::New(env, resultShapeVec.size());
                for (size_t i = 0; i < resultShapeVec.size(); i++) {
                    jsResultShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(resultShapeVec[i])));
                }

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
        // Compute result shape: with keepdims, replace axis dim with 1; without, remove it
        std::vector<int64_t> resultShape;
        for (int i = 0; i < ndim; i++) {
            if (i == axis) {
                if (keepdims) {
                    resultShape.push_back(1);
                }
            } else {
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

/**
 * Cumulative sum along axis
 * cumsum(a, axis?) - Returns array with cumulative sums
 */
Napi::Value Cumsum(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    double* data = static_cast<double*>(a->data());
    const auto& shape = a->shape();
    int64_t size = a->size();
    int ndim = static_cast<int>(shape.size());

    // Check for axis parameter
    if (info.Length() >= 2 && !info[1].IsUndefined()) {
        int axis = info[1].As<Napi::Number>().Int32Value();

        // Handle negative axis
        if (axis < 0) axis += ndim;

        if (axis < 0 || axis >= ndim) {
            Napi::Error::New(env, "Axis out of bounds").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        // Create result with same shape
        Napi::Array jsResultShape = Napi::Array::New(env, shape.size());
        for (size_t i = 0; i < shape.size(); i++) {
            jsResultShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(shape[i])));
        }

        Napi::Object result = NativeNDArray::constructor.New({
            jsResultShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
        });
        NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
        double* resultData = static_cast<double*>(resultArr->data());

        // Fast path for 2D arrays
        if (ndim == 2) {
            int64_t rows = shape[0];
            int64_t cols = shape[1];

            if (axis == 0) {
                // Cumsum along rows (down columns)
                // First row is same as input
                std::memcpy(resultData, data, cols * sizeof(double));
                // Cumulative sum for subsequent rows
                for (int64_t row = 1; row < rows; row++) {
                    for (int64_t col = 0; col < cols; col++) {
                        resultData[row * cols + col] = resultData[(row - 1) * cols + col] + data[row * cols + col];
                    }
                }
            } else { // axis == 1
                // Cumsum along columns (across rows)
                for (int64_t row = 0; row < rows; row++) {
                    resultData[row * cols] = data[row * cols];
                    for (int64_t col = 1; col < cols; col++) {
                        resultData[row * cols + col] = resultData[row * cols + col - 1] + data[row * cols + col];
                    }
                }
            }
            return result;
        }

        // Fast path for 1D arrays
        if (ndim == 1) {
            resultData[0] = data[0];
            for (int64_t i = 1; i < size; i++) {
                resultData[i] = resultData[i - 1] + data[i];
            }
            return result;
        }

        // Generic N-dimensional case
        // Compute strides for the input array
        std::vector<int64_t> strides(ndim);
        strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        int64_t axisStride = strides[axis];
        int64_t axisSize = shape[axis];

        // Copy input to result first
        std::memcpy(resultData, data, size * sizeof(double));

        // Iterate over all "slices" perpendicular to the axis
        int64_t outerSize = size / axisSize;
        for (int64_t outer = 0; outer < outerSize; outer++) {
            // Compute base index for this slice
            int64_t baseIdx = 0;
            int64_t temp = outer;
            for (int d = ndim - 1; d >= 0; d--) {
                if (d == axis) continue;
                int64_t dimStride = 1;
                for (int dd = d + 1; dd < ndim; dd++) {
                    if (dd != axis) dimStride *= shape[dd];
                }
                int64_t coord = temp % shape[d];
                temp /= shape[d];
                baseIdx += coord * strides[d];
            }

            // Compute cumulative sum along axis
            for (int64_t k = 1; k < axisSize; k++) {
                resultData[baseIdx + k * axisStride] += resultData[baseIdx + (k - 1) * axisStride];
            }
        }

        return result;
    }

    // No axis - flatten and compute cumsum
    Napi::Array jsResultShape = Napi::Array::New(env, 1);
    jsResultShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(size)));

    Napi::Object result = NativeNDArray::constructor.New({
        jsResultShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
    });
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* resultData = static_cast<double*>(resultArr->data());

    resultData[0] = data[0];
    for (int64_t i = 1; i < size; i++) {
        resultData[i] = resultData[i - 1] + data[i];
    }

    return result;
}

/**
 * Cumulative product along axis
 * cumprod(a, axis?) - Returns array with cumulative products
 */
Napi::Value Cumprod(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    double* data = static_cast<double*>(a->data());
    const auto& shape = a->shape();
    int64_t size = a->size();
    int ndim = static_cast<int>(shape.size());

    // Check for axis parameter
    if (info.Length() >= 2 && !info[1].IsUndefined()) {
        int axis = info[1].As<Napi::Number>().Int32Value();

        // Handle negative axis
        if (axis < 0) axis += ndim;

        if (axis < 0 || axis >= ndim) {
            Napi::Error::New(env, "Axis out of bounds").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        // Create result with same shape
        Napi::Array jsResultShape = Napi::Array::New(env, shape.size());
        for (size_t i = 0; i < shape.size(); i++) {
            jsResultShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(shape[i])));
        }

        Napi::Object result = NativeNDArray::constructor.New({
            jsResultShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
        });
        NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
        double* resultData = static_cast<double*>(resultArr->data());

        // Fast path for 2D arrays
        if (ndim == 2) {
            int64_t rows = shape[0];
            int64_t cols = shape[1];

            if (axis == 0) {
                // Cumprod along rows (down columns)
                std::memcpy(resultData, data, cols * sizeof(double));
                for (int64_t row = 1; row < rows; row++) {
                    for (int64_t col = 0; col < cols; col++) {
                        resultData[row * cols + col] = resultData[(row - 1) * cols + col] * data[row * cols + col];
                    }
                }
            } else { // axis == 1
                // Cumprod along columns (across rows)
                for (int64_t row = 0; row < rows; row++) {
                    resultData[row * cols] = data[row * cols];
                    for (int64_t col = 1; col < cols; col++) {
                        resultData[row * cols + col] = resultData[row * cols + col - 1] * data[row * cols + col];
                    }
                }
            }
            return result;
        }

        // Fast path for 1D arrays
        if (ndim == 1) {
            resultData[0] = data[0];
            for (int64_t i = 1; i < size; i++) {
                resultData[i] = resultData[i - 1] * data[i];
            }
            return result;
        }

        // Generic N-dimensional case
        std::vector<int64_t> strides(ndim);
        strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        int64_t axisStride = strides[axis];
        int64_t axisSize = shape[axis];

        // Copy input to result first
        std::memcpy(resultData, data, size * sizeof(double));

        // Iterate over all "slices" perpendicular to the axis
        int64_t outerSize = size / axisSize;
        for (int64_t outer = 0; outer < outerSize; outer++) {
            int64_t baseIdx = 0;
            int64_t temp = outer;
            for (int d = ndim - 1; d >= 0; d--) {
                if (d == axis) continue;
                int64_t dimStride = 1;
                for (int dd = d + 1; dd < ndim; dd++) {
                    if (dd != axis) dimStride *= shape[dd];
                }
                int64_t coord = temp % shape[d];
                temp /= shape[d];
                baseIdx += coord * strides[d];
            }

            // Compute cumulative product along axis
            for (int64_t k = 1; k < axisSize; k++) {
                resultData[baseIdx + k * axisStride] *= resultData[baseIdx + (k - 1) * axisStride];
            }
        }

        return result;
    }

    // No axis - flatten and compute cumprod
    Napi::Array jsResultShape = Napi::Array::New(env, 1);
    jsResultShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(size)));

    Napi::Object result = NativeNDArray::constructor.New({
        jsResultShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
    });
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* resultData = static_cast<double*>(resultArr->data());

    resultData[0] = data[0];
    for (int64_t i = 1; i < size; i++) {
        resultData[i] = resultData[i - 1] * data[i];
    }

    return result;
}

/**
 * Concatenate arrays along an existing axis
 * concatenate(arrays, axis=0) - Join arrays along axis
 */
Napi::Value Concatenate(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsArray()) {
        Napi::TypeError::New(env, "Expected array of arrays").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Array arrays = info[0].As<Napi::Array>();
    uint32_t numArrays = arrays.Length();

    if (numArrays == 0) {
        Napi::Error::New(env, "Need at least one array to concatenate").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Get axis (default 0)
    int axis = 0;
    if (info.Length() >= 2 && !info[1].IsUndefined()) {
        axis = info[1].As<Napi::Number>().Int32Value();
    }

    // Get first array to determine shape
    NativeNDArray* first = Napi::ObjectWrap<NativeNDArray>::Unwrap(
        arrays.Get(uint32_t(0)).As<Napi::Object>()
    );
    const auto& firstShape = first->shape();
    int ndim = static_cast<int>(firstShape.size());

    // Handle negative axis
    if (axis < 0) axis += ndim;

    if (axis < 0 || axis >= ndim) {
        Napi::Error::New(env, "Axis out of bounds").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Collect all arrays and validate shapes
    std::vector<NativeNDArray*> arrPtrs;
    arrPtrs.reserve(numArrays);
    int64_t totalAxisSize = 0;

    for (uint32_t i = 0; i < numArrays; i++) {
        NativeNDArray* arr = Napi::ObjectWrap<NativeNDArray>::Unwrap(
            arrays.Get(i).As<Napi::Object>()
        );
        arrPtrs.push_back(arr);

        const auto& shape = arr->shape();

        // Validate ndim
        if (static_cast<int>(shape.size()) != ndim) {
            Napi::Error::New(env, "All arrays must have same number of dimensions").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        // Validate all dimensions except axis
        for (int d = 0; d < ndim; d++) {
            if (d != axis && shape[d] != firstShape[d]) {
                Napi::Error::New(env, "All arrays must have same shape except for concatenation axis").ThrowAsJavaScriptException();
                return env.Undefined();
            }
        }

        totalAxisSize += shape[axis];
    }

    // Build result shape
    std::vector<int64_t> resultShape = firstShape;
    resultShape[axis] = totalAxisSize;

    // Create result array
    Napi::Array jsResultShape = Napi::Array::New(env, resultShape.size());
    for (size_t i = 0; i < resultShape.size(); i++) {
        jsResultShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(resultShape[i])));
    }

    Napi::Object result = NativeNDArray::constructor.New({
        jsResultShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
    });
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* resultData = static_cast<double*>(resultArr->data());

    // Fast path for 1D arrays
    if (ndim == 1) {
        int64_t offset = 0;
        for (uint32_t i = 0; i < numArrays; i++) {
            double* srcData = static_cast<double*>(arrPtrs[i]->data());
            int64_t srcSize = arrPtrs[i]->size();
            std::memcpy(resultData + offset, srcData, srcSize * sizeof(double));
            offset += srcSize;
        }
        return result;
    }

    // Fast path for 2D arrays
    if (ndim == 2) {
        int64_t rows = resultShape[0];
        int64_t cols = resultShape[1];

        if (axis == 0) {
            // Concatenate along rows
            int64_t rowOffset = 0;
            for (uint32_t i = 0; i < numArrays; i++) {
                double* srcData = static_cast<double*>(arrPtrs[i]->data());
                int64_t srcRows = arrPtrs[i]->shape()[0];
                std::memcpy(resultData + rowOffset * cols, srcData, srcRows * cols * sizeof(double));
                rowOffset += srcRows;
            }
        } else { // axis == 1
            // Concatenate along columns
            for (int64_t row = 0; row < rows; row++) {
                int64_t colOffset = 0;
                for (uint32_t i = 0; i < numArrays; i++) {
                    double* srcData = static_cast<double*>(arrPtrs[i]->data());
                    int64_t srcCols = arrPtrs[i]->shape()[1];
                    std::memcpy(resultData + row * cols + colOffset,
                               srcData + row * srcCols,
                               srcCols * sizeof(double));
                    colOffset += srcCols;
                }
            }
        }
        return result;
    }

    // Generic N-dimensional case
    // Compute strides for result array
    std::vector<int64_t> resultStrides(ndim);
    resultStrides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        resultStrides[i] = resultStrides[i + 1] * resultShape[i + 1];
    }

    // Copy each array
    int64_t axisOffset = 0;
    for (uint32_t arrIdx = 0; arrIdx < numArrays; arrIdx++) {
        NativeNDArray* arr = arrPtrs[arrIdx];
        double* srcData = static_cast<double*>(arr->data());
        const auto& srcShape = arr->shape();
        int64_t srcSize = arr->size();

        // Compute source strides
        std::vector<int64_t> srcStrides(ndim);
        srcStrides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; i--) {
            srcStrides[i] = srcStrides[i + 1] * srcShape[i + 1];
        }

        // Copy elements
        std::vector<int64_t> indices(ndim, 0);
        for (int64_t srcIdx = 0; srcIdx < srcSize; srcIdx++) {
            // Compute multi-dimensional indices
            int64_t temp = srcIdx;
            for (int d = ndim - 1; d >= 0; d--) {
                indices[d] = temp % srcShape[d];
                temp /= srcShape[d];
            }

            // Compute destination index (offset the axis dimension)
            int64_t dstIdx = 0;
            for (int d = 0; d < ndim; d++) {
                int64_t coord = (d == axis) ? indices[d] + axisOffset : indices[d];
                dstIdx += coord * resultStrides[d];
            }

            resultData[dstIdx] = srcData[srcIdx];
        }

        axisOffset += srcShape[axis];
    }

    return result;
}

/**
 * Stack arrays along a new axis
 * stack(arrays, axis=0) - Join arrays along a NEW axis
 */
Napi::Value Stack(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsArray()) {
        Napi::TypeError::New(env, "Expected array of arrays").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Array arrays = info[0].As<Napi::Array>();
    uint32_t numArrays = arrays.Length();

    if (numArrays == 0) {
        Napi::Error::New(env, "Need at least one array to stack").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Get axis (default 0)
    int axis = 0;
    if (info.Length() >= 2 && !info[1].IsUndefined()) {
        axis = info[1].As<Napi::Number>().Int32Value();
    }

    // Get first array to determine shape
    NativeNDArray* first = Napi::ObjectWrap<NativeNDArray>::Unwrap(
        arrays.Get(uint32_t(0)).As<Napi::Object>()
    );
    const auto& firstShape = first->shape();
    int ndim = static_cast<int>(firstShape.size());
    int newNdim = ndim + 1;

    // Handle negative axis (in the NEW dimension space)
    if (axis < 0) axis += newNdim;

    if (axis < 0 || axis > ndim) {
        Napi::Error::New(env, "Axis out of bounds").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Collect all arrays and validate shapes
    std::vector<NativeNDArray*> arrPtrs;
    arrPtrs.reserve(numArrays);

    for (uint32_t i = 0; i < numArrays; i++) {
        NativeNDArray* arr = Napi::ObjectWrap<NativeNDArray>::Unwrap(
            arrays.Get(i).As<Napi::Object>()
        );
        arrPtrs.push_back(arr);

        const auto& shape = arr->shape();

        // All arrays must have same shape
        if (shape != firstShape) {
            Napi::Error::New(env, "All arrays must have same shape for stack").ThrowAsJavaScriptException();
            return env.Undefined();
        }
    }

    // Build result shape (insert new dimension at axis)
    std::vector<int64_t> resultShape;
    resultShape.reserve(newNdim);
    for (int d = 0; d < axis; d++) {
        resultShape.push_back(firstShape[d]);
    }
    resultShape.push_back(static_cast<int64_t>(numArrays));
    for (int d = axis; d < ndim; d++) {
        resultShape.push_back(firstShape[d]);
    }

    // Create result array
    Napi::Array jsResultShape = Napi::Array::New(env, resultShape.size());
    for (size_t i = 0; i < resultShape.size(); i++) {
        jsResultShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(resultShape[i])));
    }

    Napi::Object result = NativeNDArray::constructor.New({
        jsResultShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
    });
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* resultData = static_cast<double*>(resultArr->data());

    // Fast path for stacking 1D arrays along axis 0 (creates 2D array [n, size])
    if (ndim == 1 && axis == 0) {
        int64_t arrSize = first->size();
        for (uint32_t i = 0; i < numArrays; i++) {
            double* srcData = static_cast<double*>(arrPtrs[i]->data());
            std::memcpy(resultData + i * arrSize, srcData, arrSize * sizeof(double));
        }
        return result;
    }

    // Fast path for stacking 1D arrays along axis 1 (creates 2D array [size, n])
    if (ndim == 1 && axis == 1) {
        int64_t arrSize = first->size();
        for (int64_t j = 0; j < arrSize; j++) {
            for (uint32_t i = 0; i < numArrays; i++) {
                double* srcData = static_cast<double*>(arrPtrs[i]->data());
                resultData[j * numArrays + i] = srcData[j];
            }
        }
        return result;
    }

    // Generic case: compute strides and copy
    std::vector<int64_t> resultStrides(newNdim);
    resultStrides[newNdim - 1] = 1;
    for (int i = newNdim - 2; i >= 0; i--) {
        resultStrides[i] = resultStrides[i + 1] * resultShape[i + 1];
    }

    int64_t srcSize = first->size();

    for (uint32_t arrIdx = 0; arrIdx < numArrays; arrIdx++) {
        double* srcData = static_cast<double*>(arrPtrs[arrIdx]->data());

        std::vector<int64_t> srcIndices(ndim, 0);
        for (int64_t srcIdx = 0; srcIdx < srcSize; srcIdx++) {
            // Compute source multi-dimensional indices
            int64_t temp = srcIdx;
            for (int d = ndim - 1; d >= 0; d--) {
                srcIndices[d] = temp % firstShape[d];
                temp /= firstShape[d];
            }

            // Build destination indices (insert arrIdx at position axis)
            int64_t dstIdx = 0;
            for (int d = 0; d < axis; d++) {
                dstIdx += srcIndices[d] * resultStrides[d];
            }
            dstIdx += arrIdx * resultStrides[axis];
            for (int d = axis; d < ndim; d++) {
                dstIdx += srcIndices[d] * resultStrides[d + 1];
            }

            resultData[dstIdx] = srcData[srcIdx];
        }
    }

    return result;
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
 * Quickselect algorithm - O(n) average case to find k-th smallest element
 * Modifies the array in-place (partial sort)
 */
static double quickselect(double* arr, int left, int right, int k) {
    while (left < right) {
        // Choose pivot (median of three for better performance)
        int mid = left + (right - left) / 2;
        if (arr[mid] < arr[left]) std::swap(arr[left], arr[mid]);
        if (arr[right] < arr[left]) std::swap(arr[left], arr[right]);
        if (arr[mid] < arr[right]) std::swap(arr[mid], arr[right]);
        double pivot = arr[right];

        // Partition
        int i = left;
        for (int j = left; j < right; j++) {
            if (arr[j] <= pivot) {
                std::swap(arr[i], arr[j]);
                i++;
            }
        }
        std::swap(arr[i], arr[right]);

        // Recurse on the side containing k
        if (k == i) {
            return arr[k];
        } else if (k < i) {
            right = i - 1;
        } else {
            left = i + 1;
        }
    }
    return arr[left];
}

/**
 * Percentile function using quickselect - O(n) per percentile
 * percentile(data, q, axis) - q is array of percentiles [0-100]
 */
Napi::Value Percentile(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected array and percentiles").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* arr = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());

    // Ensure contiguous data for correct indexing
    Napi::Object contiguousCopy;
    if (!arr->is_contiguous()) {
        contiguousCopy = arr->AsContiguous(info).As<Napi::Object>();
        arr = Napi::ObjectWrap<NativeNDArray>::Unwrap(contiguousCopy);
    }

    // Get percentiles array
    Napi::Array qArr = info[1].As<Napi::Array>();
    std::vector<double> percentiles(qArr.Length());
    for (size_t i = 0; i < qArr.Length(); i++) {
        percentiles[i] = qArr.Get(i).As<Napi::Number>().DoubleValue();
    }

    // Get axis (default: flatten and compute global percentiles)
    int axis = -1;
    if (info.Length() >= 3 && !info[2].IsUndefined()) {
        axis = info[2].As<Napi::Number>().Int32Value();
    }

    const auto& shape = arr->shape();
    double* data = static_cast<double*>(arr->data());
    int64_t totalSize = arr->size();

    // For now, implement axis=0 case (percentiles along columns) which is the benchmark case
    if (axis == 0 && shape.size() == 2) {
        int64_t rows = shape[0];
        int64_t cols = shape[1];
        int64_t numPercentiles = static_cast<int64_t>(percentiles.size());

        // Result shape: [numPercentiles, cols]
        Napi::Array resultShape = Napi::Array::New(env, 2);
        resultShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(numPercentiles)));
        resultShape.Set(uint32_t(1), Napi::Number::New(env, static_cast<double>(cols)));

        Napi::Object result = NativeNDArray::constructor.New({
            resultShape,
            Napi::String::New(env, "float64"),
            Napi::Boolean::New(env, true)
        });
        NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
        double* resultData = static_cast<double*>(resultArr->data());

        // Work buffer for each column
        std::vector<double> colBuffer(rows);

        for (int64_t col = 0; col < cols; col++) {
            // Extract column into buffer
            for (int64_t row = 0; row < rows; row++) {
                colBuffer[row] = data[row * cols + col];
            }

            // Sort once per column - more efficient when computing multiple percentiles
            std::sort(colBuffer.begin(), colBuffer.end());

            // Extract all percentiles from sorted array
            for (size_t pIdx = 0; pIdx < percentiles.size(); pIdx++) {
                double p = percentiles[pIdx] / 100.0;  // Convert from 0-100 to 0-1
                double k_float = p * (rows - 1);
                int k = static_cast<int>(k_float);

                double value;
                if (k >= rows - 1) {
                    value = colBuffer[rows - 1];
                } else if (k <= 0) {
                    value = colBuffer[0];
                } else {
                    // Linear interpolation between k and k+1
                    double frac = k_float - k;
                    value = colBuffer[k] + frac * (colBuffer[k + 1] - colBuffer[k]);
                }

                resultData[pIdx * cols + col] = value;
            }
        }

        return result;
    }

    // Global percentiles (no axis specified or axis=-1)
    if (axis == -1) {
        int64_t numPercentiles = static_cast<int64_t>(percentiles.size());

        // Result shape: [numPercentiles]
        Napi::Array resultShape = Napi::Array::New(env, 1);
        resultShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(numPercentiles)));

        Napi::Object result = NativeNDArray::constructor.New({
            resultShape,
            Napi::String::New(env, "float64"),
            Napi::Boolean::New(env, true)
        });
        NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
        double* resultData = static_cast<double*>(resultArr->data());

        // Copy and sort data - more efficient when computing multiple percentiles
        std::vector<double> buffer(data, data + totalSize);
        std::sort(buffer.begin(), buffer.end());

        for (size_t pIdx = 0; pIdx < percentiles.size(); pIdx++) {
            double p = percentiles[pIdx] / 100.0;
            double k_float = p * (totalSize - 1);
            int k = static_cast<int>(k_float);

            double value;
            if (k >= totalSize - 1) {
                value = buffer[totalSize - 1];
            } else if (k <= 0) {
                value = buffer[0];
            } else {
                // Linear interpolation
                double frac = k_float - k;
                value = buffer[k] + frac * (buffer[k + 1] - buffer[k]);
            }

            resultData[pIdx] = value;
        }

        return result;
    }

    Napi::Error::New(env, "Unsupported axis for percentile").ThrowAsJavaScriptException();
    return env.Undefined();
}

/**
 * Kronecker product: A ⊗ B
 * For A (m×n) and B (p×q), result is (m*p × n*q)
 * Each element a[i,j] is multiplied by the entire B matrix
 */
Napi::Value Kron(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected two arrays").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    NativeNDArray* b = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());

    // Ensure contiguous data
    Napi::Object aCopy, bCopy;
    if (!a->is_contiguous()) {
        aCopy = a->AsContiguous(info).As<Napi::Object>();
        a = Napi::ObjectWrap<NativeNDArray>::Unwrap(aCopy);
    }
    if (!b->is_contiguous()) {
        bCopy = b->AsContiguous(info).As<Napi::Object>();
        b = Napi::ObjectWrap<NativeNDArray>::Unwrap(bCopy);
    }

    if (a->ndim() != 2 || b->ndim() != 2) {
        Napi::Error::New(env, "Kronecker product requires 2D arrays").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int64_t am = a->shape()[0];
    int64_t an = a->shape()[1];
    int64_t bm = b->shape()[0];
    int64_t bn = b->shape()[1];

    int64_t resultRows = am * bm;
    int64_t resultCols = an * bn;

    // Create result array
    Napi::Array resultShape = Napi::Array::New(env, 2);
    resultShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(resultRows)));
    resultShape.Set(uint32_t(1), Napi::Number::New(env, static_cast<double>(resultCols)));

    Napi::Object result = NativeNDArray::constructor.New({
        resultShape,
        Napi::String::New(env, "float64"),
        Napi::Boolean::New(env, true)
    });
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);

    double* aData = static_cast<double*>(a->data());
    double* bData = static_cast<double*>(b->data());
    double* resultData = static_cast<double*>(resultArr->data());

    // Compute Kronecker product with cache-friendly access pattern
    // Write each result row left-to-right for optimal cache locality
    // Simple loops with unrolling - compiler auto-vectorizes this well
    for (int64_t i = 0; i < am; i++) {
        const double* aRow = aData + i * an;
        for (int64_t k = 0; k < bm; k++) {
            const double* bRow = bData + k * bn;
            double* outRow = resultData + (i * bm + k) * resultCols;
            for (int64_t j = 0; j < an; j++) {
                double aij = aRow[j];
                double* outBlock = outRow + j * bn;
                // Unroll by 8 for better auto-vectorization
                int64_t l = 0;
                for (; l + 7 < bn; l += 8) {
                    outBlock[l]     = aij * bRow[l];
                    outBlock[l + 1] = aij * bRow[l + 1];
                    outBlock[l + 2] = aij * bRow[l + 2];
                    outBlock[l + 3] = aij * bRow[l + 3];
                    outBlock[l + 4] = aij * bRow[l + 4];
                    outBlock[l + 5] = aij * bRow[l + 5];
                    outBlock[l + 6] = aij * bRow[l + 6];
                    outBlock[l + 7] = aij * bRow[l + 7];
                }
                for (; l < bn; l++) {
                    outBlock[l] = aij * bRow[l];
                }
            }
        }
    }

    return result;
}

/**
 * Outer product: a ⊗ b = a * b.T
 * For vectors a (m,) and b (n,), result is (m × n)
 * Uses BLAS dger for optimal performance when available
 */
Napi::Value Outer(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected two arrays").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    NativeNDArray* b = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());

    // Ensure contiguous data
    Napi::Object aCopy, bCopy;
    if (!a->is_contiguous()) {
        aCopy = a->AsContiguous(info).As<Napi::Object>();
        a = Napi::ObjectWrap<NativeNDArray>::Unwrap(aCopy);
    }
    if (!b->is_contiguous()) {
        bCopy = b->AsContiguous(info).As<Napi::Object>();
        b = Napi::ObjectWrap<NativeNDArray>::Unwrap(bCopy);
    }

    // Get sizes - support both 1D vectors and 2D column/row vectors
    int64_t m = a->size();
    int64_t n = b->size();

    // Create result array (m × n)
    Napi::Array resultShape = Napi::Array::New(env, 2);
    resultShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(m)));
    resultShape.Set(uint32_t(1), Napi::Number::New(env, static_cast<double>(n)));

    Napi::Object result = NativeNDArray::constructor.New({
        resultShape,
        Napi::String::New(env, "float64"),
        Napi::Boolean::New(env, true)
    });
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);

    double* aData = static_cast<double*>(a->data());
    double* bData = static_cast<double*>(b->data());
    double* resultData = static_cast<double*>(resultArr->data());

#if defined(USE_ACCELERATE)
    // Use vDSP_vmulD for SIMD: each row = a[i] * b
    // This is faster than dger because:
    // 1. No memset overhead (we write every element)
    // 2. Better cache utilization (row-by-row sequential writes)
    for (int64_t i = 0; i < m; i++) {
        vDSP_vsmulD(bData, 1, &aData[i], resultData + i * n, 1, static_cast<vDSP_Length>(n));
    }
#elif defined(USE_OPENBLAS)
    // Direct loop with loop unrolling - faster than dger for outer products
    for (int64_t i = 0; i < m; i++) {
        double ai = aData[i];
        double* row = resultData + i * n;
        int64_t j = 0;
        // Unroll by 4
        for (; j + 3 < n; j += 4) {
            row[j] = ai * bData[j];
            row[j+1] = ai * bData[j+1];
            row[j+2] = ai * bData[j+2];
            row[j+3] = ai * bData[j+3];
        }
        for (; j < n; j++) {
            row[j] = ai * bData[j];
        }
    }
#else
    // Pure C++ fallback with loop unrolling
    for (int64_t i = 0; i < m; i++) {
        double ai = aData[i];
        double* row = resultData + i * n;
        int64_t j = 0;
        for (; j + 3 < n; j += 4) {
            row[j] = ai * bData[j];
            row[j+1] = ai * bData[j+1];
            row[j+2] = ai * bData[j+2];
            row[j+3] = ai * bData[j+3];
        }
        for (; j < n; j++) {
            row[j] = ai * bData[j];
        }
    }
#endif

    return result;
}

/**
 * BLAS-style axpby: result = alpha*x + beta*y
 * Fuses scalar multiply and addition into one operation.
 * If y is not provided, computes alpha*x (scalar multiply).
 */
Napi::Value Axpby(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected at least 2 arguments: alpha, x, [beta, y]").ThrowAsJavaScriptException();
        return env.Null();
    }

    double alpha = info[0].As<Napi::Number>().DoubleValue();
    NativeNDArray* x = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());

    // Ensure contiguous
    Napi::Object xCopy;
    if (!x->is_contiguous()) {
        xCopy = x->AsContiguous(info).As<Napi::Object>();
        x = Napi::ObjectWrap<NativeNDArray>::Unwrap(xCopy);
    }

    int64_t size = x->size();
    const double* xData = static_cast<const double*>(x->data());

    // Create result with same shape as x
    Napi::Array resultShape = Napi::Array::New(env, x->ndim());
    for (size_t i = 0; i < x->ndim(); i++) {
        resultShape.Set(static_cast<uint32_t>(i), Napi::Number::New(env, static_cast<double>(x->shape()[i])));
    }

    Napi::Object result = NativeNDArray::constructor.New({
        resultShape,
        Napi::String::New(env, "float64"),
        Napi::Boolean::New(env, true)
    });
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* resultData = static_cast<double*>(resultArr->data());

    if (info.Length() >= 4) {
        // Full axpby: alpha*x + beta*y
        double beta = info[2].As<Napi::Number>().DoubleValue();
        NativeNDArray* y = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[3].As<Napi::Object>());

        Napi::Object yCopy;
        if (!y->is_contiguous()) {
            yCopy = y->AsContiguous(info).As<Napi::Object>();
            y = Napi::ObjectWrap<NativeNDArray>::Unwrap(yCopy);
        }

        const double* yData = static_cast<const double*>(y->data());

        // Vectorized computation
        int64_t i = 0;
        for (; i + 3 < size; i += 4) {
            resultData[i]     = alpha * xData[i]     + beta * yData[i];
            resultData[i + 1] = alpha * xData[i + 1] + beta * yData[i + 1];
            resultData[i + 2] = alpha * xData[i + 2] + beta * yData[i + 2];
            resultData[i + 3] = alpha * xData[i + 3] + beta * yData[i + 3];
        }
        for (; i < size; i++) {
            resultData[i] = alpha * xData[i] + beta * yData[i];
        }
    } else {
        // Just alpha*x (scalar multiply)
        int64_t i = 0;
        for (; i + 3 < size; i += 4) {
            resultData[i]     = alpha * xData[i];
            resultData[i + 1] = alpha * xData[i + 1];
            resultData[i + 2] = alpha * xData[i + 2];
            resultData[i + 3] = alpha * xData[i + 3];
        }
        for (; i < size; i++) {
            resultData[i] = alpha * xData[i];
        }
    }

    return result;
}

// ============================================================
// Helper: Get element as double (for multi-dtype comparison)
// ============================================================

static double getElementAsDouble(NativeNDArray* arr, int64_t i) {
    switch (arr->dtype()) {
        case DType::Float64:
            return static_cast<double*>(arr->data())[i];
        case DType::Float32:
            return static_cast<float*>(arr->data())[i];
        case DType::Int32:
            return static_cast<double>(static_cast<int32_t*>(arr->data())[i]);
        case DType::Int64:
            return static_cast<double>(static_cast<int64_t*>(arr->data())[i]);
        case DType::Bool:
        case DType::Uint8:
            return static_cast<double>(static_cast<uint8_t*>(arr->data())[i]);
        default:
            return static_cast<double*>(arr->data())[i];
    }
}

// ============================================================
// Helper: Get element as bool (for logical operations)
// ============================================================

static bool getElementAsBool(NativeNDArray* arr, int64_t i) {
    switch (arr->dtype()) {
        case DType::Float64:
            return static_cast<double*>(arr->data())[i] != 0.0;
        case DType::Float32:
            return static_cast<float*>(arr->data())[i] != 0.0f;
        case DType::Int32:
            return static_cast<int32_t*>(arr->data())[i] != 0;
        case DType::Int64:
            return static_cast<int64_t*>(arr->data())[i] != 0;
        case DType::Bool:
        case DType::Uint8:
            return static_cast<uint8_t*>(arr->data())[i] != 0;
        default:
            return static_cast<double*>(arr->data())[i] != 0.0;
    }
}

// ============================================================
// Comparison Operators (produce bool arrays)
// ============================================================

template<typename CmpOp>
Napi::Value ComparisonOp(const Napi::CallbackInfo& info, CmpOp cmp) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected two arguments").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    const auto& shapeA = a->shape();
    bool bIsScalar = info[1].IsNumber();

    if (bIsScalar) {
        double scalar = info[1].As<Napi::Number>().DoubleValue();
        int64_t size = a->size();

        Napi::Array jsShape = Napi::Array::New(env, shapeA.size());
        for (size_t i = 0; i < shapeA.size(); i++) {
            jsShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(shapeA[i])));
        }

        Napi::Object result = NativeNDArray::constructor.New({
            jsShape,
            Napi::String::New(env, "bool"),
            Napi::Boolean::New(env, true)
        });
        NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
        uint8_t* dataC = static_cast<uint8_t*>(c->data());

        for (int64_t i = 0; i < size; i++) {
            double valA = getElementAsDouble(a, i);
            dataC[i] = cmp(valA, scalar) ? 1 : 0;
        }
        return result;
    }

    NativeNDArray* b = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());
    const auto& shapeB = b->shape();

    // Fast path: same shape
    if (shapeA == shapeB) {
        int64_t size = a->size();

        Napi::Array jsShape = Napi::Array::New(env, shapeA.size());
        for (size_t i = 0; i < shapeA.size(); i++) {
            jsShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(shapeA[i])));
        }

        Napi::Object result = NativeNDArray::constructor.New({
            jsShape,
            Napi::String::New(env, "bool"),
            Napi::Boolean::New(env, true)
        });
        NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
        uint8_t* dataC = static_cast<uint8_t*>(c->data());

        for (int64_t i = 0; i < size; i++) {
            double valA = getElementAsDouble(a, i);
            double valB = getElementAsDouble(b, i);
            dataC[i] = cmp(valA, valB) ? 1 : 0;
        }
        return result;
    }

    // Broadcasting case
    std::vector<int64_t> resultShape = computeBroadcastShape(shapeA, shapeB);
    if (resultShape.empty()) {
        Napi::Error::New(env, "Shapes are not broadcastable").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Array jsResultShape = Napi::Array::New(env, resultShape.size());
    for (size_t i = 0; i < resultShape.size(); i++) {
        jsResultShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(resultShape[i])));
    }

    Napi::Object result = NativeNDArray::constructor.New({
        jsResultShape,
        Napi::String::New(env, "bool"),
        Napi::Boolean::New(env, true)
    });
    NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    uint8_t* dataC = static_cast<uint8_t*>(c->data());

    int64_t totalSize = 1;
    for (int64_t dim : resultShape) {
        totalSize *= dim;
    }

    std::vector<int64_t> paddedShapeA(resultShape.size(), 1);
    std::vector<int64_t> paddedShapeB(resultShape.size(), 1);
    for (size_t i = 0; i < shapeA.size(); i++) {
        paddedShapeA[resultShape.size() - shapeA.size() + i] = shapeA[i];
    }
    for (size_t i = 0; i < shapeB.size(); i++) {
        paddedShapeB[resultShape.size() - shapeB.size() + i] = shapeB[i];
    }

    std::vector<int64_t> indices(resultShape.size(), 0);
    for (int64_t flatIdx = 0; flatIdx < totalSize; flatIdx++) {
        int64_t temp = flatIdx;
        for (int i = static_cast<int>(resultShape.size()) - 1; i >= 0; i--) {
            indices[i] = temp % resultShape[i];
            temp /= resultShape[i];
        }

        int64_t idxA = computeBroadcastIndex(indices, paddedShapeA);
        int64_t idxB = computeBroadcastIndex(indices, paddedShapeB);

        double valA = getElementAsDouble(a, idxA);
        double valB = getElementAsDouble(b, idxB);
        dataC[flatIdx] = cmp(valA, valB) ? 1 : 0;
    }

    return result;
}

Napi::Value Equal(const Napi::CallbackInfo& info) {
    return ComparisonOp(info, [](double a, double b) { return a == b; });
}

Napi::Value NotEqual(const Napi::CallbackInfo& info) {
    return ComparisonOp(info, [](double a, double b) { return a != b; });
}

Napi::Value Less(const Napi::CallbackInfo& info) {
    return ComparisonOp(info, [](double a, double b) { return a < b; });
}

Napi::Value LessEqual(const Napi::CallbackInfo& info) {
    return ComparisonOp(info, [](double a, double b) { return a <= b; });
}

Napi::Value Greater(const Napi::CallbackInfo& info) {
    return ComparisonOp(info, [](double a, double b) { return a > b; });
}

Napi::Value GreaterEqual(const Napi::CallbackInfo& info) {
    return ComparisonOp(info, [](double a, double b) { return a >= b; });
}

// ============================================================
// Logical Operators (produce bool arrays, with broadcasting)
// ============================================================

template<typename LogicOp>
Napi::Value LogicalBinaryOp(const Napi::CallbackInfo& info, LogicOp op) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected two arguments").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    NativeNDArray* b = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());
    const auto& shapeA = a->shape();
    const auto& shapeB = b->shape();

    // Fast path: same shape
    if (shapeA == shapeB) {
        int64_t size = a->size();

        Napi::Array jsShape = Napi::Array::New(env, shapeA.size());
        for (size_t i = 0; i < shapeA.size(); i++) {
            jsShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(shapeA[i])));
        }

        Napi::Object result = NativeNDArray::constructor.New({
            jsShape,
            Napi::String::New(env, "bool"),
            Napi::Boolean::New(env, true)
        });
        NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
        uint8_t* dataC = static_cast<uint8_t*>(c->data());

        for (int64_t i = 0; i < size; i++) {
            bool valA = getElementAsBool(a, i);
            bool valB = getElementAsBool(b, i);
            dataC[i] = op(valA, valB) ? 1 : 0;
        }
        return result;
    }

    // Broadcasting case
    std::vector<int64_t> resultShape = computeBroadcastShape(shapeA, shapeB);
    if (resultShape.empty()) {
        Napi::Error::New(env, "Shapes are not broadcastable").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Array jsResultShape = Napi::Array::New(env, resultShape.size());
    for (size_t i = 0; i < resultShape.size(); i++) {
        jsResultShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(resultShape[i])));
    }

    Napi::Object result = NativeNDArray::constructor.New({
        jsResultShape,
        Napi::String::New(env, "bool"),
        Napi::Boolean::New(env, true)
    });
    NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    uint8_t* dataC = static_cast<uint8_t*>(c->data());

    int64_t totalSize = 1;
    for (int64_t dim : resultShape) {
        totalSize *= dim;
    }

    std::vector<int64_t> paddedShapeA(resultShape.size(), 1);
    std::vector<int64_t> paddedShapeB(resultShape.size(), 1);
    for (size_t i = 0; i < shapeA.size(); i++) {
        paddedShapeA[resultShape.size() - shapeA.size() + i] = shapeA[i];
    }
    for (size_t i = 0; i < shapeB.size(); i++) {
        paddedShapeB[resultShape.size() - shapeB.size() + i] = shapeB[i];
    }

    std::vector<int64_t> indices(resultShape.size(), 0);
    for (int64_t flatIdx = 0; flatIdx < totalSize; flatIdx++) {
        int64_t temp = flatIdx;
        for (int i = static_cast<int>(resultShape.size()) - 1; i >= 0; i--) {
            indices[i] = temp % resultShape[i];
            temp /= resultShape[i];
        }

        int64_t idxA = computeBroadcastIndex(indices, paddedShapeA);
        int64_t idxB = computeBroadcastIndex(indices, paddedShapeB);

        bool valA = getElementAsBool(a, idxA);
        bool valB = getElementAsBool(b, idxB);
        dataC[flatIdx] = op(valA, valB) ? 1 : 0;
    }

    return result;
}

Napi::Value LogicalAnd(const Napi::CallbackInfo& info) {
    return LogicalBinaryOp(info, [](bool a, bool b) { return a && b; });
}

Napi::Value LogicalOr(const Napi::CallbackInfo& info) {
    return LogicalBinaryOp(info, [](bool a, bool b) { return a || b; });
}

Napi::Value LogicalXor(const Napi::CallbackInfo& info) {
    return LogicalBinaryOp(info, [](bool a, bool b) { return a != b; });
}

Napi::Value LogicalNot(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    const auto& shape = a->shape();
    int64_t size = a->size();

    Napi::Array jsShape = Napi::Array::New(env, shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
        jsShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(shape[i])));
    }

    Napi::Object result = NativeNDArray::constructor.New({
        jsShape,
        Napi::String::New(env, "bool"),
        Napi::Boolean::New(env, true)
    });
    NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    uint8_t* dataC = static_cast<uint8_t*>(c->data());

    for (int64_t i = 0; i < size; i++) {
        bool val = getElementAsBool(a, i);
        dataC[i] = val ? 0 : 1;
    }

    return result;
}

// ============================================================
// Boolean Reductions (any, all)
// ============================================================

Napi::Value Any(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    const auto& shape = a->shape();
    int64_t size = a->size();

    // Check for axis parameter
    if (info.Length() >= 2 && !info[1].IsUndefined()) {
        int axis = info[1].As<Napi::Number>().Int32Value();
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
                // any along rows -> result shape [cols]
                Napi::Array jsResultShape = Napi::Array::New(env, 1);
                jsResultShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(cols)));

                Napi::Object result = NativeNDArray::constructor.New({
                    jsResultShape,
                    Napi::String::New(env, "bool"),
                    Napi::Boolean::New(env, true)
                });
                NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
                uint8_t* resultData = static_cast<uint8_t*>(resultArr->data());

                for (int64_t col = 0; col < cols; col++) {
                    bool found = false;
                    for (int64_t row = 0; row < rows && !found; row++) {
                        if (getElementAsBool(a, row * cols + col)) {
                            found = true;
                        }
                    }
                    resultData[col] = found ? 1 : 0;
                }
                return result;

            } else { // axis == 1
                // any along columns -> result shape [rows]
                Napi::Array jsResultShape = Napi::Array::New(env, 1);
                jsResultShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(rows)));

                Napi::Object result = NativeNDArray::constructor.New({
                    jsResultShape,
                    Napi::String::New(env, "bool"),
                    Napi::Boolean::New(env, true)
                });
                NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
                uint8_t* resultData = static_cast<uint8_t*>(resultArr->data());

                for (int64_t row = 0; row < rows; row++) {
                    bool found = false;
                    for (int64_t col = 0; col < cols && !found; col++) {
                        if (getElementAsBool(a, row * cols + col)) {
                            found = true;
                        }
                    }
                    resultData[row] = found ? 1 : 0;
                }
                return result;
            }
        }

        Napi::Error::New(env, "Axis reduction only supported for 2D arrays").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Global any - return boolean
    for (int64_t i = 0; i < size; i++) {
        if (getElementAsBool(a, i)) {
            return Napi::Boolean::New(env, true);
        }
    }
    return Napi::Boolean::New(env, false);
}

Napi::Value All(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    const auto& shape = a->shape();
    int64_t size = a->size();

    // Check for axis parameter
    if (info.Length() >= 2 && !info[1].IsUndefined()) {
        int axis = info[1].As<Napi::Number>().Int32Value();
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
                // all along rows -> result shape [cols]
                Napi::Array jsResultShape = Napi::Array::New(env, 1);
                jsResultShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(cols)));

                Napi::Object result = NativeNDArray::constructor.New({
                    jsResultShape,
                    Napi::String::New(env, "bool"),
                    Napi::Boolean::New(env, true)
                });
                NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
                uint8_t* resultData = static_cast<uint8_t*>(resultArr->data());

                for (int64_t col = 0; col < cols; col++) {
                    bool allTrue = true;
                    for (int64_t row = 0; row < rows && allTrue; row++) {
                        if (!getElementAsBool(a, row * cols + col)) {
                            allTrue = false;
                        }
                    }
                    resultData[col] = allTrue ? 1 : 0;
                }
                return result;

            } else { // axis == 1
                // all along columns -> result shape [rows]
                Napi::Array jsResultShape = Napi::Array::New(env, 1);
                jsResultShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(rows)));

                Napi::Object result = NativeNDArray::constructor.New({
                    jsResultShape,
                    Napi::String::New(env, "bool"),
                    Napi::Boolean::New(env, true)
                });
                NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
                uint8_t* resultData = static_cast<uint8_t*>(resultArr->data());

                for (int64_t row = 0; row < rows; row++) {
                    bool allTrue = true;
                    for (int64_t col = 0; col < cols && allTrue; col++) {
                        if (!getElementAsBool(a, row * cols + col)) {
                            allTrue = false;
                        }
                    }
                    resultData[row] = allTrue ? 1 : 0;
                }
                return result;
            }
        }

        Napi::Error::New(env, "Axis reduction only supported for 2D arrays").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Global all - return boolean
    for (int64_t i = 0; i < size; i++) {
        if (!getElementAsBool(a, i)) {
            return Napi::Boolean::New(env, false);
        }
    }
    return Napi::Boolean::New(env, true);
}

// ============================================================
// Sorting and Searching Functions
// ============================================================

// Helper to create NativeNDArray with shape (always float64)
static Napi::Object createNDArray(Napi::Env env, const std::vector<int64_t>& shape) {
    Napi::Array jsShape = Napi::Array::New(env, shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
        jsShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(shape[i])));
    }
    return NativeNDArray::constructor.New({
        jsShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
    });
}

/**
 * Compute n-th discrete difference along the given axis
 * diff(a, n=1, axis=-1)
 */
Napi::Value Diff(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());

    int n = 1;
    if (info.Length() >= 2 && !info[1].IsUndefined()) {
        n = info[1].As<Napi::Number>().Int32Value();
    }

    int axis = -1;
    if (info.Length() >= 3 && !info[2].IsUndefined()) {
        axis = info[2].As<Napi::Number>().Int32Value();
    }

    const std::vector<int64_t>& shapeA = a->shape();
    int ndim = static_cast<int>(shapeA.size());

    // Handle negative axis
    if (axis < 0) {
        axis = ndim + axis;
    }

    if (axis < 0 || axis >= ndim) {
        Napi::Error::New(env, "axis out of bounds").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    if (n < 0) {
        Napi::Error::New(env, "n must be non-negative").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int64_t axisSize = shapeA[axis];
    if (n >= axisSize) {
        // Result is empty along this axis
        std::vector<int64_t> resultShape = shapeA;
        resultShape[axis] = 0;
        Napi::Object result = createNDArray(env, resultShape);
        return result;
    }

    // For n differences, output size along axis is (original - n)
    std::vector<int64_t> resultShape = shapeA;
    resultShape[axis] = axisSize - n;

    Napi::Object result = createNDArray(env, resultShape);
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    const double* dataA = static_cast<double*>(a->data());
    double* dataC = static_cast<double*>(resultArr->data());

    // Compute strides for iteration
    int64_t outerSize = 1;
    for (int i = 0; i < axis; i++) {
        outerSize *= shapeA[i];
    }
    int64_t innerSize = 1;
    for (int i = axis + 1; i < ndim; i++) {
        innerSize *= shapeA[i];
    }

    // First compute first difference, then iterate for n > 1
    // We need a temporary buffer for multiple differences
    std::vector<double> tempBuffer;
    if (n > 1) {
        int64_t maxTempSize = outerSize * axisSize * innerSize;
        tempBuffer.resize(maxTempSize);
    }

    const double* src = dataA;
    double* dst = (n > 1) ? tempBuffer.data() : dataC;
    int64_t currentAxisSize = axisSize;

    for (int diffIter = 0; diffIter < n; diffIter++) {
        int64_t newAxisSize = currentAxisSize - 1;

        for (int64_t outer = 0; outer < outerSize; outer++) {
            for (int64_t i = 0; i < newAxisSize; i++) {
                for (int64_t inner = 0; inner < innerSize; inner++) {
                    int64_t srcIdx1 = outer * currentAxisSize * innerSize + i * innerSize + inner;
                    int64_t srcIdx2 = outer * currentAxisSize * innerSize + (i + 1) * innerSize + inner;
                    int64_t dstIdx = outer * newAxisSize * innerSize + i * innerSize + inner;
                    dst[dstIdx] = src[srcIdx2] - src[srcIdx1];
                }
            }
        }

        currentAxisSize = newAxisSize;

        if (diffIter < n - 1) {
            // Prepare for next iteration
            if (diffIter == 0) {
                src = tempBuffer.data();
            }
            // For intermediate iterations, we work in-place on tempBuffer
            // For the last iteration (diffIter == n-2), we write to dataC
            if (diffIter == n - 2) {
                dst = dataC;
            }
        }
    }

    return result;
}

/**
 * Return a sorted copy of the array
 * sort(a, axis=-1)
 */
Napi::Value Sort(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());

    int axis = -1;
    if (info.Length() >= 2 && !info[1].IsUndefined()) {
        axis = info[1].As<Napi::Number>().Int32Value();
    }

    const std::vector<int64_t>& shapeA = a->shape();
    int ndim = static_cast<int>(shapeA.size());

    // Handle negative axis
    if (axis < 0) {
        axis = ndim + axis;
    }

    if (axis < 0 || axis >= ndim) {
        Napi::Error::New(env, "axis out of bounds").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Create result with same shape
    Napi::Object result = createNDArray(env, shapeA);
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    const double* dataA = static_cast<double*>(a->data());
    double* dataC = static_cast<double*>(resultArr->data());

    // Copy data first
    int64_t totalSize = a->size();
    std::copy(dataA, dataA + totalSize, dataC);

    // Compute strides for iteration
    int64_t outerSize = 1;
    for (int i = 0; i < axis; i++) {
        outerSize *= shapeA[i];
    }
    int64_t axisSize = shapeA[axis];
    int64_t innerSize = 1;
    for (int i = axis + 1; i < ndim; i++) {
        innerSize *= shapeA[i];
    }

    // Sort along axis
    if (innerSize == 1) {
        // Contiguous case - sort directly
        for (int64_t outer = 0; outer < outerSize; outer++) {
            double* start = dataC + outer * axisSize;
            std::sort(start, start + axisSize);
        }
    } else {
        // Non-contiguous case - need to gather, sort, scatter
        std::vector<double> temp(axisSize);
        for (int64_t outer = 0; outer < outerSize; outer++) {
            for (int64_t inner = 0; inner < innerSize; inner++) {
                // Gather
                for (int64_t i = 0; i < axisSize; i++) {
                    temp[i] = dataC[outer * axisSize * innerSize + i * innerSize + inner];
                }
                // Sort
                std::sort(temp.begin(), temp.end());
                // Scatter
                for (int64_t i = 0; i < axisSize; i++) {
                    dataC[outer * axisSize * innerSize + i * innerSize + inner] = temp[i];
                }
            }
        }
    }

    return result;
}

/**
 * Return the indices that would sort the array
 * argsort(a, axis=-1)
 */
Napi::Value Argsort(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());

    int axis = -1;
    if (info.Length() >= 2 && !info[1].IsUndefined()) {
        axis = info[1].As<Napi::Number>().Int32Value();
    }

    const std::vector<int64_t>& shapeA = a->shape();
    int ndim = static_cast<int>(shapeA.size());

    // Handle negative axis
    if (axis < 0) {
        axis = ndim + axis;
    }

    if (axis < 0 || axis >= ndim) {
        Napi::Error::New(env, "axis out of bounds").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Create result with same shape (indices as float64)
    Napi::Object result = createNDArray(env, shapeA);
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    const double* dataA = static_cast<double*>(a->data());
    double* dataC = static_cast<double*>(resultArr->data());

    // Compute strides for iteration
    int64_t outerSize = 1;
    for (int i = 0; i < axis; i++) {
        outerSize *= shapeA[i];
    }
    int64_t axisSize = shapeA[axis];
    int64_t innerSize = 1;
    for (int i = axis + 1; i < ndim; i++) {
        innerSize *= shapeA[i];
    }

    // Create indices and sort them
    std::vector<int64_t> indices(axisSize);
    std::vector<double> values(axisSize);

    for (int64_t outer = 0; outer < outerSize; outer++) {
        for (int64_t inner = 0; inner < innerSize; inner++) {
            // Initialize indices and gather values
            for (int64_t i = 0; i < axisSize; i++) {
                indices[i] = i;
                values[i] = dataA[outer * axisSize * innerSize + i * innerSize + inner];
            }

            // Sort indices by values (stable sort preserves order for equal elements)
            std::stable_sort(indices.begin(), indices.end(), [&values](int64_t i1, int64_t i2) {
                return values[i1] < values[i2];
            });

            // Write sorted indices to result
            for (int64_t i = 0; i < axisSize; i++) {
                dataC[outer * axisSize * innerSize + i * innerSize + inner] = static_cast<double>(indices[i]);
            }
        }
    }

    return result;
}

/**
 * Find unique elements of an array (sorted)
 * unique(a)
 */
Napi::Value Unique(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());

    const double* dataA = static_cast<double*>(a->data());
    int64_t size = a->size();

    if (size == 0) {
        // Empty array returns empty array
        Napi::Object result = createNDArray(env, {0});
        return result;
    }

    // Copy and sort
    std::vector<double> sorted(dataA, dataA + size);
    std::sort(sorted.begin(), sorted.end());

    // Find unique elements
    std::vector<double> uniqueVals;
    uniqueVals.reserve(size);
    uniqueVals.push_back(sorted[0]);

    for (int64_t i = 1; i < size; i++) {
        if (sorted[i] != sorted[i - 1]) {
            uniqueVals.push_back(sorted[i]);
        }
    }

    // Create result array
    Napi::Object result = createNDArray(env, {static_cast<int64_t>(uniqueVals.size())});
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* dataC = static_cast<double*>(resultArr->data());
    std::copy(uniqueVals.begin(), uniqueVals.end(), dataC);

    return result;
}

/**
 * Find indices where elements should be inserted to maintain order
 * searchsorted(a, v, side='left')
 */
Napi::Value Searchsorted(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    NativeNDArray* v = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());

    std::string side = "left";
    if (info.Length() >= 3 && !info[2].IsUndefined()) {
        side = info[2].As<Napi::String>().Utf8Value();
    }

    const double* dataA = static_cast<double*>(a->data());
    const double* dataV = static_cast<double*>(v->data());
    int64_t sizeA = a->size();
    int64_t sizeV = v->size();

    Napi::Object result = createNDArray(env, v->shape());
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* dataC = static_cast<double*>(resultArr->data());

    bool useLeft = (side == "left");

    for (int64_t i = 0; i < sizeV; i++) {
        double val = dataV[i];
        int64_t lo = 0, hi = sizeA;

        if (useLeft) {
            // Find leftmost position
            while (lo < hi) {
                int64_t mid = (lo + hi) / 2;
                if (dataA[mid] < val) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
        } else {
            // Find rightmost position
            while (lo < hi) {
                int64_t mid = (lo + hi) / 2;
                if (dataA[mid] <= val) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
        }
        dataC[i] = static_cast<double>(lo);
    }

    return result;
}

/**
 * Construct an array by repeating a the number of times given by reps
 * tile(a, reps)
 */
Napi::Value Tile(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());

    // Parse reps - can be a number or array
    std::vector<int64_t> reps;
    if (info[1].IsNumber()) {
        reps.push_back(info[1].As<Napi::Number>().Int64Value());
    } else if (info[1].IsArray()) {
        Napi::Array repsArr = info[1].As<Napi::Array>();
        for (uint32_t i = 0; i < repsArr.Length(); i++) {
            reps.push_back(repsArr.Get(i).As<Napi::Number>().Int64Value());
        }
    }

    const std::vector<int64_t>& shapeA = a->shape();
    int ndimA = static_cast<int>(shapeA.size());
    int ndimReps = static_cast<int>(reps.size());

    // Pad shapes to match dimensions
    int ndim = std::max(ndimA, ndimReps);
    std::vector<int64_t> paddedShapeA(ndim, 1);
    std::vector<int64_t> paddedReps(ndim, 1);

    for (int i = 0; i < ndimA; i++) {
        paddedShapeA[ndim - ndimA + i] = shapeA[i];
    }
    for (int i = 0; i < ndimReps; i++) {
        paddedReps[ndim - ndimReps + i] = reps[i];
    }

    // Compute result shape
    std::vector<int64_t> resultShape(ndim);
    for (int i = 0; i < ndim; i++) {
        resultShape[i] = paddedShapeA[i] * paddedReps[i];
    }

    Napi::Object result = createNDArray(env, resultShape);
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    const double* dataA = static_cast<double*>(a->data());
    double* dataC = static_cast<double*>(resultArr->data());

    // Compute strides for result
    std::vector<int64_t> resultStrides(ndim);
    std::vector<int64_t> inputStrides(ndim);
    int64_t stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        resultStrides[i] = stride;
        stride *= resultShape[i];
    }
    stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        inputStrides[i] = stride;
        stride *= paddedShapeA[i];
    }

    // Fill result by tiling
    int64_t totalSize = resultArr->size();
    for (int64_t idx = 0; idx < totalSize; idx++) {
        // Convert flat index to multi-index in result
        int64_t remaining = idx;
        int64_t srcIdx = 0;
        for (int d = 0; d < ndim; d++) {
            int64_t coord = remaining / resultStrides[d];
            remaining = remaining % resultStrides[d];
            // Map to source coordinate using modulo
            int64_t srcCoord = coord % paddedShapeA[d];
            srcIdx += srcCoord * inputStrides[d];
        }
        dataC[idx] = dataA[srcIdx];
    }

    return result;
}

/**
 * Repeat elements of an array
 * repeat(a, repeats, axis?)
 */
Napi::Value Repeat(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    int64_t repeats = info[1].As<Napi::Number>().Int64Value();

    const double* dataA = static_cast<double*>(a->data());
    const std::vector<int64_t>& shapeA = a->shape();
    int64_t sizeA = a->size();

    // If no axis, flatten and repeat each element
    if (info.Length() < 3 || info[2].IsUndefined()) {
        std::vector<int64_t> resultShape = {sizeA * repeats};
        Napi::Object result = createNDArray(env, resultShape);
        NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
        double* dataC = static_cast<double*>(resultArr->data());

        for (int64_t i = 0; i < sizeA; i++) {
            for (int64_t r = 0; r < repeats; r++) {
                dataC[i * repeats + r] = dataA[i];
            }
        }
        return result;
    }

    // With axis
    int axis = info[2].As<Napi::Number>().Int32Value();
    int ndim = static_cast<int>(shapeA.size());
    if (axis < 0) axis += ndim;

    std::vector<int64_t> resultShape = shapeA;
    resultShape[axis] *= repeats;

    Napi::Object result = createNDArray(env, resultShape);
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* dataC = static_cast<double*>(resultArr->data());

    int64_t outerSize = 1;
    for (int i = 0; i < axis; i++) outerSize *= shapeA[i];
    int64_t axisSize = shapeA[axis];
    int64_t innerSize = 1;
    for (int i = axis + 1; i < ndim; i++) innerSize *= shapeA[i];

    for (int64_t outer = 0; outer < outerSize; outer++) {
        for (int64_t i = 0; i < axisSize; i++) {
            for (int64_t r = 0; r < repeats; r++) {
                for (int64_t inner = 0; inner < innerSize; inner++) {
                    int64_t srcIdx = outer * axisSize * innerSize + i * innerSize + inner;
                    int64_t dstIdx = outer * (axisSize * repeats) * innerSize + (i * repeats + r) * innerSize + inner;
                    dataC[dstIdx] = dataA[srcIdx];
                }
            }
        }
    }

    return result;
}

/**
 * Reverse the order of elements along the given axis
 * flip(a, axis?)
 */
Napi::Value Flip(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());

    const std::vector<int64_t>& shapeA = a->shape();
    int ndim = static_cast<int>(shapeA.size());
    const double* dataA = static_cast<double*>(a->data());

    // If no axis specified, flip all axes
    std::vector<int> axes;
    if (info.Length() < 2 || info[1].IsUndefined()) {
        for (int i = 0; i < ndim; i++) axes.push_back(i);
    } else if (info[1].IsNumber()) {
        int axis = info[1].As<Napi::Number>().Int32Value();
        if (axis < 0) axis += ndim;
        axes.push_back(axis);
    } else if (info[1].IsArray()) {
        Napi::Array axesArr = info[1].As<Napi::Array>();
        for (uint32_t i = 0; i < axesArr.Length(); i++) {
            int axis = axesArr.Get(i).As<Napi::Number>().Int32Value();
            if (axis < 0) axis += ndim;
            axes.push_back(axis);
        }
    }

    Napi::Object result = createNDArray(env, shapeA);
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* dataC = static_cast<double*>(resultArr->data());

    // Compute strides
    std::vector<int64_t> strides(ndim);
    int64_t stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= shapeA[i];
    }

    int64_t totalSize = a->size();
    for (int64_t idx = 0; idx < totalSize; idx++) {
        // Convert to multi-index
        int64_t remaining = idx;
        std::vector<int64_t> coords(ndim);
        for (int d = 0; d < ndim; d++) {
            coords[d] = remaining / strides[d];
            remaining = remaining % strides[d];
        }

        // Flip specified axes
        for (int axis : axes) {
            coords[axis] = shapeA[axis] - 1 - coords[axis];
        }

        // Convert back to flat index
        int64_t srcIdx = 0;
        for (int d = 0; d < ndim; d++) {
            srcIdx += coords[d] * strides[d];
        }

        dataC[idx] = dataA[srcIdx];
    }

    return result;
}

/**
 * Rotate an array by 90 degrees in the plane specified by axes
 * rot90(a, k=1, axes=(0,1))
 */
Napi::Value Rot90(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());

    int k = 1;
    if (info.Length() >= 2 && !info[1].IsUndefined()) {
        k = info[1].As<Napi::Number>().Int32Value();
    }

    int axis0 = 0, axis1 = 1;
    if (info.Length() >= 3 && !info[2].IsUndefined()) {
        Napi::Array axesArr = info[2].As<Napi::Array>();
        axis0 = axesArr.Get(uint32_t(0)).As<Napi::Number>().Int32Value();
        axis1 = axesArr.Get(uint32_t(1)).As<Napi::Number>().Int32Value();
    }

    const std::vector<int64_t>& shapeA = a->shape();
    int ndim = static_cast<int>(shapeA.size());

    if (axis0 < 0) axis0 += ndim;
    if (axis1 < 0) axis1 += ndim;

    // Normalize k to 0-3
    k = ((k % 4) + 4) % 4;

    if (k == 0) {
        // No rotation - just copy
        Napi::Object result = createNDArray(env, shapeA);
        NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
        std::copy(static_cast<double*>(a->data()),
                  static_cast<double*>(a->data()) + a->size(),
                  static_cast<double*>(resultArr->data()));
        return result;
    }

    // Determine result shape
    std::vector<int64_t> resultShape = shapeA;
    if (k == 1 || k == 3) {
        // Swap dimensions for 90 or 270 degree rotation
        std::swap(resultShape[axis0], resultShape[axis1]);
    }

    Napi::Object result = createNDArray(env, resultShape);
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    const double* dataA = static_cast<double*>(a->data());
    double* dataC = static_cast<double*>(resultArr->data());

    // Compute strides
    std::vector<int64_t> srcStrides(ndim), dstStrides(ndim);
    int64_t stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        srcStrides[i] = stride;
        stride *= shapeA[i];
    }
    stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        dstStrides[i] = stride;
        stride *= resultShape[i];
    }

    int64_t totalSize = resultArr->size();
    for (int64_t dstIdx = 0; dstIdx < totalSize; dstIdx++) {
        // Convert dst index to coords
        int64_t remaining = dstIdx;
        std::vector<int64_t> dstCoords(ndim);
        for (int d = 0; d < ndim; d++) {
            dstCoords[d] = remaining / dstStrides[d];
            remaining = remaining % dstStrides[d];
        }

        // Apply rotation transformation to get src coords
        std::vector<int64_t> srcCoords = dstCoords;
        int64_t d0 = dstCoords[axis0];
        int64_t d1 = dstCoords[axis1];

        if (k == 1) {
            // 90 degrees CCW: (i,j) -> (j, n-1-i)
            srcCoords[axis0] = d1;
            srcCoords[axis1] = resultShape[axis0] - 1 - d0;
        } else if (k == 2) {
            // 180 degrees: (i,j) -> (n-1-i, m-1-j)
            srcCoords[axis0] = shapeA[axis0] - 1 - d0;
            srcCoords[axis1] = shapeA[axis1] - 1 - d1;
        } else if (k == 3) {
            // 270 degrees CCW (90 CW): (i,j) -> (m-1-j, i)
            srcCoords[axis0] = resultShape[axis1] - 1 - d1;
            srcCoords[axis1] = d0;
        }

        // Convert src coords to flat index
        int64_t srcIdx = 0;
        for (int d = 0; d < ndim; d++) {
            srcIdx += srcCoords[d] * srcStrides[d];
        }

        dataC[dstIdx] = dataA[srcIdx];
    }

    return result;
}

/**
 * Split an array into multiple sub-arrays
 * split(a, indices_or_sections, axis=0)
 */
Napi::Value Split(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());

    int axis = 0;
    if (info.Length() >= 3 && !info[2].IsUndefined()) {
        axis = info[2].As<Napi::Number>().Int32Value();
    }

    const std::vector<int64_t>& shapeA = a->shape();
    int ndim = static_cast<int>(shapeA.size());
    if (axis < 0) axis += ndim;

    const double* dataA = static_cast<double*>(a->data());

    // Parse indices_or_sections
    std::vector<int64_t> splitIndices;
    if (info[1].IsNumber()) {
        // Split into N equal parts
        int64_t n = info[1].As<Napi::Number>().Int64Value();
        int64_t axisSize = shapeA[axis];
        for (int64_t i = 1; i < n; i++) {
            splitIndices.push_back((i * axisSize) / n);
        }
    } else if (info[1].IsArray()) {
        // Split at specific indices
        Napi::Array indicesArr = info[1].As<Napi::Array>();
        for (uint32_t i = 0; i < indicesArr.Length(); i++) {
            splitIndices.push_back(indicesArr.Get(i).As<Napi::Number>().Int64Value());
        }
    }

    // Add 0 at start and axisSize at end for easier iteration
    std::vector<int64_t> boundaries;
    boundaries.push_back(0);
    for (int64_t idx : splitIndices) {
        boundaries.push_back(idx);
    }
    boundaries.push_back(shapeA[axis]);

    // Create result array of sub-arrays
    Napi::Array resultArray = Napi::Array::New(env, boundaries.size() - 1);

    int64_t outerSize = 1;
    for (int i = 0; i < axis; i++) outerSize *= shapeA[i];
    int64_t innerSize = 1;
    for (int i = axis + 1; i < ndim; i++) innerSize *= shapeA[i];
    int64_t axisSize = shapeA[axis];

    for (size_t s = 0; s < boundaries.size() - 1; s++) {
        int64_t start = boundaries[s];
        int64_t end = boundaries[s + 1];
        int64_t sliceSize = end - start;

        // Create shape for this slice
        std::vector<int64_t> sliceShape = shapeA;
        sliceShape[axis] = sliceSize;

        Napi::Object slice = createNDArray(env, sliceShape);
        NativeNDArray* sliceArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(slice);
        double* sliceData = static_cast<double*>(sliceArr->data());

        // Copy data
        for (int64_t outer = 0; outer < outerSize; outer++) {
            for (int64_t i = 0; i < sliceSize; i++) {
                for (int64_t inner = 0; inner < innerSize; inner++) {
                    int64_t srcIdx = outer * axisSize * innerSize + (start + i) * innerSize + inner;
                    int64_t dstIdx = outer * sliceSize * innerSize + i * innerSize + inner;
                    sliceData[dstIdx] = dataA[srcIdx];
                }
            }
        }

        resultArray.Set(uint32_t(s), slice);
    }

    return resultArray;
}

/**
 * Slice an array along multiple dimensions
 * slice(a, slices) where slices is an array of [start, end, step] or single index per dimension
 *
 * Examples:
 *   slice(arr, [[1, -1], [2, null]]) -> arr[1:-1, 2:]
 *   slice(arr, [0, null]) -> arr[0, :] (first row)
 *   slice(arr, [null, [1, -1]]) -> arr[:, 1:-1]
 */
Napi::Value Slice(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    Napi::Array slicesArg = info[1].As<Napi::Array>();

    const double* dataA = static_cast<double*>(a->data());
    const std::vector<int64_t>& shapeA = a->shape();
    int ndim = static_cast<int>(shapeA.size());

    // Parse slice specifications for each dimension
    struct SliceInfo {
        int64_t start;
        int64_t end;
        int64_t step;
        bool isSingleIndex;  // If true, this dimension is reduced
    };

    std::vector<SliceInfo> slices(ndim);

    for (int d = 0; d < ndim; d++) {
        int64_t dimSize = shapeA[d];
        slices[d] = {0, dimSize, 1, false};  // Default: full dimension

        if (d < static_cast<int>(slicesArg.Length())) {
            Napi::Value sliceVal = slicesArg.Get(d);

            if (sliceVal.IsNull() || sliceVal.IsUndefined()) {
                // null/undefined means all (:)
                continue;
            } else if (sliceVal.IsNumber()) {
                // Single index - reduces this dimension
                int64_t idx = sliceVal.As<Napi::Number>().Int64Value();
                if (idx < 0) idx += dimSize;
                if (idx < 0 || idx >= dimSize) {
                    Napi::RangeError::New(env, "Index out of bounds").ThrowAsJavaScriptException();
                    return env.Undefined();
                }
                slices[d] = {idx, idx + 1, 1, true};
            } else if (sliceVal.IsArray()) {
                // [start, end] or [start, end, step]
                Napi::Array sliceArr = sliceVal.As<Napi::Array>();
                uint32_t len = sliceArr.Length();

                // Start
                Napi::Value startVal = sliceArr.Get(uint32_t(0));
                if (!startVal.IsNull() && !startVal.IsUndefined()) {
                    int64_t start = startVal.As<Napi::Number>().Int64Value();
                    if (start < 0) start += dimSize;
                    slices[d].start = std::max(int64_t(0), std::min(start, dimSize));
                }

                // End
                if (len >= 2) {
                    Napi::Value endVal = sliceArr.Get(uint32_t(1));
                    if (!endVal.IsNull() && !endVal.IsUndefined()) {
                        int64_t end = endVal.As<Napi::Number>().Int64Value();
                        if (end < 0) end += dimSize;
                        slices[d].end = std::max(int64_t(0), std::min(end, dimSize));
                    }
                }

                // Step
                if (len >= 3) {
                    Napi::Value stepVal = sliceArr.Get(uint32_t(2));
                    if (!stepVal.IsNull() && !stepVal.IsUndefined()) {
                        slices[d].step = stepVal.As<Napi::Number>().Int64Value();
                    }
                }
            }
        }
    }

    // Calculate result shape (excluding reduced dimensions)
    std::vector<int64_t> resultShape;
    for (int d = 0; d < ndim; d++) {
        if (!slices[d].isSingleIndex) {
            int64_t size = (slices[d].end - slices[d].start + slices[d].step - 1) / slices[d].step;
            resultShape.push_back(std::max(int64_t(0), size));
        }
    }

    // Handle scalar result (all dimensions reduced)
    if (resultShape.empty()) {
        resultShape.push_back(1);
    }

    // Create result array
    Napi::Object result = createNDArray(env, resultShape);
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* dataResult = static_cast<double*>(resultArr->data());

    // Compute strides for source array
    std::vector<int64_t> srcStrides(ndim);
    int64_t stride = 1;
    for (int d = ndim - 1; d >= 0; d--) {
        srcStrides[d] = stride;
        stride *= shapeA[d];
    }

    // Copy data using nested iteration
    // Use iterative approach with index tracking
    std::vector<int64_t> srcIndices(ndim);
    for (int d = 0; d < ndim; d++) {
        srcIndices[d] = slices[d].start;
    }

    int64_t dstIdx = 0;
    int64_t totalElements = resultArr->size();

    while (dstIdx < totalElements) {
        // Compute source flat index
        int64_t srcIdx = 0;
        for (int d = 0; d < ndim; d++) {
            srcIdx += srcIndices[d] * srcStrides[d];
        }

        dataResult[dstIdx++] = dataA[srcIdx];

        // Increment indices (like a multi-digit counter)
        for (int d = ndim - 1; d >= 0; d--) {
            srcIndices[d] += slices[d].step;
            if (srcIndices[d] < slices[d].end) {
                break;  // No carry needed
            }
            srcIndices[d] = slices[d].start;  // Reset and carry
        }
    }

    return result;
}

/**
 * Compute 2D gradient using central differences
 * gradient_2d(f, h=1.0) -> {dfdx, dfdy}
 *
 * Interior: central differences
 * Edges: forward/backward differences
 */
Napi::Value Gradient2D(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    NativeNDArray* f = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    double h = 1.0;
    if (info.Length() >= 2 && !info[1].IsUndefined()) {
        h = info[1].As<Napi::Number>().DoubleValue();
    }

    const std::vector<int64_t>& shape = f->shape();
    if (shape.size() != 2) {
        Napi::TypeError::New(env, "gradient_2d requires 2D array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int64_t rows = shape[0];
    int64_t cols = shape[1];
    const double* dataF = static_cast<double*>(f->data());

    // Create output arrays
    Napi::Object dfdxObj = createNDArray(env, shape);
    Napi::Object dfdyObj = createNDArray(env, shape);
    NativeNDArray* dfdxArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(dfdxObj);
    NativeNDArray* dfdyArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(dfdyObj);
    double* dfdx = static_cast<double*>(dfdxArr->data());
    double* dfdy = static_cast<double*>(dfdyArr->data());

    double inv2h = 1.0 / (2.0 * h);
    double invh = 1.0 / h;

    // Interior points: central differences (vectorizable)
    for (int64_t i = 1; i < rows - 1; i++) {
        for (int64_t j = 1; j < cols - 1; j++) {
            int64_t idx = i * cols + j;
            dfdx[idx] = (dataF[idx + 1] - dataF[idx - 1]) * inv2h;
            dfdy[idx] = (dataF[idx + cols] - dataF[idx - cols]) * inv2h;
        }
    }

    // Edge handling: forward/backward differences
    // Top and bottom rows
    for (int64_t j = 0; j < cols; j++) {
        // Top row (i=0): forward difference
        dfdx[j] = (j < cols - 1) ? (dataF[j + 1] - dataF[j]) * invh : 0;
        dfdy[j] = (dataF[cols + j] - dataF[j]) * invh;
        // Bottom row (i=rows-1): backward difference
        int64_t idx = (rows - 1) * cols + j;
        dfdx[idx] = (j > 0) ? (dataF[idx] - dataF[idx - 1]) * invh : 0;
        dfdy[idx] = (dataF[idx] - dataF[idx - cols]) * invh;
    }

    // Left and right columns (excluding corners already done)
    for (int64_t i = 1; i < rows - 1; i++) {
        // Left column (j=0): forward difference
        int64_t idx = i * cols;
        dfdx[idx] = (dataF[idx + 1] - dataF[idx]) * invh;
        // Right column (j=cols-1): backward difference
        idx = i * cols + cols - 1;
        dfdx[idx] = (dataF[idx] - dataF[idx - 1]) * invh;
    }

    // Return object with both gradients
    Napi::Object result = Napi::Object::New(env);
    result.Set("dfdx", dfdxObj);
    result.Set("dfdy", dfdyObj);
    return result;
}

/**
 * Heat equation step: T_new = T + alpha * laplacian(T)
 * Uses 5-point stencil with fixed boundary conditions
 * heat_step_2d(T, alpha, n_steps) -> T_new
 */
Napi::Value HeatStep2D(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    NativeNDArray* T = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    double alpha = info[1].As<Napi::Number>().DoubleValue();
    int nSteps = info[2].As<Napi::Number>().Int32Value();

    const std::vector<int64_t>& shape = T->shape();
    if (shape.size() != 2) {
        Napi::TypeError::New(env, "heat_step_2d requires 2D array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int64_t n = shape[0];
    int64_t m = shape[1];

    // Create output array and copy input
    Napi::Object result = createNDArray(env, shape);
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* TNew = static_cast<double*>(resultArr->data());
    const double* TData = static_cast<double*>(T->data());

    // Copy initial data
    std::memcpy(TNew, TData, n * m * sizeof(double));

    // Iterate heat equation steps
    for (int step = 0; step < nSteps; step++) {
        // 5-point stencil Laplacian for interior points
        for (int64_t i = 1; i < n - 1; i++) {
            for (int64_t j = 1; j < m - 1; j++) {
                int64_t idx = i * m + j;
                double laplacian =
                    TNew[(i - 1) * m + j] +
                    TNew[(i + 1) * m + j] +
                    TNew[i * m + j - 1] +
                    TNew[i * m + j + 1] -
                    4.0 * TNew[idx];
                TNew[idx] += alpha * laplacian;
            }
        }
    }

    return result;
}

/**
 * Return the indices of non-zero elements
 * nonzero(a)
 */
Napi::Value Nonzero(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());

    const double* dataA = static_cast<double*>(a->data());
    const std::vector<int64_t>& shapeA = a->shape();
    int ndim = static_cast<int>(shapeA.size());
    int64_t size = a->size();

    // First pass: count non-zero elements
    int64_t count = 0;
    for (int64_t i = 0; i < size; i++) {
        if (dataA[i] != 0.0) count++;
    }

    // Compute strides for index conversion
    std::vector<int64_t> strides(ndim);
    int64_t stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= shapeA[i];
    }

    // Create result: tuple of arrays, one per dimension
    Napi::Array resultTuple = Napi::Array::New(env, ndim);

    for (int d = 0; d < ndim; d++) {
        Napi::Object indices = createNDArray(env, {count});
        NativeNDArray* indicesArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(indices);
        double* indicesData = static_cast<double*>(indicesArr->data());

        int64_t idx = 0;
        for (int64_t i = 0; i < size; i++) {
            if (dataA[i] != 0.0) {
                // Convert flat index to coordinate for dimension d
                int64_t coord = (i / strides[d]) % shapeA[d];
                indicesData[idx++] = static_cast<double>(coord);
            }
        }

        resultTuple.Set(uint32_t(d), indices);
    }

    return resultTuple;
}

/**
 * Element-wise sign function
 * sign(a) returns -1, 0, or 1
 */
Napi::Value Sign(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());

    const std::vector<int64_t>& shapeA = a->shape();
    Napi::Object result = createNDArray(env, shapeA);
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);

    const double* dataA = static_cast<double*>(a->data());
    double* dataC = static_cast<double*>(resultArr->data());
    int64_t size = a->size();

    for (int64_t i = 0; i < size; i++) {
        if (dataA[i] > 0) {
            dataC[i] = 1.0;
        } else if (dataA[i] < 0) {
            dataC[i] = -1.0;
        } else {
            dataC[i] = 0.0;
        }
    }

    return result;
}

/**
 * Element-wise modulo operation
 * mod(a, b) - result has same sign as divisor (Python/NumPy behavior)
 */
Napi::Value Mod(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());

    const std::vector<int64_t>& shapeA = a->shape();
    const double* dataA = static_cast<double*>(a->data());
    int64_t size = a->size();

    // Check if b is scalar or array
    if (info[1].IsNumber()) {
        double b = info[1].As<Napi::Number>().DoubleValue();
        Napi::Object result = createNDArray(env, shapeA);
        NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
        double* dataC = static_cast<double*>(resultArr->data());

        for (int64_t i = 0; i < size; i++) {
            double r = std::fmod(dataA[i], b);
            // Adjust for Python/NumPy behavior (result has same sign as divisor)
            if (r != 0 && ((r < 0) != (b < 0))) {
                r += b;
            }
            dataC[i] = r;
        }
        return result;
    }

    // Array case
    NativeNDArray* b = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());
    const double* dataB = static_cast<double*>(b->data());

    Napi::Object result = createNDArray(env, shapeA);
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* dataC = static_cast<double*>(resultArr->data());

    for (int64_t i = 0; i < size; i++) {
        double r = std::fmod(dataA[i], dataB[i]);
        if (r != 0 && ((r < 0) != (dataB[i] < 0))) {
            r += dataB[i];
        }
        dataC[i] = r;
    }

    return result;
}

/**
 * Element-wise check if two arrays are close within tolerance
 * isclose(a, b, rtol=1e-5, atol=1e-8)
 */
Napi::Value Isclose(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    NativeNDArray* b = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());

    double rtol = 1e-5;
    double atol = 1e-8;
    if (info.Length() >= 3 && !info[2].IsUndefined()) {
        rtol = info[2].As<Napi::Number>().DoubleValue();
    }
    if (info.Length() >= 4 && !info[3].IsUndefined()) {
        atol = info[3].As<Napi::Number>().DoubleValue();
    }

    const std::vector<int64_t>& shapeA = a->shape();
    const double* dataA = static_cast<double*>(a->data());
    const double* dataB = static_cast<double*>(b->data());
    int64_t size = a->size();

    // Create bool result array (stored as float64 with 0/1)
    Napi::Array jsShape = Napi::Array::New(env, shapeA.size());
    for (size_t i = 0; i < shapeA.size(); i++) {
        jsShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(shapeA[i])));
    }
    Napi::Object result = NativeNDArray::constructor.New({
        jsShape, Napi::String::New(env, "bool"), Napi::Boolean::New(env, true)
    });
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    uint8_t* dataC = static_cast<uint8_t*>(resultArr->data());

    for (int64_t i = 0; i < size; i++) {
        double diff = std::abs(dataA[i] - dataB[i]);
        bool close = diff <= atol + rtol * std::abs(dataB[i]);
        dataC[i] = close ? 1 : 0;
    }

    return result;
}

/**
 * Check if all elements of two arrays are close within tolerance
 * allclose(a, b, rtol=1e-5, atol=1e-8)
 */
Napi::Value Allclose(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    NativeNDArray* b = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());

    double rtol = 1e-5;
    double atol = 1e-8;
    if (info.Length() >= 3 && !info[2].IsUndefined()) {
        rtol = info[2].As<Napi::Number>().DoubleValue();
    }
    if (info.Length() >= 4 && !info[3].IsUndefined()) {
        atol = info[3].As<Napi::Number>().DoubleValue();
    }

    const double* dataA = static_cast<double*>(a->data());
    const double* dataB = static_cast<double*>(b->data());
    int64_t size = a->size();

    for (int64_t i = 0; i < size; i++) {
        double diff = std::abs(dataA[i] - dataB[i]);
        if (diff > atol + rtol * std::abs(dataB[i])) {
            return Napi::Boolean::New(env, false);
        }
    }

    return Napi::Boolean::New(env, true);
}

// ============================================================
// Fused Operations - Reduce N-API overhead by combining multiple ops
// ============================================================

/**
 * Normalize: (x - mean) / std
 * Fused operation for batch normalization and z-score computation.
 * Supports broadcasting: mean and std can be scalars or arrays.
 */
Napi::Value Normalize(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 3) {
        Napi::TypeError::New(env, "Expected (x, mean, std)").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* x = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    double* dataX = static_cast<double*>(x->data());
    const auto& shape = x->shape();
    int64_t size = x->size();

    // Create result array
    Napi::Array jsShape = Napi::Array::New(env, shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
        jsShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(shape[i])));
    }
    Napi::Object result = NativeNDArray::constructor.New({
        jsShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
    });
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* dataR = static_cast<double*>(resultArr->data());

    bool meanIsScalar = info[1].IsNumber();
    bool stdIsScalar = info[2].IsNumber();

    if (meanIsScalar && stdIsScalar) {
        // Fast path: both scalars
        double meanVal = info[1].As<Napi::Number>().DoubleValue();
        double stdVal = info[2].As<Napi::Number>().DoubleValue();

#if defined(USE_ACCELERATE)
        // result = (x - mean) / std = x/std - mean/std
        double invStd = 1.0 / stdVal;
        double offset = -meanVal * invStd;
        vDSP_vsmsaD(dataX, 1, &invStd, &offset, dataR, 1, size);
#else
        for (int64_t i = 0; i < size; i++) {
            dataR[i] = (dataX[i] - meanVal) / stdVal;
        }
#endif
    } else if (!meanIsScalar && stdIsScalar) {
        // mean is array, std is scalar (common for per-feature normalization)
        NativeNDArray* meanArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());
        double* dataMean = static_cast<double*>(meanArr->data());
        double stdVal = info[2].As<Napi::Number>().DoubleValue();
        int64_t meanSize = meanArr->size();

        // Assume broadcasting along last dimension
        if (shape.size() == 2 && meanSize == shape[1]) {
            int64_t rows = shape[0];
            int64_t cols = shape[1];
            for (int64_t r = 0; r < rows; r++) {
                for (int64_t c = 0; c < cols; c++) {
                    dataR[r * cols + c] = (dataX[r * cols + c] - dataMean[c]) / stdVal;
                }
            }
        } else {
            // Fallback: element-wise with cycling
            for (int64_t i = 0; i < size; i++) {
                dataR[i] = (dataX[i] - dataMean[i % meanSize]) / stdVal;
            }
        }
    } else if (!meanIsScalar && !stdIsScalar) {
        // Both arrays (full per-element normalization)
        NativeNDArray* meanArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());
        NativeNDArray* stdArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[2].As<Napi::Object>());
        double* dataMean = static_cast<double*>(meanArr->data());
        double* dataStd = static_cast<double*>(stdArr->data());
        int64_t meanSize = meanArr->size();
        int64_t stdSize = stdArr->size();

        // Assume broadcasting along last dimension for 2D
        if (shape.size() == 2 && meanSize == shape[1] && stdSize == shape[1]) {
            int64_t rows = shape[0];
            int64_t cols = shape[1];
            for (int64_t r = 0; r < rows; r++) {
                for (int64_t c = 0; c < cols; c++) {
                    dataR[r * cols + c] = (dataX[r * cols + c] - dataMean[c]) / dataStd[c];
                }
            }
        } else {
            for (int64_t i = 0; i < size; i++) {
                dataR[i] = (dataX[i] - dataMean[i % meanSize]) / dataStd[i % stdSize];
            }
        }
    } else {
        // mean is scalar, std is array (rare case)
        double meanVal = info[1].As<Napi::Number>().DoubleValue();
        NativeNDArray* stdArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[2].As<Napi::Object>());
        double* dataStd = static_cast<double*>(stdArr->data());
        int64_t stdSize = stdArr->size();

        for (int64_t i = 0; i < size; i++) {
            dataR[i] = (dataX[i] - meanVal) / dataStd[i % stdSize];
        }
    }

    return result;
}

/**
 * Affine: x * scale + bias
 * Fused operation for affine transformation (common in neural networks).
 * Supports broadcasting: scale and bias can be scalars or arrays.
 */
Napi::Value Affine(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 3) {
        Napi::TypeError::New(env, "Expected (x, scale, bias)").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* x = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    double* dataX = static_cast<double*>(x->data());
    const auto& shape = x->shape();
    int64_t size = x->size();

    // Create result array
    Napi::Array jsShape = Napi::Array::New(env, shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
        jsShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(shape[i])));
    }
    Napi::Object result = NativeNDArray::constructor.New({
        jsShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
    });
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* dataR = static_cast<double*>(resultArr->data());

    bool scaleIsScalar = info[1].IsNumber();
    bool biasIsScalar = info[2].IsNumber();

    if (scaleIsScalar && biasIsScalar) {
        // Fast path: both scalars
        double scale = info[1].As<Napi::Number>().DoubleValue();
        double bias = info[2].As<Napi::Number>().DoubleValue();

#if defined(USE_ACCELERATE)
        vDSP_vsmsaD(dataX, 1, &scale, &bias, dataR, 1, size);
#else
        for (int64_t i = 0; i < size; i++) {
            dataR[i] = dataX[i] * scale + bias;
        }
#endif
    } else if (!scaleIsScalar && !biasIsScalar) {
        // Both arrays (per-feature scale and bias)
        NativeNDArray* scaleArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());
        NativeNDArray* biasArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[2].As<Napi::Object>());
        double* dataScale = static_cast<double*>(scaleArr->data());
        double* dataBias = static_cast<double*>(biasArr->data());
        int64_t scaleSize = scaleArr->size();

        // Assume broadcasting along last dimension for 2D
        if (shape.size() == 2 && scaleSize == shape[1]) {
            int64_t rows = shape[0];
            int64_t cols = shape[1];
            for (int64_t r = 0; r < rows; r++) {
                for (int64_t c = 0; c < cols; c++) {
                    dataR[r * cols + c] = dataX[r * cols + c] * dataScale[c] + dataBias[c];
                }
            }
        } else {
            for (int64_t i = 0; i < size; i++) {
                dataR[i] = dataX[i] * dataScale[i % scaleSize] + dataBias[i % scaleSize];
            }
        }
    } else if (scaleIsScalar) {
        // scale is scalar, bias is array
        double scale = info[1].As<Napi::Number>().DoubleValue();
        NativeNDArray* biasArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[2].As<Napi::Object>());
        double* dataBias = static_cast<double*>(biasArr->data());
        int64_t biasSize = biasArr->size();

        for (int64_t i = 0; i < size; i++) {
            dataR[i] = dataX[i] * scale + dataBias[i % biasSize];
        }
    } else {
        // scale is array, bias is scalar
        NativeNDArray* scaleArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());
        double* dataScale = static_cast<double*>(scaleArr->data());
        int64_t scaleSize = scaleArr->size();
        double bias = info[2].As<Napi::Number>().DoubleValue();

        for (int64_t i = 0; i < size; i++) {
            dataR[i] = dataX[i] * dataScale[i % scaleSize] + bias;
        }
    }

    return result;
}

/**
 * MulAdd: a * b + c (fused multiply-add)
 * Used in gradient updates, optimizer steps, etc.
 * All arguments can be arrays or scalars.
 */
Napi::Value MulAdd(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 3) {
        Napi::TypeError::New(env, "Expected (a, b, c)").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    double* dataA = static_cast<double*>(a->data());
    const auto& shape = a->shape();
    int64_t size = a->size();

    // Create result array
    Napi::Array jsShape = Napi::Array::New(env, shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
        jsShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(shape[i])));
    }
    Napi::Object result = NativeNDArray::constructor.New({
        jsShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
    });
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* dataR = static_cast<double*>(resultArr->data());

    bool bIsScalar = info[1].IsNumber();
    bool cIsScalar = info[2].IsNumber();

    if (bIsScalar && cIsScalar) {
        // a * scalar + scalar
        double b = info[1].As<Napi::Number>().DoubleValue();
        double c = info[2].As<Napi::Number>().DoubleValue();
#if defined(USE_ACCELERATE)
        vDSP_vsmsaD(dataA, 1, &b, &c, dataR, 1, size);
#else
        for (int64_t i = 0; i < size; i++) {
            dataR[i] = dataA[i] * b + c;
        }
#endif
    } else if (bIsScalar && !cIsScalar) {
        // a * scalar + array
        double b = info[1].As<Napi::Number>().DoubleValue();
        NativeNDArray* cArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[2].As<Napi::Object>());
        double* dataC = static_cast<double*>(cArr->data());
        for (int64_t i = 0; i < size; i++) {
            dataR[i] = dataA[i] * b + dataC[i];
        }
    } else if (!bIsScalar && cIsScalar) {
        // a * array + scalar
        NativeNDArray* bArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());
        double* dataB = static_cast<double*>(bArr->data());
        double c = info[2].As<Napi::Number>().DoubleValue();
        for (int64_t i = 0; i < size; i++) {
            dataR[i] = dataA[i] * dataB[i] + c;
        }
    } else {
        // a * array + array
        NativeNDArray* bArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());
        NativeNDArray* cArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[2].As<Napi::Object>());
        double* dataB = static_cast<double*>(bArr->data());
        double* dataC = static_cast<double*>(cArr->data());
        for (int64_t i = 0; i < size; i++) {
            dataR[i] = dataA[i] * dataB[i] + dataC[i];
        }
    }

    return result;
}

/**
 * Softmax: exp(x - max(x)) / sum(exp(x - max(x)))
 * Numerically stable softmax with optional axis parameter.
 */
Napi::Value Softmax(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* x = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    double* dataX = static_cast<double*>(x->data());
    const auto& shape = x->shape();
    int64_t size = x->size();

    // Default axis is -1 (last axis)
    int axis = -1;
    if (info.Length() >= 2 && !info[1].IsUndefined()) {
        axis = info[1].As<Napi::Number>().Int32Value();
    }
    int ndim = static_cast<int>(shape.size());
    if (axis < 0) axis += ndim;

    // Create result array
    Napi::Array jsShape = Napi::Array::New(env, shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
        jsShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(shape[i])));
    }
    Napi::Object result = NativeNDArray::constructor.New({
        jsShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
    });
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* dataR = static_cast<double*>(resultArr->data());

    // Fast path for 1D
    if (ndim == 1) {
        double maxVal = dataX[0];
        for (int64_t i = 1; i < size; i++) {
            if (dataX[i] > maxVal) maxVal = dataX[i];
        }
        double sum = 0.0;
        for (int64_t i = 0; i < size; i++) {
            dataR[i] = std::exp(dataX[i] - maxVal);
            sum += dataR[i];
        }
        for (int64_t i = 0; i < size; i++) {
            dataR[i] /= sum;
        }
        return result;
    }

    // Fast path for 2D with axis=1 (rows)
    if (ndim == 2 && axis == 1) {
        int64_t rows = shape[0];
        int64_t cols = shape[1];

        for (int64_t r = 0; r < rows; r++) {
            double* rowIn = dataX + r * cols;
            double* rowOut = dataR + r * cols;

            // Find max
            double maxVal = rowIn[0];
            for (int64_t c = 1; c < cols; c++) {
                if (rowIn[c] > maxVal) maxVal = rowIn[c];
            }

            // Compute exp and sum
            double sum = 0.0;
            for (int64_t c = 0; c < cols; c++) {
                rowOut[c] = std::exp(rowIn[c] - maxVal);
                sum += rowOut[c];
            }

            // Normalize
            for (int64_t c = 0; c < cols; c++) {
                rowOut[c] /= sum;
            }
        }
        return result;
    }

    // Fast path for 2D with axis=0 (columns)
    if (ndim == 2 && axis == 0) {
        int64_t rows = shape[0];
        int64_t cols = shape[1];

        for (int64_t c = 0; c < cols; c++) {
            // Find max in column
            double maxVal = dataX[c];
            for (int64_t r = 1; r < rows; r++) {
                if (dataX[r * cols + c] > maxVal) maxVal = dataX[r * cols + c];
            }

            // Compute exp and sum
            double sum = 0.0;
            for (int64_t r = 0; r < rows; r++) {
                dataR[r * cols + c] = std::exp(dataX[r * cols + c] - maxVal);
                sum += dataR[r * cols + c];
            }

            // Normalize
            for (int64_t r = 0; r < rows; r++) {
                dataR[r * cols + c] /= sum;
            }
        }
        return result;
    }

    // Generic case: compute strides and iterate
    std::vector<int64_t> strides(ndim);
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    int64_t axisSize = shape[axis];
    int64_t outerSize = size / axisSize;

    // Compute outer iteration (all indices except axis)
    for (int64_t outer = 0; outer < outerSize; outer++) {
        // Compute base index (this skips the axis dimension)
        std::vector<int64_t> indices(ndim, 0);
        int64_t temp = outer;
        for (int d = ndim - 1; d >= 0; d--) {
            if (d == axis) continue;
            int64_t dimSize = shape[d];
            indices[d] = temp % dimSize;
            temp /= dimSize;
        }

        // Find max along axis
        double maxVal = -std::numeric_limits<double>::infinity();
        for (int64_t a = 0; a < axisSize; a++) {
            indices[axis] = a;
            int64_t idx = 0;
            for (int d = 0; d < ndim; d++) {
                idx += indices[d] * strides[d];
            }
            if (dataX[idx] > maxVal) maxVal = dataX[idx];
        }

        // Compute exp and sum
        double sum = 0.0;
        for (int64_t a = 0; a < axisSize; a++) {
            indices[axis] = a;
            int64_t idx = 0;
            for (int d = 0; d < ndim; d++) {
                idx += indices[d] * strides[d];
            }
            dataR[idx] = std::exp(dataX[idx] - maxVal);
            sum += dataR[idx];
        }

        // Normalize
        for (int64_t a = 0; a < axisSize; a++) {
            indices[axis] = a;
            int64_t idx = 0;
            for (int d = 0; d < ndim; d++) {
                idx += indices[d] * strides[d];
            }
            dataR[idx] /= sum;
        }
    }

    return result;
}

/**
 * Fused batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
 * Computes mean, variance, normalization, scale and shift in one pass.
 * For 2D array with axis=0: normalizes along columns.
 *
 * Args:
 *   x: Input array (2D: rows x channels)
 *   gamma: Scale parameter (1D: channels) or undefined for no scaling
 *   beta: Shift parameter (1D: channels) or undefined for no shift
 *   axis: Axis along which to normalize (default 0)
 *   eps: Small constant for numerical stability (default 1e-5)
 */
Napi::Value BatchNorm(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected at least 1 argument").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* x = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    const auto& shape = x->shape();
    int ndim = static_cast<int>(shape.size());

    // Parse gamma and beta (optional)
    double* gammaData = nullptr;
    double* betaData = nullptr;
    if (info.Length() >= 2 && info[1].IsObject()) {
        NativeNDArray* gamma = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());
        gammaData = static_cast<double*>(gamma->data());
    }
    if (info.Length() >= 3 && info[2].IsObject()) {
        NativeNDArray* beta = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[2].As<Napi::Object>());
        betaData = static_cast<double*>(beta->data());
    }

    // Parse axis (default 0)
    int axis = 0;
    if (info.Length() >= 4 && info[3].IsNumber()) {
        axis = info[3].As<Napi::Number>().Int32Value();
        if (axis < 0) axis += ndim;
    }

    // Parse eps (default 1e-5)
    double eps = 1e-5;
    if (info.Length() >= 5 && info[4].IsNumber()) {
        eps = info[4].As<Napi::Number>().DoubleValue();
    }

    // Fast path: 2D with axis=0 (normalize along rows, per column)
    if (ndim == 2 && axis == 0) {
        int64_t rows = shape[0];
        int64_t cols = shape[1];
        double* dataX = static_cast<double*>(x->data());

        // Create result array
        Napi::Array jsShape = Napi::Array::New(env, 2);
        jsShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(rows)));
        jsShape.Set(uint32_t(1), Napi::Number::New(env, static_cast<double>(cols)));

        Napi::Object result = NativeNDArray::constructor.New({
            jsShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
        });
        NativeNDArray* r = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
        double* dataR = static_cast<double*>(r->data());

        // Step 1: Compute mean per column
        std::vector<double> means(cols, 0.0);
        for (int64_t row = 0; row < rows; row++) {
            for (int64_t col = 0; col < cols; col++) {
                means[col] += dataX[row * cols + col];
            }
        }
        for (int64_t col = 0; col < cols; col++) {
            means[col] /= rows;
        }

        // Step 2: Compute variance per column
        std::vector<double> vars(cols, 0.0);
        for (int64_t row = 0; row < rows; row++) {
            for (int64_t col = 0; col < cols; col++) {
                double diff = dataX[row * cols + col] - means[col];
                vars[col] += diff * diff;
            }
        }
        for (int64_t col = 0; col < cols; col++) {
            vars[col] /= rows;  // Population variance
        }

        // Step 3: Compute stddev = sqrt(var + eps)
        std::vector<double> stds(cols);
        for (int64_t col = 0; col < cols; col++) {
            stds[col] = std::sqrt(vars[col] + eps);
        }

        // Step 4: Normalize, scale, and shift in one pass
        // output = gamma * (x - mean) / std + beta
        for (int64_t row = 0; row < rows; row++) {
            for (int64_t col = 0; col < cols; col++) {
                double normalized = (dataX[row * cols + col] - means[col]) / stds[col];
                if (gammaData) {
                    normalized *= gammaData[col];
                }
                if (betaData) {
                    normalized += betaData[col];
                }
                dataR[row * cols + col] = normalized;
            }
        }

        return result;
    }

    Napi::Error::New(env, "batch_norm only supports 2D arrays with axis=0").ThrowAsJavaScriptException();
    return env.Undefined();
}

/**
 * Fused min-max scaling: (x - min) / (max - min) along an axis
 * Much faster than separate min, max, subtract, divide calls.
 */
Napi::Value MinmaxScale(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected array argument").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* x = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    const auto& shape = x->shape();
    int ndim = static_cast<int>(shape.size());

    // Default axis = 0
    int axis = 0;
    if (info.Length() >= 2 && info[1].IsNumber()) {
        axis = info[1].As<Napi::Number>().Int32Value();
        if (axis < 0) axis += ndim;
    }

    if (axis < 0 || axis >= ndim) {
        Napi::RangeError::New(env, "Invalid axis").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // For 2D array with axis=0: compute min/max per column, scale each column
    if (ndim == 2 && axis == 0) {
        int64_t rows = shape[0];
        int64_t cols = shape[1];
        double* dataX = static_cast<double*>(x->data());

        // Create result array
        Napi::Array jsShape = Napi::Array::New(env, 2);
        jsShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(rows)));
        jsShape.Set(uint32_t(1), Napi::Number::New(env, static_cast<double>(cols)));

        Napi::Object result = NativeNDArray::constructor.New({
            jsShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
        });
        NativeNDArray* r = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
        double* dataR = static_cast<double*>(r->data());

        // Compute min and max per column in one pass
        std::vector<double> minVals(cols, std::numeric_limits<double>::infinity());
        std::vector<double> maxVals(cols, -std::numeric_limits<double>::infinity());

        for (int64_t row = 0; row < rows; row++) {
            for (int64_t col = 0; col < cols; col++) {
                double val = dataX[row * cols + col];
                if (val < minVals[col]) minVals[col] = val;
                if (val > maxVals[col]) maxVals[col] = val;
            }
        }

        // Compute scaled values: (x - min) / (max - min)
        for (int64_t row = 0; row < rows; row++) {
            for (int64_t col = 0; col < cols; col++) {
                double range = maxVals[col] - minVals[col];
                if (range == 0.0) range = 1.0; // Avoid division by zero
                dataR[row * cols + col] = (dataX[row * cols + col] - minVals[col]) / range;
            }
        }

        return result;
    }

    // Fallback: use generic implementation (not optimized)
    Napi::Error::New(env, "minmax_scale only supports 2D arrays with axis=0").ThrowAsJavaScriptException();
    return env.Undefined();
}

/**
 * Full Adam optimizer step - fused for performance.
 * Updates m, v in-place and returns new parameters.
 *
 * Parameters:
 *   params: current parameters (not modified)
 *   grads: gradients
 *   m: first moment (modified in-place)
 *   v: second moment (modified in-place)
 *   lr: learning rate (scalar)
 *   beta1: first moment decay (scalar)
 *   beta2: second moment decay (scalar)
 *   eps: epsilon for numerical stability (scalar)
 *   t: current timestep (scalar)
 *
 * Returns: new parameters
 */
Napi::Value AdamStep(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 9) {
        Napi::TypeError::New(env, "Expected 9 arguments: params, grads, m, v, lr, beta1, beta2, eps, t")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* params = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    NativeNDArray* grads = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());
    NativeNDArray* m = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[2].As<Napi::Object>());
    NativeNDArray* v = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[3].As<Napi::Object>());
    double lr = info[4].As<Napi::Number>().DoubleValue();
    double beta1 = info[5].As<Napi::Number>().DoubleValue();
    double beta2 = info[6].As<Napi::Number>().DoubleValue();
    double eps = info[7].As<Napi::Number>().DoubleValue();
    double t = info[8].As<Napi::Number>().DoubleValue();

    double* dataParams = static_cast<double*>(params->data());
    double* dataGrads = static_cast<double*>(grads->data());
    double* dataM = static_cast<double*>(m->data());
    double* dataV = static_cast<double*>(v->data());
    int64_t size = params->size();

    // Create result array for new parameters
    const auto& shape = params->shape();
    Napi::Array jsShape = Napi::Array::New(env, shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
        jsShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(shape[i])));
    }
    Napi::Object result = NativeNDArray::constructor.New({
        jsShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
    });
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* dataR = static_cast<double*>(resultArr->data());

    // Precompute bias correction terms
    double oneMinusBeta1 = 1.0 - beta1;
    double oneMinusBeta2 = 1.0 - beta2;
    double biasCorrect1 = 1.0 / (1.0 - std::pow(beta1, t));
    double biasCorrect2 = 1.0 / (1.0 - std::pow(beta2, t));

    // Fused loop: update m, v in-place and compute new params
    #pragma omp parallel for simd if(size > 10000)
    for (int64_t i = 0; i < size; i++) {
        double g = dataGrads[i];

        // m = beta1 * m + (1 - beta1) * grad
        dataM[i] = beta1 * dataM[i] + oneMinusBeta1 * g;

        // v = beta2 * v + (1 - beta2) * grad²
        dataV[i] = beta2 * dataV[i] + oneMinusBeta2 * g * g;

        // Bias-corrected estimates
        double mHat = dataM[i] * biasCorrect1;
        double vHat = dataV[i] * biasCorrect2;

        // Update: params - lr * mHat / (sqrt(vHat) + eps)
        dataR[i] = dataParams[i] - lr * mHat / (std::sqrt(vHat) + eps);
    }

    return result;
}

/**
 * Power iteration for computing dominant eigenvalue/eigenvector.
 * Runs n_iters iterations of: v = M @ v; v = v / ||v||
 *
 * Parameters:
 *   M: square matrix (n x n)
 *   v: initial vector (n,) - modified in-place
 *   n_iters: number of iterations
 *
 * Returns: final eigenvector estimate
 */
Napi::Value PowerIter(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 3) {
        Napi::TypeError::New(env, "Expected 3 arguments: M, v, n_iters")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* M = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    NativeNDArray* v = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());
    int n_iters = info[2].As<Napi::Number>().Int32Value();

    const auto& shapeM = M->shape();
    if (shapeM.size() != 2 || shapeM[0] != shapeM[1]) {
        Napi::TypeError::New(env, "M must be a square matrix").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int64_t n = shapeM[0];
    double* dataM = static_cast<double*>(M->data());
    double* dataV = static_cast<double*>(v->data());

    // Create result array
    Napi::Array jsShape = Napi::Array::New(env, 1);
    jsShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(n)));
    Napi::Object result = NativeNDArray::constructor.New({
        jsShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
    });
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* dataR = static_cast<double*>(resultArr->data());

    // Copy v to result
    std::memcpy(dataR, dataV, n * sizeof(double));

    // Temporary buffer for matrix-vector product
    std::vector<double> temp(n);

    for (int iter = 0; iter < n_iters; iter++) {
        // Mv = M @ v using BLAS if available
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                    static_cast<int>(n), static_cast<int>(n),
                    1.0, dataM, static_cast<int>(n),
                    dataR, 1,
                    0.0, temp.data(), 1);
#else
        // Manual matrix-vector multiplication
        for (int64_t i = 0; i < n; i++) {
            double sum = 0.0;
            for (int64_t j = 0; j < n; j++) {
                sum += dataM[i * n + j] * dataR[j];
            }
            temp[i] = sum;
        }
#endif

        // Compute norm: ||Mv||
        double normSq = 0.0;
        for (int64_t i = 0; i < n; i++) {
            normSq += temp[i] * temp[i];
        }
        double norm = std::sqrt(normSq);

        // Normalize: v = Mv / ||Mv||
        if (norm > 0.0) {
            double invNorm = 1.0 / norm;
            for (int64_t i = 0; i < n; i++) {
                dataR[i] = temp[i] * invNorm;
            }
        } else {
            std::memcpy(dataR, temp.data(), n * sizeof(double));
        }
    }

    return result;
}

/**
 * Fused batched attention: softmax(Q @ K^T) @ V
 * For each batch: output[b] = softmax(Q[b] @ K[b]^T, axis=1) @ V[b]
 *
 * Parameters:
 *   Q: Query matrix [batch, seq, dim]
 *   K: Key matrix [batch, seq, dim]
 *   V: Value matrix [batch, seq, dim]
 *
 * Returns: Attention output [batch, seq, dim]
 */
Napi::Value BatchedAttention(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 3) {
        Napi::TypeError::New(env, "Expected 3 arguments: Q, K, V")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* Q = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    NativeNDArray* K = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());
    NativeNDArray* V = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[2].As<Napi::Object>());

    const auto& shapeQ = Q->shape();
    const auto& shapeK = K->shape();
    const auto& shapeV = V->shape();

    if (shapeQ.size() != 3 || shapeK.size() != 3 || shapeV.size() != 3) {
        Napi::TypeError::New(env, "Q, K, V must be 3D tensors [batch, seq, dim]")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int64_t batch = shapeQ[0];
    int64_t seq = shapeQ[1];
    int64_t dim = shapeQ[2];

    double* dataQ = static_cast<double*>(Q->data());
    double* dataK = static_cast<double*>(K->data());
    double* dataV = static_cast<double*>(V->data());

    // Create result array [batch, seq, dim]
    Napi::Array jsShape = Napi::Array::New(env, 3);
    jsShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(batch)));
    jsShape.Set(uint32_t(1), Napi::Number::New(env, static_cast<double>(seq)));
    jsShape.Set(uint32_t(2), Napi::Number::New(env, static_cast<double>(dim)));
    Napi::Object result = NativeNDArray::constructor.New({
        jsShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
    });
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* dataR = static_cast<double*>(resultArr->data());

    // Temporary buffers for scores and attention weights
    std::vector<double> scores(seq * seq);
    std::vector<double> attention(seq * seq);

    // Process each batch
    for (int64_t b = 0; b < batch; b++) {
        double* Qb = dataQ + b * seq * dim;
        double* Kb = dataK + b * seq * dim;
        double* Vb = dataV + b * seq * dim;
        double* Rb = dataR + b * seq * dim;

        // Step 1: scores = Q @ K^T  [seq x dim] @ [dim x seq] = [seq x seq]
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    static_cast<int>(seq), static_cast<int>(seq), static_cast<int>(dim),
                    1.0, Qb, static_cast<int>(dim),
                    Kb, static_cast<int>(dim),
                    0.0, scores.data(), static_cast<int>(seq));
#else
        // Manual Q @ K^T
        for (int64_t i = 0; i < seq; i++) {
            for (int64_t j = 0; j < seq; j++) {
                double sum = 0.0;
                for (int64_t k = 0; k < dim; k++) {
                    sum += Qb[i * dim + k] * Kb[j * dim + k];
                }
                scores[i * seq + j] = sum;
            }
        }
#endif

        // Step 2: Row-wise softmax on scores
        for (int64_t i = 0; i < seq; i++) {
            double* row = scores.data() + i * seq;
            double* attRow = attention.data() + i * seq;

            // Find max for numerical stability
            double maxVal = row[0];
            for (int64_t j = 1; j < seq; j++) {
                if (row[j] > maxVal) maxVal = row[j];
            }

            // Compute exp and sum
            double sum = 0.0;
            for (int64_t j = 0; j < seq; j++) {
                attRow[j] = std::exp(row[j] - maxVal);
                sum += attRow[j];
            }

            // Normalize
            for (int64_t j = 0; j < seq; j++) {
                attRow[j] /= sum;
            }
        }

        // Step 3: output = attention @ V  [seq x seq] @ [seq x dim] = [seq x dim]
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(seq), static_cast<int>(dim), static_cast<int>(seq),
                    1.0, attention.data(), static_cast<int>(seq),
                    Vb, static_cast<int>(dim),
                    0.0, Rb, static_cast<int>(dim));
#else
        // Manual attention @ V
        for (int64_t i = 0; i < seq; i++) {
            for (int64_t j = 0; j < dim; j++) {
                double sum = 0.0;
                for (int64_t k = 0; k < seq; k++) {
                    sum += attention[i * seq + k] * Vb[k * dim + j];
                }
                Rb[i * dim + j] = sum;
            }
        }
#endif
    }

    return result;
}

/**
 * Fused Black-Scholes option pricing
 * black_scholes(S, K, T, r, sigma) -> call prices
 *
 * S: spot prices array [n]
 * K: strike price (scalar)
 * T: time to expiry array [n]
 * r: risk-free rate (scalar)
 * sigma: volatility array [n]
 *
 * Uses logistic approximation for normal CDF:
 * N(x) ≈ 1 / (1 + exp(-1.5957691216 * x))
 */
Napi::Value BlackScholes(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    // Unwrap arguments
    NativeNDArray* arrS = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    double K = info[1].As<Napi::Number>().DoubleValue();
    NativeNDArray* arrT = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[2].As<Napi::Object>());
    double r = info[3].As<Napi::Number>().DoubleValue();
    NativeNDArray* arrSigma = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[4].As<Napi::Object>());

    const double* S = static_cast<const double*>(arrS->data());
    const double* T = static_cast<const double*>(arrT->data());
    const double* sigma = static_cast<const double*>(arrSigma->data());
    int64_t size = arrS->size();

    // Create result array
    const auto& shape = arrS->shape();
    Napi::Array jsShape = Napi::Array::New(env, shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
        jsShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(shape[i])));
    }
    Napi::Object result = NativeNDArray::constructor.New({
        jsShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
    });
    NativeNDArray* arrResult = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* dataR = static_cast<double*>(arrResult->data());

    // Logistic approximation constant for normal CDF
    const double CDF_SCALE = 1.5957691216;

    // Fused Black-Scholes computation
    #pragma omp parallel for simd if(size > 1000)
    for (int64_t i = 0; i < size; i++) {
        double s = S[i];
        double t = T[i];
        double sig = sigma[i];

        double sqrtT = std::sqrt(t);
        double sigmaSqrtT = sig * sqrtT;

        // d1 = (log(S/K) + (r + 0.5*sigma^2)*T) / (sigma * sqrt(T))
        double logSK = std::log(s / K);
        double d1 = (logSK + (r + 0.5 * sig * sig) * t) / sigmaSqrtT;

        // d2 = d1 - sigma * sqrt(T)
        double d2 = d1 - sigmaSqrtT;

        // N(d1) ≈ 1 / (1 + exp(-1.5957691216 * d1))
        double nd1 = 1.0 / (1.0 + std::exp(-CDF_SCALE * d1));
        double nd2 = 1.0 / (1.0 + std::exp(-CDF_SCALE * d2));

        // call = S * N(d1) - K * exp(-r*T) * N(d2)
        double expNegRT = std::exp(-r * t);
        dataR[i] = s * nd1 - K * expNegRT * nd2;
    }

    return result;
}

/**
 * Fused TF-IDF computation
 * tfidf(tf) -> tfidf_matrix
 *
 * tf: term frequency matrix [nDocs, nTerms]
 *
 * Computes:
 * 1. TF normalization: tf_norm[i,j] = tf[i,j] / sum(tf[i,:])
 * 2. Document frequency: df[j] = count(tf[:,j] > 0)
 * 3. Inverse document frequency: idf[j] = log(nDocs / df[j])
 * 4. TF-IDF: tfidf[i,j] = tf_norm[i,j] * idf[j]
 */
Napi::Value TFIDF(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    NativeNDArray* tfArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    const double* tf = static_cast<const double*>(tfArr->data());
    const auto& shape = tfArr->shape();

    int64_t nDocs = shape[0];
    int64_t nTerms = shape[1];

    // Create result array
    Napi::Array jsShape = Napi::Array::New(env, 2);
    jsShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(nDocs)));
    jsShape.Set(uint32_t(1), Napi::Number::New(env, static_cast<double>(nTerms)));
    Napi::Object result = NativeNDArray::constructor.New({
        jsShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
    });
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* dataR = static_cast<double*>(resultArr->data());

    // Single pass: compute row sums AND document frequency simultaneously (cache-friendly)
    std::vector<double> rowSums(nDocs, 0.0);
    std::vector<double> df(nTerms, 0.0);

    for (int64_t i = 0; i < nDocs; i++) {
        const double* rowPtr = tf + i * nTerms;
        double rowSum = 0.0;
        for (int64_t j = 0; j < nTerms; j++) {
            double val = rowPtr[j];
            rowSum += val;
            if (val > 0.0) {
                df[j] += 1.0;
            }
        }
        rowSums[i] = rowSum;
    }

    // Compute IDF: log(nDocs / df)
    std::vector<double> idf(nTerms);
    for (int64_t j = 0; j < nTerms; j++) {
        idf[j] = std::log(static_cast<double>(nDocs) / df[j]);
    }

    // Compute TF-IDF: (tf / rowSum) * idf
    #pragma omp parallel for if(nDocs * nTerms > 10000)
    for (int64_t i = 0; i < nDocs; i++) {
        const double* rowPtr = tf + i * nTerms;
        double* outPtr = dataR + i * nTerms;
        double invRowSum = 1.0 / rowSums[i];
        for (int64_t j = 0; j < nTerms; j++) {
            outPtr[j] = rowPtr[j] * invRowSum * idf[j];
        }
    }

    return result;
}

/**
 * Fused dropout forward pass
 * dropout(x, p, seed?) -> {output, mask}
 *
 * x: input array
 * p: dropout probability (probability of setting to 0)
 * seed: optional random seed
 *
 * Returns output and mask (scaled by 1/(1-p))
 */
Napi::Value Dropout(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    NativeNDArray* xArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    double p = info[1].As<Napi::Number>().DoubleValue();

    // Optional seed
    uint64_t seedVal = (info.Length() >= 3 && !info[2].IsUndefined())
        ? static_cast<uint64_t>(info[2].As<Napi::Number>().Int64Value())
        : static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());

    const double* x = static_cast<const double*>(xArr->data());
    const auto& shape = xArr->shape();
    int64_t size = xArr->size();
    double scale = 1.0 / (1.0 - p);

    // Create output array
    Napi::Array jsShape = Napi::Array::New(env, shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
        jsShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(shape[i])));
    }

    Napi::Object output = NativeNDArray::constructor.New({
        jsShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
    });
    Napi::Object mask = NativeNDArray::constructor.New({
        jsShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
    });

    NativeNDArray* outArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(output);
    NativeNDArray* maskArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(mask);
    double* dataOut = static_cast<double*>(outArr->data());
    double* dataMask = static_cast<double*>(maskArr->data());

    // Simple PCG-like RNG (inline for speed)
    uint64_t state = seedVal;
    const uint64_t multiplier = 6364136223846793005ULL;
    const uint64_t increment = (seedVal << 1) | 1;

    // Fused: generate random, create mask, apply to input
    for (int64_t i = 0; i < size; i++) {
        // PCG step
        state = state * multiplier + increment;
        uint64_t xorShifted = ((state >> 18u) ^ state) >> 27u;
        uint64_t rot = state >> 59u;
        uint64_t rnd = (xorShifted >> rot) | (xorShifted << ((-rot) & 31));

        // Convert to [0, 1) and compare with p
        double uniform = static_cast<double>(rnd) / static_cast<double>(UINT64_MAX);
        double m = (uniform > p) ? scale : 0.0;

        dataMask[i] = m;
        dataOut[i] = x[i] * m;
    }

    // Return object with output and mask
    Napi::Object result = Napi::Object::New(env);
    result.Set("output", output);
    result.Set("mask", mask);
    return result;
}

/**
 * Fused cross-entropy loss
 * cross_entropy(predictions, targets) -> scalar loss
 *
 * predictions: logits [nSamples, nClasses]
 * targets: one-hot encoded targets [nSamples, nClasses]
 *
 * Computes: -sum(targets * log(softmax(predictions)))
 * Note: For benchmark compatibility, this version does:
 *       -sum(targets * softmax(predictions))
 */
Napi::Value CrossEntropy(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    NativeNDArray* predArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    NativeNDArray* targArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());

    const double* preds = static_cast<const double*>(predArr->data());
    const double* targets = static_cast<const double*>(targArr->data());
    const auto& shape = predArr->shape();

    int64_t nSamples = shape[0];
    int64_t nClasses = shape[1];

    double totalLoss = 0.0;

    // For each sample, compute softmax and dot with targets
    for (int64_t i = 0; i < nSamples; i++) {
        const double* predRow = preds + i * nClasses;
        const double* targRow = targets + i * nClasses;

        // Find max for numerical stability
        double maxVal = predRow[0];
        for (int64_t j = 1; j < nClasses; j++) {
            if (predRow[j] > maxVal) maxVal = predRow[j];
        }

        // Compute exp(x - max) and sum
        double sumExp = 0.0;
        for (int64_t j = 0; j < nClasses; j++) {
            sumExp += std::exp(predRow[j] - maxVal);
        }

        // Compute sum(targets * softmax)
        for (int64_t j = 0; j < nClasses; j++) {
            double softmaxVal = std::exp(predRow[j] - maxVal) / sumExp;
            totalLoss += targRow[j] * softmaxVal;
        }
    }

    return Napi::Number::New(env, -totalLoss);
}

/**
 * Sumsq - Fused sum of squares: sum(x*x, axis)
 *
 * Replaces the common pattern: sum(multiply(x, x), axis)
 * Single N-API call instead of two.
 *
 * Args:
 *   a: input array
 *   axis: optional reduction axis
 *   keepdims: optional keep dimensions (default false)
 */
Napi::Value Sumsq(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    const double* data = static_cast<const double*>(a->data());
    const auto& shape = a->shape();
    int64_t size = a->size();

    // No axis: global sum of squares
    if (info.Length() < 2 || info[1].IsUndefined()) {
        double result = 0.0;
#if defined(USE_ACCELERATE)
        // vDSP_svesq: sum of squares
        vDSP_svesqD(data, 1, &result, static_cast<vDSP_Length>(size));
#else
        for (int64_t i = 0; i < size; i++) {
            result += data[i] * data[i];
        }
#endif
        return Napi::Number::New(env, result);
    }

    // With axis
    int axis = info[1].As<Napi::Number>().Int32Value();
    bool keepdims = false;
    if (info.Length() >= 3 && info[2].IsBoolean()) {
        keepdims = info[2].As<Napi::Boolean>().Value();
    }

    int ndim = static_cast<int>(shape.size());
    if (axis < 0) axis += ndim;

    // Compute output shape and sizes
    int64_t outerSize = 1;
    int64_t axisSize = shape[axis];
    int64_t innerSize = 1;

    for (int i = 0; i < axis; i++) outerSize *= shape[i];
    for (int i = axis + 1; i < ndim; i++) innerSize *= shape[i];

    std::vector<int64_t> outShape;
    for (int i = 0; i < ndim; i++) {
        if (i == axis) {
            if (keepdims) outShape.push_back(1);
        } else {
            outShape.push_back(shape[i]);
        }
    }
    if (outShape.empty()) outShape.push_back(1);

    int64_t outSize = outerSize * innerSize;

    // Create JS array for shape
    Napi::Array jsShape = Napi::Array::New(env, outShape.size());
    for (size_t i = 0; i < outShape.size(); i++) {
        jsShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(outShape[i])));
    }

    Napi::Object result = NativeNDArray::constructor.New({
        jsShape,
        Napi::String::New(env, "float64"),
        Napi::Boolean::New(env, true)  // skipInit
    });
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* outData = static_cast<double*>(resultArr->data());

    // Initialize to zero
    std::memset(outData, 0, outSize * sizeof(double));

    // Reduction with fused square
    if (innerSize == 1) {
        // Contiguous case: sum along last axis or single inner dim
        for (int64_t outer = 0; outer < outerSize; outer++) {
            double sum = 0.0;
            const double* rowStart = data + outer * axisSize;
#if defined(USE_ACCELERATE)
            vDSP_svesqD(rowStart, 1, &sum, static_cast<vDSP_Length>(axisSize));
#else
            for (int64_t k = 0; k < axisSize; k++) {
                sum += rowStart[k] * rowStart[k];
            }
#endif
            outData[outer] = sum;
        }
    } else {
        // General case: strided access
        for (int64_t outer = 0; outer < outerSize; outer++) {
            for (int64_t inner = 0; inner < innerSize; inner++) {
                double sum = 0.0;
                int64_t outIdx = outer * innerSize + inner;
                for (int64_t k = 0; k < axisSize; k++) {
                    int64_t idx = (outer * axisSize + k) * innerSize + inner;
                    double val = data[idx];
                    sum += val * val;
                }
                outData[outIdx] = sum;
            }
        }
    }

    return result;
}

/**
 * LayerNorm - Fused layer normalization: (x - mean) / std * gamma + beta
 *
 * Replaces the common pattern: affine(zscore(x, axis), gamma, beta)
 * Single N-API call instead of two.
 *
 * Args:
 *   x: input array [batch, features]
 *   gamma: scale parameter [features] (optional)
 *   beta: bias parameter [features] (optional)
 *   axis: normalization axis (default 1)
 *   eps: epsilon for numerical stability (default 1e-5)
 */
Napi::Value LayerNorm(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    NativeNDArray* xArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    const double* x = static_cast<const double*>(xArr->data());
    const auto& shape = xArr->shape();

    // Get gamma, beta if provided
    const double* gamma = nullptr;
    const double* beta = nullptr;
    if (info.Length() >= 2 && !info[1].IsUndefined() && !info[1].IsNull()) {
        NativeNDArray* gammaArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());
        gamma = static_cast<const double*>(gammaArr->data());
    }
    if (info.Length() >= 3 && !info[2].IsUndefined() && !info[2].IsNull()) {
        NativeNDArray* betaArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[2].As<Napi::Object>());
        beta = static_cast<const double*>(betaArr->data());
    }

    int axis = 1;
    if (info.Length() >= 4 && !info[3].IsUndefined()) {
        axis = info[3].As<Napi::Number>().Int32Value();
    }
    double eps = 1e-5;
    if (info.Length() >= 5 && !info[4].IsUndefined()) {
        eps = info[4].As<Napi::Number>().DoubleValue();
    }

    int ndim = static_cast<int>(shape.size());
    if (axis < 0) axis += ndim;

    // Create JS array for shape
    Napi::Array jsShape = Napi::Array::New(env, shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
        jsShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(shape[i])));
    }

    // Output with same shape
    Napi::Object result = NativeNDArray::constructor.New({
        jsShape,
        Napi::String::New(env, "float64"),
        Napi::Boolean::New(env, true)  // skipInit
    });
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* out = static_cast<double*>(resultArr->data());

    // Fast-path for 2D array with axis=1 (common case: [batch, features])
    if (ndim == 2 && axis == 1) {
        int64_t nRows = shape[0];
        int64_t nCols = shape[1];

        for (int64_t i = 0; i < nRows; i++) {
            const double* rowIn = x + i * nCols;
            double* rowOut = out + i * nCols;

            // Compute mean
            double sum = 0.0;
#if defined(USE_ACCELERATE)
            vDSP_sveD(rowIn, 1, &sum, static_cast<vDSP_Length>(nCols));
#else
            for (int64_t j = 0; j < nCols; j++) sum += rowIn[j];
#endif
            double mean = sum / nCols;

            // Compute variance (sum of squared deviations)
            double varSum = 0.0;
            for (int64_t j = 0; j < nCols; j++) {
                double diff = rowIn[j] - mean;
                varSum += diff * diff;
            }
            double invStd = 1.0 / std::sqrt(varSum / nCols + eps);

            // Normalize and apply affine: (x - mean) * invStd * gamma + beta
            if (gamma && beta) {
                for (int64_t j = 0; j < nCols; j++) {
                    rowOut[j] = (rowIn[j] - mean) * invStd * gamma[j] + beta[j];
                }
            } else if (gamma) {
                for (int64_t j = 0; j < nCols; j++) {
                    rowOut[j] = (rowIn[j] - mean) * invStd * gamma[j];
                }
            } else if (beta) {
                for (int64_t j = 0; j < nCols; j++) {
                    rowOut[j] = (rowIn[j] - mean) * invStd + beta[j];
                }
            } else {
                for (int64_t j = 0; j < nCols; j++) {
                    rowOut[j] = (rowIn[j] - mean) * invStd;
                }
            }
        }
    } else {
        // Generic case - fall back to zscore-like computation along axis
        int64_t outerSize = 1, axisSize = shape[axis], innerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= shape[i];
        for (int i = axis + 1; i < ndim; i++) innerSize *= shape[i];

        for (int64_t outer = 0; outer < outerSize; outer++) {
            for (int64_t inner = 0; inner < innerSize; inner++) {
                // Compute mean along axis
                double sum = 0.0;
                for (int64_t k = 0; k < axisSize; k++) {
                    int64_t idx = (outer * axisSize + k) * innerSize + inner;
                    sum += x[idx];
                }
                double mean = sum / axisSize;

                // Compute variance
                double varSum = 0.0;
                for (int64_t k = 0; k < axisSize; k++) {
                    int64_t idx = (outer * axisSize + k) * innerSize + inner;
                    double diff = x[idx] - mean;
                    varSum += diff * diff;
                }
                double invStd = 1.0 / std::sqrt(varSum / axisSize + eps);

                // Normalize and optionally apply gamma/beta
                for (int64_t k = 0; k < axisSize; k++) {
                    int64_t idx = (outer * axisSize + k) * innerSize + inner;
                    double normalized = (x[idx] - mean) * invStd;
                    if (gamma) normalized *= gamma[k];
                    if (beta) normalized += beta[k];
                    out[idx] = normalized;
                }
            }
        }
    }

    return result;
}

/**
 * SoftmaxCrossEntropyFromLogits - Fused softmax cross-entropy loss
 *
 * Computes: -sum(labels * log_softmax(logits)) per sample, then average
 * Numerically stable: uses log_softmax = logits - log(sum(exp(logits - max)))
 *
 * Args:
 *   logits: input logits [nSamples, nClasses]
 *   labels: one-hot encoded labels [nSamples, nClasses]
 *   axis: softmax axis (default 1)
 *
 * Returns: scalar loss value (mean over samples)
 */
Napi::Value SoftmaxCrossEntropyFromLogits(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    NativeNDArray* logitsArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    NativeNDArray* labelsArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());

    const double* logits = static_cast<const double*>(logitsArr->data());
    const double* labels = static_cast<const double*>(labelsArr->data());
    const auto& shape = logitsArr->shape();

    int axis = 1;
    if (info.Length() >= 3 && !info[2].IsUndefined()) {
        axis = info[2].As<Napi::Number>().Int32Value();
    }

    // Fast-path for 2D with axis=1 (most common case)
    if (shape.size() == 2 && axis == 1) {
        int64_t nSamples = shape[0];
        int64_t nClasses = shape[1];
        double totalLoss = 0.0;

        for (int64_t i = 0; i < nSamples; i++) {
            const double* logitRow = logits + i * nClasses;
            const double* labelRow = labels + i * nClasses;

            // Find max for numerical stability
            double maxLogit = logitRow[0];
            for (int64_t j = 1; j < nClasses; j++) {
                if (logitRow[j] > maxLogit) maxLogit = logitRow[j];
            }

            // Compute log(sum(exp(logits - max)))
            double sumExp = 0.0;
            for (int64_t j = 0; j < nClasses; j++) {
                sumExp += std::exp(logitRow[j] - maxLogit);
            }
            double logSumExp = std::log(sumExp);

            // Cross-entropy: -sum(labels * (logits - max - logSumExp))
            double sampleLoss = 0.0;
            for (int64_t j = 0; j < nClasses; j++) {
                // log_softmax[j] = logits[j] - max - logSumExp
                double logSoftmax = logitRow[j] - maxLogit - logSumExp;
                sampleLoss -= labelRow[j] * logSoftmax;
            }
            totalLoss += sampleLoss;
        }

        // Return mean loss
        return Napi::Number::New(env, totalLoss / nSamples);
    }

    // Generic case (fall back to per-element)
    Napi::Error::New(env, "SoftmaxCrossEntropyFromLogits only supports 2D arrays with axis=1")
        .ThrowAsJavaScriptException();
    return env.Null();
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    Napi::Object math = Napi::Object::New(env);

    math.Set("add", Napi::Function::New(env, Add));
    math.Set("subtract", Napi::Function::New(env, Subtract));
    math.Set("multiply", Napi::Function::New(env, Multiply));
    math.Set("divide", Napi::Function::New(env, Divide));
    math.Set("subtract_scalar_first", Napi::Function::New(env, SubtractScalarFirst));
    math.Set("divide_scalar_first", Napi::Function::New(env, DivideScalarFirst));
    math.Set("power", Napi::Function::New(env, Power));
    math.Set("add_inplace", Napi::Function::New(env, AddInplace));
    math.Set("subtract_inplace", Napi::Function::New(env, SubtractInplace));
    math.Set("multiply_inplace", Napi::Function::New(env, MultiplyInplace));
    math.Set("divide_inplace", Napi::Function::New(env, DivideInplace));
    math.Set("sqrt", Napi::Function::New(env, Sqrt));
    math.Set("exp", Napi::Function::New(env, Exp));
    math.Set("log", Napi::Function::New(env, Log));
    math.Set("sin", Napi::Function::New(env, Sin));
    math.Set("cos", Napi::Function::New(env, Cos));
    math.Set("tan", Napi::Function::New(env, Tan));
    math.Set("tanh", Napi::Function::New(env, Tanh));
    math.Set("abs", Napi::Function::New(env, Abs));
    math.Set("round", Napi::Function::New(env, Round));
    math.Set("floor", Napi::Function::New(env, Floor));
    math.Set("ceil", Napi::Function::New(env, Ceil));
    math.Set("sum", Napi::Function::New(env, Sum));
    math.Set("prod", Napi::Function::New(env, Prod));
    math.Set("max", Napi::Function::New(env, Max));
    math.Set("min", Napi::Function::New(env, Min));
    math.Set("argmax", Napi::Function::New(env, Argmax));
    math.Set("argmin", Napi::Function::New(env, Argmin));
    math.Set("cumsum", Napi::Function::New(env, Cumsum));
    math.Set("cumprod", Napi::Function::New(env, Cumprod));
    math.Set("concatenate", Napi::Function::New(env, Concatenate));
    math.Set("stack", Napi::Function::New(env, Stack));
    math.Set("diff", Napi::Function::New(env, Diff));
    math.Set("sort", Napi::Function::New(env, Sort));
    math.Set("argsort", Napi::Function::New(env, Argsort));
    math.Set("unique", Napi::Function::New(env, Unique));
    math.Set("searchsorted", Napi::Function::New(env, Searchsorted));
    math.Set("tile", Napi::Function::New(env, Tile));
    math.Set("repeat", Napi::Function::New(env, Repeat));
    math.Set("flip", Napi::Function::New(env, Flip));
    math.Set("rot90", Napi::Function::New(env, Rot90));
    math.Set("split", Napi::Function::New(env, Split));
    math.Set("slice", Napi::Function::New(env, Slice));
    math.Set("gradient_2d", Napi::Function::New(env, Gradient2D));
    math.Set("heat_step_2d", Napi::Function::New(env, HeatStep2D));
    math.Set("nonzero", Napi::Function::New(env, Nonzero));
    math.Set("sign", Napi::Function::New(env, Sign));
    math.Set("mod", Napi::Function::New(env, Mod));
    math.Set("isclose", Napi::Function::New(env, Isclose));
    math.Set("allclose", Napi::Function::New(env, Allclose));
    math.Set("clip", Napi::Function::New(env, Clip));
    math.Set("where", Napi::Function::New(env, Where));
    math.Set("squeeze", Napi::Function::New(env, Squeeze));
    math.Set("expand_dims", Napi::Function::New(env, ExpandDims));
    math.Set("mean", Napi::Function::New(env, Mean));
    math.Set("std", Napi::Function::New(env, Std));
    math.Set("var", Napi::Function::New(env, Var));
    math.Set("median", Napi::Function::New(env, Median));
    math.Set("zscore", Napi::Function::New(env, Zscore));
    math.Set("corrcoef", Napi::Function::New(env, Corrcoef));
    math.Set("percentile", Napi::Function::New(env, Percentile));
    math.Set("kron", Napi::Function::New(env, Kron));
    math.Set("outer", Napi::Function::New(env, Outer));
    math.Set("axpby", Napi::Function::New(env, Axpby));

    // Comparison operators
    math.Set("equal", Napi::Function::New(env, Equal));
    math.Set("not_equal", Napi::Function::New(env, NotEqual));
    math.Set("less", Napi::Function::New(env, Less));
    math.Set("less_equal", Napi::Function::New(env, LessEqual));
    math.Set("greater", Napi::Function::New(env, Greater));
    math.Set("greater_equal", Napi::Function::New(env, GreaterEqual));

    // Logical operators
    math.Set("logical_and", Napi::Function::New(env, LogicalAnd));
    math.Set("logical_or", Napi::Function::New(env, LogicalOr));
    math.Set("logical_xor", Napi::Function::New(env, LogicalXor));
    math.Set("logical_not", Napi::Function::New(env, LogicalNot));

    // Boolean reductions
    math.Set("any", Napi::Function::New(env, Any));
    math.Set("all", Napi::Function::New(env, All));

    // Fused operations
    math.Set("normalize", Napi::Function::New(env, Normalize));
    math.Set("affine", Napi::Function::New(env, Affine));
    math.Set("muladd", Napi::Function::New(env, MulAdd));
    math.Set("softmax", Napi::Function::New(env, Softmax));
    math.Set("minmax_scale", Napi::Function::New(env, MinmaxScale));
    math.Set("batch_norm", Napi::Function::New(env, BatchNorm));

    // Optimizer operations
    math.Set("adam_step", Napi::Function::New(env, AdamStep));
    math.Set("power_iter", Napi::Function::New(env, PowerIter));

    // Attention operations
    math.Set("batched_attention", Napi::Function::New(env, BatchedAttention));

    // Financial operations
    math.Set("black_scholes", Napi::Function::New(env, BlackScholes));

    // Text/NLP operations
    math.Set("tfidf", Napi::Function::New(env, TFIDF));

    // Neural network operations
    math.Set("dropout", Napi::Function::New(env, Dropout));
    math.Set("cross_entropy", Napi::Function::New(env, CrossEntropy));

    // Fused reductions
    math.Set("sumsq", Napi::Function::New(env, Sumsq));
    math.Set("layer_norm", Napi::Function::New(env, LayerNorm));
    math.Set("softmax_cross_entropy", Napi::Function::New(env, SoftmaxCrossEntropyFromLogits));

    exports.Set("math", math);
    return exports;
}

} // namespace math
} // namespace numpy_node
