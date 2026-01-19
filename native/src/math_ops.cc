#include "math_ops.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>

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

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    Napi::Object math = Napi::Object::New(env);

    math.Set("add", Napi::Function::New(env, Add));
    math.Set("subtract", Napi::Function::New(env, Subtract));
    math.Set("multiply", Napi::Function::New(env, Multiply));
    math.Set("divide", Napi::Function::New(env, Divide));
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

    exports.Set("math", math);
    return exports;
}

} // namespace math
} // namespace numpy_node
