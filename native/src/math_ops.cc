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

        Napi::Object result = NativeNDArray::constructor.New({jsShape, Napi::String::New(env, "float64")});
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

        Napi::Object result = NativeNDArray::constructor.New({jsShape, Napi::String::New(env, "float64")});
        NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
        double* dataC = static_cast<double*>(c->data());

        for (int64_t i = 0; i < a->size(); i++) {
            dataC[i] = op(dataA[i], dataB[i]);
        }
        return result;
    }

    // Broadcasting path
    std::vector<int64_t> resultShape = computeBroadcastShape(shapeA, shapeB);

    if (resultShape.empty()) {
        Napi::Error::New(env, "Shapes are not broadcastable").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Create result array
    Napi::Array jsResultShape = Napi::Array::New(env, resultShape.size());
    for (size_t i = 0; i < resultShape.size(); i++) {
        jsResultShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(resultShape[i])));
    }

    Napi::Object result = NativeNDArray::constructor.New({jsResultShape, Napi::String::New(env, "float64")});
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

    Napi::Object result = NativeNDArray::constructor.New({shape, Napi::String::New(env, "float64")});
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

        Napi::Object result = NativeNDArray::constructor.New({shape, Napi::String::New(env, "float64")});
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

        Napi::Object result = NativeNDArray::constructor.New({shape, Napi::String::New(env, "float64")});
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

        Napi::Object result = NativeNDArray::constructor.New({shape, Napi::String::New(env, "float64")});
        NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);

        double scalar = -info[1].As<Napi::Number>().DoubleValue();
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

        Napi::Object result = NativeNDArray::constructor.New({shape, Napi::String::New(env, "float64")});
        NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);

        vDSP_vsubD(static_cast<double*>(b->data()), 1,
                   static_cast<double*>(a->data()), 1,
                   static_cast<double*>(c->data()), 1, a->size());
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

        Napi::Object result = NativeNDArray::constructor.New({shape, Napi::String::New(env, "float64")});
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

        Napi::Object result = NativeNDArray::constructor.New({shape, Napi::String::New(env, "float64")});
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

        Napi::Object result = NativeNDArray::constructor.New({shape, Napi::String::New(env, "float64")});
        NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);

        double scalar = info[1].As<Napi::Number>().DoubleValue();
        vDSP_vsdivD(static_cast<double*>(a->data()), 1, &scalar,
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

        Napi::Object result = NativeNDArray::constructor.New({shape, Napi::String::New(env, "float64")});
        NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);

        vDSP_vdivD(static_cast<double*>(b->data()), 1,
                   static_cast<double*>(a->data()), 1,
                   static_cast<double*>(c->data()), 1, a->size());
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

Napi::Value Sqrt(const Napi::CallbackInfo& info) {
    return UnaryOp(info, [](double a) { return std::sqrt(a); });
}

Napi::Value Exp(const Napi::CallbackInfo& info) {
    return UnaryOp(info, [](double a) { return std::exp(a); });
}

Napi::Value Log(const Napi::CallbackInfo& info) {
    return UnaryOp(info, [](double a) { return std::log(a); });
}

Napi::Value Sin(const Napi::CallbackInfo& info) {
    return UnaryOp(info, [](double a) { return std::sin(a); });
}

Napi::Value Cos(const Napi::CallbackInfo& info) {
    return UnaryOp(info, [](double a) { return std::cos(a); });
}

Napi::Value Tan(const Napi::CallbackInfo& info) {
    return UnaryOp(info, [](double a) { return std::tan(a); });
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

    exports.Set("math", math);
    return exports;
}

} // namespace math
} // namespace numpy_node
