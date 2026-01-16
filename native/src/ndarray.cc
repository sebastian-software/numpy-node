#include "ndarray.h"
#include <cstring>
#include <stdexcept>
#include <numeric>

namespace np_ts {

Napi::FunctionReference NativeNDArray::constructor;

size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Int8:
        case DType::Uint8:
        case DType::Bool:
            return 1;
        case DType::Int16:
        case DType::Uint16:
            return 2;
        case DType::Int32:
        case DType::Uint32:
        case DType::Float32:
            return 4;
        case DType::Int64:
        case DType::Uint64:
        case DType::Float64:
            return 8;
        default:
            return 8;
    }
}

const char* dtype_to_string(DType dtype) {
    switch (dtype) {
        case DType::Int8: return "int8";
        case DType::Int16: return "int16";
        case DType::Int32: return "int32";
        case DType::Int64: return "int64";
        case DType::Uint8: return "uint8";
        case DType::Uint16: return "uint16";
        case DType::Uint32: return "uint32";
        case DType::Uint64: return "uint64";
        case DType::Float32: return "float32";
        case DType::Float64: return "float64";
        case DType::Bool: return "bool";
        default: return "float64";
    }
}

DType string_to_dtype(const std::string& str) {
    if (str == "int8") return DType::Int8;
    if (str == "int16") return DType::Int16;
    if (str == "int32") return DType::Int32;
    if (str == "int64") return DType::Int64;
    if (str == "uint8") return DType::Uint8;
    if (str == "uint16") return DType::Uint16;
    if (str == "uint32") return DType::Uint32;
    if (str == "uint64") return DType::Uint64;
    if (str == "float32") return DType::Float32;
    if (str == "float64") return DType::Float64;
    if (str == "bool") return DType::Bool;
    return DType::Float64;
}

Napi::Object NativeNDArray::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "NativeNDArray", {
        InstanceAccessor<&NativeNDArray::GetShape>("shape"),
        InstanceAccessor<&NativeNDArray::GetStrides>("strides"),
        InstanceAccessor<&NativeNDArray::GetDType>("dtype"),
        InstanceAccessor<&NativeNDArray::GetNdim>("ndim"),
        InstanceAccessor<&NativeNDArray::GetSize>("size"),
        InstanceAccessor<&NativeNDArray::GetData>("data"),
        InstanceMethod<&NativeNDArray::Copy>("copy"),
        InstanceMethod<&NativeNDArray::Reshape>("reshape"),
        InstanceMethod<&NativeNDArray::Transpose>("transpose"),
        InstanceMethod<&NativeNDArray::AsContiguous>("asContiguous"),
    });

    constructor = Napi::Persistent(func);
    constructor.SuppressDestruct();

    exports.Set("NativeNDArray", func);

    // Export creation functions
    exports.Set("zeros", Napi::Function::New(env, CreateZeros));
    exports.Set("ones", Napi::Function::New(env, CreateOnes));
    exports.Set("full", Napi::Function::New(env, CreateFull));
    exports.Set("fromTypedArray", Napi::Function::New(env, CreateFromTypedArray));
    exports.Set("arange", Napi::Function::New(env, CreateArange));
    exports.Set("linspace", Napi::Function::New(env, CreateLinspace));
    exports.Set("eye", Napi::Function::New(env, CreateEye));

    return exports;
}

NativeNDArray::NativeNDArray(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<NativeNDArray>(info),
      data_(nullptr),
      dtype_(DType::Float64),
      owns_data_(true),
      offset_(0) {

    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected shape and dtype").ThrowAsJavaScriptException();
        return;
    }

    // Parse shape
    if (!info[0].IsArray()) {
        Napi::TypeError::New(env, "Shape must be an array").ThrowAsJavaScriptException();
        return;
    }
    Napi::Array shapeArr = info[0].As<Napi::Array>();
    shape_.resize(shapeArr.Length());
    for (size_t i = 0; i < shapeArr.Length(); i++) {
        shape_[i] = shapeArr.Get(i).As<Napi::Number>().Int64Value();
    }

    // Parse dtype
    if (!info[1].IsString()) {
        Napi::TypeError::New(env, "DType must be a string").ThrowAsJavaScriptException();
        return;
    }
    dtype_ = string_to_dtype(info[1].As<Napi::String>().Utf8Value());

    // Compute strides
    compute_strides();

    // Allocate memory
    int64_t total_size = size();
    size_t byte_size = total_size * dtype_size(dtype_);
    data_ = std::malloc(byte_size);
    if (!data_) {
        Napi::Error::New(env, "Failed to allocate memory").ThrowAsJavaScriptException();
        return;
    }
    std::memset(data_, 0, byte_size);
}

NativeNDArray::~NativeNDArray() {
    if (owns_data_ && data_) {
        std::free(data_);
        data_ = nullptr;
    }
}

void NativeNDArray::compute_strides() {
    strides_.resize(shape_.size());
    if (shape_.empty()) return;

    // C-contiguous strides
    strides_[shape_.size() - 1] = dtype_size(dtype_);
    for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; i--) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

int64_t NativeNDArray::size() const {
    if (shape_.empty()) return 1;
    return std::accumulate(shape_.begin(), shape_.end(), int64_t(1), std::multiplies<int64_t>());
}

bool NativeNDArray::is_contiguous() const {
    if (shape_.empty()) return true;

    int64_t expected_stride = dtype_size(dtype_);
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; i--) {
        if (strides_[i] != expected_stride) return false;
        expected_stride *= shape_[i];
    }
    return true;
}

Napi::Value NativeNDArray::GetShape(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Array result = Napi::Array::New(env, shape_.size());
    for (size_t i = 0; i < shape_.size(); i++) {
        result.Set(i, Napi::Number::New(env, static_cast<double>(shape_[i])));
    }
    return result;
}

Napi::Value NativeNDArray::GetStrides(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Array result = Napi::Array::New(env, strides_.size());
    for (size_t i = 0; i < strides_.size(); i++) {
        result.Set(i, Napi::Number::New(env, static_cast<double>(strides_[i])));
    }
    return result;
}

Napi::Value NativeNDArray::GetDType(const Napi::CallbackInfo& info) {
    return Napi::String::New(info.Env(), dtype_to_string(dtype_));
}

Napi::Value NativeNDArray::GetNdim(const Napi::CallbackInfo& info) {
    return Napi::Number::New(info.Env(), static_cast<double>(shape_.size()));
}

Napi::Value NativeNDArray::GetSize(const Napi::CallbackInfo& info) {
    return Napi::Number::New(info.Env(), static_cast<double>(size()));
}

Napi::Value NativeNDArray::GetData(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    size_t byte_length = size() * dtype_size(dtype_);

    // Create an ArrayBuffer from our data
    // Note: We're creating a copy here for safety - the caller gets their own buffer
    Napi::ArrayBuffer buffer = Napi::ArrayBuffer::New(env, byte_length);
    std::memcpy(buffer.Data(), data_, byte_length);

    // Create appropriate TypedArray view
    switch (dtype_) {
        case DType::Int8:
            return Napi::Int8Array::New(env, byte_length, buffer, 0);
        case DType::Int16:
            return Napi::Int16Array::New(env, byte_length / 2, buffer, 0);
        case DType::Int32:
            return Napi::Int32Array::New(env, byte_length / 4, buffer, 0);
        case DType::Uint8:
        case DType::Bool:
            return Napi::Uint8Array::New(env, byte_length, buffer, 0);
        case DType::Uint16:
            return Napi::Uint16Array::New(env, byte_length / 2, buffer, 0);
        case DType::Uint32:
            return Napi::Uint32Array::New(env, byte_length / 4, buffer, 0);
        case DType::Float32:
            return Napi::Float32Array::New(env, byte_length / 4, buffer, 0);
        case DType::Float64:
        default:
            return Napi::Float64Array::New(env, byte_length / 8, buffer, 0);
    }
}

Napi::Value NativeNDArray::Copy(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Array shapeArr = Napi::Array::New(env, shape_.size());
    for (size_t i = 0; i < shape_.size(); i++) {
        shapeArr.Set(i, Napi::Number::New(env, static_cast<double>(shape_[i])));
    }

    Napi::Object result = constructor.New({
        shapeArr,
        Napi::String::New(env, dtype_to_string(dtype_))
    });

    NativeNDArray* copy = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    std::memcpy(copy->data_, data_, size() * dtype_size(dtype_));

    return result;
}

Napi::Value NativeNDArray::Reshape(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsArray()) {
        Napi::TypeError::New(env, "Expected new shape array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Array newShapeArr = info[0].As<Napi::Array>();
    std::vector<int64_t> newShape(newShapeArr.Length());
    int64_t inferIndex = -1;
    int64_t knownSize = 1;

    for (size_t i = 0; i < newShapeArr.Length(); i++) {
        newShape[i] = newShapeArr.Get(i).As<Napi::Number>().Int64Value();
        if (newShape[i] == -1) {
            if (inferIndex != -1) {
                Napi::Error::New(env, "Can only specify one unknown dimension").ThrowAsJavaScriptException();
                return env.Undefined();
            }
            inferIndex = i;
        } else {
            knownSize *= newShape[i];
        }
    }

    int64_t currentSize = size();
    if (inferIndex != -1) {
        if (currentSize % knownSize != 0) {
            Napi::Error::New(env, "Cannot reshape array").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        newShape[inferIndex] = currentSize / knownSize;
    }

    // Check sizes match
    int64_t newSize = std::accumulate(newShape.begin(), newShape.end(), int64_t(1), std::multiplies<int64_t>());
    if (newSize != currentSize) {
        Napi::Error::New(env, "New shape must have same total size").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Create new array with reshaped view
    Napi::Array shapeArr = Napi::Array::New(env, newShape.size());
    for (size_t i = 0; i < newShape.size(); i++) {
        shapeArr.Set(i, Napi::Number::New(env, static_cast<double>(newShape[i])));
    }

    Napi::Object result = constructor.New({
        shapeArr,
        Napi::String::New(env, dtype_to_string(dtype_))
    });

    NativeNDArray* reshaped = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    std::memcpy(reshaped->data_, data_, size() * dtype_size(dtype_));

    return result;
}

Napi::Value NativeNDArray::Transpose(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    std::vector<int64_t> newShape(shape_.rbegin(), shape_.rend());
    std::vector<int64_t> newStrides(strides_.rbegin(), strides_.rend());

    Napi::Array shapeArr = Napi::Array::New(env, newShape.size());
    for (size_t i = 0; i < newShape.size(); i++) {
        shapeArr.Set(i, Napi::Number::New(env, static_cast<double>(newShape[i])));
    }

    Napi::Object result = constructor.New({
        shapeArr,
        Napi::String::New(env, dtype_to_string(dtype_))
    });

    // For now, just copy data - proper strided transpose would be more complex
    NativeNDArray* transposed = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    // This is a simplified implementation - full transpose would need proper striding
    std::memcpy(transposed->data_, data_, size() * dtype_size(dtype_));

    return result;
}

Napi::Value NativeNDArray::AsContiguous(const Napi::CallbackInfo& info) {
    if (is_contiguous()) {
        return Copy(info);
    }
    // TODO: Implement proper contiguous copy for non-contiguous arrays
    return Copy(info);
}

// Creation functions
Napi::Value CreateZeros(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsArray()) {
        Napi::TypeError::New(env, "Expected shape array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    std::string dtype = "float64";
    if (info.Length() >= 2 && info[1].IsString()) {
        dtype = info[1].As<Napi::String>().Utf8Value();
    }

    return NativeNDArray::constructor.New({info[0], Napi::String::New(env, dtype)});
}

Napi::Value CreateOnes(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Object arr = CreateZeros(info).As<Napi::Object>();
    NativeNDArray* ndarr = Napi::ObjectWrap<NativeNDArray>::Unwrap(arr);

    // Fill with ones
    size_t total = ndarr->size();
    switch (ndarr->dtype()) {
        case DType::Float64: {
            double* data = static_cast<double*>(ndarr->data());
            for (size_t i = 0; i < total; i++) data[i] = 1.0;
            break;
        }
        case DType::Float32: {
            float* data = static_cast<float*>(ndarr->data());
            for (size_t i = 0; i < total; i++) data[i] = 1.0f;
            break;
        }
        case DType::Int32: {
            int32_t* data = static_cast<int32_t*>(ndarr->data());
            for (size_t i = 0; i < total; i++) data[i] = 1;
            break;
        }
        default: {
            // Generic case for other types
            uint8_t* data = static_cast<uint8_t*>(ndarr->data());
            size_t item_size = dtype_size(ndarr->dtype());
            for (size_t i = 0; i < total * item_size; i += item_size) {
                data[i] = 1;
            }
            break;
        }
    }

    return arr;
}

Napi::Value CreateFull(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected shape and fill value").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    double fillValue = info[1].As<Napi::Number>().DoubleValue();

    std::string dtype = "float64";
    if (info.Length() >= 3 && info[2].IsString()) {
        dtype = info[2].As<Napi::String>().Utf8Value();
    }

    Napi::Object arr = NativeNDArray::constructor.New({info[0], Napi::String::New(env, dtype)});
    NativeNDArray* ndarr = Napi::ObjectWrap<NativeNDArray>::Unwrap(arr);

    size_t total = ndarr->size();
    switch (ndarr->dtype()) {
        case DType::Float64: {
            double* data = static_cast<double*>(ndarr->data());
            for (size_t i = 0; i < total; i++) data[i] = fillValue;
            break;
        }
        case DType::Float32: {
            float* data = static_cast<float*>(ndarr->data());
            for (size_t i = 0; i < total; i++) data[i] = static_cast<float>(fillValue);
            break;
        }
        case DType::Int32: {
            int32_t* data = static_cast<int32_t*>(ndarr->data());
            for (size_t i = 0; i < total; i++) data[i] = static_cast<int32_t>(fillValue);
            break;
        }
        default:
            break;
    }

    return arr;
}

Napi::Value CreateFromTypedArray(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected TypedArray and shape").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    if (!info[0].IsTypedArray()) {
        Napi::TypeError::New(env, "First argument must be a TypedArray").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::TypedArray typedArr = info[0].As<Napi::TypedArray>();
    Napi::Array shapeArr = info[1].As<Napi::Array>();

    std::string dtype;
    switch (typedArr.TypedArrayType()) {
        case napi_int8_array: dtype = "int8"; break;
        case napi_int16_array: dtype = "int16"; break;
        case napi_int32_array: dtype = "int32"; break;
        case napi_uint8_array: dtype = "uint8"; break;
        case napi_uint16_array: dtype = "uint16"; break;
        case napi_uint32_array: dtype = "uint32"; break;
        case napi_float32_array: dtype = "float32"; break;
        case napi_float64_array: dtype = "float64"; break;
        case napi_bigint64_array: dtype = "int64"; break;
        case napi_biguint64_array: dtype = "uint64"; break;
        default: dtype = "float64"; break;
    }

    Napi::Object arr = NativeNDArray::constructor.New({shapeArr, Napi::String::New(env, dtype)});
    NativeNDArray* ndarr = Napi::ObjectWrap<NativeNDArray>::Unwrap(arr);

    // Copy data from TypedArray
    std::memcpy(ndarr->data(),
                reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(typedArr.ArrayBuffer().Data()) + typedArr.ByteOffset()),
                typedArr.ByteLength());

    return arr;
}

Napi::Value CreateArange(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    double start = 0, stop, step = 1;

    if (info.Length() == 1) {
        stop = info[0].As<Napi::Number>().DoubleValue();
    } else if (info.Length() >= 2) {
        start = info[0].As<Napi::Number>().DoubleValue();
        stop = info[1].As<Napi::Number>().DoubleValue();
        if (info.Length() >= 3) {
            step = info[2].As<Napi::Number>().DoubleValue();
        }
    } else {
        Napi::TypeError::New(env, "Expected at least one argument").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    if (step == 0) {
        Napi::Error::New(env, "Step cannot be zero").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int64_t length = static_cast<int64_t>(std::ceil((stop - start) / step));
    if (length < 0) length = 0;

    std::string dtype = "float64";
    if (info.Length() >= 4 && info[3].IsString()) {
        dtype = info[3].As<Napi::String>().Utf8Value();
    }

    Napi::Array shape = Napi::Array::New(env, 1);
    shape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(length)));

    Napi::Object arr = NativeNDArray::constructor.New({shape, Napi::String::New(env, dtype)});
    NativeNDArray* ndarr = Napi::ObjectWrap<NativeNDArray>::Unwrap(arr);

    double* data = static_cast<double*>(ndarr->data());
    for (int64_t i = 0; i < length; i++) {
        data[i] = start + i * step;
    }

    return arr;
}

Napi::Value CreateLinspace(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 3) {
        Napi::TypeError::New(env, "Expected start, stop, and num").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    double start = info[0].As<Napi::Number>().DoubleValue();
    double stop = info[1].As<Napi::Number>().DoubleValue();
    int64_t num = info[2].As<Napi::Number>().Int64Value();

    bool endpoint = true;
    if (info.Length() >= 4 && info[3].IsBoolean()) {
        endpoint = info[3].As<Napi::Boolean>().Value();
    }

    std::string dtype = "float64";
    if (info.Length() >= 5 && info[4].IsString()) {
        dtype = info[4].As<Napi::String>().Utf8Value();
    }

    Napi::Array shape = Napi::Array::New(env, 1);
    shape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(num)));

    Napi::Object arr = NativeNDArray::constructor.New({shape, Napi::String::New(env, dtype)});
    NativeNDArray* ndarr = Napi::ObjectWrap<NativeNDArray>::Unwrap(arr);

    double* data = static_cast<double*>(ndarr->data());
    double divisor = endpoint ? (num - 1) : num;
    double step = (stop - start) / divisor;

    for (int64_t i = 0; i < num; i++) {
        data[i] = start + i * step;
    }

    return arr;
}

Napi::Value CreateEye(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected at least n").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int64_t n = info[0].As<Napi::Number>().Int64Value();
    int64_t m = n;
    int64_t k = 0;

    if (info.Length() >= 2 && !info[1].IsUndefined()) {
        m = info[1].As<Napi::Number>().Int64Value();
    }
    if (info.Length() >= 3) {
        k = info[2].As<Napi::Number>().Int64Value();
    }

    std::string dtype = "float64";
    if (info.Length() >= 4 && info[3].IsString()) {
        dtype = info[3].As<Napi::String>().Utf8Value();
    }

    Napi::Array shape = Napi::Array::New(env, 2);
    shape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(n)));
    shape.Set(uint32_t(1), Napi::Number::New(env, static_cast<double>(m)));

    Napi::Object arr = NativeNDArray::constructor.New({shape, Napi::String::New(env, dtype)});
    NativeNDArray* ndarr = Napi::ObjectWrap<NativeNDArray>::Unwrap(arr);

    double* data = static_cast<double*>(ndarr->data());

    // Set diagonal
    for (int64_t i = 0; i < n; i++) {
        int64_t j = i + k;
        if (j >= 0 && j < m) {
            data[i * m + j] = 1.0;
        }
    }

    return arr;
}

} // namespace np_ts
