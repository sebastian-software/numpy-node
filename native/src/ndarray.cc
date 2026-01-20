#include "ndarray.h"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <numeric>
#include <algorithm>

#if defined(USE_ACCELERATE)
    #include <Accelerate/Accelerate.h>
#endif

namespace numpy_node {

// ============================================================
// DataBuffer Implementation
// ============================================================

DataBuffer::DataBuffer(size_t byte_size)
    : data_(nullptr), byte_size_(byte_size), owns_data_(true) {
    if (byte_size > 0) {
        data_ = std::calloc(1, byte_size);
        if (!data_) {
            throw std::bad_alloc();
        }
    }
}

DataBuffer::DataBuffer(void* external_data, size_t byte_size, bool owns)
    : data_(external_data), byte_size_(byte_size), owns_data_(owns) {}

DataBuffer::~DataBuffer() {
    if (owns_data_ && data_) {
        std::free(data_);
        data_ = nullptr;
    }
}

// ============================================================
// NativeNDArray Implementation
// ============================================================

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
        InstanceAccessor<&NativeNDArray::GetIsContiguous>("isContiguous"),
        InstanceAccessor<&NativeNDArray::GetIsView>("isView"),
        InstanceMethod<&NativeNDArray::Copy>("copy"),
        InstanceMethod<&NativeNDArray::Reshape>("reshape"),
        InstanceMethod<&NativeNDArray::Transpose>("transpose"),
        InstanceMethod<&NativeNDArray::AsContiguous>("asContiguous"),
        InstanceMethod<&NativeNDArray::SetValue>("set"),
        InstanceMethod<&NativeNDArray::Fill>("fill"),
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

Napi::Object NativeNDArray::NewView(Napi::Env env,
                                    std::shared_ptr<DataBuffer> buffer,
                                    int64_t offset,
                                    const std::vector<int64_t>& shape,
                                    const std::vector<int64_t>& strides,
                                    DType dtype) {
    // Create array with minimal allocation (will be replaced)
    Napi::Array shapeArr = Napi::Array::New(env, shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
        shapeArr.Set(i, Napi::Number::New(env, static_cast<double>(shape[i])));
    }

    // Create with "view" marker (4th arg = true means view mode)
    Napi::Object result = constructor.New({
        shapeArr,
        Napi::String::New(env, dtype_to_string(dtype)),
        Napi::Boolean::New(env, true),   // skipInit
        Napi::Boolean::New(env, true)    // isView marker
    });

    NativeNDArray* arr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    arr->setViewData(buffer, offset, shape, strides, dtype);

    return result;
}

void NativeNDArray::setViewData(std::shared_ptr<DataBuffer> buffer,
                                int64_t offset,
                                const std::vector<int64_t>& shape,
                                const std::vector<int64_t>& strides,
                                DType dtype) {
    buffer_ = buffer;
    offset_ = offset;
    shape_ = shape;
    strides_ = strides;
    dtype_ = dtype;
    is_view_ = true;
    cached_buffer_.reset();  // Clear cache - views need special handling
}

NativeNDArray::NativeNDArray(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<NativeNDArray>(info),
      buffer_(nullptr),
      dtype_(DType::Float64),
      offset_(0),
      is_view_(false) {

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

    // Check if this is a view creation (4th arg = true)
    bool isViewCreation = (info.Length() >= 4 && info[3].IsBoolean() && info[3].As<Napi::Boolean>().Value());

    if (isViewCreation) {
        // View mode: buffer will be set by setViewData()
        // Create minimal buffer that will be replaced
        is_view_ = true;
        buffer_ = std::make_shared<DataBuffer>(0);
        compute_contiguous_strides();
        return;
    }

    // Compute strides for contiguous array
    compute_contiguous_strides();

    // Allocate memory
    int64_t total_size = size();
    size_t byte_size = total_size * dtype_size(dtype_);

    // Check if we should skip zero-initialization (3rd argument = skipInit)
    bool skipInit = (info.Length() >= 3 && info[2].IsBoolean() && info[2].As<Napi::Boolean>().Value());

    try {
        if (skipInit) {
            // For ones/full: allocate without zeroing
            void* raw = std::malloc(byte_size);
            if (!raw) throw std::bad_alloc();
            buffer_ = std::make_shared<DataBuffer>(raw, byte_size, true);
        } else {
            // For zeros: DataBuffer constructor uses calloc
            buffer_ = std::make_shared<DataBuffer>(byte_size);
        }
    } catch (const std::bad_alloc&) {
        Napi::Error::New(env, "Failed to allocate memory").ThrowAsJavaScriptException();
        return;
    }
}

NativeNDArray::~NativeNDArray() {
    // shared_ptr handles cleanup automatically
    buffer_.reset();
}

void NativeNDArray::compute_contiguous_strides() {
    strides_.resize(shape_.size());
    if (shape_.empty()) return;

    // C-contiguous strides (in bytes)
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

    // For non-contiguous views, we need to return a contiguous copy
    if (!is_contiguous()) {
        // Create contiguous copy and return its data
        Napi::Object contiguous = AsContiguous(info).As<Napi::Object>();
        NativeNDArray* contArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(contiguous);
        return contArr->GetData(info);
    }

    size_t byte_length = size() * dtype_size(dtype_);

    // Create or reuse cached ArrayBuffer (zero-copy for contiguous arrays)
    if (!cached_buffer_) {
        Napi::ArrayBuffer buffer = Napi::ArrayBuffer::New(
            env,
            data(),  // Now uses offset-corrected data()
            byte_length
        );
        cached_buffer_ = std::make_unique<Napi::Reference<Napi::ArrayBuffer>>(
            Napi::Persistent(buffer)
        );
    }

    Napi::ArrayBuffer buffer = cached_buffer_->Value();

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

Napi::Value NativeNDArray::GetIsContiguous(const Napi::CallbackInfo& info) {
    return Napi::Boolean::New(info.Env(), is_contiguous());
}

Napi::Value NativeNDArray::GetIsView(const Napi::CallbackInfo& info) {
    return Napi::Boolean::New(info.Env(), is_view_);
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

    // If contiguous, simple memcpy
    if (is_contiguous()) {
        std::memcpy(copy->data(), data(), size() * dtype_size(dtype_));
    } else {
        // Non-contiguous: copy element by element using strides
        int64_t totalSize = size();
        int ndim = static_cast<int>(shape_.size());
        size_t elemSize = dtype_size(dtype_);

        const char* src = static_cast<const char*>(data());
        char* dst = static_cast<char*>(copy->data());

        // Iterate over all elements
        std::vector<int64_t> indices(ndim, 0);
        for (int64_t i = 0; i < totalSize; i++) {
            // Calculate source offset using strides
            int64_t srcOffset = 0;
            for (int d = 0; d < ndim; d++) {
                srcOffset += indices[d] * strides_[d];
            }

            // Copy element (destination is contiguous, so use elemSize * i)
            std::memcpy(dst + i * elemSize, src + srcOffset, elemSize);

            // Increment indices (like a multi-digit counter)
            for (int d = ndim - 1; d >= 0; d--) {
                indices[d]++;
                if (indices[d] < shape_[d]) break;
                indices[d] = 0;
            }
        }
    }

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

    // If contiguous, we can return a view with new shape
    if (is_contiguous()) {
        // Compute new C-contiguous strides for the new shape
        std::vector<int64_t> newStrides(newShape.size());
        if (!newShape.empty()) {
            newStrides[newShape.size() - 1] = dtype_size(dtype_);
            for (int i = static_cast<int>(newShape.size()) - 2; i >= 0; i--) {
                newStrides[i] = newStrides[i + 1] * newShape[i + 1];
            }
        }

        return NewView(env, buffer_, offset_, newShape, newStrides, dtype_);
    }

    // Non-contiguous: need to make a contiguous copy first
    Napi::Object contiguous = AsContiguous(info).As<Napi::Object>();
    NativeNDArray* contArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(contiguous);

    // Compute new strides for the new shape
    std::vector<int64_t> newStrides(newShape.size());
    if (!newShape.empty()) {
        newStrides[newShape.size() - 1] = dtype_size(dtype_);
        for (int i = static_cast<int>(newShape.size()) - 2; i >= 0; i--) {
            newStrides[i] = newStrides[i + 1] * newShape[i + 1];
        }
    }

    return NewView(env, contArr->buffer_, contArr->offset_, newShape, newStrides, dtype_);
}

Napi::Value NativeNDArray::Transpose(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    // Transpose is a VIEW operation - just permute shape and strides!
    // No data copy needed. This is O(1) instead of O(n).

    // 1D arrays - transpose is identity, return view with same shape/strides
    if (shape_.size() <= 1) {
        return NewView(env, buffer_, offset_, shape_, strides_, dtype_);
    }

    std::vector<int64_t> newShape;
    std::vector<int64_t> newStrides;

    // Check if axes are provided
    if (info.Length() >= 1 && info[0].IsArray()) {
        Napi::Array jsAxes = info[0].As<Napi::Array>();
        size_t ndim = shape_.size();

        // Validate axes length
        if (jsAxes.Length() != ndim) {
            Napi::Error::New(env, "axes must have same length as ndim").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        // Get axes permutation
        std::vector<int64_t> axes(ndim);
        for (size_t i = 0; i < ndim; i++) {
            axes[i] = jsAxes.Get(static_cast<uint32_t>(i)).As<Napi::Number>().Int64Value();
            // Handle negative axes
            if (axes[i] < 0) {
                axes[i] += ndim;
            }
            if (axes[i] < 0 || axes[i] >= static_cast<int64_t>(ndim)) {
                Napi::Error::New(env, "Invalid axis in transpose").ThrowAsJavaScriptException();
                return env.Undefined();
            }
        }

        // Permute shape and strides according to axes
        newShape.resize(ndim);
        newStrides.resize(ndim);
        for (size_t i = 0; i < ndim; i++) {
            newShape[i] = shape_[axes[i]];
            newStrides[i] = strides_[axes[i]];
        }
    } else {
        // Default: reverse all axes
        newShape = std::vector<int64_t>(shape_.rbegin(), shape_.rend());
        newStrides = std::vector<int64_t>(strides_.rbegin(), strides_.rend());
    }

    // Return a view sharing the same buffer with permuted shape/strides
    return NewView(env, buffer_, offset_, newShape, newStrides, dtype_);
}

Napi::Value NativeNDArray::AsContiguous(const Napi::CallbackInfo& info) {
    // If already contiguous, just return a copy (or could return self for optimization)
    // Return a new contiguous array
    return Copy(info);  // Copy now handles both contiguous and non-contiguous
}

void NativeNDArray::SetValue(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected indices and value").ThrowAsJavaScriptException();
        return;
    }

    // Get indices array
    Napi::Array jsIndices = info[0].As<Napi::Array>();
    double value = info[1].As<Napi::Number>().DoubleValue();

    // Calculate byte offset from indices using strides (strides are in bytes)
    int64_t byteOffset = 0;
    for (uint32_t i = 0; i < jsIndices.Length(); i++) {
        int64_t idx = jsIndices.Get(i).As<Napi::Number>().Int64Value();
        byteOffset += idx * strides_[i];
    }

    // Get pointer to element
    char* basePtr = static_cast<char*>(data());
    void* elemPtr = basePtr + byteOffset;

    // Set value based on dtype
    switch (dtype_) {
        case DType::Float64: {
            *static_cast<double*>(elemPtr) = value;
            break;
        }
        case DType::Float32: {
            *static_cast<float*>(elemPtr) = static_cast<float>(value);
            break;
        }
        case DType::Int32: {
            *static_cast<int32_t*>(elemPtr) = static_cast<int32_t>(value);
            break;
        }
        case DType::Int64: {
            *static_cast<int64_t*>(elemPtr) = static_cast<int64_t>(value);
            break;
        }
        case DType::Bool: {
            *static_cast<uint8_t*>(elemPtr) = (value != 0.0) ? 1 : 0;
            break;
        }
        case DType::Uint8: {
            *static_cast<uint8_t*>(elemPtr) = static_cast<uint8_t>(value);
            break;
        }
        default: {
            *static_cast<double*>(elemPtr) = value;
            break;
        }
    }
}

Napi::Value NativeNDArray::Fill(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected fill value").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    double value = info[0].As<Napi::Number>().DoubleValue();
    int64_t totalSize = size();

    // For contiguous arrays, use fast path
    if (is_contiguous()) {
        switch (dtype_) {
            case DType::Float64: {
                double* ptr = static_cast<double*>(data());
                for (int64_t i = 0; i < totalSize; i++) {
                    ptr[i] = value;
                }
                break;
            }
            case DType::Float32: {
                float* ptr = static_cast<float*>(data());
                float fval = static_cast<float>(value);
                for (int64_t i = 0; i < totalSize; i++) {
                    ptr[i] = fval;
                }
                break;
            }
            case DType::Int32: {
                int32_t* ptr = static_cast<int32_t*>(data());
                int32_t ival = static_cast<int32_t>(value);
                for (int64_t i = 0; i < totalSize; i++) {
                    ptr[i] = ival;
                }
                break;
            }
            case DType::Int64: {
                int64_t* ptr = static_cast<int64_t*>(data());
                int64_t ival = static_cast<int64_t>(value);
                for (int64_t i = 0; i < totalSize; i++) {
                    ptr[i] = ival;
                }
                break;
            }
            case DType::Bool: {
                uint8_t* ptr = static_cast<uint8_t*>(data());
                uint8_t bval = (value != 0.0) ? 1 : 0;
                for (int64_t i = 0; i < totalSize; i++) {
                    ptr[i] = bval;
                }
                break;
            }
            case DType::Uint8: {
                uint8_t* ptr = static_cast<uint8_t*>(data());
                uint8_t uval = static_cast<uint8_t>(value);
                for (int64_t i = 0; i < totalSize; i++) {
                    ptr[i] = uval;
                }
                break;
            }
            default: {
                double* ptr = static_cast<double*>(data());
                for (int64_t i = 0; i < totalSize; i++) {
                    ptr[i] = value;
                }
                break;
            }
        }
    } else {
        // Non-contiguous: iterate using strides
        int ndim = static_cast<int>(shape_.size());
        char* basePtr = static_cast<char*>(data());
        std::vector<int64_t> indices(ndim, 0);

        for (int64_t i = 0; i < totalSize; i++) {
            // Calculate byte offset
            int64_t byteOffset = 0;
            for (int d = 0; d < ndim; d++) {
                byteOffset += indices[d] * strides_[d];
            }

            // Set value based on dtype
            switch (dtype_) {
                case DType::Float64:
                    *reinterpret_cast<double*>(basePtr + byteOffset) = value;
                    break;
                case DType::Float32:
                    *reinterpret_cast<float*>(basePtr + byteOffset) = static_cast<float>(value);
                    break;
                case DType::Int32:
                    *reinterpret_cast<int32_t*>(basePtr + byteOffset) = static_cast<int32_t>(value);
                    break;
                case DType::Int64:
                    *reinterpret_cast<int64_t*>(basePtr + byteOffset) = static_cast<int64_t>(value);
                    break;
                case DType::Bool:
                    *reinterpret_cast<uint8_t*>(basePtr + byteOffset) = (value != 0.0) ? 1 : 0;
                    break;
                case DType::Uint8:
                    *reinterpret_cast<uint8_t*>(basePtr + byteOffset) = static_cast<uint8_t>(value);
                    break;
                default:
                    *reinterpret_cast<double*>(basePtr + byteOffset) = value;
                    break;
            }

            // Increment indices
            for (int d = ndim - 1; d >= 0; d--) {
                indices[d]++;
                if (indices[d] < shape_[d]) break;
                indices[d] = 0;
            }
        }
    }

    return info.This();
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

    if (info.Length() < 1 || !info[0].IsArray()) {
        Napi::TypeError::New(env, "Expected shape array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    std::string dtype = "float64";
    if (info.Length() >= 2 && info[1].IsString()) {
        dtype = info[1].As<Napi::String>().Utf8Value();
    }

    // Create array with skipInit=true to avoid double-initialization
    Napi::Object arr = NativeNDArray::constructor.New({
        info[0],
        Napi::String::New(env, dtype),
        Napi::Boolean::New(env, true)  // skipInit
    });
    NativeNDArray* ndarr = Napi::ObjectWrap<NativeNDArray>::Unwrap(arr);

    // Fill with ones using vectorized operations where available
    size_t total = ndarr->size();
    switch (ndarr->dtype()) {
        case DType::Float64: {
            double* data = static_cast<double*>(ndarr->data());
#if defined(USE_ACCELERATE)
            double one = 1.0;
            vDSP_vfillD(&one, data, 1, total);
#else
            for (size_t i = 0; i < total; i++) data[i] = 1.0;
#endif
            break;
        }
        case DType::Float32: {
            float* data = static_cast<float*>(ndarr->data());
#if defined(USE_ACCELERATE)
            float one = 1.0f;
            vDSP_vfill(&one, data, 1, total);
#else
            for (size_t i = 0; i < total; i++) data[i] = 1.0f;
#endif
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

    // Create array with skipInit=true to avoid double-initialization
    Napi::Object arr = NativeNDArray::constructor.New({
        info[0],
        Napi::String::New(env, dtype),
        Napi::Boolean::New(env, true)  // skipInit
    });
    NativeNDArray* ndarr = Napi::ObjectWrap<NativeNDArray>::Unwrap(arr);

    size_t total = ndarr->size();
    switch (ndarr->dtype()) {
        case DType::Float64: {
            double* data = static_cast<double*>(ndarr->data());
#if defined(USE_ACCELERATE)
            vDSP_vfillD(&fillValue, data, 1, total);
#else
            for (size_t i = 0; i < total; i++) data[i] = fillValue;
#endif
            break;
        }
        case DType::Float32: {
            float* data = static_cast<float*>(ndarr->data());
#if defined(USE_ACCELERATE)
            float fillF = static_cast<float>(fillValue);
            vDSP_vfill(&fillF, data, 1, total);
#else
            for (size_t i = 0; i < total; i++) data[i] = static_cast<float>(fillValue);
#endif
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
#if defined(USE_ACCELERATE)
    // vDSP_vrampD creates: data[i] = start + i * step
    vDSP_vrampD(&start, &step, data, 1, static_cast<vDSP_Length>(length));
#else
    for (int64_t i = 0; i < length; i++) {
        data[i] = start + i * step;
    }
#endif

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
    double step = (divisor > 0) ? (stop - start) / divisor : 0;

#if defined(USE_ACCELERATE)
    vDSP_vrampD(&start, &step, data, 1, static_cast<vDSP_Length>(num));
#else
    for (int64_t i = 0; i < num; i++) {
        data[i] = start + i * step;
    }
#endif

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

} // namespace numpy_node
