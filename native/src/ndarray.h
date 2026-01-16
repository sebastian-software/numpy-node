#ifndef NUMPY_NODE_NDARRAY_H
#define NUMPY_NODE_NDARRAY_H

#include <napi.h>
#include <vector>
#include <memory>
#include <cstdint>

namespace numpy_node {

/**
 * Data type enumeration matching TypeScript DTypeName
 */
enum class DType {
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Float32,
    Float64,
    Bool
};

/**
 * Get the size in bytes for a dtype
 */
size_t dtype_size(DType dtype);

/**
 * Convert dtype enum to string
 */
const char* dtype_to_string(DType dtype);

/**
 * Convert string to dtype enum
 */
DType string_to_dtype(const std::string& str);

/**
 * Native NDArray class
 * Wraps a contiguous memory buffer with shape/stride information
 */
class NativeNDArray : public Napi::ObjectWrap<NativeNDArray> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    static Napi::Object NewInstance(Napi::Env env,
                                    void* data,
                                    const std::vector<int64_t>& shape,
                                    DType dtype,
                                    bool owns_data = true);

    NativeNDArray(const Napi::CallbackInfo& info);
    ~NativeNDArray();

    // Property getters
    Napi::Value GetShape(const Napi::CallbackInfo& info);
    Napi::Value GetStrides(const Napi::CallbackInfo& info);
    Napi::Value GetDType(const Napi::CallbackInfo& info);
    Napi::Value GetNdim(const Napi::CallbackInfo& info);
    Napi::Value GetSize(const Napi::CallbackInfo& info);
    Napi::Value GetData(const Napi::CallbackInfo& info);

    // Methods
    Napi::Value Copy(const Napi::CallbackInfo& info);
    Napi::Value Reshape(const Napi::CallbackInfo& info);
    Napi::Value Transpose(const Napi::CallbackInfo& info);
    Napi::Value AsContiguous(const Napi::CallbackInfo& info);
    void SetValue(const Napi::CallbackInfo& info);
    Napi::Value Fill(const Napi::CallbackInfo& info);

    // Data access for native operations
    void* data() { return data_; }
    const void* data() const { return data_; }
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    DType dtype() const { return dtype_; }
    size_t ndim() const { return shape_.size(); }
    int64_t size() const;
    bool is_contiguous() const;

    // Constructor reference for creating new instances
    static Napi::FunctionReference constructor;

private:
    void* data_;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    DType dtype_;
    bool owns_data_;
    int64_t offset_;

    void compute_strides();
};

/**
 * Array creation functions
 */
Napi::Value CreateZeros(const Napi::CallbackInfo& info);
Napi::Value CreateOnes(const Napi::CallbackInfo& info);
Napi::Value CreateFull(const Napi::CallbackInfo& info);
Napi::Value CreateFromTypedArray(const Napi::CallbackInfo& info);
Napi::Value CreateArange(const Napi::CallbackInfo& info);
Napi::Value CreateLinspace(const Napi::CallbackInfo& info);
Napi::Value CreateEye(const Napi::CallbackInfo& info);

} // namespace numpy_node

#endif // NUMPY_NODE_NDARRAY_H
