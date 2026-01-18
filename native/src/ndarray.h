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
 * Shared data buffer with reference counting.
 * Multiple NDArrays can share the same underlying memory.
 */
class DataBuffer : public std::enable_shared_from_this<DataBuffer> {
public:
    explicit DataBuffer(size_t byte_size);
    DataBuffer(void* external_data, size_t byte_size, bool owns);
    ~DataBuffer();

    void* data() { return data_; }
    const void* data() const { return data_; }
    size_t byte_size() const { return byte_size_; }

    // Non-copyable
    DataBuffer(const DataBuffer&) = delete;
    DataBuffer& operator=(const DataBuffer&) = delete;

private:
    void* data_;
    size_t byte_size_;
    bool owns_data_;
};

/**
 * Native NDArray class
 * Wraps a contiguous memory buffer with shape/stride information.
 * Supports views: multiple NDArrays can share the same DataBuffer.
 */
class NativeNDArray : public Napi::ObjectWrap<NativeNDArray> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);

    // Create a new view sharing the same buffer
    static Napi::Object NewView(Napi::Env env,
                                std::shared_ptr<DataBuffer> buffer,
                                int64_t offset,
                                const std::vector<int64_t>& shape,
                                const std::vector<int64_t>& strides,
                                DType dtype);

    NativeNDArray(const Napi::CallbackInfo& info);
    ~NativeNDArray();

    // Property getters
    Napi::Value GetShape(const Napi::CallbackInfo& info);
    Napi::Value GetStrides(const Napi::CallbackInfo& info);
    Napi::Value GetDType(const Napi::CallbackInfo& info);
    Napi::Value GetNdim(const Napi::CallbackInfo& info);
    Napi::Value GetSize(const Napi::CallbackInfo& info);
    Napi::Value GetData(const Napi::CallbackInfo& info);
    Napi::Value GetIsContiguous(const Napi::CallbackInfo& info);
    Napi::Value GetIsView(const Napi::CallbackInfo& info);

    // Methods
    Napi::Value Copy(const Napi::CallbackInfo& info);
    Napi::Value Reshape(const Napi::CallbackInfo& info);
    Napi::Value Transpose(const Napi::CallbackInfo& info);
    Napi::Value AsContiguous(const Napi::CallbackInfo& info);
    void SetValue(const Napi::CallbackInfo& info);
    Napi::Value Fill(const Napi::CallbackInfo& info);

    // Data access for native operations
    void* data() { return static_cast<char*>(buffer_->data()) + offset_; }
    const void* data() const { return static_cast<const char*>(buffer_->data()) + offset_; }
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    DType dtype() const { return dtype_; }
    size_t ndim() const { return shape_.size(); }
    int64_t size() const;
    bool is_contiguous() const;
    bool is_view() const { return is_view_; }
    std::shared_ptr<DataBuffer> buffer() const { return buffer_; }
    int64_t offset() const { return offset_; }

    // Constructor reference for creating new instances
    static Napi::FunctionReference constructor;

    // Internal: set view data (called by NewView)
    void setViewData(std::shared_ptr<DataBuffer> buffer,
                     int64_t offset,
                     const std::vector<int64_t>& shape,
                     const std::vector<int64_t>& strides,
                     DType dtype);

private:
    std::shared_ptr<DataBuffer> buffer_;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;  // In bytes
    DType dtype_;
    int64_t offset_;  // Byte offset into buffer
    bool is_view_;

    // Cached ArrayBuffer for zero-copy data access (only for contiguous)
    std::unique_ptr<Napi::Reference<Napi::ArrayBuffer>> cached_buffer_;

    void compute_contiguous_strides();
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
