#include <napi.h>
#include "ndarray.h"
#include "linalg.h"
#include "math_ops.h"
#include "random.h"
#include "fft_ops.h"

/**
 * Module initialization
 */
Napi::Object Init(Napi::Env env, Napi::Object exports) {
    // Initialize core NDArray
    numpy_node::NativeNDArray::Init(env, exports);

    // Initialize submodules
    numpy_node::linalg::Init(env, exports);
    numpy_node::math::Init(env, exports);
    numpy_node::random::Init(env, exports);
    numpy_node::fft::Init(env, exports);

    // Version info
    exports.Set("version", Napi::String::New(env, "0.1.0"));

    // Platform info
#if defined(USE_ACCELERATE)
    exports.Set("backend", Napi::String::New(env, "accelerate"));
#elif defined(USE_OPENBLAS)
    exports.Set("backend", Napi::String::New(env, "openblas"));
#else
    exports.Set("backend", Napi::String::New(env, "pure"));
#endif

    return exports;
}

NODE_API_MODULE(numpy_node_native, Init)
