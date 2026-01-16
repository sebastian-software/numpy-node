#include <napi.h>
#include "ndarray.h"
#include "linalg.h"
#include "math_ops.h"
#include "random.h"

/**
 * Module initialization
 */
Napi::Object Init(Napi::Env env, Napi::Object exports) {
    // Initialize core NDArray
    np_ts::NativeNDArray::Init(env, exports);

    // Initialize submodules
    np_ts::linalg::Init(env, exports);
    np_ts::math::Init(env, exports);
    np_ts::random::Init(env, exports);

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

NODE_API_MODULE(np_ts_native, Init)
