#include "random.h"
#include <cmath>
#include <cstring>
#include <chrono>

namespace numpy_node {
namespace random {

// PCG64 implementation
PCG64::PCG64(uint64_t seedVal) {
    seed(seedVal);
}

void PCG64::seed(uint64_t seedVal) {
    state_ = 0;
    inc_ = (seedVal << 1) | 1;
    next();
    state_ += seedVal;
    next();
}

uint64_t PCG64::next() {
    uint64_t oldState = state_;
    state_ = oldState * multiplier + inc_;

    uint64_t xorShifted = ((oldState >> 18u) ^ oldState) >> 27u;
    uint64_t rot = oldState >> 59u;
    return (xorShifted >> rot) | (xorShifted << ((-rot) & 31));
}

double PCG64::uniform() {
    return static_cast<double>(next()) / static_cast<double>(UINT64_MAX);
}

double PCG64::normal() {
    // Box-Muller transform
    double u1 = uniform();
    double u2 = uniform();

    // Avoid log(0)
    while (u1 == 0.0) {
        u1 = uniform();
    }

    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}

// Generator class
Napi::FunctionReference Generator::constructor;

Napi::Object Generator::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "Generator", {
        InstanceMethod<&Generator::Random>("random"),
        InstanceMethod<&Generator::Uniform>("uniform"),
        InstanceMethod<&Generator::Normal>("normal"),
        InstanceMethod<&Generator::Integers>("integers"),
        InstanceMethod<&Generator::Choice>("choice"),
        InstanceMethod<&Generator::Shuffle>("shuffle"),
        InstanceMethod<&Generator::Permutation>("permutation"),
        InstanceMethod<&Generator::Exponential>("exponential"),
        InstanceMethod<&Generator::Poisson>("poisson"),
        InstanceMethod<&Generator::Binomial>("binomial"),
    });

    constructor = Napi::Persistent(func);
    constructor.SuppressDestruct();

    exports.Set("Generator", func);
    return exports;
}

Generator::Generator(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<Generator>(info),
      rng_(0) {

    Napi::Env env = info.Env();

    uint64_t seedVal;
    if (info.Length() > 0 && info[0].IsNumber()) {
        seedVal = static_cast<uint64_t>(info[0].As<Napi::Number>().Int64Value());
    } else {
        // Use current time as seed
        auto now = std::chrono::high_resolution_clock::now();
        seedVal = static_cast<uint64_t>(now.time_since_epoch().count());
    }

    rng_.seed(seedVal);
}

Napi::Value Generator::Random(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    // Parse shape
    std::vector<int64_t> shape;
    if (info.Length() > 0) {
        if (info[0].IsArray()) {
            Napi::Array shapeArr = info[0].As<Napi::Array>();
            shape.resize(shapeArr.Length());
            for (size_t i = 0; i < shapeArr.Length(); i++) {
                shape[i] = shapeArr.Get(i).As<Napi::Number>().Int64Value();
            }
        } else if (info[0].IsNumber()) {
            shape.push_back(info[0].As<Napi::Number>().Int64Value());
        }
    }

    if (shape.empty()) {
        // Return scalar
        return Napi::Number::New(env, rng_.uniform());
    }

    // Create array
    Napi::Array shapeNapi = Napi::Array::New(env, shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
        shapeNapi.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(shape[i])));
    }

    Napi::Object result = NativeNDArray::constructor.New({shapeNapi, Napi::String::New(env, "float64")});
    NativeNDArray* arr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);

    double* data = static_cast<double*>(arr->data());
    int64_t size = arr->size();

    for (int64_t i = 0; i < size; i++) {
        data[i] = rng_.uniform();
    }

    return result;
}

Napi::Value Generator::Uniform(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    double low = 0.0, high = 1.0;
    std::vector<int64_t> shape;

    size_t argIdx = 0;
    if (info.Length() > argIdx && info[argIdx].IsNumber()) {
        low = info[argIdx++].As<Napi::Number>().DoubleValue();
    }
    if (info.Length() > argIdx && info[argIdx].IsNumber()) {
        high = info[argIdx++].As<Napi::Number>().DoubleValue();
    }
    if (info.Length() > argIdx) {
        if (info[argIdx].IsArray()) {
            Napi::Array shapeArr = info[argIdx].As<Napi::Array>();
            shape.resize(shapeArr.Length());
            for (size_t i = 0; i < shapeArr.Length(); i++) {
                shape[i] = shapeArr.Get(i).As<Napi::Number>().Int64Value();
            }
        } else if (info[argIdx].IsNumber()) {
            shape.push_back(info[argIdx].As<Napi::Number>().Int64Value());
        }
    }

    if (shape.empty()) {
        return Napi::Number::New(env, low + (high - low) * rng_.uniform());
    }

    Napi::Array shapeNapi = Napi::Array::New(env, shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
        shapeNapi.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(shape[i])));
    }

    Napi::Object result = NativeNDArray::constructor.New({shapeNapi, Napi::String::New(env, "float64")});
    NativeNDArray* arr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);

    double* data = static_cast<double*>(arr->data());
    int64_t size = arr->size();
    double range = high - low;

    for (int64_t i = 0; i < size; i++) {
        data[i] = low + range * rng_.uniform();
    }

    return result;
}

Napi::Value Generator::Normal(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    double loc = 0.0, scale = 1.0;
    std::vector<int64_t> shape;

    size_t argIdx = 0;
    if (info.Length() > argIdx && info[argIdx].IsNumber()) {
        loc = info[argIdx++].As<Napi::Number>().DoubleValue();
    }
    if (info.Length() > argIdx && info[argIdx].IsNumber()) {
        scale = info[argIdx++].As<Napi::Number>().DoubleValue();
    }
    if (info.Length() > argIdx) {
        if (info[argIdx].IsArray()) {
            Napi::Array shapeArr = info[argIdx].As<Napi::Array>();
            shape.resize(shapeArr.Length());
            for (size_t i = 0; i < shapeArr.Length(); i++) {
                shape[i] = shapeArr.Get(i).As<Napi::Number>().Int64Value();
            }
        } else if (info[argIdx].IsNumber()) {
            shape.push_back(info[argIdx].As<Napi::Number>().Int64Value());
        }
    }

    if (shape.empty()) {
        return Napi::Number::New(env, loc + scale * rng_.normal());
    }

    Napi::Array shapeNapi = Napi::Array::New(env, shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
        shapeNapi.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(shape[i])));
    }

    Napi::Object result = NativeNDArray::constructor.New({shapeNapi, Napi::String::New(env, "float64")});
    NativeNDArray* arr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);

    double* data = static_cast<double*>(arr->data());
    int64_t size = arr->size();

    for (int64_t i = 0; i < size; i++) {
        data[i] = loc + scale * rng_.normal();
    }

    return result;
}

Napi::Value Generator::Integers(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected at least low value").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int64_t low = 0, high;
    std::vector<int64_t> shape;

    if (info.Length() == 1 || (info.Length() > 1 && !info[1].IsNumber())) {
        // Only high provided
        high = info[0].As<Napi::Number>().Int64Value();
        if (info.Length() > 1) {
            if (info[1].IsArray()) {
                Napi::Array shapeArr = info[1].As<Napi::Array>();
                shape.resize(shapeArr.Length());
                for (size_t i = 0; i < shapeArr.Length(); i++) {
                    shape[i] = shapeArr.Get(i).As<Napi::Number>().Int64Value();
                }
            } else if (info[1].IsNumber()) {
                shape.push_back(info[1].As<Napi::Number>().Int64Value());
            }
        }
    } else {
        low = info[0].As<Napi::Number>().Int64Value();
        high = info[1].As<Napi::Number>().Int64Value();
        if (info.Length() > 2) {
            if (info[2].IsArray()) {
                Napi::Array shapeArr = info[2].As<Napi::Array>();
                shape.resize(shapeArr.Length());
                for (size_t i = 0; i < shapeArr.Length(); i++) {
                    shape[i] = shapeArr.Get(i).As<Napi::Number>().Int64Value();
                }
            } else if (info[2].IsNumber()) {
                shape.push_back(info[2].As<Napi::Number>().Int64Value());
            }
        }
    }

    int64_t range = high - low;
    if (range <= 0) {
        Napi::Error::New(env, "high must be greater than low").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    if (shape.empty()) {
        int64_t val = low + static_cast<int64_t>(rng_.next() % static_cast<uint64_t>(range));
        return Napi::Number::New(env, static_cast<double>(val));
    }

    Napi::Array shapeNapi = Napi::Array::New(env, shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
        shapeNapi.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(shape[i])));
    }

    Napi::Object result = NativeNDArray::constructor.New({shapeNapi, Napi::String::New(env, "int64")});
    NativeNDArray* arr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);

    int64_t* data = static_cast<int64_t*>(arr->data());
    int64_t size = arr->size();

    for (int64_t i = 0; i < size; i++) {
        data[i] = low + static_cast<int64_t>(rng_.next() % static_cast<uint64_t>(range));
    }

    return result;
}

Napi::Value Generator::Choice(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Error::New(env, "choice not yet implemented").ThrowAsJavaScriptException();
    return env.Undefined();
}

Napi::Value Generator::Shuffle(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Error::New(env, "shuffle not yet implemented").ThrowAsJavaScriptException();
    return env.Undefined();
}

Napi::Value Generator::Permutation(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Error::New(env, "permutation not yet implemented").ThrowAsJavaScriptException();
    return env.Undefined();
}

Napi::Value Generator::Exponential(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    double scale = 1.0;
    std::vector<int64_t> shape;

    if (info.Length() > 0 && info[0].IsNumber()) {
        scale = info[0].As<Napi::Number>().DoubleValue();
    }
    if (info.Length() > 1) {
        if (info[1].IsArray()) {
            Napi::Array shapeArr = info[1].As<Napi::Array>();
            shape.resize(shapeArr.Length());
            for (size_t i = 0; i < shapeArr.Length(); i++) {
                shape[i] = shapeArr.Get(i).As<Napi::Number>().Int64Value();
            }
        }
    }

    if (shape.empty()) {
        double u = rng_.uniform();
        while (u == 0.0) u = rng_.uniform();
        return Napi::Number::New(env, -scale * std::log(u));
    }

    Napi::Array shapeNapi = Napi::Array::New(env, shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
        shapeNapi.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(shape[i])));
    }

    Napi::Object result = NativeNDArray::constructor.New({shapeNapi, Napi::String::New(env, "float64")});
    NativeNDArray* arr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);

    double* data = static_cast<double*>(arr->data());
    int64_t size = arr->size();

    for (int64_t i = 0; i < size; i++) {
        double u = rng_.uniform();
        while (u == 0.0) u = rng_.uniform();
        data[i] = -scale * std::log(u);
    }

    return result;
}

Napi::Value Generator::Poisson(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Error::New(env, "poisson not yet implemented").ThrowAsJavaScriptException();
    return env.Undefined();
}

Napi::Value Generator::Binomial(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Error::New(env, "binomial not yet implemented").ThrowAsJavaScriptException();
    return env.Undefined();
}

// Global RNG for legacy functions
static PCG64 globalRng(static_cast<uint64_t>(
    std::chrono::high_resolution_clock::now().time_since_epoch().count()));

Napi::Value Rand(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    std::vector<int64_t> shape;
    for (size_t i = 0; i < info.Length(); i++) {
        if (info[i].IsNumber()) {
            shape.push_back(info[i].As<Napi::Number>().Int64Value());
        }
    }

    if (shape.empty()) {
        return Napi::Number::New(env, globalRng.uniform());
    }

    Napi::Array shapeNapi = Napi::Array::New(env, shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
        shapeNapi.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(shape[i])));
    }

    Napi::Object result = NativeNDArray::constructor.New({shapeNapi, Napi::String::New(env, "float64")});
    NativeNDArray* arr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);

    double* data = static_cast<double*>(arr->data());
    int64_t size = arr->size();

    for (int64_t i = 0; i < size; i++) {
        data[i] = globalRng.uniform();
    }

    return result;
}

Napi::Value Randn(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    std::vector<int64_t> shape;
    for (size_t i = 0; i < info.Length(); i++) {
        if (info[i].IsNumber()) {
            shape.push_back(info[i].As<Napi::Number>().Int64Value());
        }
    }

    if (shape.empty()) {
        return Napi::Number::New(env, globalRng.normal());
    }

    Napi::Array shapeNapi = Napi::Array::New(env, shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
        shapeNapi.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(shape[i])));
    }

    Napi::Object result = NativeNDArray::constructor.New({shapeNapi, Napi::String::New(env, "float64")});
    NativeNDArray* arr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);

    double* data = static_cast<double*>(arr->data());
    int64_t size = arr->size();

    for (int64_t i = 0; i < size; i++) {
        data[i] = globalRng.normal();
    }

    return result;
}

Napi::Value Randint(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected at least one argument").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int64_t low = 0, high;
    std::vector<int64_t> shape;

    if (info.Length() >= 2 && info[1].IsNumber()) {
        low = info[0].As<Napi::Number>().Int64Value();
        high = info[1].As<Napi::Number>().Int64Value();
        if (info.Length() > 2) {
            if (info[2].IsArray()) {
                Napi::Array shapeArr = info[2].As<Napi::Array>();
                shape.resize(shapeArr.Length());
                for (size_t i = 0; i < shapeArr.Length(); i++) {
                    shape[i] = shapeArr.Get(i).As<Napi::Number>().Int64Value();
                }
            }
        }
    } else {
        high = info[0].As<Napi::Number>().Int64Value();
        if (info.Length() > 1 && info[1].IsArray()) {
            Napi::Array shapeArr = info[1].As<Napi::Array>();
            shape.resize(shapeArr.Length());
            for (size_t i = 0; i < shapeArr.Length(); i++) {
                shape[i] = shapeArr.Get(i).As<Napi::Number>().Int64Value();
            }
        }
    }

    int64_t range = high - low;
    if (range <= 0) {
        Napi::Error::New(env, "high must be greater than low").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    if (shape.empty()) {
        int64_t val = low + static_cast<int64_t>(globalRng.next() % static_cast<uint64_t>(range));
        return Napi::Number::New(env, static_cast<double>(val));
    }

    Napi::Array shapeNapi = Napi::Array::New(env, shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
        shapeNapi.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(shape[i])));
    }

    Napi::Object result = NativeNDArray::constructor.New({shapeNapi, Napi::String::New(env, "int32")});
    NativeNDArray* arr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);

    int32_t* data = static_cast<int32_t*>(arr->data());
    int64_t size = arr->size();

    for (int64_t i = 0; i < size; i++) {
        data[i] = static_cast<int32_t>(low + static_cast<int64_t>(globalRng.next() % static_cast<uint64_t>(range)));
    }

    return result;
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    Napi::Object random = Napi::Object::New(env);

    Generator::Init(env, random);

    random.Set("rand", Napi::Function::New(env, Rand));
    random.Set("randn", Napi::Function::New(env, Randn));
    random.Set("randint", Napi::Function::New(env, Randint));

    exports.Set("random", random);
    return exports;
}

} // namespace random
} // namespace numpy_node
