#ifndef NUMPY_TS_RANDOM_H
#define NUMPY_TS_RANDOM_H

#include <napi.h>
#include <cstdint>
#include "ndarray.h"

namespace numpy_node {
namespace random {

/**
 * PCG64 Random Number Generator
 * Implements the same algorithm as NumPy 2.x
 */
class PCG64 {
public:
    PCG64(uint64_t seed = 0);

    uint64_t next();
    double uniform();
    double normal();

    void seed(uint64_t seed);

private:
    uint64_t state_;
    uint64_t inc_;

    static constexpr uint64_t multiplier = 6364136223846793005ULL;
};

/**
 * Generator class wrapping PCG64
 */
class Generator : public Napi::ObjectWrap<Generator> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    Generator(const Napi::CallbackInfo& info);

    // Random generation methods
    Napi::Value Random(const Napi::CallbackInfo& info);
    Napi::Value Uniform(const Napi::CallbackInfo& info);
    Napi::Value Normal(const Napi::CallbackInfo& info);
    Napi::Value Integers(const Napi::CallbackInfo& info);
    Napi::Value Choice(const Napi::CallbackInfo& info);
    Napi::Value Shuffle(const Napi::CallbackInfo& info);
    Napi::Value Permutation(const Napi::CallbackInfo& info);

    // Distribution methods
    Napi::Value Exponential(const Napi::CallbackInfo& info);
    Napi::Value Poisson(const Napi::CallbackInfo& info);
    Napi::Value Binomial(const Napi::CallbackInfo& info);

private:
    static Napi::FunctionReference constructor;
    PCG64 rng_;
};

/**
 * Legacy global random functions (for compatibility)
 */
Napi::Value Rand(const Napi::CallbackInfo& info);
Napi::Value Randn(const Napi::CallbackInfo& info);
Napi::Value Randint(const Napi::CallbackInfo& info);

/**
 * Initialize random module
 */
Napi::Object Init(Napi::Env env, Napi::Object exports);

} // namespace random
} // namespace numpy_node

#endif // NUMPY_TS_RANDOM_H
