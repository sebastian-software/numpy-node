#include "fft_ops.h"
#include <cmath>
#include <vector>
#include <algorithm>

#if defined(USE_ACCELERATE)
    #include <Accelerate/Accelerate.h>
#endif

namespace numpy_node {
namespace fft {

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Helper to check if n is a power of 2
static bool isPowerOfTwo(int64_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// Helper to find next power of 2
static int64_t nextPowerOfTwo(int64_t n) {
    int64_t power = 1;
    while (power < n) power *= 2;
    return power;
}

// Cooley-Tukey FFT (radix-2, in-place)
// real and imag arrays are modified in place
static void cooleyTukeyFft(double* real, double* imag, int64_t n, bool inverse) {
    // Bit-reversal permutation
    int64_t j = 0;
    for (int64_t i = 0; i < n - 1; i++) {
        if (i < j) {
            std::swap(real[i], real[j]);
            std::swap(imag[i], imag[j]);
        }
        int64_t k = n / 2;
        while (k <= j) {
            j -= k;
            k /= 2;
        }
        j += k;
    }

    // Cooley-Tukey iterative FFT
    for (int64_t len = 2; len <= n; len *= 2) {
        double angle = (inverse ? 2.0 : -2.0) * M_PI / len;
        double wReal = std::cos(angle);
        double wImag = std::sin(angle);

        for (int64_t i = 0; i < n; i += len) {
            double curReal = 1.0;
            double curImag = 0.0;

            for (int64_t k = 0; k < len / 2; k++) {
                int64_t u = i + k;
                int64_t v = i + k + len / 2;

                double tReal = curReal * real[v] - curImag * imag[v];
                double tImag = curReal * imag[v] + curImag * real[v];

                real[v] = real[u] - tReal;
                imag[v] = imag[u] - tImag;
                real[u] = real[u] + tReal;
                imag[u] = imag[u] + tImag;

                double newReal = curReal * wReal - curImag * wImag;
                double newImag = curReal * wImag + curImag * wReal;
                curReal = newReal;
                curImag = newImag;
            }
        }
    }

    // Normalize for inverse FFT
    if (inverse) {
        for (int64_t i = 0; i < n; i++) {
            real[i] /= n;
            imag[i] /= n;
        }
    }
}

// Helper to create result object with real and imag arrays
static Napi::Object createComplexResult(Napi::Env env,
                                         const std::vector<int64_t>& shape,
                                         double* realData,
                                         double* imagData,
                                         int64_t size) {
    // Create real array
    Napi::Array jsShape = Napi::Array::New(env, shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
        jsShape.Set(uint32_t(i), Napi::Number::New(env, static_cast<double>(shape[i])));
    }

    Napi::Object realArr = NativeNDArray::constructor.New({
        jsShape,
        Napi::String::New(env, "float64"),
        Napi::Boolean::New(env, true)
    });
    NativeNDArray* realNative = Napi::ObjectWrap<NativeNDArray>::Unwrap(realArr);
    std::memcpy(realNative->data(), realData, size * sizeof(double));

    // Create imag array
    Napi::Object imagArr = NativeNDArray::constructor.New({
        jsShape,
        Napi::String::New(env, "float64"),
        Napi::Boolean::New(env, true)
    });
    NativeNDArray* imagNative = Napi::ObjectWrap<NativeNDArray>::Unwrap(imagArr);
    std::memcpy(imagNative->data(), imagData, size * sizeof(double));

    // Return {real, imag}
    Napi::Object result = Napi::Object::New(env);
    result.Set("real", realArr);
    result.Set("imag", imagArr);
    return result;
}

/**
 * fft(a, n?) - Compute 1-D DFT
 */
Napi::Value Fft(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsObject()) {
        Napi::TypeError::New(env, "Expected NDArray as first argument").ThrowAsJavaScriptException();
        return env.Null();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());

    // Get input size
    int64_t inputSize = a->size();

    // Optional n parameter (output size)
    int64_t n = inputSize;
    if (info.Length() >= 2 && info[1].IsNumber()) {
        n = info[1].As<Napi::Number>().Int64Value();
    }

    // For Cooley-Tukey, we need power of 2
    int64_t fftSize = nextPowerOfTwo(n);

    // Allocate working arrays
    std::vector<double> real(fftSize, 0.0);
    std::vector<double> imag(fftSize, 0.0);

    // Copy input data (zero-pad if needed, truncate if n < inputSize)
    const double* inputData = static_cast<const double*>(a->data());
    int64_t copySize = std::min(inputSize, n);
    for (int64_t i = 0; i < copySize; i++) {
        real[i] = inputData[i];
    }

#if defined(USE_ACCELERATE)
    // Use Accelerate vDSP FFT
    int log2n = static_cast<int>(std::log2(fftSize));
    FFTSetupD fftSetup = vDSP_create_fftsetupD(log2n, FFT_RADIX2);

    if (fftSetup) {
        DSPDoubleSplitComplex splitComplex;
        splitComplex.realp = real.data();
        splitComplex.imagp = imag.data();

        vDSP_fft_zipD(fftSetup, &splitComplex, 1, log2n, FFT_FORWARD);
        vDSP_destroy_fftsetupD(fftSetup);
    } else {
        // Fallback to Cooley-Tukey
        cooleyTukeyFft(real.data(), imag.data(), fftSize, false);
    }
#else
    // Use Cooley-Tukey FFT
    cooleyTukeyFft(real.data(), imag.data(), fftSize, false);
#endif

    // Return only first n elements if n was specified
    std::vector<int64_t> resultShape = {n};

    // If fftSize > n, truncate
    if (fftSize > n) {
        real.resize(n);
        imag.resize(n);
    }

    return createComplexResult(env, resultShape, real.data(), imag.data(), n);
}

/**
 * ifft(a, n?) - Compute inverse 1-D DFT
 * a can be {real, imag} object or just real array
 */
Napi::Value Ifft(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected at least one argument").ThrowAsJavaScriptException();
        return env.Null();
    }

    std::vector<double> real;
    std::vector<double> imag;
    int64_t inputSize;

    // Check if input is {real, imag} object or just an array
    if (info[0].IsObject() && !info[0].As<Napi::Object>().InstanceOf(NativeNDArray::constructor.Value())) {
        // It's a {real, imag} object
        Napi::Object complexInput = info[0].As<Napi::Object>();

        if (!complexInput.Has("real") || !complexInput.Has("imag")) {
            Napi::TypeError::New(env, "Expected {real, imag} object or NDArray").ThrowAsJavaScriptException();
            return env.Null();
        }

        NativeNDArray* realArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(complexInput.Get("real").As<Napi::Object>());
        NativeNDArray* imagArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(complexInput.Get("imag").As<Napi::Object>());

        inputSize = realArr->size();
        real.resize(inputSize);
        imag.resize(inputSize);

        std::memcpy(real.data(), realArr->data(), inputSize * sizeof(double));
        std::memcpy(imag.data(), imagArr->data(), inputSize * sizeof(double));
    } else {
        // It's just a real array
        NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
        inputSize = a->size();
        real.resize(inputSize);
        imag.resize(inputSize, 0.0);
        std::memcpy(real.data(), a->data(), inputSize * sizeof(double));
    }

    // Optional n parameter
    int64_t n = inputSize;
    if (info.Length() >= 2 && info[1].IsNumber()) {
        n = info[1].As<Napi::Number>().Int64Value();
    }

    int64_t fftSize = nextPowerOfTwo(n);

    // Resize/pad arrays
    real.resize(fftSize, 0.0);
    imag.resize(fftSize, 0.0);

#if defined(USE_ACCELERATE)
    int log2n = static_cast<int>(std::log2(fftSize));
    FFTSetupD fftSetup = vDSP_create_fftsetupD(log2n, FFT_RADIX2);

    if (fftSetup) {
        DSPDoubleSplitComplex splitComplex;
        splitComplex.realp = real.data();
        splitComplex.imagp = imag.data();

        vDSP_fft_zipD(fftSetup, &splitComplex, 1, log2n, FFT_INVERSE);

        // vDSP doesn't normalize, so we need to divide by fftSize
        double scale = 1.0 / fftSize;
        vDSP_vsmulD(real.data(), 1, &scale, real.data(), 1, fftSize);
        vDSP_vsmulD(imag.data(), 1, &scale, imag.data(), 1, fftSize);

        vDSP_destroy_fftsetupD(fftSetup);
    } else {
        cooleyTukeyFft(real.data(), imag.data(), fftSize, true);
    }
#else
    cooleyTukeyFft(real.data(), imag.data(), fftSize, true);
#endif

    std::vector<int64_t> resultShape = {n};
    real.resize(n);
    imag.resize(n);

    return createComplexResult(env, resultShape, real.data(), imag.data(), n);
}

/**
 * rfft(a, n?) - Real FFT (returns n/2+1 complex values)
 */
Napi::Value Rfft(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsObject()) {
        Napi::TypeError::New(env, "Expected NDArray as first argument").ThrowAsJavaScriptException();
        return env.Null();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    int64_t inputSize = a->size();

    int64_t n = inputSize;
    if (info.Length() >= 2 && info[1].IsNumber()) {
        n = info[1].As<Napi::Number>().Int64Value();
    }

    int64_t fftSize = nextPowerOfTwo(n);
    int64_t outputSize = n / 2 + 1;

    std::vector<double> real(fftSize, 0.0);
    std::vector<double> imag(fftSize, 0.0);

    const double* inputData = static_cast<const double*>(a->data());
    int64_t copySize = std::min(inputSize, n);
    for (int64_t i = 0; i < copySize; i++) {
        real[i] = inputData[i];
    }

#if defined(USE_ACCELERATE)
    int log2n = static_cast<int>(std::log2(fftSize));
    FFTSetupD fftSetup = vDSP_create_fftsetupD(log2n, FFT_RADIX2);

    if (fftSetup) {
        DSPDoubleSplitComplex splitComplex;
        splitComplex.realp = real.data();
        splitComplex.imagp = imag.data();

        vDSP_fft_zipD(fftSetup, &splitComplex, 1, log2n, FFT_FORWARD);
        vDSP_destroy_fftsetupD(fftSetup);
    } else {
        cooleyTukeyFft(real.data(), imag.data(), fftSize, false);
    }
#else
    cooleyTukeyFft(real.data(), imag.data(), fftSize, false);
#endif

    // Return only first n/2+1 elements (due to Hermitian symmetry)
    std::vector<int64_t> resultShape = {outputSize};
    real.resize(outputSize);
    imag.resize(outputSize);

    return createComplexResult(env, resultShape, real.data(), imag.data(), outputSize);
}

/**
 * irfft(a, n?) - Inverse real FFT
 */
Napi::Value Irfft(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected at least one argument").ThrowAsJavaScriptException();
        return env.Null();
    }

    std::vector<double> real;
    std::vector<double> imag;
    int64_t inputSize;

    // Check if input is {real, imag} object
    if (info[0].IsObject() && !info[0].As<Napi::Object>().InstanceOf(NativeNDArray::constructor.Value())) {
        Napi::Object complexInput = info[0].As<Napi::Object>();

        if (!complexInput.Has("real") || !complexInput.Has("imag")) {
            Napi::TypeError::New(env, "Expected {real, imag} object").ThrowAsJavaScriptException();
            return env.Null();
        }

        NativeNDArray* realArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(complexInput.Get("real").As<Napi::Object>());
        NativeNDArray* imagArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(complexInput.Get("imag").As<Napi::Object>());

        inputSize = realArr->size();
        real.resize(inputSize);
        imag.resize(inputSize);

        std::memcpy(real.data(), realArr->data(), inputSize * sizeof(double));
        std::memcpy(imag.data(), imagArr->data(), inputSize * sizeof(double));
    } else {
        Napi::TypeError::New(env, "Expected {real, imag} object").ThrowAsJavaScriptException();
        return env.Null();
    }

    // Output size: default is 2*(inputSize-1)
    int64_t n = 2 * (inputSize - 1);
    if (info.Length() >= 2 && info[1].IsNumber()) {
        n = info[1].As<Napi::Number>().Int64Value();
    }

    int64_t fftSize = nextPowerOfTwo(n);

    // Reconstruct full spectrum from half spectrum using Hermitian symmetry
    std::vector<double> fullReal(fftSize, 0.0);
    std::vector<double> fullImag(fftSize, 0.0);

    // Copy positive frequencies
    for (int64_t i = 0; i < inputSize && i < fftSize; i++) {
        fullReal[i] = real[i];
        fullImag[i] = imag[i];
    }

    // Mirror negative frequencies (conjugate symmetry)
    for (int64_t i = 1; i < inputSize - 1 && (fftSize - i) < fftSize; i++) {
        fullReal[fftSize - i] = real[i];
        fullImag[fftSize - i] = -imag[i];
    }

#if defined(USE_ACCELERATE)
    int log2n = static_cast<int>(std::log2(fftSize));
    FFTSetupD fftSetup = vDSP_create_fftsetupD(log2n, FFT_RADIX2);

    if (fftSetup) {
        DSPDoubleSplitComplex splitComplex;
        splitComplex.realp = fullReal.data();
        splitComplex.imagp = fullImag.data();

        vDSP_fft_zipD(fftSetup, &splitComplex, 1, log2n, FFT_INVERSE);

        double scale = 1.0 / fftSize;
        vDSP_vsmulD(fullReal.data(), 1, &scale, fullReal.data(), 1, fftSize);

        vDSP_destroy_fftsetupD(fftSetup);
    } else {
        cooleyTukeyFft(fullReal.data(), fullImag.data(), fftSize, true);
    }
#else
    cooleyTukeyFft(fullReal.data(), fullImag.data(), fftSize, true);
#endif

    // Return only real part with size n
    std::vector<int64_t> resultShape = {n};

    Napi::Array jsShape = Napi::Array::New(env, 1);
    jsShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(n)));

    Napi::Object result = NativeNDArray::constructor.New({
        jsShape,
        Napi::String::New(env, "float64"),
        Napi::Boolean::New(env, true)
    });
    NativeNDArray* resultNative = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);

    double* resultData = static_cast<double*>(resultNative->data());
    for (int64_t i = 0; i < n; i++) {
        resultData[i] = fullReal[i];
    }

    return result;
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    Napi::Object fftModule = Napi::Object::New(env);

    fftModule.Set("fft", Napi::Function::New(env, Fft));
    fftModule.Set("ifft", Napi::Function::New(env, Ifft));
    fftModule.Set("rfft", Napi::Function::New(env, Rfft));
    fftModule.Set("irfft", Napi::Function::New(env, Irfft));

    exports.Set("fft", fftModule);
    return exports;
}

} // namespace fft
} // namespace numpy_node
