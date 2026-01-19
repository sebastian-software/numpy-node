#ifndef NUMPY_TS_FFT_OPS_H
#define NUMPY_TS_FFT_OPS_H

#include <napi.h>
#include "ndarray.h"

namespace numpy_node {
namespace fft {

/**
 * Compute the 1-D discrete Fourier Transform.
 * Returns object with {real: NDArray, imag: NDArray}
 */
Napi::Value Fft(const Napi::CallbackInfo& info);

/**
 * Compute the 1-D inverse discrete Fourier Transform.
 * Takes object with {real: NDArray, imag: NDArray} or single real array.
 * Returns object with {real: NDArray, imag: NDArray}
 */
Napi::Value Ifft(const Napi::CallbackInfo& info);

/**
 * Compute the 1-D FFT for real input.
 * Returns object with {real: NDArray, imag: NDArray} of size n/2+1
 */
Napi::Value Rfft(const Napi::CallbackInfo& info);

/**
 * Compute the inverse of rfft.
 * Takes object with {real: NDArray, imag: NDArray} of size n/2+1
 * Returns real NDArray of size n
 */
Napi::Value Irfft(const Napi::CallbackInfo& info);

/**
 * Initialize FFT module
 */
Napi::Object Init(Napi::Env env, Napi::Object exports);

} // namespace fft
} // namespace numpy_node

#endif // NUMPY_TS_FFT_OPS_H
