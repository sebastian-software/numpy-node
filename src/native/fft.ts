/**
 * FFT (Fast Fourier Transform) operations
 *
 * These functions compute discrete Fourier transforms.
 * Results are returned as {real, imag} objects since the library
 * doesn't have native complex number support.
 */

import { NDArray, array as createArray } from './ndarray.js';
import { native, type ComplexResult as NativeComplexResult } from './loader.js';

const fftModule = native.fft;

/**
 * Complex array result from FFT operations
 */
export interface ComplexArray {
  real: NDArray;
  imag: NDArray;
}

/**
 * Compute the 1-D discrete Fourier Transform.
 *
 * @param a - Input array (real)
 * @param n - Length of the transformed axis (default: length of a)
 * @returns Complex result {real, imag}
 *
 * @example
 * ```typescript
 * const x = array([1, 2, 3, 4]);
 * const result = fft(x);
 * console.log(result.real.toArray()); // [10, -2, -2, -2]
 * console.log(result.imag.toArray()); // [0, 2, 0, -2]
 * ```
 */
export function fft(a: NDArray, n?: number): ComplexArray {
  const result = fftModule.fft(a._native, n);
  return {
    real: new NDArray(result.real),
    imag: new NDArray(result.imag),
  };
}

/**
 * Compute the 1-D inverse discrete Fourier Transform.
 *
 * @param a - Input complex array {real, imag} or real array
 * @param n - Length of the transformed axis
 * @returns Complex result {real, imag}
 *
 * @example
 * ```typescript
 * const freq = fft(array([1, 2, 3, 4]));
 * const recovered = ifft(freq);
 * console.log(recovered.real.toArray()); // [1, 2, 3, 4]
 * ```
 */
function isComplexArray(a: ComplexArray | NDArray): a is ComplexArray {
  return 'real' in a && 'imag' in a;
}

export function ifft(a: ComplexArray | NDArray, n?: number): ComplexArray {
  let input: NativeComplexResult | typeof NDArray.prototype._native;

  if (isComplexArray(a)) {
    input = {
      real: a.real._native,
      imag: a.imag._native,
    };
  } else {
    input = a._native;
  }

  const result = fftModule.ifft(input, n);
  return {
    real: new NDArray(result.real),
    imag: new NDArray(result.imag),
  };
}

/**
 * Compute the 1-D FFT for real input.
 *
 * Due to Hermitian symmetry, only n/2+1 unique complex values are returned.
 *
 * @param a - Input array (real)
 * @param n - Number of points (default: length of a)
 * @returns Complex result {real, imag} of size n/2+1
 *
 * @example
 * ```typescript
 * const x = array([1, 2, 3, 4]);
 * const result = rfft(x);
 * // Returns only positive frequencies
 * console.log(result.real.shape); // [3] for n=4
 * ```
 */
export function rfft(a: NDArray, n?: number): ComplexArray {
  const result = fftModule.rfft(a._native, n);
  return {
    real: new NDArray(result.real),
    imag: new NDArray(result.imag),
  };
}

/**
 * Compute the inverse of rfft.
 *
 * @param a - Input complex array {real, imag} of size n/2+1
 * @param n - Length of the output (default: 2*(input_length-1))
 * @returns Real array of size n
 *
 * @example
 * ```typescript
 * const x = array([1, 2, 3, 4]);
 * const freq = rfft(x);
 * const recovered = irfft(freq, 4);
 * console.log(recovered.toArray()); // [1, 2, 3, 4]
 * ```
 */
export function irfft(a: ComplexArray, n?: number): NDArray {
  const input = {
    real: a.real._native,
    imag: a.imag._native,
  };
  return new NDArray(fftModule.irfft(input, n));
}

/**
 * Compute the FFT frequencies for a signal of given length.
 *
 * @param n - Window length
 * @param d - Sample spacing (default: 1.0)
 * @returns Array of sample frequencies
 */
export function fftfreq(n: number, d: number = 1.0): NDArray {
  const freq: number[] = [];
  const val = 1.0 / (n * d);

  // Positive frequencies
  for (let i = 0; i < Math.floor((n + 1) / 2); i++) {
    freq.push(i * val);
  }
  // Negative frequencies
  for (let i = Math.floor((n + 1) / 2); i < n; i++) {
    freq.push((i - n) * val);
  }

  return createArray(freq);
}

/**
 * Compute the FFT frequencies for rfft output.
 *
 * @param n - Window length
 * @param d - Sample spacing (default: 1.0)
 * @returns Array of sample frequencies (only positive)
 */
export function rfftfreq(n: number, d: number = 1.0): NDArray {
  const outputSize = Math.floor(n / 2) + 1;
  const freq: number[] = [];
  const val = 1.0 / (n * d);

  for (let i = 0; i < outputSize; i++) {
    freq.push(i * val);
  }

  return createArray(freq);
}
