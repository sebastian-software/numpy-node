/**
 * NDArray - Thin wrapper around native NativeNDArray
 */

import { native, type NativeNDArray } from './loader.js';

/**
 * Supported data types
 */
export type DTypeName =
  | 'int8'
  | 'int16'
  | 'int32'
  | 'int64'
  | 'uint8'
  | 'uint16'
  | 'uint32'
  | 'uint64'
  | 'float32'
  | 'float64'
  | 'bool';

/**
 * Shape type
 */
export type Shape = readonly number[];

/**
 * TypedArray types
 */
export type TypedArray =
  | Int8Array
  | Int16Array
  | Int32Array
  | BigInt64Array
  | Uint8Array
  | Uint16Array
  | Uint32Array
  | BigUint64Array
  | Float32Array
  | Float64Array;

/**
 * Array input types for creating NDArrays
 */
export type ArrayInput = number[] | number[][] | number[][][] | TypedArray | NDArray;

/**
 * NDArray class - wraps native NativeNDArray
 */
export class NDArray {
  /** @internal */
  readonly _native: NativeNDArray;

  /**
   * Create NDArray from native instance
   * @internal
   */
  constructor(nativeArray: NativeNDArray) {
    this._native = nativeArray;
  }

  // ============================================================
  // Properties
  // ============================================================

  /** The shape of the array */
  get shape(): Shape {
    return this._native.shape;
  }

  /** The strides of the array */
  get strides(): readonly number[] {
    return this._native.strides;
  }

  /** The data type */
  get dtype(): DTypeName {
    return this._native.dtype as DTypeName;
  }

  /** Number of dimensions */
  get ndim(): number {
    return this._native.ndim;
  }

  /** Total number of elements */
  get size(): number {
    return this._native.size;
  }

  /** The underlying data buffer */
  get data(): ArrayBuffer {
    return this._native.data;
  }

  /** Transpose of the array */
  get T(): NDArray {
    return this.transpose();
  }

  // ============================================================
  // Methods
  // ============================================================

  /**
   * Return a copy of the array
   */
  copy(): NDArray {
    return new NDArray(this._native.copy());
  }

  /**
   * Return array with new shape
   */
  reshape(shape: number[]): NDArray {
    return new NDArray(this._native.reshape(shape));
  }

  /**
   * Transpose the array
   */
  transpose(axes?: number[]): NDArray {
    return new NDArray(this._native.transpose(axes));
  }

  /**
   * Return contiguous array
   */
  asContiguous(): NDArray {
    return new NDArray(this._native.asContiguous());
  }

  /**
   * Get element size in bytes for this dtype
   */
  private get elementSize(): number {
    switch (this.dtype) {
      case 'int8':
      case 'uint8':
      case 'bool':
        return 1;
      case 'int16':
      case 'uint16':
        return 2;
      case 'int32':
      case 'uint32':
      case 'float32':
        return 4;
      case 'int64':
      case 'uint64':
      case 'float64':
      default:
        return 8;
    }
  }

  /**
   * Get element at indices
   */
  at(...indices: number[]): number {
    // Access via typed array view
    const view = this.toTypedArray();
    if (indices.length === 1) {
      return Number(view[indices[0]!]);
    }
    // Calculate flat index from indices
    // Note: C++ strides are in bytes, so divide by element size
    const elemSize = this.elementSize;
    let flatIndex = 0;
    for (let i = 0; i < indices.length; i++) {
      flatIndex += indices[i]! * (this.strides[i]! / elemSize);
    }
    return Number(view[flatIndex]);
  }

  /**
   * Set element at indices
   */
  set(indices: number[], value: number): void {
    this._native.set(indices, value);
  }

  /**
   * Return a flattened copy of the array
   */
  flatten(): NDArray {
    return this.reshape([this.size]);
  }

  /**
   * Return a flattened view/copy of the array (alias for flatten)
   */
  ravel(): NDArray {
    return this.flatten();
  }

  /**
   * Fill array with a scalar value
   */
  fill(value: number): this {
    this._native.fill(value);
    return this;
  }

  /**
   * Remove axes with size 1
   */
  squeeze(axis?: number): NDArray {
    const newShape: number[] = [];
    for (let i = 0; i < this.shape.length; i++) {
      if (axis !== undefined) {
        if (i === axis && this.shape[i] === 1) {
          continue;
        }
        newShape.push(this.shape[i]!);
      } else {
        if (this.shape[i] !== 1) {
          newShape.push(this.shape[i]!);
        }
      }
    }
    return this.reshape(newShape.length > 0 ? newShape : [1]);
  }

  /**
   * Convert to nested JavaScript array
   */
  toArray(): unknown {
    // Note: _native.data returns a CONTIGUOUS copy of the data,
    // so we must use C-contiguous strides to read it, not the view's strides
    const data = this.toTypedArray();

    if (this.ndim === 1) {
      const result: number[] = [];
      for (let i = 0; i < data.length; i++) {
        result.push(Number(data[i]));
      }
      return result;
    }

    // Compute C-contiguous element strides for the returned data
    // (GetData always returns contiguous data regardless of view strides)
    const contiguousStrides: number[] = new Array(this.ndim);
    contiguousStrides[this.ndim - 1] = 1;
    for (let i = this.ndim - 2; i >= 0; i--) {
      contiguousStrides[i] = contiguousStrides[i + 1]! * this.shape[i + 1]!;
    }

    const buildArray = (dim: number, offset: number): unknown => {
      if (dim === this.ndim - 1) {
        const result: number[] = [];
        for (let i = 0; i < this.shape[dim]!; i++) {
          result.push(Number(data[offset + i * contiguousStrides[dim]!]));
        }
        return result;
      }

      const result: unknown[] = [];
      for (let i = 0; i < this.shape[dim]!; i++) {
        result.push(buildArray(dim + 1, offset + i * contiguousStrides[dim]!));
      }
      return result;
    };

    return buildArray(0, 0);
  }

  /**
   * Convert to flat array
   */
  toFlatArray(): number[] {
    const data = this.toTypedArray();
    const result: number[] = [];
    for (let i = 0; i < data.length; i++) {
      result.push(Number(data[i]));
    }
    return result;
  }

  /**
   * Get TypedArray view of data
   */
  toTypedArray(): TypedArray {
    const buffer = this._native.data;
    switch (this.dtype) {
      case 'int8':
        return new Int8Array(buffer);
      case 'int16':
        return new Int16Array(buffer);
      case 'int32':
        return new Int32Array(buffer);
      case 'int64':
        return new BigInt64Array(buffer);
      case 'uint8':
        return new Uint8Array(buffer);
      case 'uint16':
        return new Uint16Array(buffer);
      case 'uint32':
        return new Uint32Array(buffer);
      case 'uint64':
        return new BigUint64Array(buffer);
      case 'float32':
        return new Float32Array(buffer);
      case 'float64':
        return new Float64Array(buffer);
      case 'bool':
        return new Uint8Array(buffer);
      default:
        return new Float64Array(buffer);
    }
  }

  /**
   * String representation
   */
  toString(): string {
    return `NDArray(${JSON.stringify(this.toArray())}, dtype='${this.dtype}')`;
  }

  /**
   * Iterator over elements
   */
  *[Symbol.iterator](): Generator<number> {
    const data = this.toTypedArray();
    for (let i = 0; i < data.length; i++) {
      yield Number(data[i]);
    }
  }

  // ============================================================
  // In-place Operations
  // ============================================================

  /**
   * In-place addition: this += b
   * @returns this (for chaining)
   */
  iadd(b: NDArray | number): this {
    const { math } = native;
    math.add_inplace(this._native, b instanceof NDArray ? b._native : b);
    return this;
  }

  /**
   * In-place subtraction: this -= b
   * @returns this (for chaining)
   */
  isub(b: NDArray | number): this {
    const { math } = native;
    math.subtract_inplace(this._native, b instanceof NDArray ? b._native : b);
    return this;
  }

  /**
   * In-place multiplication: this *= b
   * @returns this (for chaining)
   */
  imul(b: NDArray | number): this {
    const { math } = native;
    math.multiply_inplace(this._native, b instanceof NDArray ? b._native : b);
    return this;
  }

  /**
   * In-place division: this /= b
   * @returns this (for chaining)
   */
  idiv(b: NDArray | number): this {
    const { math } = native;
    math.divide_inplace(this._native, b instanceof NDArray ? b._native : b);
    return this;
  }
}

// ============================================================
// Creation Functions
// ============================================================

/**
 * Create array from nested arrays or TypedArray
 */
export function array(data: ArrayInput, dtype?: DTypeName): NDArray {
  if (data instanceof NDArray) {
    return dtype ? astype(data, dtype) : data.copy();
  }

  if (ArrayBuffer.isView(data)) {
    return new NDArray(native.fromTypedArray(data));
  }

  // Flatten nested arrays and infer shape
  const { flat, shape } = flattenNested(data);
  const typedArray = new Float64Array(flat);
  const arr = new NDArray(native.fromTypedArray(typedArray, shape));

  return dtype ? astype(arr, dtype) : arr;
}

/**
 * Helper to flatten nested arrays
 */
function flattenNested(data: unknown): { flat: number[]; shape: number[] } {
  const shape: number[] = [];
  let current: unknown = data;

  while (Array.isArray(current)) {
    shape.push(current.length);
    current = current[0];
  }

  const flat: number[] = [];
  const flatten = (arr: unknown): void => {
    if (Array.isArray(arr)) {
      for (const item of arr) {
        flatten(item);
      }
    } else {
      flat.push(Number(arr));
    }
  };
  flatten(data);

  return { flat, shape };
}

/**
 * Cast array to different dtype
 */
export function astype(arr: NDArray, dtype: DTypeName): NDArray {
  if (arr.dtype === dtype) {
    return arr.copy();
  }
  // Create new array with desired dtype and copy data
  const result = zeros(arr.shape as number[], dtype);
  const srcData = arr.toTypedArray();
  const dstData = result.toTypedArray();
  for (let i = 0; i < srcData.length; i++) {
    (dstData as Float64Array)[i] = Number(srcData[i]);
  }
  return result;
}

/**
 * Create array filled with zeros
 */
export function zeros(shape: number[], dtype: DTypeName = 'float64'): NDArray {
  return new NDArray(native.zeros(shape, dtype));
}

/**
 * Create array filled with ones
 */
export function ones(shape: number[], dtype: DTypeName = 'float64'): NDArray {
  return new NDArray(native.ones(shape, dtype));
}

/**
 * Create array filled with value
 */
export function full(shape: number[], value: number, dtype: DTypeName = 'float64'): NDArray {
  return new NDArray(native.full(shape, value, dtype));
}

/**
 * Create array with evenly spaced values
 */
export function arange(
  startOrStop: number,
  stop?: number,
  step = 1,
  dtype: DTypeName = 'float64'
): NDArray {
  if (stop === undefined) {
    return new NDArray(native.arange(0, startOrStop, step, dtype));
  }
  return new NDArray(native.arange(startOrStop, stop, step, dtype));
}

/**
 * Create array with evenly spaced values over interval
 */
export function linspace(
  start: number,
  stop: number,
  num = 50,
  dtype: DTypeName = 'float64'
): NDArray {
  return new NDArray(native.linspace(start, stop, num, dtype));
}

/**
 * Create identity matrix
 */
export function eye(n: number, m?: number, k = 0, dtype: DTypeName = 'float64'): NDArray {
  return new NDArray(native.eye(n, m ?? n, k, dtype));
}

/**
 * Create identity matrix (alias)
 */
export function identity(n: number, dtype: DTypeName = 'float64'): NDArray {
  return eye(n, n, 0, dtype);
}

/**
 * Create empty array (uninitialized)
 */
export function empty(shape: number[], dtype: DTypeName = 'float64'): NDArray {
  return zeros(shape, dtype);
}

/**
 * Create zeros array with same shape as input
 */
export function zerosLike(arr: NDArray, dtype?: DTypeName): NDArray {
  return zeros(arr.shape as number[], dtype ?? arr.dtype);
}

/**
 * Create ones array with same shape as input
 */
export function onesLike(arr: NDArray, dtype?: DTypeName): NDArray {
  return ones(arr.shape as number[], dtype ?? arr.dtype);
}

/**
 * Create empty array with same shape as input
 */
export function emptyLike(arr: NDArray, dtype?: DTypeName): NDArray {
  return empty(arr.shape as number[], dtype ?? arr.dtype);
}
