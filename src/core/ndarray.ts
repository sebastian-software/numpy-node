/**
 * NDArray - Core n-dimensional array class for numpy-node
 * Provides NumPy-compatible array operations with TypedArray backing
 */

import {
  type DTypeName,
  type TypedArray,
  type DTypeToTypedArray,
  type DTypeToElement,
  getDTypeDescriptor,
  inferDTypeFromArray,
} from './dtype.js';
import {
  type Shape,
  shapeToSize,
  shapeToNdim,
  shapesEqual,
  normalizeShape,
  inferShape,
  flattenData,
  validateShape,
  transposeShape,
  normalizeAxis,
} from './shape.js';
import {
  type Strides,
  type Order,
  calculateStrides,
  indicesToOffset,
  isCContiguous,
  isFContiguous,
  transposeStrides,
} from './strides.js';
import {
  type IndexSpec,
  applyIndex,
  slice,
  normalizeIndex,
  IndexIterator,
  indicesToFlat,
} from './indexing.js';
import {
  broadcastShapes,
  getBroadcastStrides,
  needsBroadcast,
} from './broadcasting.js';

/**
 * Options for creating an NDArray
 */
export interface NDArrayOptions {
  dtype?: DTypeName;
  order?: Order;
  copy?: boolean;
}

/**
 * Input data types that can be used to create an NDArray
 */
export type ArrayInput =
  | readonly number[]
  | readonly bigint[]
  | readonly boolean[]
  | readonly (readonly number[])[]
  | readonly (readonly (readonly number[])[])[]
  | TypedArray
  | NDArray;

/**
 * NDArray - The core n-dimensional array class
 */
export class NDArray<D extends DTypeName = DTypeName> {
  private readonly _data: DTypeToTypedArray[D];
  private readonly _shape: Shape;
  private readonly _strides: Strides;
  private readonly _dtype: DTypeName;
  private readonly _offset: number;
  private readonly _isView: boolean;

  /**
   * Private constructor - use static factory methods
   */
  private constructor(
    data: TypedArray,
    shape: Shape,
    strides: Strides,
    dtype: DTypeName,
    offset: number = 0,
    isView: boolean = false
  ) {
    this._data = data as DTypeToTypedArray[D];
    this._shape = shape;
    this._strides = strides;
    this._dtype = dtype;
    this._offset = offset;
    this._isView = isView;
  }

  // ============================================================
  // Static Factory Methods
  // ============================================================

  /**
   * Create an NDArray from nested JavaScript arrays or TypedArrays
   */
  public static from<D extends DTypeName = 'float64'>(
    data: ArrayInput,
    options: NDArrayOptions & { dtype?: D } = {}
  ): NDArray<D> {
    if (data instanceof NDArray) {
      if (options.copy === true || (options.dtype !== undefined && options.dtype !== data._dtype)) {
        return data.astype(options.dtype ?? (data._dtype as D));
      }
      return data as unknown as NDArray<D>;
    }

    // Infer shape from nested arrays
    const shape = inferShape(data);
    const flatData = flattenData<number | bigint | boolean>(data);

    // Determine dtype
    const dtype = options.dtype ?? (inferDTypeFromArray(flatData) as D);
    const descriptor = getDTypeDescriptor(dtype);

    // Create TypedArray
    const size = shapeToSize(shape);
    const typedArray = new descriptor.arrayConstructor(size);

    // Copy data
    if (descriptor.isBigInt) {
      for (let i = 0; i < flatData.length; i++) {
        const value = flatData[i];
        (typedArray as BigInt64Array | BigUint64Array)[i] =
          typeof value === 'bigint' ? value : BigInt(value as number);
      }
    } else {
      for (let i = 0; i < flatData.length; i++) {
        const value = flatData[i];
        (typedArray as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] =
          typeof value === 'boolean' ? (value ? 1 : 0) : Number(value);
      }
    }

    const order = options.order ?? 'C';
    const strides = calculateStrides(shape, order, 1);

    return new NDArray<D>(typedArray, shape, strides, dtype, 0, false);
  }

  /**
   * Create an NDArray filled with zeros
   */
  public static zeros<D extends DTypeName = 'float64'>(
    shape: Shape,
    options: { dtype?: D; order?: Order } = {}
  ): NDArray<D> {
    validateShape(shape);
    const dtype = options.dtype ?? ('float64' as D);
    const descriptor = getDTypeDescriptor(dtype);
    const size = shapeToSize(shape);
    const data = new descriptor.arrayConstructor(size);
    const order = options.order ?? 'C';
    const strides = calculateStrides(shape, order, 1);

    return new NDArray<D>(data, shape, strides, dtype, 0, false);
  }

  /**
   * Create an NDArray filled with ones
   */
  public static ones<D extends DTypeName = 'float64'>(
    shape: Shape,
    options: { dtype?: D; order?: Order } = {}
  ): NDArray<D> {
    const arr = NDArray.zeros(shape, options);
    arr.fill(1);
    return arr;
  }

  /**
   * Create an NDArray filled with a specific value
   */
  public static full<D extends DTypeName = 'float64'>(
    shape: Shape,
    fillValue: number | bigint,
    options: { dtype?: D; order?: Order } = {}
  ): NDArray<D> {
    const arr = NDArray.zeros(shape, options);
    arr.fill(fillValue);
    return arr;
  }

  /**
   * Create an NDArray with uninitialized values
   */
  public static empty<D extends DTypeName = 'float64'>(
    shape: Shape,
    options: { dtype?: D; order?: Order } = {}
  ): NDArray<D> {
    return NDArray.zeros(shape, options);
  }

  /**
   * Create an array with evenly spaced values within a given interval
   */
  public static arange<D extends DTypeName = 'float64'>(
    startOrStop: number,
    stop?: number,
    step: number = 1,
    options: { dtype?: D } = {}
  ): NDArray<D> {
    let start: number;
    let end: number;

    if (stop === undefined) {
      start = 0;
      end = startOrStop;
    } else {
      start = startOrStop;
      end = stop;
    }

    if (step === 0) {
      throw new Error('Step cannot be zero');
    }

    const length = Math.max(0, Math.ceil((end - start) / step));
    const dtype = options.dtype ?? ('float64' as D);
    const arr = NDArray.zeros([length], { dtype });

    for (let i = 0; i < length; i++) {
      arr.setFlat(i, start + i * step);
    }

    return arr;
  }

  /**
   * Create an array with evenly spaced values over a specified interval
   */
  public static linspace<D extends DTypeName = 'float64'>(
    start: number,
    stop: number,
    num: number = 50,
    options: { dtype?: D; endpoint?: boolean } = {}
  ): NDArray<D> {
    const endpoint = options.endpoint ?? true;
    const dtype = options.dtype ?? ('float64' as D);

    if (num < 0) {
      throw new Error('Number of samples must be non-negative');
    }

    if (num === 0) {
      return NDArray.zeros([0], { dtype });
    }

    if (num === 1) {
      return NDArray.from([start], { dtype });
    }

    const arr = NDArray.zeros([num], { dtype });
    const divisor = endpoint ? num - 1 : num;
    const step = (stop - start) / divisor;

    for (let i = 0; i < num; i++) {
      arr.setFlat(i, start + i * step);
    }

    return arr;
  }

  /**
   * Create an identity matrix
   */
  public static eye<D extends DTypeName = 'float64'>(
    n: number,
    m?: number,
    k: number = 0,
    options: { dtype?: D } = {}
  ): NDArray<D> {
    const rows = n;
    const cols = m ?? n;
    const dtype = options.dtype ?? ('float64' as D);
    const arr = NDArray.zeros([rows, cols], { dtype });

    const maxDiag = Math.min(rows, cols - Math.max(0, k), rows + Math.min(0, k));

    for (let i = 0; i < maxDiag; i++) {
      const row = k >= 0 ? i : i - k;
      const col = k >= 0 ? i + k : i;
      if (row >= 0 && row < rows && col >= 0 && col < cols) {
        arr.set([row, col], 1);
      }
    }

    return arr;
  }

  /**
   * Create an array from an existing buffer
   */
  public static fromBuffer<D extends DTypeName>(
    buffer: TypedArray,
    shape: Shape,
    dtype: D,
    options: { strides?: Strides; offset?: number; copy?: boolean } = {}
  ): NDArray<D> {
    validateShape(shape);
    const strides = options.strides ?? calculateStrides(shape, 'C', 1);
    const offset = options.offset ?? 0;
    const copy = options.copy ?? false;

    const data = copy ? buffer.slice() : buffer;
    return new NDArray<D>(data, shape, strides, dtype, offset, !copy);
  }

  // ============================================================
  // Properties
  // ============================================================

  /** The shape of the array */
  public get shape(): Shape {
    return this._shape;
  }

  /** The strides of the array */
  public get strides(): Strides {
    return this._strides;
  }

  /** The data type of the array */
  public get dtype(): DTypeName {
    return this._dtype;
  }

  /** The number of dimensions */
  public get ndim(): number {
    return shapeToNdim(this._shape);
  }

  /** The total number of elements */
  public get size(): number {
    return shapeToSize(this._shape);
  }

  /** The underlying TypedArray buffer */
  public get data(): DTypeToTypedArray[D] {
    return this._data;
  }

  /** The byte offset into the buffer */
  public get offset(): number {
    return this._offset;
  }

  /** Whether this array is a view of another array */
  public get isView(): boolean {
    return this._isView;
  }

  /** The total number of bytes */
  public get nbytes(): number {
    return this.size * getDTypeDescriptor(this._dtype).bytes;
  }

  /** The number of bytes per element */
  public get itemsize(): number {
    return getDTypeDescriptor(this._dtype).bytes;
  }

  /** Check if the array is C-contiguous */
  public get isCContiguous(): boolean {
    return isCContiguous(this._shape, this._strides, 1);
  }

  /** Check if the array is Fortran-contiguous */
  public get isFContiguous(): boolean {
    return isFContiguous(this._shape, this._strides, 1);
  }

  /** The transpose of the array */
  public get T(): NDArray<D> {
    return this.transpose();
  }

  // ============================================================
  // Element Access
  // ============================================================

  /**
   * Get a single element by indices
   */
  public at(...indices: number[]): DTypeToElement[D] {
    if (indices.length !== this._shape.length) {
      throw new Error(
        `Expected ${this._shape.length} indices, got ${indices.length}`
      );
    }

    // Normalize negative indices
    const normalizedIndices = indices.map((idx, i) => {
      const size = this._shape[i];
      if (size === undefined) {
        throw new Error(`Invalid shape at dimension ${i}`);
      }
      return normalizeIndex(idx, size);
    });

    const offset = this._offset + indicesToOffset(normalizedIndices, this._strides);
    return this._data[offset] as DTypeToElement[D];
  }

  /**
   * Set a single element by indices
   */
  public set(indices: readonly number[], value: number | bigint): void {
    if (indices.length !== this._shape.length) {
      throw new Error(
        `Expected ${this._shape.length} indices, got ${indices.length}`
      );
    }

    const normalizedIndices = indices.map((idx, i) => {
      const size = this._shape[i];
      if (size === undefined) {
        throw new Error(`Invalid shape at dimension ${i}`);
      }
      return normalizeIndex(idx, size);
    });

    const offset = this._offset + indicesToOffset(normalizedIndices, this._strides);
    this.setFlat(offset, value);
  }

  /**
   * Get element by flat index
   */
  public getFlat(index: number): DTypeToElement[D] {
    return this._data[index] as DTypeToElement[D];
  }

  /**
   * Set element by flat index
   */
  public setFlat(index: number, value: number | bigint): void {
    const descriptor = getDTypeDescriptor(this._dtype);
    if (descriptor.isBigInt) {
      (this._data as BigInt64Array | BigUint64Array)[index] =
        typeof value === 'bigint' ? value : BigInt(value);
    } else {
      (this._data as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[index] =
        Number(value);
    }
  }

  /**
   * Fill the array with a value
   */
  public fill(value: number | bigint): void {
    const descriptor = getDTypeDescriptor(this._dtype);
    if (descriptor.isBigInt) {
      const bigValue = typeof value === 'bigint' ? value : BigInt(value);
      (this._data as BigInt64Array | BigUint64Array).fill(bigValue);
    } else {
      (this._data as Exclude<TypedArray, BigInt64Array | BigUint64Array>).fill(
        Number(value)
      );
    }
  }

  // ============================================================
  // Slicing and Indexing
  // ============================================================

  /**
   * Get a slice of the array
   * @param indexSpec - Array of indices or slice specifications
   */
  public slice(indexSpec: IndexSpec): NDArray<D> {
    const result = applyIndex(this._shape, this._strides, indexSpec);

    return new NDArray<D>(
      this._data,
      result.shape,
      result.strides,
      this._dtype,
      this._offset + result.offset,
      true
    );
  }

  /**
   * Set values in a slice of the array
   */
  public setSlice(indexSpec: IndexSpec, value: NDArray | number | bigint): void {
    const target = this.slice(indexSpec);

    if (typeof value === 'number' || typeof value === 'bigint') {
      // Fill with scalar
      for (const indices of new IndexIterator(target._shape)) {
        const offset = target._offset + indicesToFlat(indices, target._strides);
        target.setFlat(offset, value);
      }
    } else {
      // Copy from array with broadcasting
      const broadcastShape = broadcastShapes(target._shape, value._shape);

      if (!shapesEqual(broadcastShape, target._shape)) {
        throw new Error(
          `Cannot assign array with shape [${value._shape.join(', ')}] to slice with shape [${target._shape.join(', ')}]`
        );
      }

      const srcStrides = needsBroadcast(value._shape, target._shape)
        ? getBroadcastStrides(value._shape, value._strides, target._shape)
        : value._strides;

      for (const indices of new IndexIterator(target._shape)) {
        const targetOffset = target._offset + indicesToFlat(indices, target._strides);
        const srcOffset = value._offset + indicesToFlat(indices, srcStrides);
        target.setFlat(targetOffset, value.getFlat(srcOffset) as number);
      }
    }
  }

  // ============================================================
  // Shape Manipulation
  // ============================================================

  /**
   * Return a copy of the array with a new shape
   */
  public reshape(newShape: Shape): NDArray<D> {
    const normalizedShape = normalizeShape(newShape, this.size);

    // Try to create a view if possible
    if (this.isCContiguous) {
      const newStrides = calculateStrides(normalizedShape, 'C', 1);
      return new NDArray<D>(
        this._data,
        normalizedShape,
        newStrides,
        this._dtype,
        this._offset,
        true
      );
    }

    // Need to make a copy
    const copy = this.copy();
    const newStrides = calculateStrides(normalizedShape, 'C', 1);
    return new NDArray<D>(
      copy._data,
      normalizedShape,
      newStrides,
      this._dtype,
      0,
      false
    );
  }

  /**
   * Return a flattened copy of the array
   */
  public flatten(): NDArray<D> {
    return this.reshape([this.size]);
  }

  /**
   * Return a flattened view if possible, otherwise a copy
   */
  public ravel(): NDArray<D> {
    if (this.isCContiguous) {
      return new NDArray<D>(
        this._data,
        [this.size],
        [1],
        this._dtype,
        this._offset,
        true
      );
    }
    return this.flatten();
  }

  /**
   * Transpose the array
   */
  public transpose(axes?: readonly number[]): NDArray<D> {
    if (axes === undefined) {
      return new NDArray<D>(
        this._data,
        transposeShape(this._shape),
        transposeStrides(this._strides),
        this._dtype,
        this._offset,
        true
      );
    }

    // Validate axes
    if (axes.length !== this._shape.length) {
      throw new Error(
        `Axes length ${axes.length} does not match array dimensions ${this._shape.length}`
      );
    }

    const seen = new Set<number>();
    for (const axis of axes) {
      if (axis < 0 || axis >= this._shape.length) {
        throw new Error(`Invalid axis ${axis}`);
      }
      if (seen.has(axis)) {
        throw new Error(`Duplicate axis ${axis}`);
      }
      seen.add(axis);
    }

    const newShape = axes.map((axis) => this._shape[axis]!);
    const newStrides = axes.map((axis) => this._strides[axis]!);

    return new NDArray<D>(
      this._data,
      newShape,
      newStrides,
      this._dtype,
      this._offset,
      true
    );
  }

  /**
   * Remove single-dimensional entries from the shape
   */
  public squeeze(axis?: number | readonly number[]): NDArray<D> {
    let axes: number[];

    if (axis === undefined) {
      axes = this._shape
        .map((dim, i) => (dim === 1 ? i : -1))
        .filter((i) => i !== -1);
    } else if (typeof axis === 'number') {
      const normalizedAxis = normalizeAxis(axis, this._shape.length);
      if (this._shape[normalizedAxis] !== 1) {
        throw new Error(`Cannot squeeze axis ${axis} with size ${this._shape[normalizedAxis]}`);
      }
      axes = [normalizedAxis];
    } else {
      axes = axis.map((a) => normalizeAxis(a, this._shape.length));
      for (const a of axes) {
        if (this._shape[a] !== 1) {
          throw new Error(`Cannot squeeze axis ${a} with size ${this._shape[a]}`);
        }
      }
    }

    const axesSet = new Set(axes);
    const newShape = this._shape.filter((_, i) => !axesSet.has(i));
    const newStrides = this._strides.filter((_, i) => !axesSet.has(i));

    return new NDArray<D>(
      this._data,
      newShape,
      newStrides,
      this._dtype,
      this._offset,
      true
    );
  }

  /**
   * Expand the shape of the array by inserting a new axis
   */
  public expandDims(axis: number): NDArray<D> {
    const normalizedAxis = axis < 0 ? axis + this._shape.length + 1 : axis;
    if (normalizedAxis < 0 || normalizedAxis > this._shape.length) {
      throw new Error(`Invalid axis ${axis} for expand_dims`);
    }

    const newShape = [...this._shape];
    newShape.splice(normalizedAxis, 0, 1);

    const newStrides = [...this._strides];
    newStrides.splice(normalizedAxis, 0, 0);

    return new NDArray<D>(
      this._data,
      newShape,
      newStrides,
      this._dtype,
      this._offset,
      true
    );
  }

  // ============================================================
  // Copying and Type Conversion
  // ============================================================

  /**
   * Return a deep copy of the array
   */
  public copy(): NDArray<D> {
    const newData = new (getDTypeDescriptor(this._dtype).arrayConstructor)(this.size);
    const newStrides = calculateStrides(this._shape, 'C', 1);

    // Copy data
    let i = 0;
    for (const indices of new IndexIterator(this._shape)) {
      const srcOffset = this._offset + indicesToFlat(indices, this._strides);
      newData[i++] = this._data[srcOffset]!;
    }

    return new NDArray<D>(newData, [...this._shape], newStrides, this._dtype, 0, false);
  }

  /**
   * Cast the array to a different dtype
   */
  public astype<NewD extends DTypeName>(dtype: NewD): NDArray<NewD> {
    if (dtype === (this._dtype as unknown as NewD)) {
      return this.copy() as unknown as NDArray<NewD>;
    }

    const newDescriptor = getDTypeDescriptor(dtype);
    const newData = new newDescriptor.arrayConstructor(this.size);
    const newStrides = calculateStrides(this._shape, 'C', 1);

    const srcIsBigInt = getDTypeDescriptor(this._dtype).isBigInt;
    const dstIsBigInt = newDescriptor.isBigInt;

    let i = 0;
    for (const indices of new IndexIterator(this._shape)) {
      const srcOffset = this._offset + indicesToFlat(indices, this._strides);
      let value = this._data[srcOffset];

      if (srcIsBigInt && !dstIsBigInt) {
        value = Number(value as bigint) as typeof value;
      } else if (!srcIsBigInt && dstIsBigInt) {
        value = BigInt(value as number) as typeof value;
      }

      if (dstIsBigInt) {
        (newData as BigInt64Array | BigUint64Array)[i++] = value as bigint;
      } else {
        (newData as Float64Array)[i++] = value as number;
      }
    }

    return new NDArray<NewD>(
      newData,
      [...this._shape],
      newStrides,
      dtype,
      0,
      false
    );
  }

  // ============================================================
  // Iteration
  // ============================================================

  /**
   * Iterate over flat indices
   */
  public *flatIndices(): Generator<number> {
    for (const indices of new IndexIterator(this._shape)) {
      yield this._offset + indicesToFlat(indices, this._strides);
    }
  }

  /**
   * Iterate over elements
   */
  public *[Symbol.iterator](): Generator<DTypeToElement[D]> {
    for (const flatIndex of this.flatIndices()) {
      yield this._data[flatIndex] as DTypeToElement[D];
    }
  }

  /**
   * Iterate over elements with their indices
   */
  public *entries(): Generator<[readonly number[], DTypeToElement[D]]> {
    for (const indices of new IndexIterator(this._shape)) {
      const flatIndex = this._offset + indicesToFlat(indices, this._strides);
      yield [indices, this._data[flatIndex] as DTypeToElement[D]];
    }
  }

  // ============================================================
  // Conversion
  // ============================================================

  /**
   * Convert to a nested JavaScript array
   */
  public toArray(): unknown {
    if (this._shape.length === 0) {
      return this._data[this._offset];
    }

    const buildArray = (depth: number, indices: number[]): unknown => {
      if (depth === this._shape.length) {
        const flatIndex = this._offset + indicesToFlat(indices, this._strides);
        return this._data[flatIndex];
      }

      const size = this._shape[depth]!;
      const result: unknown[] = new Array(size);
      for (let i = 0; i < size; i++) {
        result[i] = buildArray(depth + 1, [...indices, i]);
      }
      return result;
    };

    return buildArray(0, []);
  }

  /**
   * Convert to a flat JavaScript array
   */
  public toFlatArray(): (number | bigint)[] {
    const result: (number | bigint)[] = [];
    for (const value of this) {
      result.push(value as number | bigint);
    }
    return result;
  }

  /**
   * String representation
   */
  public toString(): string {
    const arrayStr = JSON.stringify(this.toArray());
    return `NDArray(${arrayStr}, dtype='${this._dtype}')`;
  }

  // ============================================================
  // Comparison
  // ============================================================

  /**
   * Check if two arrays have equal values
   */
  public equals(other: NDArray): boolean {
    if (!shapesEqual(this._shape, other._shape)) {
      return false;
    }

    const iter1 = this[Symbol.iterator]();
    const iter2 = other[Symbol.iterator]();

    while (true) {
      const r1 = iter1.next();
      const r2 = iter2.next();

      if (r1.done === true && r2.done === true) {
        return true;
      }

      if (r1.done === true || r2.done === true) {
        return false;
      }

      if (r1.value !== r2.value) {
        return false;
      }
    }
  }

  /**
   * Check if arrays are approximately equal (for floating point)
   */
  public allClose(
    other: NDArray,
    rtol: number = 1e-5,
    atol: number = 1e-8
  ): boolean {
    if (!shapesEqual(this._shape, other._shape)) {
      return false;
    }

    const iter1 = this[Symbol.iterator]();
    const iter2 = other[Symbol.iterator]();

    while (true) {
      const r1 = iter1.next();
      const r2 = iter2.next();

      if (r1.done === true && r2.done === true) {
        return true;
      }

      if (r1.done === true || r2.done === true) {
        return false;
      }

      const v1 = Number(r1.value);
      const v2 = Number(r2.value);

      if (Math.abs(v1 - v2) > atol + rtol * Math.abs(v2)) {
        return false;
      }
    }
  }
}

// Re-export slice helper
export { slice };
