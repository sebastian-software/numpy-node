/**
 * Data type system for numpy-ts
 * Provides NumPy-compatible data types with TypedArray backing
 */

/**
 * String identifiers for all supported data types
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
 * TypedArray constructors for each dtype
 */
export type TypedArrayConstructor =
  | Int8ArrayConstructor
  | Int16ArrayConstructor
  | Int32ArrayConstructor
  | BigInt64ArrayConstructor
  | Uint8ArrayConstructor
  | Uint16ArrayConstructor
  | Uint32ArrayConstructor
  | BigUint64ArrayConstructor
  | Float32ArrayConstructor
  | Float64ArrayConstructor;

/**
 * TypedArray instances for each dtype
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
 * Map dtype names to their TypedArray types
 */
export interface DTypeToTypedArray {
  int8: Int8Array;
  int16: Int16Array;
  int32: Int32Array;
  int64: BigInt64Array;
  uint8: Uint8Array;
  uint16: Uint16Array;
  uint32: Uint32Array;
  uint64: BigUint64Array;
  float32: Float32Array;
  float64: Float64Array;
  bool: Uint8Array;
}

/**
 * Map dtype names to their element types
 */
export interface DTypeToElement {
  int8: number;
  int16: number;
  int32: number;
  int64: bigint;
  uint8: number;
  uint16: number;
  uint32: number;
  uint64: bigint;
  float32: number;
  float64: number;
  bool: number;
}

/**
 * Numeric element types (excludes bigint for most operations)
 */
export type NumericDTypeName = Exclude<DTypeName, 'int64' | 'uint64'>;

/**
 * Data type descriptor containing metadata about a dtype
 */
export interface DTypeDescriptor {
  readonly name: DTypeName;
  readonly bytes: number;
  readonly arrayConstructor: TypedArrayConstructor;
  readonly isInteger: boolean;
  readonly isSigned: boolean;
  readonly isFloat: boolean;
  readonly isBigInt: boolean;
  readonly min: number | bigint;
  readonly max: number | bigint;
}

/**
 * DType descriptors for all supported types
 */
const DTYPE_DESCRIPTORS: Record<DTypeName, DTypeDescriptor> = {
  int8: {
    name: 'int8',
    bytes: 1,
    arrayConstructor: Int8Array,
    isInteger: true,
    isSigned: true,
    isFloat: false,
    isBigInt: false,
    min: -128,
    max: 127,
  },
  int16: {
    name: 'int16',
    bytes: 2,
    arrayConstructor: Int16Array,
    isInteger: true,
    isSigned: true,
    isFloat: false,
    isBigInt: false,
    min: -32768,
    max: 32767,
  },
  int32: {
    name: 'int32',
    bytes: 4,
    arrayConstructor: Int32Array,
    isInteger: true,
    isSigned: true,
    isFloat: false,
    isBigInt: false,
    min: -2147483648,
    max: 2147483647,
  },
  int64: {
    name: 'int64',
    bytes: 8,
    arrayConstructor: BigInt64Array,
    isInteger: true,
    isSigned: true,
    isFloat: false,
    isBigInt: true,
    min: BigInt('-9223372036854775808'),
    max: BigInt('9223372036854775807'),
  },
  uint8: {
    name: 'uint8',
    bytes: 1,
    arrayConstructor: Uint8Array,
    isInteger: true,
    isSigned: false,
    isFloat: false,
    isBigInt: false,
    min: 0,
    max: 255,
  },
  uint16: {
    name: 'uint16',
    bytes: 2,
    arrayConstructor: Uint16Array,
    isInteger: true,
    isSigned: false,
    isFloat: false,
    isBigInt: false,
    min: 0,
    max: 65535,
  },
  uint32: {
    name: 'uint32',
    bytes: 4,
    arrayConstructor: Uint32Array,
    isInteger: true,
    isSigned: false,
    isFloat: false,
    isBigInt: false,
    min: 0,
    max: 4294967295,
  },
  uint64: {
    name: 'uint64',
    bytes: 8,
    arrayConstructor: BigUint64Array,
    isInteger: true,
    isSigned: false,
    isFloat: false,
    isBigInt: true,
    min: BigInt(0),
    max: BigInt('18446744073709551615'),
  },
  float32: {
    name: 'float32',
    bytes: 4,
    arrayConstructor: Float32Array,
    isInteger: false,
    isSigned: true,
    isFloat: true,
    isBigInt: false,
    min: -3.4028235e38,
    max: 3.4028235e38,
  },
  float64: {
    name: 'float64',
    bytes: 8,
    arrayConstructor: Float64Array,
    isInteger: false,
    isSigned: true,
    isFloat: true,
    isBigInt: false,
    min: -Number.MAX_VALUE,
    max: Number.MAX_VALUE,
  },
  bool: {
    name: 'bool',
    bytes: 1,
    arrayConstructor: Uint8Array,
    isInteger: true,
    isSigned: false,
    isFloat: false,
    isBigInt: false,
    min: 0,
    max: 1,
  },
};

/**
 * DType class representing a data type
 */
export class DType {
  private readonly _descriptor: DTypeDescriptor;

  private constructor(descriptor: DTypeDescriptor) {
    this._descriptor = descriptor;
  }

  /**
   * Create a DType from a name string
   */
  public static from(name: DTypeName): DType {
    const descriptor = DTYPE_DESCRIPTORS[name];
    return new DType(descriptor);
  }

  /**
   * Get the dtype name
   */
  public get name(): DTypeName {
    return this._descriptor.name;
  }

  /**
   * Get the size in bytes
   */
  public get bytes(): number {
    return this._descriptor.bytes;
  }

  /**
   * Get the TypedArray constructor
   */
  public get arrayConstructor(): TypedArrayConstructor {
    return this._descriptor.arrayConstructor;
  }

  /**
   * Check if this is an integer type
   */
  public get isInteger(): boolean {
    return this._descriptor.isInteger;
  }

  /**
   * Check if this is a signed type
   */
  public get isSigned(): boolean {
    return this._descriptor.isSigned;
  }

  /**
   * Check if this is a floating point type
   */
  public get isFloat(): boolean {
    return this._descriptor.isFloat;
  }

  /**
   * Check if this is a BigInt type
   */
  public get isBigInt(): boolean {
    return this._descriptor.isBigInt;
  }

  /**
   * Get the minimum value for this dtype
   */
  public get min(): number | bigint {
    return this._descriptor.min;
  }

  /**
   * Get the maximum value for this dtype
   */
  public get max(): number | bigint {
    return this._descriptor.max;
  }

  /**
   * Create a new TypedArray of this dtype
   */
  public createArray(length: number): TypedArray {
    return new this._descriptor.arrayConstructor(length);
  }

  /**
   * Create a TypedArray from existing data
   */
  public createArrayFrom(data: ArrayLike<number> | ArrayLike<bigint>): TypedArray {
    if (this._descriptor.isBigInt) {
      const bigintData = data as ArrayLike<bigint>;
      return new (this._descriptor.arrayConstructor as BigInt64ArrayConstructor | BigUint64ArrayConstructor)(
        Array.from(bigintData)
      );
    }
    return new (this._descriptor.arrayConstructor as Exclude<TypedArrayConstructor, BigInt64ArrayConstructor | BigUint64ArrayConstructor>)(
      data as ArrayLike<number>
    );
  }

  /**
   * Check equality with another dtype
   */
  public equals(other: DType): boolean {
    return this._descriptor.name === other._descriptor.name;
  }

  /**
   * String representation
   */
  public toString(): string {
    return `dtype('${this._descriptor.name}')`;
  }
}

/**
 * Get the DType descriptor for a given name
 */
export function getDTypeDescriptor(name: DTypeName): DTypeDescriptor {
  return DTYPE_DESCRIPTORS[name];
}

/**
 * Check if a string is a valid dtype name
 */
export function isValidDTypeName(name: string): name is DTypeName {
  return name in DTYPE_DESCRIPTORS;
}

/**
 * Infer the dtype from a JavaScript value
 */
export function inferDType(value: unknown): DTypeName {
  if (typeof value === 'bigint') {
    return value >= 0 ? 'uint64' : 'int64';
  }
  if (typeof value === 'number') {
    if (Number.isInteger(value)) {
      if (value >= 0 && value <= 255) return 'uint8';
      if (value >= -128 && value <= 127) return 'int8';
      if (value >= 0 && value <= 65535) return 'uint16';
      if (value >= -32768 && value <= 32767) return 'int16';
      if (value >= 0 && value <= 4294967295) return 'uint32';
      if (value >= -2147483648 && value <= 2147483647) return 'int32';
    }
    return 'float64';
  }
  if (typeof value === 'boolean') {
    return 'bool';
  }
  return 'float64';
}

/**
 * Infer the dtype from an array of values
 */
export function inferDTypeFromArray(values: readonly unknown[]): DTypeName {
  if (values.length === 0) {
    return 'float64';
  }

  let hasBigInt = false;
  let hasFloat = false;
  let hasSigned = false;
  let maxUint = 0;
  let minInt = 0;
  let maxInt = 0;

  for (const value of values) {
    if (typeof value === 'bigint') {
      hasBigInt = true;
      if (value < 0) hasSigned = true;
    } else if (typeof value === 'number') {
      if (!Number.isInteger(value)) {
        hasFloat = true;
      } else {
        if (value < 0) {
          hasSigned = true;
          minInt = Math.min(minInt, value);
        } else {
          maxUint = Math.max(maxUint, value);
        }
        maxInt = Math.max(maxInt, value);
      }
    } else if (typeof value === 'boolean') {
      maxUint = Math.max(maxUint, value ? 1 : 0);
    }
  }

  if (hasFloat) return 'float64';
  if (hasBigInt) return hasSigned ? 'int64' : 'uint64';

  if (hasSigned) {
    if (minInt >= -128 && maxInt <= 127) return 'int8';
    if (minInt >= -32768 && maxInt <= 32767) return 'int16';
    if (minInt >= -2147483648 && maxInt <= 2147483647) return 'int32';
    return 'int64';
  }

  if (maxUint <= 255) return 'uint8';
  if (maxUint <= 65535) return 'uint16';
  if (maxUint <= 4294967295) return 'uint32';
  return 'uint64';
}

/**
 * Determine the result dtype for a binary operation
 * Follows NumPy type promotion rules
 */
export function promoteDTypes(dtype1: DTypeName, dtype2: DTypeName): DTypeName {
  if (dtype1 === dtype2) return dtype1;

  const d1 = DTYPE_DESCRIPTORS[dtype1];
  const d2 = DTYPE_DESCRIPTORS[dtype2];

  // Float promotion
  if (d1.isFloat || d2.isFloat) {
    if (d1.bytes >= 8 || d2.bytes >= 8) return 'float64';
    return 'float64'; // Always use float64 for mixed operations
  }

  // BigInt promotion
  if (d1.isBigInt || d2.isBigInt) {
    if (d1.isSigned || d2.isSigned) return 'int64';
    return 'uint64';
  }

  // Integer promotion
  const maxBytes = Math.max(d1.bytes, d2.bytes);
  const needsSigned = d1.isSigned || d2.isSigned;

  // If mixing signed and unsigned, may need larger type
  if (d1.isSigned !== d2.isSigned) {
    const unsignedType = d1.isSigned ? d2 : d1;
    const signedType = d1.isSigned ? d1 : d2;

    if (unsignedType.bytes >= signedType.bytes) {
      // Need to promote to larger signed type
      if (unsignedType.bytes >= 4) return 'int64';
      if (unsignedType.bytes >= 2) return 'int32';
      return 'int16';
    }
  }

  if (needsSigned) {
    if (maxBytes === 1) return 'int8';
    if (maxBytes === 2) return 'int16';
    if (maxBytes === 4) return 'int32';
    return 'int64';
  }

  if (maxBytes === 1) return 'uint8';
  if (maxBytes === 2) return 'uint16';
  if (maxBytes === 4) return 'uint32';
  return 'uint64';
}

/**
 * Check if a dtype can be safely cast to another
 */
export function canCast(from: DTypeName, to: DTypeName, casting: 'no' | 'equiv' | 'safe' | 'same_kind' | 'unsafe' = 'safe'): boolean {
  if (from === to) return true;
  if (casting === 'no') return false;
  if (casting === 'unsafe') return true;

  const fromDesc = DTYPE_DESCRIPTORS[from];
  const toDesc = DTYPE_DESCRIPTORS[to];

  if (casting === 'equiv') {
    return fromDesc.bytes === toDesc.bytes && fromDesc.isFloat === toDesc.isFloat;
  }

  if (casting === 'same_kind') {
    if (fromDesc.isFloat && toDesc.isFloat) return toDesc.bytes >= fromDesc.bytes;
    if (fromDesc.isInteger && toDesc.isInteger) return toDesc.bytes >= fromDesc.bytes;
    return false;
  }

  // 'safe' casting
  if (fromDesc.isFloat) {
    return toDesc.isFloat && toDesc.bytes >= fromDesc.bytes;
  }

  if (toDesc.isFloat) {
    // Integer to float is safe if float has enough precision
    if (fromDesc.isBigInt) return false; // BigInt to float is not safe
    if (fromDesc.bytes <= 3) return toDesc.bytes >= 4; // int24 fits in float32
    return toDesc.bytes >= 8; // int32 needs float64
  }

  // Integer to integer
  if (fromDesc.isBigInt !== toDesc.isBigInt) return false;

  if (fromDesc.isSigned === toDesc.isSigned) {
    return toDesc.bytes >= fromDesc.bytes;
  }

  // Unsigned to signed needs extra bit
  if (!fromDesc.isSigned && toDesc.isSigned) {
    return toDesc.bytes > fromDesc.bytes;
  }

  // Signed to unsigned - only safe if value is non-negative (can't determine statically)
  return false;
}

/**
 * Predefined dtype instances for convenience
 */
export const dtypes = {
  int8: DType.from('int8'),
  int16: DType.from('int16'),
  int32: DType.from('int32'),
  int64: DType.from('int64'),
  uint8: DType.from('uint8'),
  uint16: DType.from('uint16'),
  uint32: DType.from('uint32'),
  uint64: DType.from('uint64'),
  float32: DType.from('float32'),
  float64: DType.from('float64'),
  bool: DType.from('bool'),
} as const;

/**
 * Default dtype for operations
 */
export const DEFAULT_DTYPE: DTypeName = 'float64';
