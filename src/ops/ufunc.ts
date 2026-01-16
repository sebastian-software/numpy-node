/**
 * Universal function (ufunc) base implementation for np-ts
 * Provides infrastructure for element-wise operations with broadcasting
 */

import {
  NDArray,
  type DTypeName,
  broadcastShapes,
  getBroadcastStrides,
  needsBroadcast,
  promoteDTypes,
  IndexIterator,
  indicesToFlat,
} from '../core/index.js';

/**
 * Unary function type
 */
export type UnaryOp = (x: number) => number;
export type UnaryOpBigInt = (x: bigint) => bigint;

/**
 * Binary function type
 */
export type BinaryOp = (x: number, y: number) => number;
export type BinaryOpBigInt = (x: bigint, y: bigint) => bigint;

/**
 * Apply a unary operation element-wise
 */
export function applyUnary<D extends DTypeName>(
  arr: NDArray<D>,
  op: UnaryOp,
  opBigInt?: UnaryOpBigInt,
  outDtype?: DTypeName
): NDArray {
  const dtype = outDtype ?? arr.dtype;
  const result = NDArray.zeros(arr.shape, { dtype });

  const isBigIntInput = arr.dtype === 'int64' || arr.dtype === 'uint64';
  const isBigIntOutput = dtype === 'int64' || dtype === 'uint64';

  let i = 0;
  for (const indices of new IndexIterator(arr.shape)) {
    const srcOffset = arr.offset + indicesToFlat(indices, arr.strides);
    const value = arr.data[srcOffset];

    let resultValue: number | bigint;
    if (isBigIntInput && opBigInt !== undefined) {
      resultValue = opBigInt(value as bigint);
    } else {
      resultValue = op(Number(value));
    }

    if (isBigIntOutput && typeof resultValue === 'number') {
      resultValue = BigInt(Math.round(resultValue));
    } else if (!isBigIntOutput && typeof resultValue === 'bigint') {
      resultValue = Number(resultValue);
    }

    result.setFlat(i++, resultValue);
  }

  return result;
}

/**
 * Apply a binary operation element-wise with broadcasting
 */
export function applyBinary<D1 extends DTypeName, D2 extends DTypeName>(
  arr1: NDArray<D1>,
  arr2: NDArray<D2>,
  op: BinaryOp,
  opBigInt?: BinaryOpBigInt,
  outDtype?: DTypeName
): NDArray {
  const broadcastShape = broadcastShapes(arr1.shape, arr2.shape);
  const dtype = outDtype ?? promoteDTypes(arr1.dtype, arr2.dtype);
  const result = NDArray.zeros(broadcastShape, { dtype });

  const strides1 = needsBroadcast(arr1.shape, broadcastShape)
    ? getBroadcastStrides(arr1.shape, arr1.strides, broadcastShape)
    : arr1.strides;

  const strides2 = needsBroadcast(arr2.shape, broadcastShape)
    ? getBroadcastStrides(arr2.shape, arr2.strides, broadcastShape)
    : arr2.strides;

  const isBigInt1 = arr1.dtype === 'int64' || arr1.dtype === 'uint64';
  const isBigInt2 = arr2.dtype === 'int64' || arr2.dtype === 'uint64';
  const isBigIntOutput = dtype === 'int64' || dtype === 'uint64';

  let i = 0;
  for (const indices of new IndexIterator(broadcastShape)) {
    const offset1 = arr1.offset + indicesToFlat(indices, strides1);
    const offset2 = arr2.offset + indicesToFlat(indices, strides2);

    const val1 = arr1.data[offset1];
    const val2 = arr2.data[offset2];

    let resultValue: number | bigint;
    if ((isBigInt1 || isBigInt2) && opBigInt !== undefined) {
      const bigVal1 = isBigInt1 ? (val1 as bigint) : BigInt(val1 as number);
      const bigVal2 = isBigInt2 ? (val2 as bigint) : BigInt(val2 as number);
      resultValue = opBigInt(bigVal1, bigVal2);
    } else {
      resultValue = op(Number(val1), Number(val2));
    }

    if (isBigIntOutput && typeof resultValue === 'number') {
      resultValue = BigInt(Math.round(resultValue));
    } else if (!isBigIntOutput && typeof resultValue === 'bigint') {
      resultValue = Number(resultValue);
    }

    result.setFlat(i++, resultValue);
  }

  return result;
}

/**
 * Apply a binary operation with a scalar
 */
export function applyBinaryScalar<D extends DTypeName>(
  arr: NDArray<D>,
  scalar: number | bigint,
  op: BinaryOp,
  opBigInt?: BinaryOpBigInt,
  outDtype?: DTypeName,
  scalarFirst: boolean = false
): NDArray {
  const dtype = outDtype ?? arr.dtype;
  const result = NDArray.zeros(arr.shape, { dtype });

  const isBigIntInput = arr.dtype === 'int64' || arr.dtype === 'uint64';
  const isBigIntScalar = typeof scalar === 'bigint';
  const isBigIntOutput = dtype === 'int64' || dtype === 'uint64';

  let i = 0;
  for (const indices of new IndexIterator(arr.shape)) {
    const srcOffset = arr.offset + indicesToFlat(indices, arr.strides);
    const value = arr.data[srcOffset];

    let resultValue: number | bigint;
    if ((isBigIntInput || isBigIntScalar) && opBigInt !== undefined) {
      const bigVal = isBigIntInput ? (value as bigint) : BigInt(value as number);
      const bigScalar = isBigIntScalar ? scalar : BigInt(scalar as number);
      resultValue = scalarFirst
        ? opBigInt(bigScalar, bigVal)
        : opBigInt(bigVal, bigScalar);
    } else {
      const numVal = Number(value);
      const numScalar = Number(scalar);
      resultValue = scalarFirst ? op(numScalar, numVal) : op(numVal, numScalar);
    }

    if (isBigIntOutput && typeof resultValue === 'number') {
      resultValue = BigInt(Math.round(resultValue));
    } else if (!isBigIntOutput && typeof resultValue === 'bigint') {
      resultValue = Number(resultValue);
    }

    result.setFlat(i++, resultValue);
  }

  return result;
}

/**
 * Apply a comparison operation element-wise with broadcasting
 * Returns a boolean array (stored as uint8)
 */
export function applyComparison<D1 extends DTypeName, D2 extends DTypeName>(
  arr1: NDArray<D1>,
  arr2: NDArray<D2>,
  op: (x: number, y: number) => boolean
): NDArray<'bool'> {
  const broadcastShape = broadcastShapes(arr1.shape, arr2.shape);
  const result = NDArray.zeros<'bool'>(broadcastShape, { dtype: 'bool' });

  const strides1 = needsBroadcast(arr1.shape, broadcastShape)
    ? getBroadcastStrides(arr1.shape, arr1.strides, broadcastShape)
    : arr1.strides;

  const strides2 = needsBroadcast(arr2.shape, broadcastShape)
    ? getBroadcastStrides(arr2.shape, arr2.strides, broadcastShape)
    : arr2.strides;

  let i = 0;
  for (const indices of new IndexIterator(broadcastShape)) {
    const offset1 = arr1.offset + indicesToFlat(indices, strides1);
    const offset2 = arr2.offset + indicesToFlat(indices, strides2);

    const val1 = Number(arr1.data[offset1]);
    const val2 = Number(arr2.data[offset2]);

    result.setFlat(i++, op(val1, val2) ? 1 : 0);
  }

  return result;
}

/**
 * Apply a comparison operation with a scalar
 */
export function applyComparisonScalar<D extends DTypeName>(
  arr: NDArray<D>,
  scalar: number | bigint,
  op: (x: number, y: number) => boolean,
  scalarFirst: boolean = false
): NDArray<'bool'> {
  const result = NDArray.zeros<'bool'>(arr.shape, { dtype: 'bool' });
  const numScalar = Number(scalar);

  let i = 0;
  for (const indices of new IndexIterator(arr.shape)) {
    const srcOffset = arr.offset + indicesToFlat(indices, arr.strides);
    const value = Number(arr.data[srcOffset]);

    const boolResult = scalarFirst ? op(numScalar, value) : op(value, numScalar);
    result.setFlat(i++, boolResult ? 1 : 0);
  }

  return result;
}

/**
 * Reduce an array along an axis using a binary operation
 */
export function reduceAxis<D extends DTypeName>(
  arr: NDArray<D>,
  op: BinaryOp,
  axis: number,
  initial?: number,
  keepdims: boolean = false
): NDArray {
  const normalizedAxis = axis < 0 ? axis + arr.ndim : axis;
  if (normalizedAxis < 0 || normalizedAxis >= arr.ndim) {
    throw new Error(`Invalid axis ${axis} for array with ${arr.ndim} dimensions`);
  }

  // Calculate output shape
  const outShape: number[] = [];
  for (let i = 0; i < arr.ndim; i++) {
    if (i === normalizedAxis) {
      if (keepdims) {
        outShape.push(1);
      }
    } else {
      outShape.push(arr.shape[i]!);
    }
  }

  if (outShape.length === 0) {
    outShape.push(1);
  }

  const result = NDArray.zeros(outShape, { dtype: arr.dtype });
  const axisSize = arr.shape[normalizedAxis]!;

  // Iterate over output positions
  let outIdx = 0;
  for (const outIndices of new IndexIterator(outShape)) {
    let acc = initial ?? Number(arr.data[0]);
    let first = initial === undefined;

    // Build input indices template
    const inIndices: number[] = [];
    let j = 0;
    for (let i = 0; i < arr.ndim; i++) {
      if (i === normalizedAxis) {
        inIndices.push(0); // Will be replaced
        if (keepdims) j++; // Skip the reduced dimension in outIndices when keepdims
      } else {
        inIndices.push(outIndices[j]!);
        j++;
      }
    }

    // Reduce along axis
    for (let k = 0; k < axisSize; k++) {
      inIndices[normalizedAxis] = k;
      const offset = arr.offset + indicesToFlat(inIndices, arr.strides);
      const value = Number(arr.data[offset]);

      if (first) {
        acc = value;
        first = false;
      } else {
        acc = op(acc, value);
      }
    }

    result.setFlat(outIdx++, acc);
  }

  return result;
}

/**
 * Reduce all elements of an array using a binary operation
 */
export function reduceAll<D extends DTypeName>(
  arr: NDArray<D>,
  op: BinaryOp,
  initial?: number
): number {
  let acc = initial;
  let first = initial === undefined;

  for (const indices of new IndexIterator(arr.shape)) {
    const offset = arr.offset + indicesToFlat(indices, arr.strides);
    const value = Number(arr.data[offset]);

    if (first) {
      acc = value;
      first = false;
    } else {
      acc = op(acc!, value);
    }
  }

  return acc ?? 0;
}
