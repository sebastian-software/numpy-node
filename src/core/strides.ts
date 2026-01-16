/**
 * Strides utilities for numpy-ts
 * Handles stride calculation for array memory layout
 */

import type { Shape } from './shape.js';

/**
 * Strides type - number of bytes/elements to step in each dimension
 */
export type Strides = readonly number[];

/**
 * Mutable strides for internal operations
 */
export type MutableStrides = number[];

/**
 * Memory order for array layout
 */
export type Order = 'C' | 'F';

/**
 * Calculate strides for a given shape in C-contiguous (row-major) order
 * Each stride represents the number of elements to step in that dimension
 */
export function calculateStridesC(shape: Shape, itemSize: number = 1): Strides {
  const ndim = shape.length;
  if (ndim === 0) {
    return [];
  }

  const strides: MutableStrides = new Array(ndim);
  strides[ndim - 1] = itemSize;

  for (let i = ndim - 2; i >= 0; i--) {
    const nextShape = shape[i + 1];
    const nextStride = strides[i + 1];
    if (nextShape === undefined || nextStride === undefined) {
      throw new Error('Invalid shape or stride during calculation');
    }
    strides[i] = nextStride * nextShape;
  }

  return strides;
}

/**
 * Calculate strides for a given shape in Fortran-contiguous (column-major) order
 */
export function calculateStridesF(shape: Shape, itemSize: number = 1): Strides {
  const ndim = shape.length;
  if (ndim === 0) {
    return [];
  }

  const strides: MutableStrides = new Array(ndim);
  strides[0] = itemSize;

  for (let i = 1; i < ndim; i++) {
    const prevShape = shape[i - 1];
    const prevStride = strides[i - 1];
    if (prevShape === undefined || prevStride === undefined) {
      throw new Error('Invalid shape or stride during calculation');
    }
    strides[i] = prevStride * prevShape;
  }

  return strides;
}

/**
 * Calculate strides based on order
 */
export function calculateStrides(shape: Shape, order: Order = 'C', itemSize: number = 1): Strides {
  return order === 'C' ? calculateStridesC(shape, itemSize) : calculateStridesF(shape, itemSize);
}

/**
 * Calculate the flat index from multi-dimensional indices
 */
export function indicesToOffset(indices: readonly number[], strides: Strides): number {
  if (indices.length !== strides.length) {
    throw new Error(
      `Index dimension ${indices.length} does not match strides dimension ${strides.length}`
    );
  }

  let offset = 0;
  for (let i = 0; i < indices.length; i++) {
    const idx = indices[i];
    const stride = strides[i];
    if (idx === undefined || stride === undefined) {
      throw new Error(`Invalid index or stride at dimension ${i}`);
    }
    offset += idx * stride;
  }
  return offset;
}

/**
 * Calculate multi-dimensional indices from a flat index
 */
export function offsetToIndices(offset: number, shape: Shape, strides: Strides): number[] {
  if (shape.length !== strides.length) {
    throw new Error(
      `Shape dimension ${shape.length} does not match strides dimension ${strides.length}`
    );
  }

  const indices: number[] = new Array(shape.length);
  let remaining = offset;

  // Sort dimensions by stride (descending) to handle non-contiguous arrays
  const dimOrder = shape
    .map((_, i) => i)
    .sort((a, b) => {
      const strideA = strides[a];
      const strideB = strides[b];
      if (strideA === undefined || strideB === undefined) {
        throw new Error('Invalid stride during dimension sorting');
      }
      return strideB - strideA;
    });

  for (const dim of dimOrder) {
    const stride = strides[dim];
    if (stride === undefined) {
      throw new Error(`Invalid stride at dimension ${dim}`);
    }
    if (stride === 0) {
      indices[dim] = 0;
    } else {
      indices[dim] = Math.floor(remaining / stride);
      remaining = remaining % stride;
    }
  }

  return indices;
}

/**
 * Check if an array is C-contiguous
 */
export function isCContiguous(shape: Shape, strides: Strides, itemSize: number = 1): boolean {
  const expectedStrides = calculateStridesC(shape, itemSize);
  return stridesEqual(strides, expectedStrides);
}

/**
 * Check if an array is F-contiguous
 */
export function isFContiguous(shape: Shape, strides: Strides, itemSize: number = 1): boolean {
  const expectedStrides = calculateStridesF(shape, itemSize);
  return stridesEqual(strides, expectedStrides);
}

/**
 * Check if an array is contiguous (either C or F)
 */
export function isContiguous(shape: Shape, strides: Strides, itemSize: number = 1): boolean {
  return isCContiguous(shape, strides, itemSize) || isFContiguous(shape, strides, itemSize);
}

/**
 * Check if two strides are equal
 */
export function stridesEqual(strides1: Strides, strides2: Strides): boolean {
  if (strides1.length !== strides2.length) {
    return false;
  }
  for (let i = 0; i < strides1.length; i++) {
    if (strides1[i] !== strides2[i]) {
      return false;
    }
  }
  return true;
}

/**
 * Calculate strides for a transposed array
 */
export function transposeStrides(strides: Strides): Strides {
  return [...strides].reverse();
}

/**
 * Calculate strides after axes permutation
 */
export function permuteStrides(strides: Strides, axes: readonly number[]): Strides {
  const result: MutableStrides = new Array(strides.length);
  for (let i = 0; i < axes.length; i++) {
    const axis = axes[i];
    if (axis === undefined) {
      throw new Error(`Invalid axis at index ${i}`);
    }
    const stride = strides[axis];
    if (stride === undefined) {
      throw new Error(`Invalid stride at axis ${axis}`);
    }
    result[i] = stride;
  }
  return result;
}

/**
 * Calculate strides for a broadcast view
 * Dimensions of size 1 get stride 0 (they will be repeated)
 */
export function broadcastStrides(
  shape: Shape,
  strides: Strides,
  targetShape: Shape
): Strides {
  const offset = targetShape.length - shape.length;
  const result: MutableStrides = new Array(targetShape.length).fill(0);

  for (let i = shape.length - 1; i >= 0; i--) {
    const targetIndex = i + offset;
    const dim = shape[i];
    const targetDim = targetShape[targetIndex];

    if (dim === undefined || targetDim === undefined) {
      throw new Error('Invalid shape dimension');
    }

    if (dim === targetDim) {
      const stride = strides[i];
      if (stride === undefined) {
        throw new Error(`Invalid stride at index ${i}`);
      }
      result[targetIndex] = stride;
    } else if (dim === 1) {
      result[targetIndex] = 0; // Broadcast dimension
    } else {
      throw new Error(
        `Cannot broadcast shape [${shape.join(', ')}] to [${targetShape.join(', ')}]`
      );
    }
  }

  return result;
}

/**
 * Calculate strides for a reshaped view (if possible)
 * Returns null if reshaping requires a copy
 */
export function reshapeStrides(
  shape: Shape,
  strides: Strides,
  newShape: Shape
): Strides | null {
  // Check if we can create a view
  if (!isCContiguous(shape, strides, 1) && !isFContiguous(shape, strides, 1)) {
    return null;
  }

  const order = isCContiguous(shape, strides, 1) ? 'C' : 'F';
  return calculateStrides(newShape, order, 1);
}

/**
 * Calculate strides for a sliced view
 */
export function sliceStrides(
  strides: Strides,
  sliceSteps: readonly number[]
): Strides {
  if (strides.length !== sliceSteps.length) {
    throw new Error('Strides and slice steps must have same length');
  }

  const result: MutableStrides = new Array(strides.length);
  for (let i = 0; i < strides.length; i++) {
    const stride = strides[i];
    const step = sliceSteps[i];
    if (stride === undefined || step === undefined) {
      throw new Error(`Invalid stride or step at index ${i}`);
    }
    result[i] = stride * step;
  }
  return result;
}

/**
 * Calculate strides after expanding dimensions
 */
export function expandDimsStrides(strides: Strides, axis: number): Strides {
  const normalizedAxis = axis < 0 ? axis + strides.length + 1 : axis;
  const result: MutableStrides = [...strides];
  // New dimension has stride 0 (size 1, no need to step)
  result.splice(normalizedAxis, 0, 0);
  return result;
}

/**
 * Calculate strides after squeezing dimensions
 */
export function squeezeStrides(strides: Strides, axes: readonly number[]): Strides {
  const axesSet = new Set(axes);
  const result: MutableStrides = [];

  for (let i = 0; i < strides.length; i++) {
    if (!axesSet.has(i)) {
      const stride = strides[i];
      if (stride !== undefined) {
        result.push(stride);
      }
    }
  }

  return result;
}
