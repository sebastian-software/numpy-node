/**
 * Indexing and slicing utilities for numpy-ts
 * Provides type-safe method-based slicing as an alternative to Python's bracket syntax
 */

import type { Shape, MutableShape } from './shape.js';
import type { Strides, MutableStrides } from './strides.js';
import { normalizeAxis } from './shape.js';

/**
 * Slice specification for a single dimension
 */
export interface SliceSpec {
  readonly start: number | null;
  readonly stop: number | null;
  readonly step: number;
}

/**
 * Index type for a single dimension:
 * - number: single element selection
 * - SliceSpec: range selection
 * - ':' string: select all (equivalent to SliceSpec with nulls)
 * - null/undefined: select all (newaxis when used with expand_dims)
 */
export type DimensionIndex = number | SliceSpec | ':' | null | undefined;

/**
 * Full index specification for an array
 */
export type IndexSpec = readonly DimensionIndex[];

/**
 * Result of applying an index to a shape
 */
export interface IndexResult {
  readonly shape: Shape;
  readonly strides: Strides;
  readonly offset: number;
  readonly isView: boolean;
}

/**
 * Create a slice specification
 */
export function slice(
  start: number | null = null,
  stop: number | null = null,
  step: number = 1
): SliceSpec {
  if (step === 0) {
    throw new Error('Slice step cannot be zero');
  }
  return { start, stop, step };
}

/**
 * Parse a string slice notation like "1:5:2" or "::2"
 */
export function parseSliceString(s: string): SliceSpec {
  if (s === ':') {
    return slice();
  }

  const parts = s.split(':');
  if (parts.length < 1 || parts.length > 3) {
    throw new Error(`Invalid slice string: ${s}`);
  }

  const parseNumber = (part: string | undefined): number | null => {
    if (part === undefined || part === '') {
      return null;
    }
    const num = parseInt(part, 10);
    if (isNaN(num)) {
      throw new Error(`Invalid slice component: ${part}`);
    }
    return num;
  };

  const start = parseNumber(parts[0]);
  const stop = parseNumber(parts[1]);
  const step = parts.length > 2 ? (parseNumber(parts[2]) ?? 1) : 1;

  if (step === 0) {
    throw new Error('Slice step cannot be zero');
  }

  return slice(start, stop, step);
}

/**
 * Normalize a dimension index (handle negatives, validate bounds)
 */
export function normalizeIndex(index: number, size: number): number {
  const normalized = index < 0 ? index + size : index;
  if (normalized < 0 || normalized >= size) {
    throw new Error(`Index ${index} is out of bounds for axis with size ${size}`);
  }
  return normalized;
}

/**
 * Normalize a slice for a dimension of given size
 * Returns [start, stop, step, length]
 */
export function normalizeSlice(
  sliceSpec: SliceSpec,
  size: number
): [number, number, number, number] {
  const { step } = sliceSpec;
  let { start, stop } = sliceSpec;

  // Handle negative step
  if (step > 0) {
    // Positive step
    if (start === null) {
      start = 0;
    } else if (start < 0) {
      start = Math.max(0, start + size);
    } else {
      start = Math.min(start, size);
    }

    if (stop === null) {
      stop = size;
    } else if (stop < 0) {
      stop = Math.max(0, stop + size);
    } else {
      stop = Math.min(stop, size);
    }

    const length = Math.max(0, Math.ceil((stop - start) / step));
    return [start, stop, step, length];
  } else {
    // Negative step
    if (start === null) {
      start = size - 1;
    } else if (start < 0) {
      start = Math.max(-1, start + size);
    } else {
      start = Math.min(start, size - 1);
    }

    if (stop === null) {
      stop = -1;
    } else if (stop < 0) {
      stop = Math.max(-1, stop + size);
    } else {
      stop = Math.min(stop, size - 1);
    }

    const length = Math.max(0, Math.ceil((start - stop) / -step));
    return [start, stop, step, length];
  }
}

/**
 * Check if a dimension index selects a single element (reduces dimension)
 */
export function isScalarIndex(index: DimensionIndex): index is number {
  return typeof index === 'number';
}

/**
 * Check if a dimension index is a slice (preserves dimension)
 */
export function isSliceIndex(
  index: DimensionIndex
): index is SliceSpec | ':' | null | undefined {
  return !isScalarIndex(index);
}

/**
 * Normalize a dimension index to a canonical form
 */
export function normalizeDimensionIndex(
  index: DimensionIndex,
  size: number
): { type: 'scalar'; value: number } | { type: 'slice'; value: SliceSpec } {
  if (typeof index === 'number') {
    return { type: 'scalar', value: normalizeIndex(index, size) };
  }

  if (typeof index === 'string') {
    if (index === ':') {
      return { type: 'slice', value: slice() };
    }
    return { type: 'slice', value: parseSliceString(index) };
  }

  if (index === null || index === undefined) {
    return { type: 'slice', value: slice() };
  }

  return { type: 'slice', value: index };
}

/**
 * Apply an index specification to a shape and strides
 */
export function applyIndex(
  shape: Shape,
  strides: Strides,
  indexSpec: IndexSpec
): IndexResult {
  if (indexSpec.length > shape.length) {
    throw new Error(
      `Too many indices (${indexSpec.length}) for array with ${shape.length} dimensions`
    );
  }

  const resultShape: MutableShape = [];
  const resultStrides: MutableStrides = [];
  let offset = 0;

  // Process each dimension
  for (let i = 0; i < shape.length; i++) {
    const size = shape[i];
    const stride = strides[i];
    const index = i < indexSpec.length ? indexSpec[i] : ':';

    if (size === undefined || stride === undefined) {
      throw new Error(`Invalid shape or stride at dimension ${i}`);
    }

    const normalized = normalizeDimensionIndex(index, size);

    if (normalized.type === 'scalar') {
      // Single element - this dimension is removed
      offset += normalized.value * stride;
    } else {
      // Slice - dimension is preserved (possibly with different size)
      const [start, _stop, step, length] = normalizeSlice(normalized.value, size);
      offset += start * stride;
      resultShape.push(length);
      resultStrides.push(stride * step);
    }
  }

  return {
    shape: resultShape,
    strides: resultStrides,
    offset,
    isView: true,
  };
}

/**
 * Calculate the flat indices for a full slice operation
 * Used when a view cannot be created and data must be copied
 */
export function getSliceIndices(
  shape: Shape,
  strides: Strides,
  indexSpec: IndexSpec,
  baseOffset: number = 0
): number[] {
  const result = applyIndex(shape, strides, indexSpec);
  const indices: number[] = [];

  // Generate all indices in the result shape
  const totalSize = result.shape.reduce((a, b) => a * b, 1);

  if (totalSize === 0) {
    return [];
  }

  const currentIndex = new Array(result.shape.length).fill(0);

  for (let i = 0; i < totalSize; i++) {
    // Calculate flat index
    let flatIndex = baseOffset + result.offset;
    for (let j = 0; j < result.shape.length; j++) {
      const idx = currentIndex[j];
      const stride = result.strides[j];
      if (idx !== undefined && stride !== undefined) {
        flatIndex += idx * stride;
      }
    }
    indices.push(flatIndex);

    // Increment current index
    for (let j = result.shape.length - 1; j >= 0; j--) {
      const current = currentIndex[j];
      const shapeDim = result.shape[j];

      if (current === undefined || shapeDim === undefined) {
        continue;
      }

      currentIndex[j] = current + 1;
      if (currentIndex[j] < shapeDim) {
        break;
      }
      currentIndex[j] = 0;
    }
  }

  return indices;
}

/**
 * Parse a multi-dimensional index specification
 * Supports arrays like [[0, 2], ':'] for x[[0, 2], :]
 */
export function parseIndexSpec(spec: readonly (DimensionIndex | string)[]): IndexSpec {
  return spec.map((item) => {
    if (typeof item === 'string' && item !== ':' && item.includes(':')) {
      return parseSliceString(item);
    }
    return item as DimensionIndex;
  });
}

/**
 * Generate indices for iteration over an array in a given order
 */
export class IndexIterator implements IterableIterator<readonly number[]> {
  private readonly _shape: Shape;
  private readonly _order: 'C' | 'F';
  private readonly _size: number;
  private readonly _indices: number[];
  private _position: number;

  constructor(shape: Shape, order: 'C' | 'F' = 'C') {
    this._shape = shape;
    this._order = order;
    this._size = shape.reduce((a, b) => a * b, 1);
    this._indices = new Array(shape.length).fill(0);
    this._position = 0;
  }

  public [Symbol.iterator](): IterableIterator<readonly number[]> {
    return this;
  }

  public next(): IteratorResult<readonly number[]> {
    if (this._position >= this._size) {
      return { done: true, value: undefined };
    }

    const result = [...this._indices];
    this._position++;

    // Increment indices based on order
    if (this._order === 'C') {
      // Row-major: rightmost index changes fastest
      for (let i = this._indices.length - 1; i >= 0; i--) {
        const current = this._indices[i];
        const shapeDim = this._shape[i];

        if (current === undefined || shapeDim === undefined) {
          throw new Error('Invalid index state');
        }

        const newIndex = current + 1;
        this._indices[i] = newIndex;
        if (newIndex < shapeDim) {
          break;
        }
        this._indices[i] = 0;
      }
    } else {
      // Column-major: leftmost index changes fastest
      for (let i = 0; i < this._indices.length; i++) {
        const current = this._indices[i];
        const shapeDim = this._shape[i];

        if (current === undefined || shapeDim === undefined) {
          throw new Error('Invalid index state');
        }

        const newIndex = current + 1;
        this._indices[i] = newIndex;
        if (newIndex < shapeDim) {
          break;
        }
        this._indices[i] = 0;
      }
    }

    return { done: false, value: result };
  }

  public reset(): void {
    this._indices.fill(0);
    this._position = 0;
  }
}

/**
 * Convert multi-dimensional indices to a flat index
 */
export function indicesToFlat(indices: readonly number[], strides: Strides): number {
  let flat = 0;
  for (let i = 0; i < indices.length; i++) {
    const idx = indices[i];
    const stride = strides[i];
    if (idx !== undefined && stride !== undefined) {
      flat += idx * stride;
    }
  }
  return flat;
}

/**
 * Convert a flat index to multi-dimensional indices
 */
export function flatToIndices(flat: number, shape: Shape): number[] {
  const indices: number[] = new Array(shape.length);
  let remaining = flat;

  for (let i = shape.length - 1; i >= 0; i--) {
    const dim = shape[i];
    if (dim === undefined) {
      throw new Error(`Invalid shape dimension at index ${i}`);
    }
    indices[i] = remaining % dim;
    remaining = Math.floor(remaining / dim);
  }

  return indices;
}

/**
 * Validate that indices are within bounds for a shape
 */
export function validateIndices(indices: readonly number[], shape: Shape): void {
  if (indices.length !== shape.length) {
    throw new Error(
      `Index dimension ${indices.length} does not match array dimension ${shape.length}`
    );
  }

  for (let i = 0; i < indices.length; i++) {
    const idx = indices[i];
    const dim = shape[i];

    if (idx === undefined || dim === undefined) {
      throw new Error(`Invalid index or shape at dimension ${i}`);
    }

    if (idx < 0 || idx >= dim) {
      throw new Error(`Index ${idx} is out of bounds for dimension ${i} with size ${dim}`);
    }
  }
}

/**
 * Apply fancy indexing with integer arrays
 * Returns the flat indices to gather
 */
export function applyFancyIndex(
  shape: Shape,
  strides: Strides,
  indexArrays: readonly (readonly number[])[],
  baseOffset: number = 0
): { indices: number[]; shape: Shape } {
  if (indexArrays.length === 0) {
    throw new Error('At least one index array required');
  }

  // All index arrays must have the same length
  const length = indexArrays[0]?.length ?? 0;
  for (let i = 1; i < indexArrays.length; i++) {
    const arr = indexArrays[i];
    if (arr === undefined || arr.length !== length) {
      throw new Error('All index arrays must have the same length');
    }
  }

  const indices: number[] = new Array(length);

  for (let i = 0; i < length; i++) {
    let offset = baseOffset;

    for (let dim = 0; dim < indexArrays.length; dim++) {
      const indexArr = indexArrays[dim];
      const idx = indexArr?.[i];
      const dimSize = shape[dim];
      const stride = strides[dim];

      if (idx === undefined || dimSize === undefined || stride === undefined) {
        throw new Error('Invalid index array or shape');
      }

      const normalizedIdx = normalizeIndex(idx, dimSize);
      offset += normalizedIdx * stride;
    }

    indices[i] = offset;
  }

  return { indices, shape: [length] };
}

/**
 * Apply boolean mask indexing
 * Returns the flat indices where mask is true
 */
export function applyBooleanMask(
  shape: Shape,
  strides: Strides,
  mask: readonly boolean[],
  baseOffset: number = 0
): { indices: number[]; count: number } {
  const totalSize = shape.reduce((a, b) => a * b, 1);

  if (mask.length !== totalSize) {
    throw new Error(
      `Boolean mask length ${mask.length} does not match array size ${totalSize}`
    );
  }

  const indices: number[] = [];
  const currentIndex = new Array(shape.length).fill(0);

  for (let i = 0; i < totalSize; i++) {
    if (mask[i] === true) {
      let flatIndex = baseOffset;
      for (let j = 0; j < shape.length; j++) {
        const idx = currentIndex[j];
        const stride = strides[j];
        if (idx !== undefined && stride !== undefined) {
          flatIndex += idx * stride;
        }
      }
      indices.push(flatIndex);
    }

    // Increment current index
    for (let j = shape.length - 1; j >= 0; j--) {
      const current = currentIndex[j];
      const shapeDim = shape[j];

      if (current === undefined || shapeDim === undefined) {
        continue;
      }

      currentIndex[j] = current + 1;
      if (currentIndex[j] < shapeDim) {
        break;
      }
      currentIndex[j] = 0;
    }
  }

  return { indices, count: indices.length };
}

/**
 * Axis selection utilities
 */
export function selectAxis(shape: Shape, axis: number, index: number): Shape {
  const normalizedAxis = normalizeAxis(axis, shape.length);
  normalizeIndex(index, shape[normalizedAxis] ?? 0); // Validate index is in bounds

  const result: MutableShape = [];
  for (let i = 0; i < shape.length; i++) {
    if (i !== normalizedAxis) {
      const dim = shape[i];
      if (dim !== undefined) {
        result.push(dim);
      }
    }
  }

  return result;
}
