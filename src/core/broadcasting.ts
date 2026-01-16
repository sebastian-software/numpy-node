/**
 * Broadcasting utilities for numpy-node
 * Implements NumPy-compatible broadcasting rules
 */

import type { Shape, MutableShape } from './shape.js';
import type { Strides, MutableStrides } from './strides.js';

/**
 * Broadcasting result containing the output shape and information
 * about how each input maps to the output
 */
export interface BroadcastResult {
  readonly shape: Shape;
  readonly inputMappings: readonly BroadcastMapping[];
}

/**
 * Mapping information for how an input broadcasts to the output
 */
export interface BroadcastMapping {
  readonly originalShape: Shape;
  readonly broadcastShape: Shape;
  readonly strideMultipliers: readonly number[];
}

/**
 * Check if two shapes are broadcastable
 */
export function areBroadcastable(shape1: Shape, shape2: Shape): boolean {
  const maxDim = Math.max(shape1.length, shape2.length);

  for (let i = 0; i < maxDim; i++) {
    const dim1 = shape1[shape1.length - 1 - i] ?? 1;
    const dim2 = shape2[shape2.length - 1 - i] ?? 1;

    if (dim1 !== dim2 && dim1 !== 1 && dim2 !== 1) {
      return false;
    }
  }

  return true;
}

/**
 * Check if multiple shapes are mutually broadcastable
 */
export function areAllBroadcastable(shapes: readonly Shape[]): boolean {
  if (shapes.length <= 1) {
    return true;
  }

  try {
    broadcastShapes(...shapes);
    return true;
  } catch {
    return false;
  }
}

/**
 * Calculate the broadcast shape of two shapes
 */
export function broadcastTwoShapes(shape1: Shape, shape2: Shape): Shape {
  const maxDim = Math.max(shape1.length, shape2.length);
  const result: MutableShape = new Array(maxDim);

  for (let i = 0; i < maxDim; i++) {
    const dim1 = shape1[shape1.length - 1 - i] ?? 1;
    const dim2 = shape2[shape2.length - 1 - i] ?? 1;

    if (dim1 === dim2) {
      result[maxDim - 1 - i] = dim1;
    } else if (dim1 === 1) {
      result[maxDim - 1 - i] = dim2;
    } else if (dim2 === 1) {
      result[maxDim - 1 - i] = dim1;
    } else {
      throw new Error(
        `Cannot broadcast shapes [${shape1.join(', ')}] and [${shape2.join(', ')}]: ` +
          `incompatible dimensions ${dim1} and ${dim2}`
      );
    }
  }

  return result;
}

/**
 * Calculate the broadcast shape of multiple shapes
 */
export function broadcastShapes(...shapes: readonly Shape[]): Shape {
  if (shapes.length === 0) {
    return [];
  }

  const firstShape = shapes[0];
  if (firstShape === undefined) {
    return [];
  }

  let result: Shape = firstShape;

  for (let i = 1; i < shapes.length; i++) {
    const shape = shapes[i];
    if (shape !== undefined) {
      result = broadcastTwoShapes(result, shape);
    }
  }

  return result;
}

/**
 * Get detailed broadcast information for multiple shapes
 */
export function getBroadcastInfo(...shapes: readonly Shape[]): BroadcastResult {
  const outputShape = broadcastShapes(...shapes);
  const inputMappings: BroadcastMapping[] = [];

  for (const shape of shapes) {
    const mapping = getBroadcastMapping(shape, outputShape);
    inputMappings.push(mapping);
  }

  return {
    shape: outputShape,
    inputMappings,
  };
}

/**
 * Get the mapping for how a single shape broadcasts to a target shape
 */
export function getBroadcastMapping(shape: Shape, targetShape: Shape): BroadcastMapping {
  const offset = targetShape.length - shape.length;
  const broadcastShape: MutableShape = new Array(targetShape.length);
  const strideMultipliers: MutableStrides = new Array(targetShape.length);

  // Fill leading dimensions (dimensions added by broadcasting)
  for (let i = 0; i < offset; i++) {
    const targetDim = targetShape[i];
    if (targetDim === undefined) {
      throw new Error(`Invalid target shape dimension at index ${i}`);
    }
    broadcastShape[i] = targetDim;
    strideMultipliers[i] = 0; // No data in input for this dimension
  }

  // Map existing dimensions
  for (let i = 0; i < shape.length; i++) {
    const dim = shape[i];
    const targetDim = targetShape[i + offset];

    if (dim === undefined || targetDim === undefined) {
      throw new Error('Invalid shape dimension');
    }

    broadcastShape[i + offset] = targetDim;

    if (dim === targetDim) {
      strideMultipliers[i + offset] = 1;
    } else if (dim === 1) {
      strideMultipliers[i + offset] = 0; // Broadcast this dimension
    } else {
      throw new Error(
        `Cannot broadcast shape [${shape.join(', ')}] to [${targetShape.join(', ')}]`
      );
    }
  }

  return {
    originalShape: shape,
    broadcastShape,
    strideMultipliers,
  };
}

/**
 * Calculate broadcast strides for a shape being broadcast to a target shape
 */
export function getBroadcastStrides(
  shape: Shape,
  strides: Strides,
  targetShape: Shape
): Strides {
  const offset = targetShape.length - shape.length;
  const result: MutableStrides = new Array(targetShape.length);

  // Leading dimensions get stride 0
  for (let i = 0; i < offset; i++) {
    result[i] = 0;
  }

  // Map existing dimensions
  for (let i = 0; i < shape.length; i++) {
    const dim = shape[i];
    const targetDim = targetShape[i + offset];
    const stride = strides[i];

    if (dim === undefined || targetDim === undefined || stride === undefined) {
      throw new Error('Invalid shape or stride');
    }

    if (dim === targetDim) {
      result[i + offset] = stride;
    } else if (dim === 1) {
      result[i + offset] = 0; // Broadcast: repeat this element
    } else {
      throw new Error(
        `Cannot broadcast shape [${shape.join(', ')}] to [${targetShape.join(', ')}]`
      );
    }
  }

  return result;
}

/**
 * Iterator for broadcast indices
 * Yields indices in the broadcast output space
 */
export class BroadcastIterator implements IterableIterator<readonly number[]> {
  private readonly _shape: Shape;
  private readonly _size: number;
  private readonly _indices: number[];
  private _position: number;

  constructor(shape: Shape) {
    this._shape = shape;
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

    // Increment indices (rightmost first)
    for (let i = this._indices.length - 1; i >= 0; i--) {
      const currentIndex = this._indices[i];
      const shapeDim = this._shape[i];

      if (currentIndex === undefined || shapeDim === undefined) {
        throw new Error('Invalid index state');
      }

      const newIndex = currentIndex + 1;
      this._indices[i] = newIndex;

      if (newIndex < shapeDim) {
        break;
      }
      this._indices[i] = 0;
    }

    return { done: false, value: result };
  }

  public reset(): void {
    this._indices.fill(0);
    this._position = 0;
  }
}

/**
 * Map broadcast output index to input index
 */
export function mapBroadcastIndex(
  outputIndex: readonly number[],
  inputShape: Shape,
  targetShape: Shape
): number[] {
  const offset = targetShape.length - inputShape.length;
  const result: number[] = new Array(inputShape.length);

  for (let i = 0; i < inputShape.length; i++) {
    const inputDim = inputShape[i];
    const outputIdx = outputIndex[i + offset];

    if (inputDim === undefined || outputIdx === undefined) {
      throw new Error('Invalid index or shape');
    }

    // If input dimension is 1, it broadcasts and we always use index 0
    result[i] = inputDim === 1 ? 0 : outputIdx;
  }

  return result;
}

/**
 * Calculate flat offset for a broadcast index
 */
export function broadcastIndexToOffset(
  outputIndex: readonly number[],
  inputShape: Shape,
  inputStrides: Strides,
  targetShape: Shape
): number {
  const inputIndex = mapBroadcastIndex(outputIndex, inputShape, targetShape);
  let offset = 0;

  for (let i = 0; i < inputIndex.length; i++) {
    const idx = inputIndex[i];
    const stride = inputStrides[i];

    if (idx === undefined || stride === undefined) {
      throw new Error('Invalid index or stride');
    }

    offset += idx * stride;
  }

  return offset;
}

/**
 * Check if a shape needs broadcasting to match a target
 */
export function needsBroadcast(shape: Shape, targetShape: Shape): boolean {
  if (shape.length !== targetShape.length) {
    return true;
  }

  for (let i = 0; i < shape.length; i++) {
    if (shape[i] !== targetShape[i]) {
      return true;
    }
  }

  return false;
}

/**
 * Get the axes that are broadcast (have size 1 in input but larger in output)
 */
export function getBroadcastAxes(shape: Shape, targetShape: Shape): number[] {
  const offset = targetShape.length - shape.length;
  const axes: number[] = [];

  // Leading dimensions are always broadcast
  for (let i = 0; i < offset; i++) {
    axes.push(i);
  }

  // Check remaining dimensions
  for (let i = 0; i < shape.length; i++) {
    if (shape[i] === 1 && (targetShape[i + offset] ?? 0) > 1) {
      axes.push(i + offset);
    }
  }

  return axes;
}
