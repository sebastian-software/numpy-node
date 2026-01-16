/**
 * Shape utilities for np-ts
 * Handles array shape validation, manipulation, and calculations
 */

/**
 * Shape type - immutable tuple of dimension sizes
 */
export type Shape = readonly number[];

/**
 * Mutable shape for internal operations
 */
export type MutableShape = number[];

/**
 * Validate that a shape is valid (all non-negative integers)
 */
export function validateShape(shape: Shape): void {
  for (let i = 0; i < shape.length; i++) {
    const dim = shape[i];
    if (dim === undefined || !Number.isInteger(dim) || dim < 0) {
      throw new Error(`Invalid shape dimension at index ${i}: ${String(dim)}`);
    }
  }
}

/**
 * Calculate the total number of elements from a shape
 */
export function shapeToSize(shape: Shape): number {
  if (shape.length === 0) {
    return 1; // Scalar
  }
  let size = 1;
  for (const dim of shape) {
    size *= dim;
  }
  return size;
}

/**
 * Get the number of dimensions (rank) of a shape
 */
export function shapeToNdim(shape: Shape): number {
  return shape.length;
}

/**
 * Check if two shapes are equal
 */
export function shapesEqual(shape1: Shape, shape2: Shape): boolean {
  if (shape1.length !== shape2.length) {
    return false;
  }
  for (let i = 0; i < shape1.length; i++) {
    if (shape1[i] !== shape2[i]) {
      return false;
    }
  }
  return true;
}

/**
 * Normalize a shape by converting -1 dimension to the inferred value
 * Used for reshape operations
 */
export function normalizeShape(newShape: Shape, currentSize: number): Shape {
  let unknownIndex = -1;
  let knownSize = 1;

  for (let i = 0; i < newShape.length; i++) {
    const dim = newShape[i];
    if (dim === undefined) {
      throw new Error(`Invalid shape dimension at index ${i}`);
    }
    if (dim === -1) {
      if (unknownIndex !== -1) {
        throw new Error('Can only specify one unknown dimension (-1)');
      }
      unknownIndex = i;
    } else if (dim < 0) {
      throw new Error(`Invalid shape dimension at index ${i}: ${dim}`);
    } else {
      knownSize *= dim;
    }
  }

  if (unknownIndex === -1) {
    if (knownSize !== currentSize) {
      throw new Error(
        `Cannot reshape array of size ${currentSize} into shape [${newShape.join(', ')}]`
      );
    }
    return newShape;
  }

  if (currentSize % knownSize !== 0) {
    throw new Error(
      `Cannot reshape array of size ${currentSize} into shape [${newShape.join(', ')}]`
    );
  }

  const inferredDim = currentSize / knownSize;
  const result: MutableShape = [...newShape];
  result[unknownIndex] = inferredDim;
  return result;
}

/**
 * Infer the shape of nested JavaScript arrays
 */
export function inferShape(data: unknown): Shape {
  const shape: MutableShape = [];
  let current: unknown = data;

  while (Array.isArray(current)) {
    shape.push(current.length);
    if (current.length === 0) {
      break;
    }
    current = current[0];
  }

  return shape;
}

/**
 * Validate that nested array data matches a shape
 */
export function validateDataShape(data: unknown, shape: Shape, depth: number = 0): void {
  if (depth >= shape.length) {
    if (Array.isArray(data)) {
      throw new Error('Data has more dimensions than specified shape');
    }
    return;
  }

  if (!Array.isArray(data)) {
    throw new Error(`Expected array at depth ${depth}, got ${typeof data}`);
  }

  const expectedLength = shape[depth];
  if (data.length !== expectedLength) {
    throw new Error(
      `Shape mismatch at depth ${depth}: expected ${String(expectedLength)}, got ${data.length}`
    );
  }

  for (const element of data) {
    validateDataShape(element, shape, depth + 1);
  }
}

/**
 * Flatten nested array to 1D
 */
export function flattenData<T>(data: unknown): T[] {
  const result: T[] = [];

  function recurse(current: unknown): void {
    if (Array.isArray(current)) {
      for (const element of current) {
        recurse(element);
      }
    } else {
      result.push(current as T);
    }
  }

  recurse(data);
  return result;
}

/**
 * Get the transpose of a shape (reverse dimensions)
 */
export function transposeShape(shape: Shape): Shape {
  return [...shape].reverse();
}

/**
 * Get the shape after applying axes permutation
 */
export function permuteShape(shape: Shape, axes: readonly number[]): Shape {
  if (axes.length !== shape.length) {
    throw new Error(
      `Axes length ${axes.length} does not match shape dimensions ${shape.length}`
    );
  }

  const seen = new Set<number>();
  for (const axis of axes) {
    if (axis < 0 || axis >= shape.length) {
      throw new Error(`Invalid axis ${axis} for shape with ${shape.length} dimensions`);
    }
    if (seen.has(axis)) {
      throw new Error(`Duplicate axis ${axis} in permutation`);
    }
    seen.add(axis);
  }

  const result: MutableShape = new Array(shape.length);
  for (let i = 0; i < axes.length; i++) {
    const axis = axes[i];
    if (axis === undefined) {
      throw new Error(`Invalid axis at index ${i}`);
    }
    const shapeValue = shape[axis];
    if (shapeValue === undefined) {
      throw new Error(`Invalid shape value at axis ${axis}`);
    }
    result[i] = shapeValue;
  }
  return result;
}

/**
 * Normalize a negative axis index
 */
export function normalizeAxis(axis: number, ndim: number): number {
  if (axis < -ndim || axis >= ndim) {
    throw new Error(`Axis ${axis} is out of bounds for array with ${ndim} dimensions`);
  }
  return axis < 0 ? axis + ndim : axis;
}

/**
 * Normalize multiple axes
 */
export function normalizeAxes(axes: readonly number[], ndim: number): number[] {
  return axes.map((axis) => normalizeAxis(axis, ndim));
}

/**
 * Get shape after reducing along specified axes
 */
export function getReducedShape(
  shape: Shape,
  axes: readonly number[],
  keepdims: boolean
): Shape {
  const normalizedAxes = normalizeAxes(axes, shape.length);
  const axesSet = new Set(normalizedAxes);

  if (keepdims) {
    return shape.map((dim, i) => (axesSet.has(i) ? 1 : dim));
  }

  const result: MutableShape = [];
  for (let i = 0; i < shape.length; i++) {
    if (!axesSet.has(i)) {
      const dim = shape[i];
      if (dim !== undefined) {
        result.push(dim);
      }
    }
  }
  return result;
}

/**
 * Get shape after squeezing (removing dimensions of size 1)
 */
export function getSqueezeShape(shape: Shape, axes?: readonly number[]): Shape {
  if (axes === undefined) {
    return shape.filter((dim) => dim !== 1);
  }

  const normalizedAxes = normalizeAxes(axes, shape.length);
  const axesSet = new Set(normalizedAxes);

  for (const axis of axesSet) {
    if (shape[axis] !== 1) {
      throw new Error(`Cannot squeeze axis ${axis} with size ${String(shape[axis])}`);
    }
  }

  const result: MutableShape = [];
  for (let i = 0; i < shape.length; i++) {
    if (!axesSet.has(i)) {
      const dim = shape[i];
      if (dim !== undefined) {
        result.push(dim);
      }
    }
  }
  return result;
}

/**
 * Get shape after expanding dimensions
 */
export function getExpandDimsShape(shape: Shape, axis: number): Shape {
  const normalizedAxis = axis < 0 ? axis + shape.length + 1 : axis;
  if (normalizedAxis < 0 || normalizedAxis > shape.length) {
    throw new Error(
      `Axis ${axis} is out of bounds for expand_dims on array with ${shape.length} dimensions`
    );
  }

  const result: MutableShape = [...shape];
  result.splice(normalizedAxis, 0, 1);
  return result;
}

/**
 * Calculate the shape of concatenated arrays along an axis
 */
export function getConcatShape(shapes: readonly Shape[], axis: number): Shape {
  if (shapes.length === 0) {
    throw new Error('Cannot concatenate zero arrays');
  }

  const firstShape = shapes[0];
  if (firstShape === undefined) {
    throw new Error('First shape is undefined');
  }

  const ndim = firstShape.length;
  const normalizedAxis = normalizeAxis(axis, ndim);

  // Validate all shapes have same dimensions except along concat axis
  for (let i = 1; i < shapes.length; i++) {
    const shape = shapes[i];
    if (shape === undefined || shape.length !== ndim) {
      throw new Error(
        `All arrays must have the same number of dimensions. Array 0 has ${ndim} dimensions, array ${i} has ${shape?.length ?? 0} dimensions`
      );
    }
    for (let j = 0; j < ndim; j++) {
      if (j !== normalizedAxis && shape[j] !== firstShape[j]) {
        throw new Error(
          `All arrays must have the same shape except along the concatenation axis. ` +
            `Array 0 has shape [${firstShape.join(', ')}], array ${i} has shape [${shape.join(', ')}]`
        );
      }
    }
  }

  // Calculate result shape
  const result: MutableShape = [...firstShape];
  let totalDim = firstShape[normalizedAxis] ?? 0;
  for (let i = 1; i < shapes.length; i++) {
    const shape = shapes[i];
    if (shape !== undefined) {
      totalDim += shape[normalizedAxis] ?? 0;
    }
  }
  result[normalizedAxis] = totalDim;
  return result;
}

/**
 * Calculate the shape of stacked arrays along a new axis
 */
export function getStackShape(shapes: readonly Shape[], axis: number): Shape {
  if (shapes.length === 0) {
    throw new Error('Cannot stack zero arrays');
  }

  const firstShape = shapes[0];
  if (firstShape === undefined) {
    throw new Error('First shape is undefined');
  }

  // Validate all shapes are equal
  for (let i = 1; i < shapes.length; i++) {
    const shape = shapes[i];
    if (shape === undefined || !shapesEqual(shape, firstShape)) {
      throw new Error(
        `All arrays must have the same shape for stack. ` +
          `Array 0 has shape [${firstShape.join(', ')}], array ${i} has shape [${shape?.join(', ') ?? ''}]`
      );
    }
  }

  // Insert new dimension
  const normalizedAxis = axis < 0 ? axis + firstShape.length + 1 : axis;
  if (normalizedAxis < 0 || normalizedAxis > firstShape.length) {
    throw new Error(
      `Axis ${axis} is out of bounds for stack on arrays with ${firstShape.length} dimensions`
    );
  }

  const result: MutableShape = [...firstShape];
  result.splice(normalizedAxis, 0, shapes.length);
  return result;
}
