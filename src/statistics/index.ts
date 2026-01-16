/**
 * Statistical functions for np-ts
 * Provides statistical operations on arrays
 */

import { NDArray, type DTypeName, IndexIterator } from '../core/index.js';
import { sum, subtract, multiply, divide, sqrt } from '../ops/index.js';

/**
 * Compute the arithmetic mean along the specified axis
 */
export function mean<D extends DTypeName>(
  a: NDArray<D>,
  axis?: number,
  keepdims: boolean = false
): number | NDArray<'float64'> {
  if (axis === undefined) {
    const total = sum(a) as number;
    return total / a.size;
  }

  const sumResult = sum(a, axis, keepdims) as NDArray;
  const normalizedAxis = axis < 0 ? axis + a.ndim : axis;
  const axisSize = a.shape[normalizedAxis]!;
  return divide(sumResult, axisSize) as NDArray<'float64'>;
}

/**
 * Compute the weighted average along the specified axis
 */
export function average<D extends DTypeName>(
  a: NDArray<D>,
  options: { axis?: number; weights?: NDArray; returned?: boolean } = {}
): number | NDArray<'float64'> | [number | NDArray<'float64'>, number | NDArray<'float64'>] {
  const { axis, weights, returned = false } = options;

  if (weights === undefined) {
    const avg = mean(a, axis);
    if (returned) {
      const wsum = axis === undefined ? a.size : a.shape[axis < 0 ? axis + a.ndim : axis]!;
      return [avg, wsum] as [number | NDArray<'float64'>, number | NDArray<'float64'>];
    }
    return avg;
  }

  // Weighted average
  const weighted = multiply(a, weights);
  const weightedSum = sum(weighted, axis);
  const weightSum = sum(weights, axis);

  // Handle scalar case (no axis specified)
  if (typeof weightedSum === 'number' && typeof weightSum === 'number') {
    const avg = weightedSum / weightSum;
    if (returned) {
      return [avg, weightSum];
    }
    return avg;
  }

  // Handle array case (axis specified)
  const avg = divide(weightedSum as NDArray, weightSum as NDArray | number);

  if (returned) {
    return [avg, weightSum] as [number | NDArray<'float64'>, number | NDArray<'float64'>];
  }
  return avg as number | NDArray<'float64'>;
}

/**
 * Compute the variance along the specified axis
 */
export function variance<D extends DTypeName>(
  a: NDArray<D>,
  axis?: number,
  ddof: number = 0,
  keepdims: boolean = false
): number | NDArray<'float64'> {
  const meanVal = mean(a, axis, true);

  if (axis === undefined) {
    let sumSq = 0;
    for (const val of a) {
      const diff = Number(val) - (meanVal as number);
      sumSq += diff * diff;
    }
    return sumSq / (a.size - ddof);
  }

  // Compute variance along axis
  const normalizedAxis = axis < 0 ? axis + a.ndim : axis;
  const axisSize = a.shape[normalizedAxis]!;
  const outShape = keepdims
    ? a.shape.map((d, i) => (i === normalizedAxis ? 1 : d))
    : a.shape.filter((_, i) => i !== normalizedAxis);

  if (outShape.length === 0) {
    outShape.push(1);
  }

  const result = NDArray.zeros<'float64'>(outShape, { dtype: 'float64' });
  const meanArr = meanVal as NDArray<'float64'>;
  // Extract mean values as flat array for reliable indexing
  const meanValues = meanArr.toFlatArray().map(Number);

  let outIdx = 0;
  for (const outIndices of new IndexIterator(outShape)) {
    // Build indices template for input array
    const inIndices: number[] = [];
    let j = 0;
    for (let i = 0; i < a.ndim; i++) {
      if (i === normalizedAxis) {
        inIndices.push(0);
      } else {
        const idx = keepdims ? outIndices[i] : outIndices[j];
        inIndices.push(idx ?? 0);
        if (!keepdims) j++;
      }
    }

    const meanValue = meanValues[outIdx]!;
    let sumSq = 0;

    for (let k = 0; k < axisSize; k++) {
      inIndices[normalizedAxis] = k;
      const val = Number(a.at(...inIndices));
      const diff = val - meanValue;
      sumSq += diff * diff;
    }

    result.setFlat(outIdx++, sumSq / (axisSize - ddof));
  }

  return result;
}

/**
 * Alias for variance
 */
export const var_ = variance;

/**
 * Compute the standard deviation along the specified axis
 */
export function std<D extends DTypeName>(
  a: NDArray<D>,
  axis?: number,
  ddof: number = 0,
  keepdims: boolean = false
): number | NDArray<'float64'> {
  const varResult = variance(a, axis, ddof, keepdims);

  if (typeof varResult === 'number') {
    return Math.sqrt(varResult);
  }

  return sqrt(varResult) as NDArray<'float64'>;
}

/**
 * Compute the median along the specified axis
 */
export function median<D extends DTypeName>(
  a: NDArray<D>,
  axis?: number,
  keepdims: boolean = false
): number | NDArray<'float64'> {
  if (axis === undefined) {
    // Flatten and sort
    const values = a.toFlatArray().map(Number);
    values.sort((x, y) => x - y);
    const mid = Math.floor(values.length / 2);
    if (values.length % 2 === 0) {
      return (values[mid - 1]! + values[mid]!) / 2;
    }
    return values[mid]!;
  }

  const normalizedAxis = axis < 0 ? axis + a.ndim : axis;
  const axisSize = a.shape[normalizedAxis]!;
  const outShape = keepdims
    ? a.shape.map((d, i) => (i === normalizedAxis ? 1 : d))
    : a.shape.filter((_, i) => i !== normalizedAxis);

  if (outShape.length === 0) {
    outShape.push(1);
  }

  const result = NDArray.zeros<'float64'>(outShape, { dtype: 'float64' });

  let outIdx = 0;
  for (const outIndices of new IndexIterator(outShape)) {
    const inIndices: number[] = [];
    let j = 0;
    for (let i = 0; i < a.ndim; i++) {
      if (i === normalizedAxis) {
        inIndices.push(0);
      } else {
        const idx = keepdims ? (i < normalizedAxis ? outIndices[i] : outIndices[i]) : outIndices[j]!;
        inIndices.push(idx ?? 0);
        if (!keepdims) j++;
      }
    }

    // Collect values along axis
    const values: number[] = [];
    for (let k = 0; k < axisSize; k++) {
      inIndices[normalizedAxis] = k;
      values.push(Number(a.at(...inIndices)));
    }

    values.sort((x, y) => x - y);
    const mid = Math.floor(values.length / 2);
    const medianVal =
      values.length % 2 === 0 ? (values[mid - 1]! + values[mid]!) / 2 : values[mid]!;

    result.setFlat(outIdx++, medianVal);
  }

  return result;
}

/**
 * Compute the qth percentile of the data along the specified axis
 */
export function percentile<D extends DTypeName>(
  a: NDArray<D>,
  q: number | number[],
  axis?: number,
  keepdims: boolean = false
): number | NDArray<'float64'> {
  const qArray = Array.isArray(q) ? q : [q];

  // Validate q values
  for (const qVal of qArray) {
    if (qVal < 0 || qVal > 100) {
      throw new Error(`Percentile must be in range [0, 100], got ${qVal}`);
    }
  }

  if (axis === undefined) {
    const values = a.toFlatArray().map(Number);
    values.sort((x, y) => x - y);

    if (qArray.length === 1) {
      return computePercentile(values, qArray[0]!);
    }

    const result = NDArray.zeros<'float64'>([qArray.length], { dtype: 'float64' });
    for (let i = 0; i < qArray.length; i++) {
      result.setFlat(i, computePercentile(values, qArray[i]!));
    }
    return result;
  }

  // Percentile along axis
  const normalizedAxis = axis < 0 ? axis + a.ndim : axis;
  const axisSize = a.shape[normalizedAxis]!;
  const baseOutShape = keepdims
    ? a.shape.map((d, i) => (i === normalizedAxis ? 1 : d))
    : a.shape.filter((_, i) => i !== normalizedAxis);

  if (baseOutShape.length === 0) {
    baseOutShape.push(1);
  }

  const outShape = qArray.length > 1 ? [qArray.length, ...baseOutShape] : baseOutShape;
  const result = NDArray.zeros<'float64'>(outShape, { dtype: 'float64' });

  let outIdx = 0;
  for (const qVal of qArray) {
    for (const outIndices of new IndexIterator(baseOutShape)) {
      const inIndices: number[] = [];
      let j = 0;
      for (let i = 0; i < a.ndim; i++) {
        if (i === normalizedAxis) {
          inIndices.push(0);
        } else {
          const idx = keepdims ? (i < normalizedAxis ? outIndices[i] : outIndices[i]) : outIndices[j]!;
          inIndices.push(idx ?? 0);
          if (!keepdims) j++;
        }
      }

      const values: number[] = [];
      for (let k = 0; k < axisSize; k++) {
        inIndices[normalizedAxis] = k;
        values.push(Number(a.at(...inIndices)));
      }

      values.sort((x, y) => x - y);
      result.setFlat(outIdx++, computePercentile(values, qVal));
    }
  }

  return result;
}

/**
 * Helper function to compute percentile from sorted array
 */
function computePercentile(sortedValues: number[], q: number): number {
  const n = sortedValues.length;
  const idx = (q / 100) * (n - 1);
  const lower = Math.floor(idx);
  const upper = Math.ceil(idx);
  const frac = idx - lower;

  if (lower === upper || upper >= n) {
    return sortedValues[lower]!;
  }

  return sortedValues[lower]! * (1 - frac) + sortedValues[upper]! * frac;
}

/**
 * Compute the qth quantile of the data along the specified axis
 */
export function quantile<D extends DTypeName>(
  a: NDArray<D>,
  q: number | number[],
  axis?: number,
  keepdims: boolean = false
): number | NDArray<'float64'> {
  const qArray = Array.isArray(q) ? q : [q];
  const percentiles = qArray.map((v) => v * 100);
  return percentile(a, percentiles, axis, keepdims);
}

/**
 * Compute the histogram of a set of data
 */
export function histogram<D extends DTypeName>(
  a: NDArray<D>,
  bins: number | NDArray = 10,
  range?: [number, number]
): { hist: NDArray<'int32'>; binEdges: NDArray<'float64'> } {
  const values = a.toFlatArray().map(Number);

  // Determine range
  let minVal: number, maxVal: number;
  if (range !== undefined) {
    [minVal, maxVal] = range;
  } else {
    minVal = Math.min(...values);
    maxVal = Math.max(...values);
    // Slightly expand range to include max value in last bin
    maxVal = maxVal + (maxVal - minVal) * 1e-10;
  }

  // Determine bin edges
  let binEdges: number[];
  let numBins: number;

  if (typeof bins === 'number') {
    numBins = bins;
    binEdges = [];
    const step = (maxVal - minVal) / numBins;
    for (let i = 0; i <= numBins; i++) {
      binEdges.push(minVal + i * step);
    }
  } else {
    binEdges = bins.toFlatArray().map(Number);
    numBins = binEdges.length - 1;
  }

  // Count values in each bin
  const counts = new Array(numBins).fill(0);
  for (const val of values) {
    for (let i = 0; i < numBins; i++) {
      const lower = binEdges[i]!;
      const upper = binEdges[i + 1]!;
      if ((i === numBins - 1 && val >= lower && val <= upper) ||
          (val >= lower && val < upper)) {
        counts[i]++;
        break;
      }
    }
  }

  return {
    hist: NDArray.from(counts, { dtype: 'int32' }),
    binEdges: NDArray.from(binEdges, { dtype: 'float64' }),
  };
}

/**
 * Compute the bi-dimensional histogram of two data samples
 */
export function histogram2d<D1 extends DTypeName, D2 extends DTypeName>(
  x: NDArray<D1>,
  y: NDArray<D2>,
  bins: number | [number, number] = 10,
  range?: [[number, number], [number, number]]
): { hist: NDArray<'int32'>; xEdges: NDArray<'float64'>; yEdges: NDArray<'float64'> } {
  if (x.size !== y.size) {
    throw new Error('x and y must have the same size');
  }

  const xVals = x.toFlatArray().map(Number);
  const yVals = y.toFlatArray().map(Number);

  const [xBins, yBins] = Array.isArray(bins) ? bins : [bins, bins];

  // Determine ranges
  let xRange: [number, number], yRange: [number, number];
  if (range !== undefined) {
    [xRange, yRange] = range;
  } else {
    xRange = [Math.min(...xVals), Math.max(...xVals) + 1e-10];
    yRange = [Math.min(...yVals), Math.max(...yVals) + 1e-10];
  }

  // Create bin edges
  const xEdges: number[] = [];
  const yEdges: number[] = [];
  const xStep = (xRange[1] - xRange[0]) / xBins;
  const yStep = (yRange[1] - yRange[0]) / yBins;

  for (let i = 0; i <= xBins; i++) {
    xEdges.push(xRange[0] + i * xStep);
  }
  for (let i = 0; i <= yBins; i++) {
    yEdges.push(yRange[0] + i * yStep);
  }

  // Count
  const hist = NDArray.zeros<'int32'>([xBins, yBins], { dtype: 'int32' });

  for (let i = 0; i < xVals.length; i++) {
    const xVal = xVals[i]!;
    const yVal = yVals[i]!;

    // Find bin indices
    let xIdx = -1, yIdx = -1;

    for (let j = 0; j < xBins; j++) {
      const lower = xEdges[j]!;
      const upper = xEdges[j + 1]!;
      if ((j === xBins - 1 && xVal >= lower && xVal <= upper) ||
          (xVal >= lower && xVal < upper)) {
        xIdx = j;
        break;
      }
    }

    for (let j = 0; j < yBins; j++) {
      const lower = yEdges[j]!;
      const upper = yEdges[j + 1]!;
      if ((j === yBins - 1 && yVal >= lower && yVal <= upper) ||
          (yVal >= lower && yVal < upper)) {
        yIdx = j;
        break;
      }
    }

    if (xIdx >= 0 && yIdx >= 0) {
      hist.set([xIdx, yIdx], (hist.at(xIdx, yIdx) as number) + 1);
    }
  }

  return {
    hist,
    xEdges: NDArray.from(xEdges, { dtype: 'float64' }),
    yEdges: NDArray.from(yEdges, { dtype: 'float64' }),
  };
}

/**
 * Compute the covariance matrix
 */
export function cov<D extends DTypeName>(
  m: NDArray<D>,
  rowvar: boolean = true,
  ddof: number = 1
): NDArray<'float64'> {
  let data: NDArray;

  if (m.ndim === 1) {
    data = m.reshape([1, m.size]);
    rowvar = true;
  } else if (m.ndim !== 2) {
    throw new Error('Input must be 1-D or 2-D array');
  } else {
    data = m;
  }

  // If variables are in columns, transpose
  if (!rowvar) {
    data = data.T;
  }

  const [numVars, numObs] = data.shape as [number, number];

  // Center the data
  const means = mean(data, 1, true) as NDArray<'float64'>;
  const centered = subtract(data, means);

  // Compute covariance
  const result = NDArray.zeros<'float64'>([numVars, numVars], { dtype: 'float64' });
  const divisor = numObs - ddof;

  for (let i = 0; i < numVars; i++) {
    for (let j = i; j < numVars; j++) {
      let cov = 0;
      for (let k = 0; k < numObs; k++) {
        cov += Number(centered.at(i, k)) * Number(centered.at(j, k));
      }
      cov /= divisor;
      result.set([i, j], cov);
      if (i !== j) {
        result.set([j, i], cov);
      }
    }
  }

  return result;
}

/**
 * Compute the correlation coefficient matrix
 */
export function corrcoef<D extends DTypeName>(
  x: NDArray<D>,
  rowvar: boolean = true
): NDArray<'float64'> {
  const c = cov(x, rowvar);
  const d = NDArray.zeros<'float64'>([c.shape[0]!], { dtype: 'float64' });

  // Get diagonal
  for (let i = 0; i < c.shape[0]!; i++) {
    d.setFlat(i, Math.sqrt(Number(c.at(i, i))));
  }

  // Normalize
  const result = NDArray.zeros<'float64'>(c.shape, { dtype: 'float64' });
  for (let i = 0; i < c.shape[0]!; i++) {
    for (let j = 0; j < c.shape[1]!; j++) {
      const corr = Number(c.at(i, j)) / (Number(d.at(i)) * Number(d.at(j)));
      result.set([i, j], corr);
    }
  }

  return result;
}
