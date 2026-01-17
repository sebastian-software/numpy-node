import { array, subtract, divide, min, max, NDArray } from './src/index.js';

function randomMatrix(rows: number, cols: number): NDArray {
  const data: number[][] = [];
  for (let i = 0; i < rows; i++) {
    const row: number[] = [];
    for (let j = 0; j < cols; j++) {
      row.push(Math.random());
    }
    data.push(row);
  }
  return array(data);
}

console.log('Creating data...');
const data = randomMatrix(100, 10);
console.log('Data shape:', data.shape);

console.log('Computing min...');
const minVals = min(data, 0) as NDArray;
console.log('Min shape:', minVals.shape);

console.log('Computing max...');
const maxVals = max(data, 0) as NDArray;
console.log('Max shape:', maxVals.shape);

console.log('Computing range...');
const range = subtract(maxVals, minVals);
console.log('Range shape:', range.shape);

console.log('Centering data (subtract)...');
const centered = subtract(data, minVals);
console.log('Centered shape:', centered.shape);

console.log('Dividing (scaling)...');
const scaled = divide(centered, range);
console.log('Scaled shape:', scaled.shape);

console.log('Done!');
