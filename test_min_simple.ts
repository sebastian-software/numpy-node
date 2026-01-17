import { array, min, max, NDArray } from './src/index.js';

const data = array([
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9],
]);
console.log('Data:', data.toFlatArray());
console.log('Data shape:', data.shape);

console.log('\nTesting min without axis...');
const minAll = min(data);
console.log('Min all:', minAll);

console.log('\nTesting max without axis...');
const maxAll = max(data);
console.log('Max all:', maxAll);

console.log('\nTesting min with axis=0...');
const minAxis0 = min(data, 0);
console.log('Min axis=0:', minAxis0);
console.log('Type:', typeof minAxis0);
if (minAxis0 instanceof NDArray) {
  console.log('Shape:', minAxis0.shape);
} else {
  console.log('Not an NDArray!');
}

console.log('\nTesting min with axis=1...');
const minAxis1 = min(data, 1);
console.log('Min axis=1:', minAxis1);

console.log('\nDone!');
