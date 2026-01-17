import {
  array,
  zeros,
  linspace,
  add,
  subtract,
  multiply,
  divide,
  matmul,
  sqrt,
  sum,
  mean,
  std,
  min,
  max,
  svd,
  qr,
  eig,
  solve,
  normal_equations,
  zscore,
  corrcoef,
  gram_matrix,
  softmax,
  NDArray,
} from './src/index.js';

function randomMatrix(rows: number, cols: number): NDArray {
  const data: number[][] = [];
  for (let i = 0; i < rows; i++) {
    const row: number[] = [];
    for (let j = 0; j < cols; j++) {
      const u1 = Math.random();
      const u2 = Math.random();
      row.push(Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2));
    }
    data.push(row);
  }
  return array(data);
}

function randomVector(n: number): NDArray {
  const data: number[] = [];
  for (let i = 0; i < n; i++) {
    const u1 = Math.random();
    const u2 = Math.random();
    data.push(Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2));
  }
  return array(data);
}

function testScenario(name: string, fn: () => void) {
  console.log(`Testing: ${name}...`);
  try {
    fn();
    console.log(`  ✓ OK`);
    return true;
  } catch (e) {
    console.log(`  ✗ ERROR: ${e}`);
    return false;
  }
}

// Test scenarios one by one
console.log('Testing scenarios...\n');

// 1. Data Normalization
testScenario('Data Normalization', () => {
  const data = randomMatrix(100, 10);
  zscore(data, 0);
});

// 2. Linear Regression
testScenario('Linear Regression', () => {
  const X = randomMatrix(100, 10);
  const y = randomMatrix(100, 1);
  normal_equations(X, y);
});

// 3. PCA
testScenario('PCA', () => {
  const data = randomMatrix(100, 20);
  const μ = mean(data, 0) as NDArray;
  const centered = subtract(data, μ);
  const { vh } = svd(centered);
  const V = vh.T;
  matmul(centered, V);
});

// 4. Correlation Matrix
testScenario('Correlation Matrix', () => {
  const data = randomMatrix(100, 10);
  corrcoef(data);
});

// 5. Matrix Chain
testScenario('Matrix Chain', () => {
  const X = randomMatrix(100, 64);
  const W1 = randomMatrix(64, 32);
  const W2 = randomMatrix(32, 10);
  const h1 = matmul(X, W1);
  const out = matmul(h1, W2);
  softmax(out);
});

// 6. Covariance Eigendecomp
testScenario('Covariance Eigendecomp', () => {
  const data = randomMatrix(100, 20);
  const μ = mean(data, 0) as NDArray;
  const centered = subtract(data, μ);
  const Xt = centered.T;
  const XtX = matmul(Xt, centered);
  const cov = divide(XtX, 99);
  eig(cov);
});

// 7. Least Squares QR
testScenario('Least Squares QR', () => {
  const A = randomMatrix(100, 10);
  const b = randomMatrix(100, 1);
  const { q, r } = qr(A);
  const Qt = q.T;
  const Qtb = matmul(Qt, b);
  solve(r, Qtb);
});

// 8. Batch Statistics
testScenario('Batch Statistics', () => {
  const batches: NDArray[] = [];
  for (let i = 0; i < 10; i++) {
    batches.push(randomMatrix(100, 5));
  }
  const means: NDArray[] = [];
  for (const batch of batches) {
    means.push(mean(batch, 0) as NDArray);
  }
});

// 9. Distance Matrix
testScenario('Distance Matrix', () => {
  const points = randomMatrix(100, 10);
  const sq = multiply(points, points);
  sum(sq, 1);
  gram_matrix(points);
});

// 10. Polynomial Fit
testScenario('Polynomial Fit', () => {
  const xArr = linspace(-5, 5, 100);
  const xData = xArr.data as Float64Array;
  const yData: number[][] = [];
  for (let i = 0; i < 100; i++) {
    yData.push([Math.sin(xData[i])]);
  }
  const y = array(yData);
  const vData: number[][] = [];
  for (let i = 0; i < 100; i++) {
    const row: number[] = [];
    for (let j = 5; j >= 0; j--) {
      row.push(Math.pow(xData[i], j));
    }
    vData.push(row);
  }
  const V = array(vData);
  normal_equations(V, y);
});

// 11. Min-Max Scaling
testScenario('Min-Max Scaling', () => {
  const data = randomMatrix(100, 10);
  const minVals = min(data, 0) as NDArray;
  const maxVals = max(data, 0) as NDArray;
  const range = subtract(maxVals, minVals);
  const centered = subtract(data, minVals);
  divide(centered, range);
});

// 12. Outer Product Sum
testScenario('Outer Product Sum', () => {
  const vectors = randomMatrix(100, 10);
  const Xt = vectors.T;
  matmul(Xt, vectors);
});

// 13. Weighted Mean
testScenario('Weighted Mean', () => {
  const values = randomMatrix(100, 20);
  const weights = randomVector(100);
  const totalWeight = sum(weights) as number;
  const normWeights = divide(weights, totalWeight);
  const wReshaped = normWeights.reshape([1, 100]);
  matmul(wReshaped, values);
});

// 14. Gradient Descent
testScenario('Gradient Descent', () => {
  const X = randomMatrix(100, 10);
  const y = randomMatrix(100, 1);
  const theta = randomMatrix(10, 1);
  const predictions = matmul(X, theta);
  const errors = subtract(predictions, y);
  const Xt = X.T;
  const grad = matmul(Xt, errors);
  divide(grad, 100);
});

// 15. Cross-Entropy
testScenario('Cross-Entropy', () => {
  const predictions = randomMatrix(100, 10);
  const targets = randomMatrix(100, 10);
  const softmaxPreds = softmax(predictions);
  const logPreds = multiply(targets, softmaxPreds);
  sum(logPreds);
});

// 16. Cosine Similarity
testScenario('Cosine Similarity', () => {
  const vectors = randomMatrix(100, 20);
  const sq = multiply(vectors, vectors);
  const norms = sqrt(sum(sq, 1) as NDArray);
  const dotProds = gram_matrix(vectors);
  const normCol = norms.reshape([100, 1]);
  const normRow = norms.reshape([1, 100]);
  const normOuter = matmul(normCol, normRow);
  divide(dotProds, normOuter);
});

// 17. K-Means Step
testScenario('K-Means Step', () => {
  const data = randomMatrix(100, 10);
  const centroids = randomMatrix(5, 10);
  sum(multiply(data, data), 1);
  sum(multiply(centroids, centroids), 1);
  const centroidsT = centroids.T;
  matmul(data, centroidsT);
});

// 18. Image Filter
testScenario('Image Filter', () => {
  const image = randomMatrix(64, 64);
  const blurred = add(image, multiply(image, 0.1));
  subtract(multiply(image, 2), blurred);
});

// 19. Power Iteration
testScenario('Power Iteration', () => {
  const A = randomMatrix(50, 50);
  const At = A.T;
  const M = matmul(At, A);
  let v = randomVector(50);
  for (let i = 0; i < 3; i++) {
    const vCol = v.reshape([50, 1]);
    const Mv = matmul(M, vCol);
    const norm = Math.sqrt(sum(multiply(Mv, Mv)) as number);
    v = divide(Mv, norm).reshape([50]);
  }
});

console.log('\nAll scenarios tested!');
