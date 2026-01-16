/**
 * numpy-node - NumPy for Node.js with native bindings
 *
 * This module provides NumPy-like array operations backed by
 * native C++ code with BLAS/LAPACK acceleration.
 */

// Re-export everything from native module
export * from './native/index.js';

// Default export as np-like namespace
import * as native from './native/index.js';

/**
 * NumPy-like namespace object
 * Use: import np from 'numpy-node'
 */
const np = {
  ...native,
};

export default np;
