# NumPy Feature Roadmap

Prioritized list of missing NumPy functions for numpy-node.

## Already Implemented

These functions are already available:

- **Array Creation:** `array`, `zeros`, `ones`, `full`, `arange`, `linspace`, `eye`, `identity`, `empty`, `zerosLike`, `onesLike`, `emptyLike`
- **Arithmetic:** `add`, `subtract`, `multiply`, `divide`, `power` (+ in-place variants)
- **Unary Math:** `sqrt`, `exp`, `log`, `sin`, `cos`, `tan`, `abs`, `negative`, `round`, `floor`, `ceil`, `sign`, `mod`
- **Reductions:** `sum`, `prod`, `mean`, `std`, `variance`, `median`, `min`, `max`, `argmin`, `argmax`, `cumsum`, `cumprod`
- **Comparison:** `equal`, `not_equal`, `less`, `less_equal`, `greater`, `greater_equal`, `isclose`, `allclose`
- **Logical:** `logical_and`, `logical_or`, `logical_xor`, `logical_not`
- **Boolean Reductions:** `any`, `all`
- **Linear Algebra:** `matmul`, `dot`, `inv`, `det`, `solve`, `eig`, `eigvals`, `svd`, `qr`, `cholesky`, `norm`, `matrix_rank`, `trace`, `cond`, `lstsq`
- **Array Manipulation:** `clip`, `where`, `squeeze`, `expand_dims`, `concatenate`, `stack`, `vstack`, `hstack`, `tile`, `repeat`, `flip`, `rot90`, `split`, `nonzero`
- **Sorting/Searching:** `diff`, `sort`, `argsort`, `unique`, `searchsorted`
- **Advanced:** `outer`, `kron`, `percentile`, `corrcoef`, `zscore`
- **Random:** `random.random`, `random.uniform`, `random.normal`, `random.randint`, `random.seed`

---

## Tier 1 - Essential (Highest Priority)

These functions are used in almost every NumPy project.

_All Tier 1 functions implemented: `argmin`, `argmax`, `clip`, `where`, `concatenate`, `stack`_

## Tier 2 - Very Common

Regularly used in Data Science and ML.

_All Tier 2 functions implemented: `cumsum`, `cumprod`, `diff`, `sort`, `argsort`, `unique`_

## Tier 3 - Common

Useful for many use cases.

_All Tier 3 functions implemented: `searchsorted`, `tile`, `repeat`, `round`, `floor`, `ceil`, `squeeze`, `expand_dims`, `vstack`, `hstack`_

## Tier 4 - Useful

Less common but important for specific domains.

_All Tier 4 functions implemented: `flip`, `rot90`, `sign`, `mod`, `allclose`, `isclose`, `nonzero`, `split`_

## Tier 5 - Advanced

Complex to implement but enables new domains.

| Function                        | Description            | Use Case                 | Complexity |
| ------------------------------- | ---------------------- | ------------------------ | ---------- |
| `fft(a)`                        | Fast Fourier Transform | Audio, Signal Processing | High       |
| `ifft(a)`                       | Inverse FFT            | Frequency to Time        | High       |
| `rfft(a)`                       | Real FFT               | More efficient for real  | High       |
| `einsum(subscripts, *operands)` | Einstein Summation     | Flexible Tensor Ops      | High       |

---

## Implementation Plan

### Phase 1: Next Priorities (Tier 1)

1. ~~`argmin`, `argmax` - C++ with vDSP/BLAS optimization~~
2. ~~`clip` - C++ with vDSP_vclipD~~
3. ~~`where` - Boolean Indexing~~
4. ~~`concatenate`, `stack` - Array joining~~

### Phase 2: Cumulative & Sorting (Tier 2)

1. ~~`cumsum`, `cumprod` - sequential, possibly parallel prefix sum~~
2. ~~`diff` - simple loop~~
3. ~~`sort`, `argsort` - std::sort~~
4. ~~`unique` - sort + deduplicate~~

### Phase 3: Utilities (Tier 3)

1. ~~`round`, `floor`, `ceil` - vDSP or std::round~~
2. ~~`squeeze`, `expand_dims` - pure shape manipulation~~
3. ~~`tile`, `repeat` - memory operations~~
4. ~~`searchsorted` - Binary Search~~

### Phase 4: Extended (Tier 4 & 5)

1. ~~Array manipulation: `flip`, `rot90`, `split`, `sign`, `mod`, `nonzero`~~
2. ~~Approximate comparison: `allclose`, `isclose`~~
3. FFT: Accelerate Framework has vDSP*fft*\*
4. `einsum`: Complex parser + optimized contractions

---

## Coverage Summary

| Category           | Implemented | Missing             | Coverage |
| ------------------ | ----------- | ------------------- | -------- |
| Array Creation     | 12          | 0                   | 100%     |
| Arithmetic         | 5           | 0                   | 100%     |
| Unary Math         | 13          | 0                   | 100%     |
| Reductions         | 12          | 0                   | 100%     |
| Comparison         | 8           | 0                   | 100%     |
| Logical            | 4           | 0                   | 100%     |
| Boolean Reductions | 2           | 0                   | 100%     |
| Linear Algebra     | 15          | 0                   | 100%     |
| Sorting/Searching  | 5           | 0                   | 100%     |
| Array Manipulation | 18          | 0                   | 100%     |
| **Total (Core)**   | **~94**     | **4 (FFT, einsum)** | **~96%** |

_Note: "Core" refers to the most commonly used NumPy functions. Only advanced FFT and einsum remain unimplemented._
