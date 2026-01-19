# ADR-0005: NumPy Compatibility Policy

## Status

Accepted

## Context

numpy-node aims to be the Node.js equivalent of NumPy. Users expect the same behavior when they migrate code from Python to TypeScript. Any deviation from NumPy's behavior causes confusion, bugs, and reduces trust in the library.

Currently, there is no formal policy ensuring that all functions behave identically to their NumPy counterparts. This has led to:

- Inconsistent implementations where behavior differs from NumPy
- Missing features that NumPy provides (e.g., broadcasting in logical operators)
- No systematic testing against NumPy reference outputs
- Unclear expectations for contributors adding new functions

## Decision

**All numpy-node functions MUST be 1:1 compatible with their NumPy counterparts.**

This means:

### 1. Behavioral Compatibility

- Same function signatures (accounting for language differences)
- Same output values given the same inputs
- Same edge case handling (NaN, Inf, empty arrays, etc.)
- Same broadcasting semantics
- Same default parameter values

### 2. Verification Process

Before merging any new function or modification:

1. **Create NumPy verification test**: Document the exact NumPy command and expected output
2. **Add to verification script**: Add test cases to `scripts/generate_numpy_reference.py`
3. **Add conformity test**: Add corresponding test in `tests/numpy-conformity.test.ts`
4. **Test edge cases**: Verify behavior with edge cases matches NumPy

**Note:** Reference values are generated fresh from the latest NumPy version on every CI run and `pnpm install`. They are NOT committed to git.

### 3. Test Format

All tests MUST include the NumPy verification command as a comment:

```typescript
// >>> np.greater(np.array([1, 2, 3, 4, 5]), 3)
// array([False, False, False,  True,  True])
it('should compare array with scalar', () => {
  const a = array([1, 2, 3, 4, 5]);
  const result = greater(a, 3);
  expect(result.dtype).toBe('bool');
  expect(Array.from(result.toFlatArray())).toEqual([0, 0, 0, 1, 1]);
});
```

### 4. Documentation

- README clearly states NumPy compatibility guarantee
- Each function's docstring references its NumPy equivalent
- Known differences (if any) are documented prominently

### 5. Exceptions

Some differences are acceptable and must be documented:

- **Return types**: JavaScript/TypeScript types instead of Python types
- **Method naming**: camelCase alternatives may be provided alongside snake_case
- **Module structure**: Flat imports instead of nested modules (e.g., `linalg.inv` vs `np.linalg.inv`)
- **Performance optimizations**: Fused operations that don't exist in NumPy (must be clearly named differently)

## Consequences

### Positive

- **Trust**: Users can confidently migrate NumPy code to numpy-node
- **Documentation**: NumPy's extensive documentation applies to numpy-node
- **Consistency**: All contributors follow the same standard
- **Quality**: Systematic testing catches regressions
- **Ecosystem**: Tutorials and examples from NumPy work with numpy-node

### Negative

- **Development overhead**: More testing and verification required
- **Constraints**: Cannot deviate from NumPy behavior even if "better" alternatives exist
- **Maintenance burden**: Must track NumPy changes across versions
- **Edge cases**: Must implement even rarely-used edge case behaviors

## References

- [NumPy Documentation](https://numpy.org/doc/stable/)
- [NumPy API Reference](https://numpy.org/doc/stable/reference/index.html)
- [NumPy Source Code](https://github.com/numpy/numpy)
- `scripts/generate_numpy_reference.py` - Local verification script for generating NumPy reference values
- `tests/numpy-conformity.test.ts` - Automated conformity tests run in CI
