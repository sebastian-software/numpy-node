# ADR-0006: Platform-Specific Package Distribution

## Status

Accepted

## Context

Native Node.js addons (compiled C++ code) are platform and architecture specific. Users need prebuilt binaries for their platform, otherwise they must have a C++ compiler and all dependencies (BLAS/LAPACK) installed to build from source.

Options considered:

1. **Single package with prebuildify** - Bundles all binaries in one package
2. **Platform-specific packages with optionalDependencies** - esbuild-style distribution
3. **node-pre-gyp** - Downloads binaries at install time from external host
4. **Build from source only** - Require users to compile

## Decision

Use **platform-specific packages** distributed as optionalDependencies, following the pattern established by esbuild, SWC, and other modern native Node.js packages.

### Package Structure

```
numpy-node (main package)
├── @numpy-node/darwin-arm64
├── @numpy-node/linux-x64
├── @numpy-node/linux-arm64
└── @numpy-node/win32-x64
```

### Versioning with pnpm Workspaces

Use **pnpm workspaces** with `workspace:*` protocol for automatic version synchronization:

```yaml
# pnpm-workspace.yaml
packages:
  - 'packages/*'
```

```json
// package.json
"optionalDependencies": {
  "@numpy-node/darwin-arm64": "workspace:*",
  "@numpy-node/linux-arm64": "workspace:*",
  "@numpy-node/linux-x64": "workspace:*",
  "@numpy-node/win32-x64": "workspace:*"
}
```

When publishing:

- pnpm resolves `workspace:*` to the actual version
- Release Please updates versions in all package.json files via `extra-files` config
- All packages publish with identical versions

### Native Module Loading

The main package loads the native module at runtime:

```typescript
function loadNativeModule(): NativeModule | null {
  const platform = process.platform;
  const arch = process.arch;
  const packageName = `@numpy-node/${platform}-${arch}`;

  try {
    const modulePath = require.resolve(`${packageName}/numpy_node_native.node`);
    return require(modulePath);
  } catch {
    return null; // Graceful fallback to pure JS
  }
}
```

## Consequences

### Positive

- **Smaller install size**: Users only download their platform's binary (~5-10MB vs ~40-50MB for all platforms)
- **npm handles platform selection**: optionalDependencies install only for matching platforms
- **Version sync automatic**: pnpm workspace protocol eliminates manual version management
- **CI/CD friendly**: Each platform builds independently, then publishes in parallel
- **No external binary hosting**: Everything goes through npm registry
- **Graceful degradation**: Pure TypeScript fallback when native module unavailable

### Negative

- **Multiple packages to maintain**: 5 packages instead of 1
- **Release complexity**: Must publish all packages atomically (handled by CI)
- **npm organization required**: Scoped packages require npm org (@numpy-node)
- **Workspace tooling dependency**: Relies on pnpm workspace features

## Implementation Details

### Release Please Configuration

```json
// release-please-config.json
{
  "packages": {
    ".": {
      "extra-files": [
        "packages/darwin-arm64/package.json",
        "packages/linux-x64/package.json",
        "packages/linux-arm64/package.json",
        "packages/win32-x64/package.json"
      ]
    }
  }
}
```

### Publish Workflow

1. Build native modules on each platform (macOS, Linux, Windows)
2. Copy binaries to platform packages
3. Publish platform packages with npm provenance
4. Publish main package with pnpm (resolves workspace:\* to versions)

### npm Trusted Publishing

Each package must be configured separately on npmjs.com for OIDC/provenance:

- Repository: `owner/repo`
- Workflow: `publish.yml`

## References

- [esbuild platform packages](https://github.com/evanw/esbuild)
- [SWC platform packages](https://github.com/swc-project/swc)
- [pnpm workspaces](https://pnpm.io/workspaces)
- [npm optionalDependencies](https://docs.npmjs.com/cli/v10/configuring-npm/package-json#optionaldependencies)
- [Release Please monorepo support](https://github.com/googleapis/release-please/blob/main/docs/manifest-releaser.md)
