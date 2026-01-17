# Claude Code Context

## Project Overview

numpy-node is a TypeScript NumPy implementation with optional native C++ bindings (N-API).

## Release Workflow

This project uses **Release Please** with **npm Trusted Publishing (OIDC)** for fully automated releases.

### How It Works

1. **Conventional Commits** trigger version bumps:
   - `feat:` → minor bump (0.1.0 → 0.2.0)
   - `fix:` → patch bump (0.1.0 → 0.1.1)
   - `feat!:` or `BREAKING CHANGE:` → major bump

2. **Release Please** (on push to main):
   - Analyzes commits since last release
   - Creates/updates a release PR with changelog
   - When PR is merged → creates GitHub Release

3. **Publish Workflow** (on release published):
   - Builds native modules for all platforms
   - Publishes to npm using OIDC (no tokens needed)

### Key Files

- `release-please-config.json` - Release Please configuration
- `.release-please-manifest.json` - Current version tracking
- `.github/workflows/release-please.yml` - Creates release PRs
- `.github/workflows/publish.yml` - Builds and publishes to npm

## Important Learnings

### npm Trusted Publishing (OIDC)

1. **Requires npm v11+** - Node.js 22 ships with npm v10 which does NOT work. Use Node.js 24 for publish jobs.

2. **Configure on npm for EACH package separately**:
   - Go to npmjs.com → Package → Settings → Trusted Publisher
   - Repository: `owner/repo`
   - Workflow: `publish.yml` (filename only, not full path)

3. **package.json must have `repository` field** for provenance validation:

   ```json
   "repository": {
     "type": "git",
     "url": "https://github.com/owner/repo.git"
   }
   ```

4. **Workflow permissions required**:

   ```yaml
   permissions:
     contents: read
     id-token: write
   ```

5. **Don't use `workflow_call`** for publish workflows - OIDC tokens may contain claims from the calling workflow that don't match npm's expectations. Trigger via `release: [published]` event instead.

### Release Please

1. **Use a PAT (Classic Personal Access Token)** with `repo` scope, not GITHUB_TOKEN. Store as `RELEASE_PLEASE_TOKEN` secret.

2. **If release was created manually**, update the PR label from `autorelease: pending` to `autorelease: tagged` to tell Release Please to move on.

3. **Workflow should NOT call publish.yml** via `workflow_call`. Let the release event trigger publish.yml directly.

## Platform-Specific Packages (for native modules)

Uses esbuild-style distribution with optional dependencies:

- `@numpy-node/darwin-arm64`
- `@numpy-node/linux-x64`
- `@numpy-node/linux-arm64`
- `@numpy-node/win32-x64`

Main package has these as `optionalDependencies` - npm installs only the matching platform.

## CI/CD Notes

- **macOS**: Use `macos-15` (Intel runners retired)
- **Windows LAPACK**: Install `openblas:x64-windows` AND `lapack-reference:x64-windows` separately via vcpkg
- **Windows M_PI**: Add `#define _USE_MATH_DEFINES` before ANY includes

## Commands

```bash
pnpm build          # Build TypeScript
pnpm build:native   # Build native module
pnpm test           # Run tests
pnpm lint           # Lint code
```
