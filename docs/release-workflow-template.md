# Automated Release Workflow Template

This guide describes how to set up fully automated npm releases using Release Please and Trusted Publishing (OIDC).

## Overview

```
Push to main → Release Please creates PR → Merge PR → GitHub Release → npm Publish
     │                    │                     │              │              │
     └─ Conventional      └─ Changelog          └─ Tag         └─ Triggers    └─ OIDC auth
        Commits              generated             created        publish.yml     (no tokens!)
```

## Setup Steps

### 1. Create Release Please Config

**release-please-config.json:**

```json
{
  "$schema": "https://raw.githubusercontent.com/googleapis/release-please/main/schemas/config.json",
  "packages": {
    ".": {
      "release-type": "node",
      "changelog-path": "CHANGELOG.md",
      "bump-minor-pre-major": true,
      "bump-patch-for-minor-pre-major": true
    }
  }
}
```

**.release-please-manifest.json:**

```json
{
  ".": "0.0.0"
}
```

(Set to your current version)

### 2. Create GitHub PAT

1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Create token with `repo` scope
3. Add as repository secret: `RELEASE_PLEASE_TOKEN`

### 3. Create Workflows

**.github/workflows/release-please.yml:**

```yaml
name: Release Please

on:
  push:
    branches:
      - main

permissions:
  contents: write
  pull-requests: write

jobs:
  release-please:
    runs-on: ubuntu-latest
    steps:
      - uses: googleapis/release-please-action@v4
        with:
          token: ${{ secrets.RELEASE_PLEASE_TOKEN }}
          config-file: release-please-config.json
          manifest-file: .release-please-manifest.json
```

**.github/workflows/publish.yml:**

```yaml
name: Publish

on:
  release:
    types: [published]
  workflow_dispatch:
    inputs:
      dry-run:
        description: 'Dry run (skip actual publish)'
        required: false
        default: 'true'
        type: boolean

permissions:
  contents: read
  id-token: write # Required for OIDC

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: '24' # IMPORTANT: npm v11+ required for OIDC
          registry-url: 'https://registry.npmjs.org'

      - run: npm ci
      - run: npm run build
      - run: npm test

      - name: Publish (dry run)
        if: ${{ inputs.dry-run == true }}
        run: npm publish --dry-run

      - name: Publish
        if: ${{ inputs.dry-run != true }}
        run: npm publish --access public --provenance
```

### 4. Configure npm Trusted Publishing

1. Publish initial version manually: `npm publish --access public`
2. Go to npmjs.com → Your Package → Settings → Publishing access
3. Under "Trusted Publisher", click "Add new provider"
4. Configure:
   - **Repository owner**: your-github-username
   - **Repository name**: your-repo
   - **Workflow filename**: `publish.yml`
   - **Environment**: (leave empty)

### 5. Add Repository to package.json

```json
{
  "repository": {
    "type": "git",
    "url": "https://github.com/owner/repo.git"
  }
}
```

This is **required** for npm provenance validation.

## Conventional Commits Cheatsheet

```
feat: add new feature        → minor bump (0.1.0 → 0.2.0)
fix: fix a bug               → patch bump (0.1.0 → 0.1.1)
feat!: breaking change       → major bump (0.1.0 → 1.0.0)
docs: update readme          → no release
chore: update deps           → no release
refactor: restructure code   → no release (unless configured)

# Scopes are optional
feat(api): add endpoint
fix(ui): button alignment
```

## Troubleshooting

### "Access token expired or revoked" + 404 Error

**Cause**: npm v10 doesn't support OIDC properly.
**Fix**: Use Node.js 24 (ships with npm v11).

### Release Please not creating PRs

**Check**:

1. Are you using conventional commit format?
2. Does the PAT have `repo` scope?
3. Is there an old PR with `autorelease: pending` label? Change it to `autorelease: tagged`.

### Trusted Publishing 404 Error

**Check**:

1. Workflow filename matches exactly (e.g., `publish.yml`)
2. Repository owner/name match exactly
3. `id-token: write` permission is set
4. `repository` field exists in package.json

### Release was created manually

If you created a release manually and Release Please is stuck:

1. Find the Release Please PR
2. Change label from `autorelease: pending` to `autorelease: tagged`
3. Push a new commit to trigger Release Please again

## Benefits

- **No npm tokens to manage** - OIDC handles authentication
- **Automatic changelogs** - Generated from commit messages
- **Version bumps automated** - Based on conventional commits
- **Provenance** - Cryptographic proof of build origin
- **PR-based releases** - Review before releasing
