# Changelog

## [0.3.0](https://github.com/sebastian-software/numpy-node/compare/numpy-node-v0.2.0...numpy-node-v0.3.0) (2026-01-20)


### ⚠ BREAKING CHANGES

* The library now uses native C++ bindings for all array operations. The pure TypeScript implementation has been removed.

### Features

* add 18 new real-world benchmark scenarios ([f99bf82](https://github.com/sebastian-software/numpy-node/commit/f99bf8219a5c9f676f0e40972689189dd3e696fb))
* add 20 new scenarios to NumPy Python benchmark ([7d5345a](https://github.com/sebastian-software/numpy-node/commit/7d5345a36610a6ad5e9b4264a11bee6dfc2afb17))
* add boolean array support and NumPy conformity testing ([4c86a50](https://github.com/sebastian-software/numpy-node/commit/4c86a50cde30b7e015ef5a438da4bac42a4f70e2))
* add completeness check for NumPy conformity tests ([29b1139](https://github.com/sebastian-software/numpy-node/commit/29b11394038d6215db469141155064740a68ae89))
* add fused arithmetic operations for reduced N-API overhead ([d65a182](https://github.com/sebastian-software/numpy-node/commit/d65a1826f36ff1c364977486a468f664f5a50332))
* add fused C++ operations for performance optimization ([01d04e9](https://github.com/sebastian-software/numpy-node/commit/01d04e94a00bf79a333b259bda7e9f0307cf09a5))
* add in-place arithmetic operations ([6ca7dc1](https://github.com/sebastian-software/numpy-node/commit/6ca7dc1649a984dbe616f09c769cf1021542b9cd))
* add keepdims parameter to reduction operations ([0d1bad6](https://github.com/sebastian-software/numpy-node/commit/0d1bad63a36708f8b09c6f87519b74af753950e5))
* add main entry point with np namespace ([fcfa5df](https://github.com/sebastian-software/numpy-node/commit/fcfa5dfc20214181899ac3bbf1ebab1940b8a104))
* add matrix-vector dot, norm variants, statistics, and NDArray methods ([a6eeaf4](https://github.com/sebastian-software/numpy-node/commit/a6eeaf46dfcb3d694ee86756a8a343b0dc33a31c))
* add NumPy vs numpy-node benchmark comparison ([b834dd7](https://github.com/sebastian-software/numpy-node/commit/b834dd767bde413c55beea56fa9871a51d0de745))
* add platform-specific package distribution ([60f005f](https://github.com/sebastian-software/numpy-node/commit/60f005f2a56a8385fa58168d5fc43f5cce8818fd))
* add release-please for automated releases ([79f0962](https://github.com/sebastian-software/numpy-node/commit/79f0962f8b6531712ef3ed981d4f3a6bf5733611))
* auto-generate NumPy reference on pnpm install ([21caf0e](https://github.com/sebastian-software/numpy-node/commit/21caf0ee0fd6d8b2107f028cf62c9d55f90eab71))
* **core:** add NDArray class and type system ([3dbc6eb](https://github.com/sebastian-software/numpy-node/commit/3dbc6ebaddc2369f7afd28e0b28454da2655f672))
* **creation:** add array creation functions ([5f62ff5](https://github.com/sebastian-software/numpy-node/commit/5f62ff5b5f7414d65c76fb3966192b0f41a917df))
* export linalg module from main entry point ([87a9e38](https://github.com/sebastian-software/numpy-node/commit/87a9e388854545b00aa03f37cab1b7c67a525c3b))
* implement einsum (Einstein summation) ([91e6dca](https://github.com/sebastian-software/numpy-node/commit/91e6dca8a5068c9c52e77cb2c941677d4f234a11))
* implement FFT module with fft, ifft, rfft, irfft ([06aca8c](https://github.com/sebastian-software/numpy-node/commit/06aca8c3b8352854e3d5b15b0d78b01f4988ce2b))
* implement NumPy-style views for transpose and reshape ([25e7d2b](https://github.com/sebastian-software/numpy-node/commit/25e7d2b11e6447807950a123cf00f34616ab50d3))
* implement Tier 3+4 NumPy functions ([67ba81a](https://github.com/sebastian-software/numpy-node/commit/67ba81aa006681269fe0045cf08a29270a0a1f07))
* **linalg:** add linear algebra module ([2d36f0a](https://github.com/sebastian-software/numpy-node/commit/2d36f0a21daf4b39afe6fd4db9e1df10c58d5e5c))
* **linalg:** implement eigvals, eig, svd, qr, cholesky, matrix_rank ([f4cf8f9](https://github.com/sebastian-software/numpy-node/commit/f4cf8f901e471fed348ab18eeee26c237a4a7e11))
* **math:** implement axis-based reductions for sum, mean, std ([15c4b0c](https://github.com/sebastian-software/numpy-node/commit/15c4b0cbfb7e2099a3612315e19d37342852c2ff))
* **math:** implement NumPy-style broadcasting for binary operations ([86969bc](https://github.com/sebastian-software/numpy-node/commit/86969bcef68692c6770a83732c08a7a70f0081de))
* **native:** add C++ N-API module scaffolding ([1f7492d](https://github.com/sebastian-software/numpy-node/commit/1f7492d00c054e778689144f2c3ba6e7b8b710c5))
* **ops:** add array operations ([15a393f](https://github.com/sebastian-software/numpy-node/commit/15a393f90c5e890ff070ebe7ca42054d3e4c498b))
* **stats:** add statistical functions ([81f961b](https://github.com/sebastian-software/numpy-node/commit/81f961b684ccee0116f96758f57a1d1726aac475))
* switch to npm trusted publishing with provenance ([549fe9e](https://github.com/sebastian-software/numpy-node/commit/549fe9e11d198c625198b01c960adcac4ab29290))
* use local .venv for NumPy installation ([3e8e259](https://github.com/sebastian-software/numpy-node/commit/3e8e25998cfcc2f6b669feeeb1234f7cf0150d35))


### Bug Fixes

* add id-token permission to release-please workflow ([2abc41b](https://github.com/sebastian-software/numpy-node/commit/2abc41b134d6a4a0419c069282aea514fcc3de5a))
* add missing cmath include and extend CI matrix ([6010e20](https://github.com/sebastian-software/numpy-node/commit/6010e2014eb7f2ec2ead92ff86455c87d552e64d))
* add repository field for npm provenance validation ([fcf96c1](https://github.com/sebastian-software/numpy-node/commit/fcf96c18751f82c2d106121d7ffc6854019bdd2b))
* also trigger publish on manual releases ([a0d42d4](https://github.com/sebastian-software/numpy-node/commit/a0d42d4054bd73b24762080496c3c2894bce55df))
* **ci:** install lapack-reference separately on Windows ([4c7717b](https://github.com/sebastian-software/numpy-node/commit/4c7717bae2f0469144124a24e3d26e994cfbfd2f))
* **ci:** trigger publish directly via release event, not workflow_call ([deafae0](https://github.com/sebastian-software/numpy-node/commit/deafae039df605bda134571c3803ccb105618060))
* **ci:** upgrade npm for OIDC trusted publishing support ([d0534be](https://github.com/sebastian-software/numpy-node/commit/d0534be75d2fd54f0fc31e6a89b0f9457d53ebc0))
* **ci:** use Node.js 24 for publish job (includes npm v11) ([dc7948e](https://github.com/sebastian-software/numpy-node/commit/dc7948ed23dd44d5404201993b1ae24ceae8b4d7))
* correct softmax cross-entropy benchmark ([d20f1c9](https://github.com/sebastian-software/numpy-node/commit/d20f1c96c0b4b588e5fec61da1181116600b08ad))
* define _USE_MATH_DEFINES for Windows M_PI ([b5cb4e8](https://github.com/sebastian-software/numpy-node/commit/b5cb4e8abf3d1673c1ca11b17eaed6dceccd7e21))
* install openblas with lapack feature on Windows ([c78f631](https://github.com/sebastian-software/numpy-node/commit/c78f631e7429dd908712a28c63c34576cc36109d))
* lockfile ([c329a45](https://github.com/sebastian-software/numpy-node/commit/c329a453b3cb37c79e5b64f638df32e4d5498ba7))
* M_PI must be defined before any includes on MSVC ([8ac036a](https://github.com/sebastian-software/numpy-node/commit/8ac036ad9cc3eab11df1b7e56ca526b025ad7f3f))
* **native:** configure cmake-js and fix build errors ([2929e6c](https://github.com/sebastian-software/numpy-node/commit/2929e6c910b1d8907029784260cdf03318b7d1f5))
* **ops:** fix reduceAxis keepdims indexing bug ([88a4c02](https://github.com/sebastian-software/numpy-node/commit/88a4c02535dc6baf27bde997792a2ba818fbdeda))
* remove darwin-x64, use macos-15 for arm64 ([77072cf](https://github.com/sebastian-software/numpy-node/commit/77072cfafcf9db24e8313985865b5e93a5d3faaf))
* **stats:** fix average and variance functions ([00e5ff4](https://github.com/sebastian-software/numpy-node/commit/00e5ff4dc66ca91c8b5c23c6dea46dd9f272a790))
* update CI to use Node 22 and refresh lockfile ([15b5176](https://github.com/sebastian-software/numpy-node/commit/15b5176c627436ebb227881fbe12f37583547b89))
* update macOS runner for x64 builds ([2ffb447](https://github.com/sebastian-software/numpy-node/commit/2ffb44761bce9effbd866b655c92bb9451818deb))
* use [@numpy-node](https://github.com/numpy-node) scope for platform packages ([94a563f](https://github.com/sebastian-software/numpy-node/commit/94a563ff3063bf04264decbadde74593852236ea))
* use PAT for release-please ([265123a](https://github.com/sebastian-software/numpy-node/commit/265123a755b9389c1e1f9fe1e6badf313b6b5507))


### Performance Improvements

* add batch_matmul for reduced N-API overhead ([3ddc8fd](https://github.com/sebastian-software/numpy-node/commit/3ddc8fd016b5b9346ce7da01552013b6be87c041))
* add broadcasting optimization and fused operations ([8057cae](https://github.com/sebastian-software/numpy-node/commit/8057cae2358ab5b25afd02c1edd53e25584f8bc9))
* add fused ML operations and optimize benchmarks to 70% wins ([9543d95](https://github.com/sebastian-software/numpy-node/commit/9543d95b2016ff29a2c778600138745f8a50efa6))
* add fused native operations for numerical methods ([fc84fac](https://github.com/sebastian-software/numpy-node/commit/fc84fac2cabaf7f819c3af91b8a8ac60a6a2642a))
* add native percentile with O(n) quickselect algorithm ([7b96aec](https://github.com/sebastian-software/numpy-node/commit/7b96aec900f960bb02d028acf10d9778664a0f20))
* add native percentile with optimized sort ([14e94a4](https://github.com/sebastian-software/numpy-node/commit/14e94a4537b28909f20b74616ddfe5b6263c0369))
* add vDSP-accelerated 2D-1D broadcasting for add and multiply ([b6fb482](https://github.com/sebastian-software/numpy-node/commit/b6fb4827a5d0353d234aaa3a3a8cf250ba5f9137))
* add vDSP/vecLib optimizations for macOS ([7c04bea](https://github.com/sebastian-software/numpy-node/commit/7c04bea2e1481bfb52c22ac958f55a5ba379d91f))


### Code Refactoring

* replace pure TypeScript with native C++ bindings ([a8fa1ce](https://github.com/sebastian-software/numpy-node/commit/a8fa1cec88e7d550cb15ed4e52df066e72b665bf))

## [0.2.0](https://github.com/sebastian-software/numpy-node/compare/numpy-node-v0.1.0...numpy-node-v0.2.0) (2026-01-16)


### ⚠ BREAKING CHANGES

* The library now uses native C++ bindings for all array operations. The pure TypeScript implementation has been removed.

### Features

* add main entry point with np namespace ([fcfa5df](https://github.com/sebastian-software/numpy-node/commit/fcfa5dfc20214181899ac3bbf1ebab1940b8a104))
* add matrix-vector dot, norm variants, statistics, and NDArray methods ([a6eeaf4](https://github.com/sebastian-software/numpy-node/commit/a6eeaf46dfcb3d694ee86756a8a343b0dc33a31c))
* add platform-specific package distribution ([60f005f](https://github.com/sebastian-software/numpy-node/commit/60f005f2a56a8385fa58168d5fc43f5cce8818fd))
* add release-please for automated releases ([79f0962](https://github.com/sebastian-software/numpy-node/commit/79f0962f8b6531712ef3ed981d4f3a6bf5733611))
* **core:** add NDArray class and type system ([3dbc6eb](https://github.com/sebastian-software/numpy-node/commit/3dbc6ebaddc2369f7afd28e0b28454da2655f672))
* **creation:** add array creation functions ([5f62ff5](https://github.com/sebastian-software/numpy-node/commit/5f62ff5b5f7414d65c76fb3966192b0f41a917df))
* export linalg module from main entry point ([87a9e38](https://github.com/sebastian-software/numpy-node/commit/87a9e388854545b00aa03f37cab1b7c67a525c3b))
* **linalg:** add linear algebra module ([2d36f0a](https://github.com/sebastian-software/numpy-node/commit/2d36f0a21daf4b39afe6fd4db9e1df10c58d5e5c))
* **linalg:** implement eigvals, eig, svd, qr, cholesky, matrix_rank ([f4cf8f9](https://github.com/sebastian-software/numpy-node/commit/f4cf8f901e471fed348ab18eeee26c237a4a7e11))
* **math:** implement axis-based reductions for sum, mean, std ([15c4b0c](https://github.com/sebastian-software/numpy-node/commit/15c4b0cbfb7e2099a3612315e19d37342852c2ff))
* **math:** implement NumPy-style broadcasting for binary operations ([86969bc](https://github.com/sebastian-software/numpy-node/commit/86969bcef68692c6770a83732c08a7a70f0081de))
* **native:** add C++ N-API module scaffolding ([1f7492d](https://github.com/sebastian-software/numpy-node/commit/1f7492d00c054e778689144f2c3ba6e7b8b710c5))
* **ops:** add array operations ([15a393f](https://github.com/sebastian-software/numpy-node/commit/15a393f90c5e890ff070ebe7ca42054d3e4c498b))
* **stats:** add statistical functions ([81f961b](https://github.com/sebastian-software/numpy-node/commit/81f961b684ccee0116f96758f57a1d1726aac475))


### Bug Fixes

* add missing cmath include and extend CI matrix ([6010e20](https://github.com/sebastian-software/numpy-node/commit/6010e2014eb7f2ec2ead92ff86455c87d552e64d))
* **native:** configure cmake-js and fix build errors ([2929e6c](https://github.com/sebastian-software/numpy-node/commit/2929e6c910b1d8907029784260cdf03318b7d1f5))
* **ops:** fix reduceAxis keepdims indexing bug ([88a4c02](https://github.com/sebastian-software/numpy-node/commit/88a4c02535dc6baf27bde997792a2ba818fbdeda))
* **stats:** fix average and variance functions ([00e5ff4](https://github.com/sebastian-software/numpy-node/commit/00e5ff4dc66ca91c8b5c23c6dea46dd9f272a790))
* update CI to use Node 22 and refresh lockfile ([15b5176](https://github.com/sebastian-software/numpy-node/commit/15b5176c627436ebb227881fbe12f37583547b89))
* use [@numpy-node](https://github.com/numpy-node) scope for platform packages ([94a563f](https://github.com/sebastian-software/numpy-node/commit/94a563ff3063bf04264decbadde74593852236ea))


### Code Refactoring

* replace pure TypeScript with native C++ bindings ([a8fa1ce](https://github.com/sebastian-software/numpy-node/commit/a8fa1cec88e7d550cb15ed4e52df066e72b665bf))
