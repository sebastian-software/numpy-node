# Changelog

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
