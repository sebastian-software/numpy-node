#include "linalg.h"
#include <cmath>
#include <limits>
#include <cstring>
#include <algorithm>

// Platform-specific includes
#if defined(USE_ACCELERATE)
    #include <Accelerate/Accelerate.h>
#elif defined(USE_OPENBLAS)
    extern "C" {
        // BLAS functions
        void dgemm_(const char* transa, const char* transb,
                   const int* m, const int* n, const int* k,
                   const double* alpha, const double* a, const int* lda,
                   const double* b, const int* ldb,
                   const double* beta, double* c, const int* ldc);

        double ddot_(const int* n, const double* x, const int* incx,
                    const double* y, const int* incy);

        void dgemv_(const char* trans, const int* m, const int* n,
                   const double* alpha, const double* a, const int* lda,
                   const double* x, const int* incx,
                   const double* beta, double* y, const int* incy);

        // LAPACK functions
        void dgetrf_(const int* m, const int* n, double* a, const int* lda,
                    int* ipiv, int* info);

        void dgetri_(const int* n, double* a, const int* lda,
                    const int* ipiv, double* work, const int* lwork, int* info);

        void dgesv_(const int* n, const int* nrhs, double* a, const int* lda,
                   int* ipiv, double* b, const int* ldb, int* info);

        void dgeev_(const char* jobvl, const char* jobvr,
                   const int* n, double* a, const int* lda,
                   double* wr, double* wi,
                   double* vl, const int* ldvl,
                   double* vr, const int* ldvr,
                   double* work, const int* lwork, int* info);

        void dgesvd_(const char* jobu, const char* jobvt,
                    const int* m, const int* n, double* a, const int* lda,
                    double* s, double* u, const int* ldu,
                    double* vt, const int* ldvt,
                    double* work, const int* lwork, int* info);

        void dgeqrf_(const int* m, const int* n, double* a, const int* lda,
                    double* tau, double* work, const int* lwork, int* info);

        void dorgqr_(const int* m, const int* n, const int* k,
                    double* a, const int* lda, const double* tau,
                    double* work, const int* lwork, int* info);

        void dpotrf_(const char* uplo, const int* n, double* a, const int* lda,
                    int* info);

        // Least squares solver
        void dgels_(const char* trans, const int* m, const int* n, const int* nrhs,
                   double* a, const int* lda, double* b, const int* ldb,
                   double* work, const int* lwork, int* info);

        // Symmetric rank-k update: C = alpha * A' * A + beta * C
        void dsyrk_(const char* uplo, const char* trans, const int* n, const int* k,
                   const double* alpha, const double* a, const int* lda,
                   const double* beta, double* c, const int* ldc);

        // Symmetric positive definite solve
        void dposv_(const char* uplo, const int* n, const int* nrhs,
                   double* a, const int* lda, double* b, const int* ldb, int* info);
    }
#endif

namespace numpy_node {
namespace linalg {

/**
 * Helper: ensure array is contiguous, making a copy if needed.
 * Returns the contiguous array (may be same as input if already contiguous).
 * If a copy was made, outCopy will hold a reference to prevent deletion.
 */
static NativeNDArray* ensureContiguous(const Napi::CallbackInfo& info,
                                       NativeNDArray* arr,
                                       Napi::Object& outCopy) {
    if (arr->is_contiguous()) {
        return arr;
    }

    // Make contiguous copy
    outCopy = arr->AsContiguous(info).As<Napi::Object>();
    return Napi::ObjectWrap<NativeNDArray>::Unwrap(outCopy);
}

Napi::Value Matmul(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected two arrays").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* aOrig = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    NativeNDArray* bOrig = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());

    if (aOrig->ndim() != 2 || bOrig->ndim() != 2) {
        Napi::Error::New(env, "matmul requires 2D arrays").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Ensure inputs are contiguous for BLAS
    Napi::Object aCopy, bCopy;
    NativeNDArray* a = ensureContiguous(info, aOrig, aCopy);
    NativeNDArray* b = ensureContiguous(info, bOrig, bCopy);

    int m = static_cast<int>(a->shape()[0]);
    int k = static_cast<int>(a->shape()[1]);
    int n = static_cast<int>(b->shape()[1]);

    if (k != static_cast<int>(b->shape()[0])) {
        Napi::Error::New(env, "Matrix dimensions incompatible for multiplication").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Create result array
    Napi::Array shape = Napi::Array::New(env, 2);
    shape.Set(uint32_t(0), Napi::Number::New(env, m));
    shape.Set(uint32_t(1), Napi::Number::New(env, n));

    Napi::Object result = NativeNDArray::constructor.New({shape, Napi::String::New(env, "float64")});
    NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);

    double* dataA = static_cast<double*>(a->data());
    double* dataB = static_cast<double*>(b->data());
    double* dataC = static_cast<double*>(c->data());

#if defined(USE_ACCELERATE)
    // Use Accelerate's BLAS
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k,
                1.0, dataA, k,
                dataB, n,
                0.0, dataC, n);
#elif defined(USE_OPENBLAS)
    // Use OpenBLAS - note: FORTRAN column-major order
    char transA = 'N', transB = 'N';
    double alpha = 1.0, beta = 0.0;
    dgemm_(&transB, &transA, &n, &m, &k, &alpha, dataB, &n, dataA, &k, &beta, dataC, &n);
#else
    // Pure C++ fallback
    std::memset(dataC, 0, m * n * sizeof(double));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int l = 0; l < k; l++) {
                sum += dataA[i * k + l] * dataB[l * n + j];
            }
            dataC[i * n + j] = sum;
        }
    }
#endif

    return result;
}

/**
 * Compute A @ B.T without explicit transpose.
 * Uses BLAS dgemm with transB='T' for better performance.
 * Common in attention mechanisms: Q @ K.T
 */
Napi::Value MatmulNT(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected two arrays").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* aOrig = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    NativeNDArray* bOrig = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());

    if (aOrig->ndim() != 2 || bOrig->ndim() != 2) {
        Napi::Error::New(env, "matmul_nt requires 2D arrays").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Ensure inputs are contiguous for BLAS
    Napi::Object aCopy, bCopy;
    NativeNDArray* a = ensureContiguous(info, aOrig, aCopy);
    NativeNDArray* b = ensureContiguous(info, bOrig, bCopy);

    // A is m×k, B is n×k (will be transposed to k×n), result is m×n
    int m = static_cast<int>(a->shape()[0]);
    int k = static_cast<int>(a->shape()[1]);
    int n = static_cast<int>(b->shape()[0]);  // B's rows become result columns after transpose

    if (k != static_cast<int>(b->shape()[1])) {
        Napi::Error::New(env, "Matrix dimensions incompatible: A columns must match B columns").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Create result array (m × n)
    Napi::Array shape = Napi::Array::New(env, 2);
    shape.Set(uint32_t(0), Napi::Number::New(env, m));
    shape.Set(uint32_t(1), Napi::Number::New(env, n));

    Napi::Object result = NativeNDArray::constructor.New({shape, Napi::String::New(env, "float64")});
    NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);

    double* dataA = static_cast<double*>(a->data());
    double* dataB = static_cast<double*>(b->data());
    double* dataC = static_cast<double*>(c->data());

#if defined(USE_ACCELERATE)
    // Use Accelerate's BLAS with transpose on B
    // C = A @ B.T where A is m×k, B is n×k
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                m, n, k,
                1.0, dataA, k,
                dataB, k,  // B is n×k, leading dim is k
                0.0, dataC, n);
#elif defined(USE_OPENBLAS)
    // Use OpenBLAS - note: FORTRAN column-major order
    // Row-major C = A @ B.T becomes column-major C.T = B @ A.T
    char transA = 'N', transB = 'T';
    double alpha = 1.0, beta = 0.0;
    dgemm_(&transA, &transB, &n, &m, &k, &alpha, dataB, &k, dataA, &k, &beta, dataC, &n);
#else
    // Pure C++ fallback: C[i,j] = sum_l A[i,l] * B[j,l]
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int l = 0; l < k; l++) {
                sum += dataA[i * k + l] * dataB[j * k + l];
            }
            dataC[i * n + j] = sum;
        }
    }
#endif

    return result;
}

Napi::Value Dot(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected two arrays").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    NativeNDArray* b = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());

    // 1D dot product
    if (a->ndim() == 1 && b->ndim() == 1) {
        if (a->size() != b->size()) {
            Napi::Error::New(env, "Vectors must have same length").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        double* dataA = static_cast<double*>(a->data());
        double* dataB = static_cast<double*>(b->data());
        int n = static_cast<int>(a->size());

#if defined(USE_ACCELERATE)
        double result = cblas_ddot(n, dataA, 1, dataB, 1);
        return Napi::Number::New(env, result);
#elif defined(USE_OPENBLAS)
        int incx = 1, incy = 1;
        double result = ddot_(&n, dataA, &incx, dataB, &incy);
        return Napi::Number::New(env, result);
#else
        double result = 0.0;
        for (int i = 0; i < n; i++) {
            result += dataA[i] * dataB[i];
        }
        return Napi::Number::New(env, result);
#endif
    }

    // 2D case: matrix multiplication
    if (a->ndim() == 2 && b->ndim() == 2) {
        return Matmul(info);
    }

    // Matrix-vector: (m, k) @ (k,) -> (m,)
    if (a->ndim() == 2 && b->ndim() == 1) {
        int m = static_cast<int>(a->shape()[0]);
        int k = static_cast<int>(a->shape()[1]);

        if (k != static_cast<int>(b->size())) {
            Napi::Error::New(env, "Matrix columns must match vector length").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        // Create result array of shape (m,)
        Napi::Array resultShape = Napi::Array::New(env, 1);
        resultShape.Set(uint32_t(0), Napi::Number::New(env, m));
        Napi::Object result = NativeNDArray::constructor.New({resultShape, Napi::String::New(env, "float64")});
        NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);

        double* dataA = static_cast<double*>(a->data());
        double* dataB = static_cast<double*>(b->data());
        double* dataC = static_cast<double*>(c->data());

#if defined(USE_ACCELERATE)
        // y = alpha * A * x + beta * y (row-major)
        cblas_dgemv(CblasRowMajor, CblasNoTrans, m, k, 1.0, dataA, k, dataB, 1, 0.0, dataC, 1);
#elif defined(USE_OPENBLAS)
        // For Fortran dgemv, we need column-major, but our data is row-major
        // So we compute A^T * x with column-major interpretation
        char trans = 'T';
        double alpha = 1.0, beta = 0.0;
        int incx = 1, incy = 1;
        dgemv_(&trans, &k, &m, &alpha, dataA, &k, dataB, &incx, &beta, dataC, &incy);
#else
        // Manual computation
        for (int i = 0; i < m; i++) {
            double sum = 0.0;
            for (int j = 0; j < k; j++) {
                sum += dataA[i * k + j] * dataB[j];
            }
            dataC[i] = sum;
        }
#endif
        return result;
    }

    // Vector-matrix: (k,) @ (k, n) -> (n,)
    if (a->ndim() == 1 && b->ndim() == 2) {
        int k = static_cast<int>(a->size());
        int n = static_cast<int>(b->shape()[1]);

        if (k != static_cast<int>(b->shape()[0])) {
            Napi::Error::New(env, "Vector length must match matrix rows").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        // Create result array of shape (n,)
        Napi::Array resultShape = Napi::Array::New(env, 1);
        resultShape.Set(uint32_t(0), Napi::Number::New(env, n));
        Napi::Object result = NativeNDArray::constructor.New({resultShape, Napi::String::New(env, "float64")});
        NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);

        double* dataA = static_cast<double*>(a->data());
        double* dataB = static_cast<double*>(b->data());
        double* dataC = static_cast<double*>(c->data());

#if defined(USE_ACCELERATE)
        // y = alpha * A^T * x + beta * y (row-major), but we want x^T * A = (A^T * x)^T
        // In row-major: y = A^T * x where A is (k, n), x is (k,), y is (n,)
        cblas_dgemv(CblasRowMajor, CblasTrans, k, n, 1.0, dataB, n, dataA, 1, 0.0, dataC, 1);
#elif defined(USE_OPENBLAS)
        char trans = 'N';
        double alpha = 1.0, beta = 0.0;
        int incx = 1, incy = 1;
        dgemv_(&trans, &n, &k, &alpha, dataB, &n, dataA, &incx, &beta, dataC, &incy);
#else
        // Manual computation
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int i = 0; i < k; i++) {
                sum += dataA[i] * dataB[i * n + j];
            }
            dataC[j] = sum;
        }
#endif
        return result;
    }

    Napi::Error::New(env, "dot only supports 1D and 2D arrays").ThrowAsJavaScriptException();
    return env.Undefined();
}

Napi::Value Det(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());

    if (a->ndim() != 2 || a->shape()[0] != a->shape()[1]) {
        Napi::Error::New(env, "Matrix must be square").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int n = static_cast<int>(a->shape()[0]);

    // Make a copy for LU decomposition
    std::vector<double> work(n * n);
    std::memcpy(work.data(), a->data(), n * n * sizeof(double));

    std::vector<int> ipiv(n);
    int lapackInfo = 0;

#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
    dgetrf_(&n, &n, work.data(), &n, ipiv.data(), &lapackInfo);

    if (lapackInfo != 0) {
        return Napi::Number::New(env, 0.0);
    }

    double det = 1.0;
    int signSwaps = 0;
    for (int i = 0; i < n; i++) {
        det *= work[i * n + i];
        if (ipiv[i] != i + 1) signSwaps++;
    }
    if (signSwaps % 2 != 0) det = -det;

    return Napi::Number::New(env, det);
#else
    // Simple determinant for small matrices
    double* data = static_cast<double*>(a->data());
    if (n == 2) {
        double det = data[0] * data[3] - data[1] * data[2];
        return Napi::Number::New(env, det);
    } else if (n == 3) {
        double det = data[0] * (data[4] * data[8] - data[5] * data[7])
                   - data[1] * (data[3] * data[8] - data[5] * data[6])
                   + data[2] * (data[3] * data[7] - data[4] * data[6]);
        return Napi::Number::New(env, det);
    }

    Napi::Error::New(env, "Determinant requires LAPACK for matrices larger than 3x3").ThrowAsJavaScriptException();
    return env.Undefined();
#endif
}

Napi::Value Inv(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());

    if (a->ndim() != 2 || a->shape()[0] != a->shape()[1]) {
        Napi::Error::New(env, "Matrix must be square").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int n = static_cast<int>(a->shape()[0]);

    // Create result array with copy of input
    Napi::Array shape = Napi::Array::New(env, 2);
    shape.Set(uint32_t(0), Napi::Number::New(env, n));
    shape.Set(uint32_t(1), Napi::Number::New(env, n));

    Napi::Object result = NativeNDArray::constructor.New({shape, Napi::String::New(env, "float64")});
    NativeNDArray* inv = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    std::memcpy(inv->data(), a->data(), n * n * sizeof(double));

#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
    double* invData = static_cast<double*>(inv->data());
    std::vector<int> ipiv(n);
    int lapackInfo = 0;

    // LU decomposition
    dgetrf_(&n, &n, invData, &n, ipiv.data(), &lapackInfo);
    if (lapackInfo != 0) {
        Napi::Error::New(env, "Matrix is singular").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Query workspace size
    int lwork = -1;
    double workQuery;
    dgetri_(&n, invData, &n, ipiv.data(), &workQuery, &lwork, &lapackInfo);

    lwork = static_cast<int>(workQuery);
    std::vector<double> work(lwork);

    // Compute inverse
    dgetri_(&n, invData, &n, ipiv.data(), work.data(), &lwork, &lapackInfo);
    if (lapackInfo != 0) {
        Napi::Error::New(env, "Matrix inversion failed").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    return result;
#else
    // Simple 2x2 inverse
    if (n == 2) {
        double* data = static_cast<double*>(a->data());
        double* invData = static_cast<double*>(inv->data());
        double det = data[0] * data[3] - data[1] * data[2];
        if (std::abs(det) < 1e-10) {
            Napi::Error::New(env, "Matrix is singular").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        invData[0] = data[3] / det;
        invData[1] = -data[1] / det;
        invData[2] = -data[2] / det;
        invData[3] = data[0] / det;
        return result;
    }

    Napi::Error::New(env, "Matrix inverse requires LAPACK for matrices larger than 2x2").ThrowAsJavaScriptException();
    return env.Undefined();
#endif
}

Napi::Value Solve(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected two arrays (A and b)").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    NativeNDArray* b = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());

    if (a->ndim() != 2 || a->shape()[0] != a->shape()[1]) {
        Napi::Error::New(env, "A must be a square matrix").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int n = static_cast<int>(a->shape()[0]);
    int nrhs = b->ndim() == 1 ? 1 : static_cast<int>(b->shape()[1]);

#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
    // Convert A from row-major to column-major for LAPACK
    // dgesv expects column-major storage
    std::vector<double> aCopy(n * n);
    double* aData = static_cast<double*>(a->data());
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // row-major: A[i,j] = aData[i*n + j]
            // col-major: aCopy[j*n + i]
            aCopy[j * n + i] = aData[i * n + j];
        }
    }

    // Copy b to solution vector (dgesv overwrites b with solution)
    // For 1D b: just copy directly (vector, no layout change needed)
    // For 2D b: convert from row-major to column-major
    std::vector<double> bCopy(n * nrhs);
    double* bData = static_cast<double*>(b->data());
    if (b->ndim() == 1) {
        std::memcpy(bCopy.data(), bData, n * sizeof(double));
    } else {
        // b is 2D (n x nrhs), convert to column-major
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < nrhs; j++) {
                bCopy[j * n + i] = bData[i * nrhs + j];
            }
        }
    }

    std::vector<int> ipiv(n);
    int lapackInfo = 0;

    dgesv_(&n, &nrhs, aCopy.data(), &n, ipiv.data(),
           bCopy.data(), &n, &lapackInfo);

    if (lapackInfo != 0) {
        Napi::Error::New(env, "Linear solve failed").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Create result array and copy solution
    Napi::Array shape;
    if (b->ndim() == 1) {
        shape = Napi::Array::New(env, 1);
        shape.Set(uint32_t(0), Napi::Number::New(env, n));
    } else {
        shape = Napi::Array::New(env, 2);
        shape.Set(uint32_t(0), Napi::Number::New(env, n));
        shape.Set(uint32_t(1), Napi::Number::New(env, nrhs));
    }

    Napi::Object result = NativeNDArray::constructor.New({shape, Napi::String::New(env, "float64")});
    NativeNDArray* x = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* xData = static_cast<double*>(x->data());

    if (b->ndim() == 1) {
        // 1D result: copy directly
        std::memcpy(xData, bCopy.data(), n * sizeof(double));
    } else {
        // 2D result: convert from column-major back to row-major
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < nrhs; j++) {
                xData[i * nrhs + j] = bCopy[j * n + i];
            }
        }
    }

    return result;
#else
    Napi::Error::New(env, "Linear solve requires LAPACK").ThrowAsJavaScriptException();
    return env.Undefined();
#endif
}

Napi::Value Norm(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    double* data = static_cast<double*>(a->data());
    int64_t size = a->size();

    // Determine norm type from second argument
    // Default: L2 for vectors, Frobenius for matrices (both compute same way)
    int normType = 2; // 0=L1, 1=Inf, 2=L2/Frobenius

    if (info.Length() > 1 && !info[1].IsUndefined()) {
        if (info[1].IsString()) {
            std::string ordStr = info[1].As<Napi::String>().Utf8Value();
            if (ordStr == "fro") {
                normType = 2; // Frobenius = L2 for flattened data
            } else if (ordStr == "inf" || ordStr == "Infinity") {
                normType = 1; // Infinity norm (string form)
            }
        } else if (info[1].IsNumber()) {
            double ord = info[1].As<Napi::Number>().DoubleValue();
            if (ord == 1.0) {
                normType = 0; // L1
            } else if (std::isinf(ord)) {
                normType = 1; // Infinity
            } else {
                normType = 2; // L2 (default for ord=2 or other)
            }
        }
    }

    double result = 0.0;

    switch (normType) {
        case 0: // L1 norm: sum of absolute values
            for (int64_t i = 0; i < size; i++) {
                result += std::abs(data[i]);
            }
            break;
        case 1: // Infinity norm: max absolute value
            for (int64_t i = 0; i < size; i++) {
                double absVal = std::abs(data[i]);
                if (absVal > result) {
                    result = absVal;
                }
            }
            break;
        case 2: // L2/Frobenius norm: sqrt of sum of squares
        default:
            for (int64_t i = 0; i < size; i++) {
                result += data[i] * data[i];
            }
            result = std::sqrt(result);
            break;
    }

    return Napi::Number::New(env, result);
}

Napi::Value Trace(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());

    if (a->ndim() != 2) {
        Napi::Error::New(env, "trace requires 2D array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int rows = static_cast<int>(a->shape()[0]);
    int cols = static_cast<int>(a->shape()[1]);
    int diagLen = std::min(rows, cols);

    double* data = static_cast<double*>(a->data());
    double trace = 0.0;

    for (int i = 0; i < diagLen; i++) {
        trace += data[i * cols + i];
    }

    return Napi::Number::New(env, trace);
}

// Helper to transpose array (row-major to column-major)
static void transpose(const double* src, double* dst, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

// Helper to transpose array (column-major to row-major)
static void transposeBack(const double* src, double* dst, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dst[i * cols + j] = src[i + j * rows];
        }
    }
}

Napi::Value Eigvals(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());

    if (a->ndim() != 2 || a->shape()[0] != a->shape()[1]) {
        Napi::Error::New(env, "Matrix must be square").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int n = static_cast<int>(a->shape()[0]);

#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
    // Copy and transpose to column-major order
    std::vector<double> aCopy(n * n);
    transpose(static_cast<double*>(a->data()), aCopy.data(), n, n);

    std::vector<double> wr(n), wi(n);
    std::vector<double> work(1);
    int lwork = -1;
    int lapackInfo = 0;

    char jobvl = 'N', jobvr = 'N';
    double* vl = nullptr;
    double* vr = nullptr;
    int ldvl = 1, ldvr = 1;

    // Query workspace size
    dgeev_(&jobvl, &jobvr, &n, aCopy.data(), &n, wr.data(), wi.data(),
           vl, &ldvl, vr, &ldvr, work.data(), &lwork, &lapackInfo);

    lwork = static_cast<int>(work[0]);
    work.resize(lwork);

    // Compute eigenvalues
    dgeev_(&jobvl, &jobvr, &n, aCopy.data(), &n, wr.data(), wi.data(),
           vl, &ldvl, vr, &ldvr, work.data(), &lwork, &lapackInfo);

    if (lapackInfo != 0) {
        Napi::Error::New(env, "Eigenvalue computation failed").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Create result array (real parts only for now)
    Napi::Array shape = Napi::Array::New(env, 1);
    shape.Set(uint32_t(0), Napi::Number::New(env, n));

    Napi::Object result = NativeNDArray::constructor.New({shape, Napi::String::New(env, "float64")});
    NativeNDArray* eigenvalues = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    std::memcpy(eigenvalues->data(), wr.data(), n * sizeof(double));

    return result;
#else
    Napi::Error::New(env, "eigvals requires LAPACK").ThrowAsJavaScriptException();
    return env.Undefined();
#endif
}

Napi::Value Eig(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());

    if (a->ndim() != 2 || a->shape()[0] != a->shape()[1]) {
        Napi::Error::New(env, "Matrix must be square").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int n = static_cast<int>(a->shape()[0]);

#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
    // Copy and transpose to column-major order
    std::vector<double> aCopy(n * n);
    transpose(static_cast<double*>(a->data()), aCopy.data(), n, n);

    std::vector<double> wr(n), wi(n);
    std::vector<double> vr(n * n);
    std::vector<double> work(1);
    int lwork = -1;
    int lapackInfo = 0;

    char jobvl = 'N', jobvr = 'V';
    double* vl = nullptr;
    int ldvl = 1, ldvr = n;

    // Query workspace size
    dgeev_(&jobvl, &jobvr, &n, aCopy.data(), &n, wr.data(), wi.data(),
           vl, &ldvl, vr.data(), &ldvr, work.data(), &lwork, &lapackInfo);

    lwork = static_cast<int>(work[0]);
    work.resize(lwork);

    // Compute eigenvalues and eigenvectors
    dgeev_(&jobvl, &jobvr, &n, aCopy.data(), &n, wr.data(), wi.data(),
           vl, &ldvl, vr.data(), &ldvr, work.data(), &lwork, &lapackInfo);

    if (lapackInfo != 0) {
        Napi::Error::New(env, "Eigenvalue computation failed").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Create eigenvalues array
    Napi::Array evalShape = Napi::Array::New(env, 1);
    evalShape.Set(uint32_t(0), Napi::Number::New(env, n));
    Napi::Object eigenvalues = NativeNDArray::constructor.New({evalShape, Napi::String::New(env, "float64")});
    NativeNDArray* evalArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(eigenvalues);
    std::memcpy(evalArr->data(), wr.data(), n * sizeof(double));

    // Create eigenvectors array (transpose back to row-major)
    Napi::Array evecShape = Napi::Array::New(env, 2);
    evecShape.Set(uint32_t(0), Napi::Number::New(env, n));
    evecShape.Set(uint32_t(1), Napi::Number::New(env, n));
    Napi::Object eigenvectors = NativeNDArray::constructor.New({evecShape, Napi::String::New(env, "float64")});
    NativeNDArray* evecArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(eigenvectors);
    transposeBack(vr.data(), static_cast<double*>(evecArr->data()), n, n);

    // Return object with eigenvalues and eigenvectors
    Napi::Object result = Napi::Object::New(env);
    result.Set("eigenvalues", eigenvalues);
    result.Set("eigenvectors", eigenvectors);
    return result;
#else
    Napi::Error::New(env, "eig requires LAPACK").ThrowAsJavaScriptException();
    return env.Undefined();
#endif
}

Napi::Value Svd(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());

    if (a->ndim() != 2) {
        Napi::Error::New(env, "svd requires 2D array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int m = static_cast<int>(a->shape()[0]);
    int n = static_cast<int>(a->shape()[1]);
    int minmn = std::min(m, n);

#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
    // Copy and transpose to column-major order
    std::vector<double> aCopy(m * n);
    transpose(static_cast<double*>(a->data()), aCopy.data(), m, n);

    std::vector<double> s(minmn);
    // Economy SVD: U is m x minmn, Vt is minmn x n
    std::vector<double> u(m * minmn);
    std::vector<double> vt(minmn * n);
    std::vector<double> work(1);
    int lwork = -1;
    int lapackInfo = 0;

    // Use 'S' for economy (thin) SVD - much faster for non-square matrices
    char jobu = 'S', jobvt = 'S';

    // Query workspace size
    dgesvd_(&jobu, &jobvt, &m, &n, aCopy.data(), &m,
            s.data(), u.data(), &m, vt.data(), &minmn,
            work.data(), &lwork, &lapackInfo);

    lwork = static_cast<int>(work[0]);
    work.resize(lwork);

    // Compute SVD
    dgesvd_(&jobu, &jobvt, &m, &n, aCopy.data(), &m,
            s.data(), u.data(), &m, vt.data(), &minmn,
            work.data(), &lwork, &lapackInfo);

    if (lapackInfo != 0) {
        Napi::Error::New(env, "SVD computation failed").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Create U array (transpose back to row-major): m x minmn
    Napi::Array uShape = Napi::Array::New(env, 2);
    uShape.Set(uint32_t(0), Napi::Number::New(env, m));
    uShape.Set(uint32_t(1), Napi::Number::New(env, minmn));
    Napi::Object uResult = NativeNDArray::constructor.New({uShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)});
    NativeNDArray* uArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(uResult);
    transposeBack(u.data(), static_cast<double*>(uArr->data()), minmn, m);

    // Create S array
    Napi::Array sShape = Napi::Array::New(env, 1);
    sShape.Set(uint32_t(0), Napi::Number::New(env, minmn));
    Napi::Object sResult = NativeNDArray::constructor.New({sShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)});
    NativeNDArray* sArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(sResult);
    std::memcpy(sArr->data(), s.data(), minmn * sizeof(double));

    // Create Vh array (transpose back to row-major): minmn x n
    Napi::Array vhShape = Napi::Array::New(env, 2);
    vhShape.Set(uint32_t(0), Napi::Number::New(env, minmn));
    vhShape.Set(uint32_t(1), Napi::Number::New(env, n));
    Napi::Object vhResult = NativeNDArray::constructor.New({vhShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)});
    NativeNDArray* vhArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(vhResult);
    transposeBack(vt.data(), static_cast<double*>(vhArr->data()), n, minmn);

    // Return object with u, s, vh
    Napi::Object result = Napi::Object::New(env);
    result.Set("u", uResult);
    result.Set("s", sResult);
    result.Set("vh", vhResult);
    return result;
#else
    Napi::Error::New(env, "svd requires LAPACK").ThrowAsJavaScriptException();
    return env.Undefined();
#endif
}

Napi::Value Qr(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());

    if (a->ndim() != 2) {
        Napi::Error::New(env, "qr requires 2D array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int m = static_cast<int>(a->shape()[0]);
    int n = static_cast<int>(a->shape()[1]);
    int k = std::min(m, n);

#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
    // Copy and transpose to column-major order
    std::vector<double> aCopy(m * n);
    transpose(static_cast<double*>(a->data()), aCopy.data(), m, n);

    std::vector<double> tau(k);
    std::vector<double> work(1);
    int lwork = -1;
    int lapackInfo = 0;

    // Query workspace size for QR
    dgeqrf_(&m, &n, aCopy.data(), &m, tau.data(), work.data(), &lwork, &lapackInfo);
    lwork = static_cast<int>(work[0]);
    work.resize(lwork);

    // Compute QR factorization
    dgeqrf_(&m, &n, aCopy.data(), &m, tau.data(), work.data(), &lwork, &lapackInfo);

    if (lapackInfo != 0) {
        Napi::Error::New(env, "QR factorization failed").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Extract R (upper triangular part) - still in column-major
    std::vector<double> rColMajor(m * n, 0.0);
    for (int i = 0; i < m; i++) {
        for (int j = i; j < n; j++) {
            rColMajor[i + j * m] = aCopy[i + j * m];
        }
    }

    // Generate Q from Householder reflectors
    lwork = -1;
    dorgqr_(&m, &k, &k, aCopy.data(), &m, tau.data(), work.data(), &lwork, &lapackInfo);
    lwork = static_cast<int>(work[0]);
    work.resize(lwork);

    dorgqr_(&m, &k, &k, aCopy.data(), &m, tau.data(), work.data(), &lwork, &lapackInfo);

    if (lapackInfo != 0) {
        Napi::Error::New(env, "Q generation failed").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Create Q array (transpose back to row-major)
    Napi::Array qShape = Napi::Array::New(env, 2);
    qShape.Set(uint32_t(0), Napi::Number::New(env, m));
    qShape.Set(uint32_t(1), Napi::Number::New(env, k));
    Napi::Object qResult = NativeNDArray::constructor.New({qShape, Napi::String::New(env, "float64")});
    NativeNDArray* qArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(qResult);
    transposeBack(aCopy.data(), static_cast<double*>(qArr->data()), m, k);

    // Create R array (transpose back to row-major)
    Napi::Array rShape = Napi::Array::New(env, 2);
    rShape.Set(uint32_t(0), Napi::Number::New(env, k));
    rShape.Set(uint32_t(1), Napi::Number::New(env, n));
    Napi::Object rResult = NativeNDArray::constructor.New({rShape, Napi::String::New(env, "float64")});
    NativeNDArray* rArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(rResult);
    // Only copy the top k rows of R
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            static_cast<double*>(rArr->data())[i * n + j] = rColMajor[i + j * m];
        }
    }

    // Return object with q and r
    Napi::Object result = Napi::Object::New(env);
    result.Set("q", qResult);
    result.Set("r", rResult);
    return result;
#else
    Napi::Error::New(env, "qr requires LAPACK").ThrowAsJavaScriptException();
    return env.Undefined();
#endif
}

Napi::Value Cholesky(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());

    if (a->ndim() != 2 || a->shape()[0] != a->shape()[1]) {
        Napi::Error::New(env, "Matrix must be square").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int n = static_cast<int>(a->shape()[0]);

#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
    // Copy and transpose to column-major order
    std::vector<double> aCopy(n * n);
    transpose(static_cast<double*>(a->data()), aCopy.data(), n, n);

    int lapackInfo = 0;
    char uplo = 'L';  // Lower triangular

    dpotrf_(&uplo, &n, aCopy.data(), &n, &lapackInfo);

    if (lapackInfo != 0) {
        Napi::Error::New(env, "Cholesky decomposition failed (matrix not positive definite)").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Zero out upper triangle in column-major
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            aCopy[i + j * n] = 0.0;
        }
    }

    // Create result array (transpose back to row-major)
    Napi::Array shape = Napi::Array::New(env, 2);
    shape.Set(uint32_t(0), Napi::Number::New(env, n));
    shape.Set(uint32_t(1), Napi::Number::New(env, n));
    Napi::Object result = NativeNDArray::constructor.New({shape, Napi::String::New(env, "float64")});
    NativeNDArray* L = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    transposeBack(aCopy.data(), static_cast<double*>(L->data()), n, n);

    return result;
#else
    Napi::Error::New(env, "cholesky requires LAPACK").ThrowAsJavaScriptException();
    return env.Undefined();
#endif
}

Napi::Value MatrixRank(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());

    if (a->ndim() != 2) {
        Napi::Error::New(env, "matrix_rank requires 2D array").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int m = static_cast<int>(a->shape()[0]);
    int n = static_cast<int>(a->shape()[1]);
    int minmn = std::min(m, n);

#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
    // Use SVD to compute rank
    std::vector<double> aCopy(m * n);
    transpose(static_cast<double*>(a->data()), aCopy.data(), m, n);

    std::vector<double> s(minmn);
    std::vector<double> work(1);
    int lwork = -1;
    int lapackInfo = 0;

    char jobu = 'N', jobvt = 'N';
    double* u = nullptr;
    double* vt = nullptr;
    int ldu = 1, ldvt = 1;

    // Query workspace size
    dgesvd_(&jobu, &jobvt, &m, &n, aCopy.data(), &m,
            s.data(), u, &ldu, vt, &ldvt,
            work.data(), &lwork, &lapackInfo);

    lwork = static_cast<int>(work[0]);
    work.resize(lwork);

    // Compute SVD (singular values only)
    dgesvd_(&jobu, &jobvt, &m, &n, aCopy.data(), &m,
            s.data(), u, &ldu, vt, &ldvt,
            work.data(), &lwork, &lapackInfo);

    if (lapackInfo != 0) {
        Napi::Error::New(env, "SVD for rank computation failed").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Count non-zero singular values
    double tol = std::max(m, n) * s[0] * 1e-14;
    int rank = 0;
    for (int i = 0; i < minmn; i++) {
        if (s[i] > tol) rank++;
    }

    return Napi::Number::New(env, rank);
#else
    Napi::Error::New(env, "matrix_rank requires LAPACK").ThrowAsJavaScriptException();
    return env.Undefined();
#endif
}

// Least squares solution: min ||b - A*x||
// This is much faster than computing (A'A)^-1 * A'b manually
Napi::Value Lstsq(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected two arrays (A, b)").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    NativeNDArray* b = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());

    if (a->ndim() != 2) {
        Napi::Error::New(env, "A must be 2D").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int m = static_cast<int>(a->shape()[0]);  // rows
    int n = static_cast<int>(a->shape()[1]);  // cols
    int nrhs = 1;  // number of right-hand sides

    // b can be 1D (m,) or 2D (m, nrhs)
    if (b->ndim() == 2) {
        nrhs = static_cast<int>(b->shape()[1]);
        if (static_cast<int>(b->shape()[0]) != m) {
            Napi::Error::New(env, "A and b have incompatible shapes").ThrowAsJavaScriptException();
            return env.Undefined();
        }
    } else if (b->ndim() == 1) {
        if (static_cast<int>(b->size()) != m) {
            Napi::Error::New(env, "A and b have incompatible shapes").ThrowAsJavaScriptException();
            return env.Undefined();
        }
    } else {
        Napi::Error::New(env, "b must be 1D or 2D").ThrowAsJavaScriptException();
        return env.Undefined();
    }

#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
    // Copy A to column-major and b to work array
    // dgels overwrites both A and b
    std::vector<double> aCopy(m * n);
    transpose(static_cast<double*>(a->data()), aCopy.data(), m, n);

    // b needs to be max(m, n) x nrhs for dgels
    int ldb = std::max(m, n);
    std::vector<double> bCopy(ldb * nrhs, 0.0);

    // Copy b (transpose if 2D)
    double* bData = static_cast<double*>(b->data());
    if (b->ndim() == 2) {
        // Transpose b to column-major
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < nrhs; j++) {
                bCopy[j * ldb + i] = bData[i * nrhs + j];
            }
        }
    } else {
        // 1D: just copy
        std::memcpy(bCopy.data(), bData, m * sizeof(double));
    }

    // Query workspace size
    char trans = 'N';  // No transpose (solve A*x = b)
    int lapackInfo = 0;
    double workQuery;
    int lwork = -1;

    dgels_(&trans, &m, &n, &nrhs, aCopy.data(), &m, bCopy.data(), &ldb,
           &workQuery, &lwork, &lapackInfo);

    lwork = static_cast<int>(workQuery);
    std::vector<double> work(lwork);

    // Solve least squares
    dgels_(&trans, &m, &n, &nrhs, aCopy.data(), &m, bCopy.data(), &ldb,
           work.data(), &lwork, &lapackInfo);

    if (lapackInfo != 0) {
        Napi::Error::New(env, "Least squares computation failed").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Create result array - solution x has shape (n,) or (n, nrhs)
    // The solution is in the first n rows of bCopy
    if (nrhs == 1) {
        Napi::Array xShape = Napi::Array::New(env, 1);
        xShape.Set(uint32_t(0), Napi::Number::New(env, n));
        Napi::Object xResult = NativeNDArray::constructor.New({
            xShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
        });
        NativeNDArray* xArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(xResult);
        std::memcpy(xArr->data(), bCopy.data(), n * sizeof(double));
        return xResult;
    } else {
        Napi::Array xShape = Napi::Array::New(env, 2);
        xShape.Set(uint32_t(0), Napi::Number::New(env, n));
        xShape.Set(uint32_t(1), Napi::Number::New(env, nrhs));
        Napi::Object xResult = NativeNDArray::constructor.New({
            xShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
        });
        NativeNDArray* xArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(xResult);
        double* xData = static_cast<double*>(xArr->data());
        // Transpose back to row-major
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < nrhs; j++) {
                xData[i * nrhs + j] = bCopy[j * ldb + i];
            }
        }
        return xResult;
    }
#else
    Napi::Error::New(env, "lstsq requires LAPACK").ThrowAsJavaScriptException();
    return env.Undefined();
#endif
}

/**
 * Solve normal equations: beta = (X'X)^(-1) X'y
 * Fuses X'X, X'y, and solve into a single native call.
 * Uses dsyrk for efficient X'X computation.
 */
Napi::Value NormalEquations(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected X and y").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* X = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    NativeNDArray* y = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());

    if (X->ndim() != 2) {
        Napi::Error::New(env, "X must be 2D").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int m = static_cast<int>(X->shape()[0]);  // Number of samples
    int n = static_cast<int>(X->shape()[1]);  // Number of features
    int nrhs = 1;
    bool y_is_2d = (y->ndim() == 2);

    if (y_is_2d) {
        if (y->shape()[0] != static_cast<size_t>(m)) {
            Napi::Error::New(env, "y must have same number of rows as X").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        nrhs = static_cast<int>(y->shape()[1]);
    } else if (y->ndim() == 1) {
        if (y->size() != static_cast<size_t>(m)) {
            Napi::Error::New(env, "y must have same length as X rows").ThrowAsJavaScriptException();
            return env.Undefined();
        }
    }

    double* dataX = static_cast<double*>(X->data());
    double* dataY = static_cast<double*>(y->data());

#if defined(USE_ACCELERATE)
    // Step 1: Compute X'X using dsyrk (symmetric rank-k update)
    // X'X where X is m x n, result is n x n
    std::vector<double> XtX(n * n, 0.0);

    // cblas_dsyrk computes C = alpha * A' * A + beta * C (when CblasTrans)
    // For row-major: A is m x n, we want A' * A which is n x n
    cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                n, m,           // n = result size, k = inner dimension (m rows of X)
                1.0, dataX, n,  // A = X, lda = n (number of columns)
                0.0, XtX.data(), n);  // C = XtX, ldc = n

    // Fill lower triangle (dsyrk only fills upper)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            XtX[i * n + j] = XtX[j * n + i];
        }
    }

    // Step 2: Compute X'y using dgemv
    // X'y where X is m x n and y is m x nrhs, result is n x nrhs
    std::vector<double> Xty(n * nrhs, 0.0);

    if (nrhs == 1) {
        // Vector case: use dgemv
        cblas_dgemv(CblasRowMajor, CblasTrans, m, n,
                    1.0, dataX, n, dataY, 1,
                    0.0, Xty.data(), 1);
    } else {
        // Matrix case: use dgemm
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    n, nrhs, m,
                    1.0, dataX, n, dataY, nrhs,
                    0.0, Xty.data(), nrhs);
    }

    // Step 3: Solve (X'X) * beta = X'y using dposv (symmetric positive definite solver)
    // dposv modifies both XtX and Xty in place
    // Convert to column-major for LAPACK
    std::vector<double> XtX_colmaj(n * n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            XtX_colmaj[j * n + i] = XtX[i * n + j];
        }
    }

    std::vector<double> Xty_colmaj(n * nrhs);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < nrhs; j++) {
            Xty_colmaj[j * n + i] = Xty[i * nrhs + j];
        }
    }

    char uplo = 'U';
    int lapackInfo = 0;
    dposv_(&uplo, &n, &nrhs, XtX_colmaj.data(), &n, Xty_colmaj.data(), &n, &lapackInfo);

    if (lapackInfo != 0) {
        // Fall back to general solver if positive definite solver fails
        // Recompute XtX and Xty since dposv modified them
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                XtX_colmaj[j * n + i] = XtX[i * n + j];
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < nrhs; j++) {
                Xty_colmaj[j * n + i] = Xty[i * nrhs + j];
            }
        }

        std::vector<int> ipiv(n);
        dgesv_(&n, &nrhs, XtX_colmaj.data(), &n, ipiv.data(), Xty_colmaj.data(), &n, &lapackInfo);

        if (lapackInfo != 0) {
            Napi::Error::New(env, "Normal equations solve failed").ThrowAsJavaScriptException();
            return env.Undefined();
        }
    }

    // Create result array - match y's dimensionality
    if (nrhs == 1 && !y_is_2d) {
        // y was 1D, return 1D result
        Napi::Array shape = Napi::Array::New(env, 1);
        shape.Set(uint32_t(0), Napi::Number::New(env, n));
        Napi::Object result = NativeNDArray::constructor.New({
            shape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
        });
        NativeNDArray* beta = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
        std::memcpy(beta->data(), Xty_colmaj.data(), n * sizeof(double));
        return result;
    } else {
        // y was 2D, return 2D result (n x nrhs)
        Napi::Array shape = Napi::Array::New(env, 2);
        shape.Set(uint32_t(0), Napi::Number::New(env, n));
        shape.Set(uint32_t(1), Napi::Number::New(env, nrhs));
        Napi::Object result = NativeNDArray::constructor.New({
            shape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
        });
        NativeNDArray* beta = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
        double* betaData = static_cast<double*>(beta->data());
        // Transpose back to row-major
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < nrhs; j++) {
                betaData[i * nrhs + j] = Xty_colmaj[j * n + i];
            }
        }
        return result;
    }
#elif defined(USE_OPENBLAS)
    // Step 1: Compute X'X using dsyrk
    std::vector<double> XtX(n * n, 0.0);

    // For Fortran dsyrk with row-major data:
    // We have X in row-major (m x n), need X' * X
    // Treat X as column-major n x m matrix, then X * X' gives n x n result
    char uplo = 'U';
    char trans = 'N';  // 'N' means A * A' for column-major input
    double alpha = 1.0, beta_val = 0.0;
    dsyrk_(&uplo, &trans, &n, &m, &alpha, dataX, &n, &beta_val, XtX.data(), &n);

    // Fill lower triangle
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            XtX[i * n + j] = XtX[j * n + i];
        }
    }

    // Step 2: Compute X'y using dgemv/dgemm
    std::vector<double> Xty(n * nrhs, 0.0);

    if (nrhs == 1) {
        // X' * y where X is m x n (stored row-major), y is m x 1
        // For Fortran dgemv with row-major X: treat as n x m column-major
        // Then 'N' gives (n x m) * (m x 1) = n x 1 which is X' * y
        char transv = 'N';
        int inc = 1;
        dgemv_(&transv, &n, &m, &alpha, dataX, &n, dataY, &inc, &beta_val, Xty.data(), &inc);
    } else {
        // Matrix case
        char transA = 'N', transB = 'N';
        dgemm_(&transB, &transA, &nrhs, &n, &m, &alpha, dataY, &nrhs, dataX, &n, &beta_val, Xty.data(), &nrhs);
    }

    // Step 3: Solve (X'X) * beta = X'y
    int lapackInfo = 0;
    dposv_(&uplo, &n, &nrhs, XtX.data(), &n, Xty.data(), &n, &lapackInfo);

    if (lapackInfo != 0) {
        // Fallback: recompute and use general solver
        std::fill(XtX.begin(), XtX.end(), 0.0);
        dsyrk_(&uplo, &trans, &n, &m, &alpha, dataX, &n, &beta_val, XtX.data(), &n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                XtX[i * n + j] = XtX[j * n + i];
            }
        }
        std::fill(Xty.begin(), Xty.end(), 0.0);
        if (nrhs == 1) {
            char transv = 'N';
            int inc = 1;
            dgemv_(&transv, &n, &m, &alpha, dataX, &n, dataY, &inc, &beta_val, Xty.data(), &inc);
        }

        std::vector<int> ipiv(n);
        dgesv_(&n, &nrhs, XtX.data(), &n, ipiv.data(), Xty.data(), &n, &lapackInfo);
        if (lapackInfo != 0) {
            Napi::Error::New(env, "Normal equations solve failed").ThrowAsJavaScriptException();
            return env.Undefined();
        }
    }

    // Create result - match y's dimensionality
    if (nrhs == 1 && !y_is_2d) {
        // y was 1D, return 1D result
        Napi::Array shape = Napi::Array::New(env, 1);
        shape.Set(uint32_t(0), Napi::Number::New(env, n));
        Napi::Object result = NativeNDArray::constructor.New({
            shape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
        });
        NativeNDArray* betaArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
        std::memcpy(betaArr->data(), Xty.data(), n * sizeof(double));
        return result;
    } else {
        // y was 2D, return 2D result (n x nrhs)
        Napi::Array shape = Napi::Array::New(env, 2);
        shape.Set(uint32_t(0), Napi::Number::New(env, n));
        shape.Set(uint32_t(1), Napi::Number::New(env, nrhs));
        Napi::Object result = NativeNDArray::constructor.New({
            shape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
        });
        NativeNDArray* betaArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
        std::memcpy(betaArr->data(), Xty.data(), n * nrhs * sizeof(double));
        return result;
    }
#else
    Napi::Error::New(env, "normal_equations requires BLAS/LAPACK").ThrowAsJavaScriptException();
    return env.Undefined();
#endif
}

/**
 * Gram matrix: X'X
 * Computes the Gram matrix using dsyrk for optimal performance.
 * For X with shape (m, n), returns (n, n) symmetric matrix.
 */
Napi::Value Gram(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected matrix X").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* X = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    const auto& shapeX = X->shape();

    if (shapeX.size() != 2) {
        Napi::TypeError::New(env, "X must be 2D").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int m = static_cast<int>(shapeX[0]); // rows
    int n = static_cast<int>(shapeX[1]); // cols
    double* dataX = static_cast<double*>(X->data());

    // Create result array (n x n)
    Napi::Array resultShape = Napi::Array::New(env, 2);
    resultShape.Set(uint32_t(0), Napi::Number::New(env, n));
    resultShape.Set(uint32_t(1), Napi::Number::New(env, n));

    Napi::Object result = NativeNDArray::constructor.New({
        resultShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
    });
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* dataR = static_cast<double*>(resultArr->data());

#if defined(USE_ACCELERATE)
    // Use dsyrk: C = alpha * A' * A + beta * C
    // For row-major X (m x n), we compute X' * X which is n x n
    cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                n, m,           // n = result size, k = inner dimension (m)
                1.0, dataX, n,  // A = X, lda = n (stride = cols)
                0.0, dataR, n); // C = result, ldc = n

    // Fill lower triangle (dsyrk only fills upper)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            dataR[i * n + j] = dataR[j * n + i];
        }
    }
#elif defined(USE_OPENBLAS) || defined(USE_LAPACK)
    // Fortran dsyrk with row-major data
    char uplo = 'L';  // Lower triangle in Fortran = Upper in C (row-major)
    char trans = 'N'; // No transpose in Fortran convention for X'X
    double alpha = 1.0;
    double beta = 0.0;

    dsyrk_(&uplo, &trans, &n, &m, &alpha, dataX, &n, &beta, dataR, &n);

    // Fill other triangle
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            dataR[i * n + j] = dataR[j * n + i];
        }
    }
#else
    // Fallback: compute X'X directly
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < m; k++) {
                sum += dataX[k * n + i] * dataX[k * n + j];
            }
            dataR[i * n + j] = sum;
            if (i != j) {
                dataR[j * n + i] = sum;
            }
        }
    }
#endif

    return result;
}

/**
 * X'y: Compute X transposed times y using optimized BLAS operations.
 * For X with shape (m, n) and y with shape (m,) or (m, k), returns (n,) or (n, k).
 * Much faster than matmul(X.T, y) for tall matrices.
 */
Napi::Value Xty(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected matrix X and vector/matrix y").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* X = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    NativeNDArray* y = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());
    const auto& shapeX = X->shape();
    const auto& shapeY = y->shape();

    if (shapeX.size() != 2) {
        Napi::TypeError::New(env, "X must be 2D").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int m = static_cast<int>(shapeX[0]); // rows of X
    int n = static_cast<int>(shapeX[1]); // cols of X
    double* dataX = static_cast<double*>(X->data());
    double* dataY = static_cast<double*>(y->data());

    // Determine nrhs (number of right-hand sides)
    int nrhs = 1;
    bool isVector = (shapeY.size() == 1);
    if (shapeY.size() == 2) {
        nrhs = static_cast<int>(shapeY[1]);
    }

    // Check dimensions match
    int yRows = isVector ? static_cast<int>(shapeY[0]) : static_cast<int>(shapeY[0]);
    if (yRows != m) {
        Napi::TypeError::New(env, "X rows must match y rows").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Create result array
    Napi::Array resultShape = Napi::Array::New(env, isVector ? 1 : 2);
    resultShape.Set(uint32_t(0), Napi::Number::New(env, n));
    if (!isVector) {
        resultShape.Set(uint32_t(1), Napi::Number::New(env, nrhs));
    }

    Napi::Object result = NativeNDArray::constructor.New({
        resultShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
    });
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* dataR = static_cast<double*>(resultArr->data());

#if defined(USE_ACCELERATE)
    if (nrhs == 1) {
        // Vector case: use dgemv
        cblas_dgemv(CblasRowMajor, CblasTrans, m, n,
                    1.0, dataX, n, dataY, 1,
                    0.0, dataR, 1);
    } else {
        // Matrix case: use dgemm
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    n, nrhs, m,
                    1.0, dataX, n, dataY, nrhs,
                    0.0, dataR, nrhs);
    }
#elif defined(USE_OPENBLAS) || defined(USE_LAPACK)
    if (nrhs == 1) {
        // For Fortran dgemv with row-major X: treat as n x m column-major
        char trans = 'N'; // No transpose in Fortran for X' in row-major
        int inc = 1;
        double alpha = 1.0;
        double beta = 0.0;
        dgemv_(&trans, &n, &m, &alpha, dataX, &n, dataY, &inc, &beta, dataR, &inc);
    } else {
        // For Fortran dgemm
        char transA = 'N';
        char transB = 'N';
        double alpha = 1.0;
        double beta = 0.0;
        dgemm_(&transA, &transB, &n, &nrhs, &m, &alpha, dataX, &n, dataY, &m, &beta, dataR, &n);
    }
#else
    // Fallback: compute X'y directly
    for (int j = 0; j < nrhs; j++) {
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int k = 0; k < m; k++) {
                sum += dataX[k * n + i] * dataY[k * nrhs + j];
            }
            dataR[i * nrhs + j] = sum;
        }
    }
#endif

    return result;
}

/**
 * Batch matrix multiplication - reduces N-API overhead by doing multiple
 * matmuls in a single native call.
 * batch_matmul(As, Bs) where As and Bs are arrays of 2D matrices
 * Returns array of result matrices
 */
Napi::Value BatchMatmul(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected two arrays of matrices").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    if (!info[0].IsArray() || !info[1].IsArray()) {
        Napi::TypeError::New(env, "Arguments must be arrays").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Array asArray = info[0].As<Napi::Array>();
    Napi::Array bsArray = info[1].As<Napi::Array>();

    uint32_t batchSize = asArray.Length();
    if (batchSize != bsArray.Length()) {
        Napi::Error::New(env, "Arrays must have same length").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    if (batchSize == 0) {
        return Napi::Array::New(env, 0);
    }

    Napi::Array results = Napi::Array::New(env, batchSize);

#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
    for (uint32_t i = 0; i < batchSize; i++) {
        NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(asArray.Get(i).As<Napi::Object>());
        NativeNDArray* b = Napi::ObjectWrap<NativeNDArray>::Unwrap(bsArray.Get(i).As<Napi::Object>());

        // Ensure contiguous
        Napi::Object aCopy, bCopy;
        a = ensureContiguous(info, a, aCopy);
        b = ensureContiguous(info, b, bCopy);

        if (a->ndim() != 2 || b->ndim() != 2) {
            Napi::Error::New(env, "All matrices must be 2D").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        int64_t m = a->shape()[0];
        int64_t ka = a->shape()[1];
        int64_t kb = b->shape()[0];
        int64_t n = b->shape()[1];

        if (ka != kb) {
            Napi::Error::New(env, "Matrix dimensions incompatible for multiplication").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        int64_t k = ka;

        // Create result matrix
        Napi::Array cShape = Napi::Array::New(env, 2);
        cShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(m)));
        cShape.Set(uint32_t(1), Napi::Number::New(env, static_cast<double>(n)));
        Napi::Object cResult = NativeNDArray::constructor.New({
            cShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
        });
        NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(cResult);

        double* aData = static_cast<double*>(a->data());
        double* bData = static_cast<double*>(b->data());
        double* cData = static_cast<double*>(c->data());

        // Perform matmul using BLAS
#if defined(USE_ACCELERATE)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                    1.0, aData, static_cast<int>(k),
                    bData, static_cast<int>(n),
                    0.0, cData, static_cast<int>(n));
#elif defined(USE_OPENBLAS)
        // For OpenBLAS with Fortran interface, we need to account for row-major storage
        // C = A * B in row-major is equivalent to C' = B' * A' in column-major
        // So we compute C' = B' * A' and the result is C in row-major
        char transA = 'N', transB = 'N';
        double alpha = 1.0, beta = 0.0;
        int mi = static_cast<int>(m);
        int ni = static_cast<int>(n);
        int ki = static_cast<int>(k);
        // dgemm expects column-major, but our data is row-major
        // C(m,n) = A(m,k) * B(k,n) row-major
        // Equivalent to C'(n,m) = B'(n,k) * A'(k,m) column-major
        // We swap A and B and swap m and n
        dgemm_(&transA, &transB, &ni, &mi, &ki,
               &alpha, bData, &ni, aData, &ki,
               &beta, cData, &ni);
#endif

        results.Set(i, cResult);
    }
#else
    // Pure fallback: use regular loop
    for (uint32_t i = 0; i < batchSize; i++) {
        NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(asArray.Get(i).As<Napi::Object>());
        NativeNDArray* b = Napi::ObjectWrap<NativeNDArray>::Unwrap(bsArray.Get(i).As<Napi::Object>());

        int64_t m = a->shape()[0];
        int64_t k = a->shape()[1];
        int64_t n = b->shape()[1];

        Napi::Array cShape = Napi::Array::New(env, 2);
        cShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(m)));
        cShape.Set(uint32_t(1), Napi::Number::New(env, static_cast<double>(n)));
        Napi::Object cResult = NativeNDArray::constructor.New({
            cShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
        });
        NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(cResult);

        double* aData = static_cast<double*>(a->data());
        double* bData = static_cast<double*>(b->data());
        double* cData = static_cast<double*>(c->data());

        // Naive matmul
        for (int64_t row = 0; row < m; row++) {
            for (int64_t col = 0; col < n; col++) {
                double sum = 0.0;
                for (int64_t inner = 0; inner < k; inner++) {
                    sum += aData[row * k + inner] * bData[inner * n + col];
                }
                cData[row * n + col] = sum;
            }
        }

        results.Set(i, cResult);
    }
#endif

    return results;
}

/**
 * Batch matrix multiplication with stacked 3D arrays.
 * Much more efficient than batch_matmul as it only creates one output array.
 *
 * batch_matmul_stacked(A, B) where:
 *   A has shape [batch, m, k]
 *   B has shape [batch, k, n]
 * Returns array with shape [batch, m, n]
 */
Napi::Value BatchMatmulStacked(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected two 3D arrays").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    NativeNDArray* b = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());

    // Ensure contiguous
    Napi::Object aCopy, bCopy;
    a = ensureContiguous(info, a, aCopy);
    b = ensureContiguous(info, b, bCopy);

    if (a->ndim() != 3 || b->ndim() != 3) {
        Napi::Error::New(env, "Both inputs must be 3D arrays [batch, rows, cols]").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int64_t batchA = a->shape()[0];
    int64_t batchB = b->shape()[0];
    if (batchA != batchB) {
        Napi::Error::New(env, "Batch dimensions must match").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int64_t batch = batchA;
    int64_t m = a->shape()[1];
    int64_t ka = a->shape()[2];
    int64_t kb = b->shape()[1];
    int64_t n = b->shape()[2];

    if (ka != kb) {
        Napi::Error::New(env, "Matrix dimensions incompatible: A[batch,m,k] @ B[batch,k,n]").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int64_t k = ka;

    // Create single output array [batch, m, n]
    Napi::Array cShape = Napi::Array::New(env, 3);
    cShape.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(batch)));
    cShape.Set(uint32_t(1), Napi::Number::New(env, static_cast<double>(m)));
    cShape.Set(uint32_t(2), Napi::Number::New(env, static_cast<double>(n)));
    Napi::Object cResult = NativeNDArray::constructor.New({
        cShape, Napi::String::New(env, "float64"), Napi::Boolean::New(env, true)
    });
    NativeNDArray* c = Napi::ObjectWrap<NativeNDArray>::Unwrap(cResult);

    double* aData = static_cast<double*>(a->data());
    double* bData = static_cast<double*>(b->data());
    double* cData = static_cast<double*>(c->data());

    int64_t aStride = m * k;
    int64_t bStride = k * n;
    int64_t cStride = m * n;

#if defined(USE_ACCELERATE)
    // Loop over batch, all BLAS calls happen in C++ without N-API overhead
    for (int64_t i = 0; i < batch; i++) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                    1.0, aData + i * aStride, static_cast<int>(k),
                    bData + i * bStride, static_cast<int>(n),
                    0.0, cData + i * cStride, static_cast<int>(n));
    }
#elif defined(USE_OPENBLAS)
    char transA = 'N', transB = 'N';
    double alpha = 1.0, beta = 0.0;
    int mi = static_cast<int>(m);
    int ni = static_cast<int>(n);
    int ki = static_cast<int>(k);

    for (int64_t i = 0; i < batch; i++) {
        dgemm_(&transA, &transB, &ni, &mi, &ki,
               &alpha, bData + i * bStride, &ni,
               aData + i * aStride, &ki,
               &beta, cData + i * cStride, &ni);
    }
#else
    // Pure fallback
    for (int64_t i = 0; i < batch; i++) {
        double* aMatrix = aData + i * aStride;
        double* bMatrix = bData + i * bStride;
        double* cMatrix = cData + i * cStride;

        for (int64_t row = 0; row < m; row++) {
            for (int64_t col = 0; col < n; col++) {
                double sum = 0.0;
                for (int64_t inner = 0; inner < k; inner++) {
                    sum += aMatrix[row * k + inner] * bMatrix[inner * n + col];
                }
                cMatrix[row * n + col] = sum;
            }
        }
    }
#endif

    return cResult;
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    Napi::Object linalg = Napi::Object::New(env);

    linalg.Set("matmul", Napi::Function::New(env, Matmul));
    linalg.Set("matmul_nt", Napi::Function::New(env, MatmulNT));
    linalg.Set("batch_matmul", Napi::Function::New(env, BatchMatmul));
    linalg.Set("batch_matmul_stacked", Napi::Function::New(env, BatchMatmulStacked));
    linalg.Set("dot", Napi::Function::New(env, Dot));
    linalg.Set("det", Napi::Function::New(env, Det));
    linalg.Set("inv", Napi::Function::New(env, Inv));
    linalg.Set("solve", Napi::Function::New(env, Solve));
    linalg.Set("eig", Napi::Function::New(env, Eig));
    linalg.Set("eigvals", Napi::Function::New(env, Eigvals));
    linalg.Set("svd", Napi::Function::New(env, Svd));
    linalg.Set("qr", Napi::Function::New(env, Qr));
    linalg.Set("cholesky", Napi::Function::New(env, Cholesky));
    linalg.Set("norm", Napi::Function::New(env, Norm));
    linalg.Set("matrix_rank", Napi::Function::New(env, MatrixRank));
    linalg.Set("trace", Napi::Function::New(env, Trace));
    linalg.Set("lstsq", Napi::Function::New(env, Lstsq));
    linalg.Set("normal_equations", Napi::Function::New(env, NormalEquations));
    linalg.Set("gram", Napi::Function::New(env, Gram));
    linalg.Set("xty", Napi::Function::New(env, Xty));

    exports.Set("linalg", linalg);
    return exports;
}

} // namespace linalg
} // namespace numpy_node
