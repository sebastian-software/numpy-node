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
    }
#endif

namespace numpy_node {
namespace linalg {

Napi::Value Matmul(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected two arrays").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    NativeNDArray* a = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[0].As<Napi::Object>());
    NativeNDArray* b = Napi::ObjectWrap<NativeNDArray>::Unwrap(info[1].As<Napi::Object>());

    if (a->ndim() != 2 || b->ndim() != 2) {
        Napi::Error::New(env, "matmul requires 2D arrays").ThrowAsJavaScriptException();
        return env.Undefined();
    }

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
    // Make copies since dgesv modifies inputs
    std::vector<double> aCopy(n * n);
    std::memcpy(aCopy.data(), a->data(), n * n * sizeof(double));

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
    std::memcpy(x->data(), b->data(), x->size() * sizeof(double));

    std::vector<int> ipiv(n);
    int lapackInfo = 0;

    dgesv_(&n, &nrhs, aCopy.data(), &n, ipiv.data(),
           static_cast<double*>(x->data()), &n, &lapackInfo);

    if (lapackInfo != 0) {
        Napi::Error::New(env, "Linear solve failed").ThrowAsJavaScriptException();
        return env.Undefined();
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
    std::vector<double> u(m * m);
    std::vector<double> vt(n * n);
    std::vector<double> work(1);
    int lwork = -1;
    int lapackInfo = 0;

    char jobu = 'A', jobvt = 'A';

    // Query workspace size
    dgesvd_(&jobu, &jobvt, &m, &n, aCopy.data(), &m,
            s.data(), u.data(), &m, vt.data(), &n,
            work.data(), &lwork, &lapackInfo);

    lwork = static_cast<int>(work[0]);
    work.resize(lwork);

    // Compute SVD
    dgesvd_(&jobu, &jobvt, &m, &n, aCopy.data(), &m,
            s.data(), u.data(), &m, vt.data(), &n,
            work.data(), &lwork, &lapackInfo);

    if (lapackInfo != 0) {
        Napi::Error::New(env, "SVD computation failed").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Create U array (transpose back to row-major)
    Napi::Array uShape = Napi::Array::New(env, 2);
    uShape.Set(uint32_t(0), Napi::Number::New(env, m));
    uShape.Set(uint32_t(1), Napi::Number::New(env, m));
    Napi::Object uResult = NativeNDArray::constructor.New({uShape, Napi::String::New(env, "float64")});
    NativeNDArray* uArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(uResult);
    transposeBack(u.data(), static_cast<double*>(uArr->data()), m, m);

    // Create S array
    Napi::Array sShape = Napi::Array::New(env, 1);
    sShape.Set(uint32_t(0), Napi::Number::New(env, minmn));
    Napi::Object sResult = NativeNDArray::constructor.New({sShape, Napi::String::New(env, "float64")});
    NativeNDArray* sArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(sResult);
    std::memcpy(sArr->data(), s.data(), minmn * sizeof(double));

    // Create Vh array (transpose back to row-major)
    Napi::Array vhShape = Napi::Array::New(env, 2);
    vhShape.Set(uint32_t(0), Napi::Number::New(env, n));
    vhShape.Set(uint32_t(1), Napi::Number::New(env, n));
    Napi::Object vhResult = NativeNDArray::constructor.New({vhShape, Napi::String::New(env, "float64")});
    NativeNDArray* vhArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(vhResult);
    transposeBack(vt.data(), static_cast<double*>(vhArr->data()), n, n);

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

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    Napi::Object linalg = Napi::Object::New(env);

    linalg.Set("matmul", Napi::Function::New(env, Matmul));
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

    exports.Set("linalg", linalg);
    return exports;
}

} // namespace linalg
} // namespace numpy_node
