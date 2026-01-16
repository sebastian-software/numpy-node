#include "linalg.h"
#include <cmath>
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

    // Default: Frobenius/L2 norm
    double* data = static_cast<double*>(a->data());
    double sum = 0.0;

    for (int64_t i = 0; i < a->size(); i++) {
        sum += data[i] * data[i];
    }

    return Napi::Number::New(env, std::sqrt(sum));
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

// Stubs for functions that require full LAPACK
Napi::Value Eig(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Error::New(env, "eig not yet implemented").ThrowAsJavaScriptException();
    return env.Undefined();
}

Napi::Value Eigvals(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Error::New(env, "eigvals not yet implemented").ThrowAsJavaScriptException();
    return env.Undefined();
}

Napi::Value Svd(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Error::New(env, "svd not yet implemented").ThrowAsJavaScriptException();
    return env.Undefined();
}

Napi::Value Qr(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Error::New(env, "qr not yet implemented").ThrowAsJavaScriptException();
    return env.Undefined();
}

Napi::Value Cholesky(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Error::New(env, "cholesky not yet implemented").ThrowAsJavaScriptException();
    return env.Undefined();
}

Napi::Value MatrixRank(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Error::New(env, "matrix_rank not yet implemented").ThrowAsJavaScriptException();
    return env.Undefined();
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
