#include "einsum.h"
#include <cstring>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <numeric>
#include <sstream>

#if defined(USE_ACCELERATE)
    #include <Accelerate/Accelerate.h>
#elif defined(USE_OPENBLAS)
    extern "C" {
        void dgemm_(const char* transa, const char* transb,
                   const int* m, const int* n, const int* k,
                   const double* alpha, const double* a, const int* lda,
                   const double* b, const int* ldb,
                   const double* beta, double* c, const int* ldc);
    }
#endif

namespace numpy_node {
namespace einsum {

// Parsed subscript for one operand
struct Subscript {
    std::vector<char> indices;  // e.g., {'i', 'j'} for "ij"
};

// Parsed einsum expression
struct EinsumExpr {
    std::vector<Subscript> inputs;
    Subscript output;
    bool hasExplicitOutput;
};

// Parse einsum subscripts string
static EinsumExpr parseSubscripts(const std::string& subscripts) {
    EinsumExpr expr;
    expr.hasExplicitOutput = false;

    // Find arrow separator
    size_t arrowPos = subscripts.find("->");
    std::string inputPart;
    std::string outputPart;

    if (arrowPos != std::string::npos) {
        inputPart = subscripts.substr(0, arrowPos);
        outputPart = subscripts.substr(arrowPos + 2);
        expr.hasExplicitOutput = true;
    } else {
        inputPart = subscripts;
    }

    // Parse input subscripts (comma-separated)
    std::stringstream ss(inputPart);
    std::string token;
    while (std::getline(ss, token, ',')) {
        Subscript sub;
        for (char c : token) {
            if (c != ' ') {
                sub.indices.push_back(c);
            }
        }
        expr.inputs.push_back(sub);
    }

    // Parse output subscripts
    if (expr.hasExplicitOutput) {
        for (char c : outputPart) {
            if (c != ' ') {
                expr.output.indices.push_back(c);
            }
        }
    } else {
        // Implicit output: sorted unique indices that appear exactly once
        std::unordered_map<char, int> indexCount;
        for (const auto& sub : expr.inputs) {
            for (char c : sub.indices) {
                indexCount[c]++;
            }
        }

        std::vector<char> uniqueIndices;
        for (const auto& pair : indexCount) {
            if (pair.second == 1) {
                uniqueIndices.push_back(pair.first);
            }
        }
        std::sort(uniqueIndices.begin(), uniqueIndices.end());
        expr.output.indices = uniqueIndices;
    }

    return expr;
}

// Get dimension size for an index from operand
static int64_t getDimSize(char index, const Subscript& sub, NativeNDArray* arr) {
    for (size_t i = 0; i < sub.indices.size(); i++) {
        if (sub.indices[i] == index) {
            return arr->shape()[i];
        }
    }
    return -1;
}

// Build index-to-dimension mapping
static std::unordered_map<char, int64_t> buildIndexDimMap(
    const EinsumExpr& expr,
    const std::vector<NativeNDArray*>& operands
) {
    std::unordered_map<char, int64_t> dimMap;

    for (size_t i = 0; i < expr.inputs.size(); i++) {
        const auto& sub = expr.inputs[i];
        for (size_t j = 0; j < sub.indices.size(); j++) {
            char idx = sub.indices[j];
            int64_t dim = operands[i]->shape()[j];
            if (dimMap.find(idx) == dimMap.end()) {
                dimMap[idx] = dim;
            }
        }
    }

    return dimMap;
}

// Compute output shape
static std::vector<int64_t> computeOutputShape(
    const EinsumExpr& expr,
    const std::unordered_map<char, int64_t>& dimMap
) {
    std::vector<int64_t> shape;
    for (char c : expr.output.indices) {
        auto it = dimMap.find(c);
        if (it != dimMap.end()) {
            shape.push_back(it->second);
        }
    }
    return shape;
}

// Get position of index in subscript
static int getIndexPos(char c, const Subscript& sub) {
    for (size_t i = 0; i < sub.indices.size(); i++) {
        if (sub.indices[i] == c) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

// Compute flat index from multi-index
static int64_t flatIndex(const std::vector<int64_t>& indices,
                          const std::vector<int64_t>& shape) {
    int64_t flat = 0;
    int64_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; i--) {
        flat += indices[i] * stride;
        stride *= shape[i];
    }
    return flat;
}

// Generic einsum implementation (slow but correct)
static void einsumGeneric(
    const EinsumExpr& expr,
    const std::vector<NativeNDArray*>& operands,
    double* output,
    const std::vector<int64_t>& outputShape,
    const std::unordered_map<char, int64_t>& dimMap
) {
    // Collect all indices (output + contracted)
    std::unordered_set<char> outputSet(expr.output.indices.begin(), expr.output.indices.end());
    std::vector<char> allIndices = expr.output.indices;
    std::vector<char> contractedIndices;

    for (const auto& sub : expr.inputs) {
        for (char c : sub.indices) {
            if (outputSet.find(c) == outputSet.end()) {
                if (std::find(contractedIndices.begin(), contractedIndices.end(), c) == contractedIndices.end()) {
                    contractedIndices.push_back(c);
                    allIndices.push_back(c);
                }
            }
        }
    }

    // Build dimension array for all indices
    std::vector<int64_t> allDims;
    for (char c : allIndices) {
        allDims.push_back(dimMap.at(c));
    }

    // Total iterations
    int64_t totalIters = 1;
    for (int64_t d : allDims) {
        totalIters *= d;
    }

    // Initialize output to zero
    int64_t outputSize = 1;
    for (int64_t d : outputShape) {
        outputSize *= d;
    }
    std::memset(output, 0, outputSize * sizeof(double));

    // Iterate over all index combinations
    std::vector<int64_t> current(allIndices.size(), 0);
    std::unordered_map<char, int64_t> indexValues;

    for (int64_t iter = 0; iter < totalIters; iter++) {
        // Set index values
        for (size_t i = 0; i < allIndices.size(); i++) {
            indexValues[allIndices[i]] = current[i];
        }

        // Compute product of operand elements
        double product = 1.0;
        for (size_t opIdx = 0; opIdx < operands.size(); opIdx++) {
            const auto& sub = expr.inputs[opIdx];
            std::vector<int64_t> opIndices;
            for (char c : sub.indices) {
                opIndices.push_back(indexValues[c]);
            }
            int64_t flat = flatIndex(opIndices, operands[opIdx]->shape());
            product *= static_cast<double*>(operands[opIdx]->data())[flat];
        }

        // Add to output
        if (!outputShape.empty()) {
            std::vector<int64_t> outIndices;
            for (char c : expr.output.indices) {
                outIndices.push_back(indexValues[c]);
            }
            int64_t outFlat = flatIndex(outIndices, outputShape);
            output[outFlat] += product;
        } else {
            // Scalar output
            output[0] += product;
        }

        // Increment counter
        for (int i = static_cast<int>(current.size()) - 1; i >= 0; i--) {
            current[i]++;
            if (current[i] < allDims[i]) {
                break;
            }
            current[i] = 0;
        }
    }
}

// Optimized matrix multiplication: ij,jk->ik
static bool tryMatmul(
    const EinsumExpr& expr,
    const std::vector<NativeNDArray*>& operands,
    double* output,
    const std::vector<int64_t>& outputShape
) {
    if (operands.size() != 2) return false;
    if (expr.inputs[0].indices.size() != 2) return false;
    if (expr.inputs[1].indices.size() != 2) return false;
    if (expr.output.indices.size() != 2) return false;

    const auto& sub0 = expr.inputs[0].indices;
    const auto& sub1 = expr.inputs[1].indices;
    const auto& subOut = expr.output.indices;

    // Check for ij,jk->ik pattern
    if (sub0[1] == sub1[0] && sub0[0] == subOut[0] && sub1[1] == subOut[1] &&
        sub0[0] != sub0[1] && sub1[0] != sub1[1]) {

        int64_t m = operands[0]->shape()[0];
        int64_t k = operands[0]->shape()[1];
        int64_t n = operands[1]->shape()[1];

        const double* A = static_cast<const double*>(operands[0]->data());
        const double* B = static_cast<const double*>(operands[1]->data());

#if defined(USE_ACCELERATE)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                    1.0, A, static_cast<int>(k),
                    B, static_cast<int>(n),
                    0.0, output, static_cast<int>(n));
#elif defined(USE_OPENBLAS)
        int mi = static_cast<int>(m);
        int ni = static_cast<int>(n);
        int ki = static_cast<int>(k);
        double alpha = 1.0;
        double beta = 0.0;
        // Note: BLAS uses column-major, we're row-major, so we compute B'A' = (AB)'
        dgemm_("N", "N", &ni, &mi, &ki, &alpha, B, &ni, A, &ki, &beta, output, &ni);
        // Transpose result in-place (only works for square matrices)
        // For non-square, we'd need a temp buffer - skip for now and fall through to generic
        if (m != n) return false;
        for (int64_t i = 0; i < m; i++) {
            for (int64_t j = i + 1; j < n; j++) {
                std::swap(output[i * n + j], output[j * n + i]);
            }
        }
#else
        // Pure C++ fallback
        for (int64_t i = 0; i < m; i++) {
            for (int64_t j = 0; j < n; j++) {
                double sum = 0.0;
                for (int64_t l = 0; l < k; l++) {
                    sum += A[i * k + l] * B[l * n + j];
                }
                output[i * n + j] = sum;
            }
        }
#endif
        return true;
    }

    return false;
}

// Optimized trace: ii->
static bool tryTrace(
    const EinsumExpr& expr,
    const std::vector<NativeNDArray*>& operands,
    double* output
) {
    if (operands.size() != 1) return false;
    if (expr.inputs[0].indices.size() != 2) return false;
    if (!expr.output.indices.empty()) return false;

    const auto& sub = expr.inputs[0].indices;
    if (sub[0] != sub[1]) return false;

    int64_t n = operands[0]->shape()[0];
    const double* data = static_cast<const double*>(operands[0]->data());

    double sum = 0.0;
    for (int64_t i = 0; i < n; i++) {
        sum += data[i * n + i];
    }
    output[0] = sum;
    return true;
}

// Optimized diagonal: ii->i
static bool tryDiagonal(
    const EinsumExpr& expr,
    const std::vector<NativeNDArray*>& operands,
    double* output
) {
    if (operands.size() != 1) return false;
    if (expr.inputs[0].indices.size() != 2) return false;
    if (expr.output.indices.size() != 1) return false;

    const auto& sub = expr.inputs[0].indices;
    if (sub[0] != sub[1]) return false;
    if (expr.output.indices[0] != sub[0]) return false;

    int64_t n = operands[0]->shape()[0];
    const double* data = static_cast<const double*>(operands[0]->data());

    for (int64_t i = 0; i < n; i++) {
        output[i] = data[i * n + i];
    }
    return true;
}

// Optimized inner product: i,i->
static bool tryInnerProduct(
    const EinsumExpr& expr,
    const std::vector<NativeNDArray*>& operands,
    double* output
) {
    if (operands.size() != 2) return false;
    if (expr.inputs[0].indices.size() != 1) return false;
    if (expr.inputs[1].indices.size() != 1) return false;
    if (!expr.output.indices.empty()) return false;

    if (expr.inputs[0].indices[0] != expr.inputs[1].indices[0]) return false;

    int64_t n = operands[0]->size();
    const double* a = static_cast<const double*>(operands[0]->data());
    const double* b = static_cast<const double*>(operands[1]->data());

    double sum = 0.0;
#if defined(USE_ACCELERATE)
    vDSP_dotprD(a, 1, b, 1, &sum, static_cast<vDSP_Length>(n));
#else
    for (int64_t i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
#endif
    output[0] = sum;
    return true;
}

// Optimized outer product: i,j->ij
static bool tryOuterProduct(
    const EinsumExpr& expr,
    const std::vector<NativeNDArray*>& operands,
    double* output,
    const std::vector<int64_t>& outputShape
) {
    if (operands.size() != 2) return false;
    if (expr.inputs[0].indices.size() != 1) return false;
    if (expr.inputs[1].indices.size() != 1) return false;
    if (expr.output.indices.size() != 2) return false;

    if (expr.inputs[0].indices[0] == expr.inputs[1].indices[0]) return false;
    if (expr.output.indices[0] != expr.inputs[0].indices[0]) return false;
    if (expr.output.indices[1] != expr.inputs[1].indices[0]) return false;

    int64_t m = operands[0]->size();
    int64_t n = operands[1]->size();
    const double* a = static_cast<const double*>(operands[0]->data());
    const double* b = static_cast<const double*>(operands[1]->data());

    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            output[i * n + j] = a[i] * b[j];
        }
    }
    return true;
}

Napi::Value Einsum(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "einsum requires at least subscripts and one operand").ThrowAsJavaScriptException();
        return env.Null();
    }

    // Parse subscripts
    std::string subscripts = info[0].As<Napi::String>().Utf8Value();
    EinsumExpr expr = parseSubscripts(subscripts);

    // Get operands
    std::vector<NativeNDArray*> operands;
    for (size_t i = 1; i < info.Length(); i++) {
        if (!info[i].IsObject()) {
            Napi::TypeError::New(env, "Operands must be NDArrays").ThrowAsJavaScriptException();
            return env.Null();
        }
        operands.push_back(Napi::ObjectWrap<NativeNDArray>::Unwrap(info[i].As<Napi::Object>()));
    }

    if (operands.size() != expr.inputs.size()) {
        Napi::TypeError::New(env, "Number of operands doesn't match subscripts").ThrowAsJavaScriptException();
        return env.Null();
    }

    // Build index dimension map
    auto dimMap = buildIndexDimMap(expr, operands);

    // Compute output shape
    auto outputShape = computeOutputShape(expr, dimMap);

    // Create output array
    Napi::Array jsShape = Napi::Array::New(env, outputShape.size());
    for (size_t i = 0; i < outputShape.size(); i++) {
        jsShape.Set(static_cast<uint32_t>(i), Napi::Number::New(env, static_cast<double>(outputShape[i])));
    }

    // Handle scalar output (empty shape means scalar, but we need at least 1D for NDArray)
    if (outputShape.empty()) {
        jsShape = Napi::Array::New(env, 1);
        jsShape.Set(uint32_t(0), Napi::Number::New(env, 1.0));
        outputShape.push_back(1);
    }

    Napi::Object result = NativeNDArray::constructor.New({
        jsShape,
        Napi::String::New(env, "float64"),
        Napi::Boolean::New(env, true)
    });
    NativeNDArray* resultArr = Napi::ObjectWrap<NativeNDArray>::Unwrap(result);
    double* outputData = static_cast<double*>(resultArr->data());

    // Try optimized paths first
    if (tryMatmul(expr, operands, outputData, outputShape)) {
        return result;
    }
    if (tryTrace(expr, operands, outputData)) {
        return result;
    }
    if (tryDiagonal(expr, operands, outputData)) {
        return result;
    }
    if (tryInnerProduct(expr, operands, outputData)) {
        return result;
    }
    if (tryOuterProduct(expr, operands, outputData, outputShape)) {
        return result;
    }

    // Fall back to generic implementation
    einsumGeneric(expr, operands, outputData, outputShape, dimMap);

    return result;
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    exports.Set("einsum", Napi::Function::New(env, Einsum));
    return exports;
}

} // namespace einsum
} // namespace numpy_node
