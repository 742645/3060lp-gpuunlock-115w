// SPDX-License-Identifier: GPL-2.0-only
// cuBLAS probe: strict FP32, FP16 input / FP32 accumulation, and strict FP16.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

// Fail the entire run on CUDA errors, including asynchronous errors at event synchronization.
static void cu(cudaError_t code) {
    if (code != cudaSuccess) { std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(code)); std::exit(2); }
}
// Fail instead of emitting a throughput result from an unsuccessful cuBLAS launch.
static void blas(cublasStatus_t code) {
    if (code != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "cuBLAS: %d\n", int(code)); std::exit(2); }
}
// Convert exactly representable dyadic input values to the measured input format.
template<class T> static T convert(float x);
template<> float convert(float x) { return x; }
template<> __half convert(float x) { return __float2half_rn(x); }

// Time five independent batches and check every output against a cheap exact reference.
template<class T, class Out = float> static bool bench(cublasHandle_t h, const char *mode,
        cudaDataType type, cublasComputeType_t compute, int n) {
    constexpr bool half_output = std::is_same<Out, __half>::value;
    const cudaDataType output_type = half_output ? CUDA_R_16F : CUDA_R_32F;
    const size_t cells = size_t(n) * n;
    std::vector<T> a(cells), b(cells);
    std::vector<Out> result(cells);
    for (int col = 0; col < n; ++col) for (int row = 0; row < n; ++row) {
        if (half_output) {
            // At most 1024 nonzero +/-1/256 products: every partial sum is exact
            // in FP16, regardless of reduction order. Dense GEMM still computes n^3 MACs.
            a[row + size_t(col)*n] = convert<T>(col < 1024 ? (row%2 ? -1.0f : 1.0f)/16 : 0);
            b[row + size_t(col)*n] = convert<T>((col%3 ? 1.0f : -1.0f)/16);
        } else {
            a[row + size_t(col)*n] = convert<T>((row%13-6)/8.0f);
            b[row + size_t(col)*n] = convert<T>((col%11-5)/8.0f);
        }
    }
    T *da = nullptr, *db = nullptr; Out *dc = nullptr;
    cu(cudaMalloc(&da, cells*sizeof(T))); cu(cudaMalloc(&db, cells*sizeof(T)));
    cu(cudaMalloc(&dc, cells*sizeof(Out)));
    cu(cudaMemcpy(da, a.data(), cells*sizeof(T), cudaMemcpyHostToDevice));
    cu(cudaMemcpy(db, b.data(), cells*sizeof(T), cudaMemcpyHostToDevice));
    const Out alpha = convert<Out>(1), beta = convert<Out>(0);
    auto launch = [&]() { blas(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
        &alpha, da, type, n, db, type, n, &beta, dc, output_type, n, compute,
        CUBLAS_GEMM_DEFAULT)); };
    for (int i = 0; i < 10; ++i) launch();
    cu(cudaDeviceSynchronize());
    std::vector<double> rates;
    std::printf("{\"mode\":\"%s\",\"n\":%d,\"warmups\":10,\"iterations\":20,\"reps\":5,", mode, n);
    if (half_output)
        std::printf("\"compute_type\":\"CUBLAS_COMPUTE_16F_PEDANTIC\",\"input_type\":\"FP16\","
                    "\"scale_type\":\"FP16\",\"output_type\":\"FP16\",\"tensor_cores_allowed\":false,");
    std::printf("\"times_ms\":[");
    for (int rep = 0; rep < 5; ++rep) {
        cudaEvent_t start, stop; cu(cudaEventCreate(&start)); cu(cudaEventCreate(&stop));
        cu(cudaEventRecord(start)); for (int i = 0; i < 20; ++i) launch();
        cu(cudaEventRecord(stop)); cu(cudaEventSynchronize(stop));
        float elapsed = 0; cu(cudaEventElapsedTime(&elapsed, start, stop));
        if (!(elapsed > 0) || !std::isfinite(elapsed)) std::exit(3);
        double ms = elapsed/20.0; rates.push_back(2.0*n*n*n/(ms*1e9));
        std::printf("%s%.8f", rep ? "," : "", ms);
        cu(cudaEventDestroy(start)); cu(cudaEventDestroy(stop));
    }
    cu(cudaMemcpy(result.data(), dc, cells*sizeof(Out), cudaMemcpyDeviceToHost));
    size_t failures = 0;
    for (int col = 0; col < n; ++col) for (int row = 0; row < n; ++row) {
        const float expected = half_output
            ? std::min(n, 1024)*(row%2 ? -1.0f : 1.0f)*(col%3 ? 1.0f : -1.0f)/256
            : n*((row%13-6)/8.0f)*((col%11-5)/8.0f);
        float actual = float(result[row + size_t(col)*n]);
        if (!std::isfinite(actual) || actual != expected) ++failures;
    }
    std::sort(rates.begin(), rates.end());
    std::printf("],\"median_TF_s\":%.6f,\"best_TF_s\":%.6f,\"cells_checked\":%zu,\"failures\":%zu}\n",
        rates[2], rates.back(), cells, failures); std::fflush(stdout);
    cu(cudaFree(da)); cu(cudaFree(db)); cu(cudaFree(dc)); return failures == 0;
}

// Generate varied exact FP16 inputs; k <= 256 keeps every partial sum representable.
static float strict_input(int row, int col, unsigned seed) {
    unsigned value = unsigned(row)*1664525u ^ unsigned(col)*1013904223u ^ seed;
    value ^= value >> 13;
    return (int(value%3)-1)/16.0f;
}

// Check half input, half scale, half compute and half output against an exact CPU reference.
static bool validate_strict16(cublasHandle_t h, int m, int n, int k, unsigned seed) {
    std::vector<__half> a(size_t(m)*k), b(size_t(k)*n), result(size_t(m)*n);
    std::vector<float> reference(size_t(m)*n, 0);
    for (int col=0; col<k; ++col) for (int row=0; row<m; ++row)
        a[row+size_t(col)*m] = convert<__half>(strict_input(row,col,seed));
    for (int col=0; col<n; ++col) for (int row=0; row<k; ++row)
        b[row+size_t(col)*k] = convert<__half>(strict_input(row,col,seed^0xa5a5a5a5u));
    for (int col=0; col<n; ++col) for (int inner=0; inner<k; ++inner) for (int row=0; row<m; ++row)
        reference[row+size_t(col)*m] += float(a[row+size_t(inner)*m])*float(b[inner+size_t(col)*k]);
    __half *da=nullptr, *db=nullptr, *dc=nullptr;
    cu(cudaMalloc(&da,a.size()*sizeof(__half))); cu(cudaMalloc(&db,b.size()*sizeof(__half)));
    cu(cudaMalloc(&dc,result.size()*sizeof(__half)));
    cu(cudaMemcpy(da,a.data(),a.size()*sizeof(__half),cudaMemcpyHostToDevice));
    cu(cudaMemcpy(db,b.data(),b.size()*sizeof(__half),cudaMemcpyHostToDevice));
    const __half alpha=convert<__half>(1), beta=convert<__half>(0);
    blas(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&alpha,da,CUDA_R_16F,m,
         db,CUDA_R_16F,k,&beta,dc,CUDA_R_16F,m,CUBLAS_COMPUTE_16F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
    cu(cudaDeviceSynchronize());
    cu(cudaMemcpy(result.data(),dc,result.size()*sizeof(__half),cudaMemcpyDeviceToHost));
    size_t failures=0;
    for (size_t i=0; i<result.size(); ++i)
        if (!std::isfinite(float(result[i])) || float(result[i]) != reference[i]) ++failures;
    std::printf("{\"mode\":\"FP16_strict\",\"m\":%d,\"n\":%d,\"k\":%d,\"seed\":%u,"
                "\"cells_checked\":%zu,\"failures\":%zu}\n",m,n,k,seed,result.size(),failures);
    cu(cudaFree(da)); cu(cudaFree(db)); cu(cudaFree(dc));
    return failures == 0;
}

// --validate-strict16 runs three seeded CPU-reference cases; the default runs all nine benchmarks.
int main(int argc, char **argv) {
    cublasHandle_t h; blas(cublasCreate(&h)); bool pass = true;
    if (argc == 2 && !std::strcmp(argv[1], "--validate-strict16")) {
        pass &= validate_strict16(h,256,256,256,1);
        pass &= validate_strict16(h,64,96,128,2);
        pass &= validate_strict16(h,127,65,95,3);
    } else if (argc == 1) {
        for (int n : {2048, 4096, 8192}) {
            pass &= bench<float>(h, "FP32_strict", CUDA_R_32F, CUBLAS_COMPUTE_32F_PEDANTIC, n);
            pass &= bench<__half>(h, "FP16_FP32acc", CUDA_R_16F, CUBLAS_COMPUTE_32F, n);
            pass &= bench<__half,__half>(h, "FP16_strict", CUDA_R_16F, CUBLAS_COMPUTE_16F_PEDANTIC, n);
        }
    } else {
        std::fprintf(stderr,"usage: %s [--validate-strict16]\n",argv[0]);
        pass = false;
    }
    blas(cublasDestroy(h)); return pass ? 0 : 1;
}
