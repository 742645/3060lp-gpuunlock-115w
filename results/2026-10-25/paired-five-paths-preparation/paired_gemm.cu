// SPDX-License-Identifier: GPL-2.0-only
// Same-input five-path GEMM comparison, with separate untimed CUPTI kernel tracing.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cupti.h>

// Stop on any synchronous/asynchronous CUDA error rather than publishing a timing.
static void cu(cudaError_t value) {
    if (value != cudaSuccess) { std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(value)); std::exit(2); }
}
// Stop if the requested cuBLAS precision/operation is unsupported or fails.
static void blas(cublasStatus_t value) {
    if (value != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr,"cuBLAS: %d\n",int(value)); std::exit(2); }
}
// Propagate CUPTI failures; tracing is separate from the performance measurement process.
static void cpti(CUptiResult value) {
    if (value != CUPTI_SUCCESS) {
        const char *message=nullptr; cuptiGetResultString(value,&message);
        std::fprintf(stderr,"CUPTI: %s\n",message ? message : "unknown"); std::exit(2);
    }
}
// Escape activity kernel names for one-record-per-line JSON output.
static std::string quote(const char *text) {
    std::string result="\"";
    for (const unsigned char *p=reinterpret_cast<const unsigned char *>(text); *p; ++p) {
        if (*p == '\\' || *p == '"') result += '\\';
        if (*p < 32) { char encoded[7]; std::snprintf(encoded,sizeof(encoded),"\\u%04x",*p); result += encoded; }
        else result += char(*p);
    }
    return result + "\"";
}

struct Mode { const char *name; cublasComputeType_t compute; const char *compute_name; bool half_in, half_out, tensor; };
static const Mode modes[] = {
    {"fp32_simt",CUBLAS_COMPUTE_32F_PEDANTIC,"CUBLAS_COMPUTE_32F_PEDANTIC",false,false,false},
    {"fp16_simt",CUBLAS_COMPUTE_16F_PEDANTIC,"CUBLAS_COMPUTE_16F_PEDANTIC",true,true,false},
    {"tf32_tensor_fp32acc",CUBLAS_COMPUTE_32F_FAST_TF32,"CUBLAS_COMPUTE_32F_FAST_TF32",false,false,true},
    {"fp16_tensor_fp16acc",CUBLAS_COMPUTE_16F,"CUBLAS_COMPUTE_16F",true,true,true},
    {"fp16_tensor_fp32acc",CUBLAS_COMPUTE_32F,"CUBLAS_COMPUTE_32F",true,false,true},
};
static const char *trace_mode="unset";
static int trace_size=0;

// Allocate an aligned CUPTI activity buffer; do not instrument measured benchmark processes.
static void CUPTIAPI buffer_requested(uint8_t **buffer, size_t *size, size_t *max_records) {
    *size=1024*1024; *max_records=0;
    if (posix_memalign(reinterpret_cast<void **>(buffer),8,*size)) std::exit(2);
}
// Record the actual launched kernels and reject dropped activity records.
static void CUPTIAPI buffer_completed(CUcontext context, uint32_t stream, uint8_t *buffer, size_t, size_t valid) {
    CUpti_Activity *record=nullptr;
    for (;;) {
        CUptiResult status=cuptiActivityGetNextRecord(buffer,valid,&record);
        if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) break;
        cpti(status);
        if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
            const auto *kernel=reinterpret_cast<const CUpti_ActivityKernel9 *>(record);
            std::printf("{\"mode\":\"%s\",\"n\":%d,\"kernel\":%s}\n",trace_mode,trace_size,quote(kernel->name).c_str());
        }
    }
    size_t dropped=0; cpti(cuptiActivityGetNumDroppedRecords(context,stream,&dropped));
    if (dropped) { std::fprintf(stderr,"dropped CUPTI kernel records: %zu\n",dropped); std::exit(3); }
    std::free(buffer);
}

// Exact signed dyadic values for small dense CPU-reference tests.
static float varied_input(int row,int col,unsigned seed) {
    unsigned value=unsigned(row)*1664525u ^ unsigned(col)*1013904223u ^ seed;
    value ^= value >> 13;
    return (int(value%3)-1)/16.0f;
}
// Convert the same mathematical input values into the selected storage precision.
template<class T> static T convert(float value);
template<> float convert(float value) { return value; }
template<> __half convert(float value) { return __float2half_rn(value); }

// Run one mode with identical dyadic inputs across all five paths; validate all outputs.
template<class In,class Out> static bool run(cublasHandle_t handle,const Mode &mode,int m,int n,int k,
        unsigned seed,bool validate_only,bool trace_only) {
    std::vector<In> a(size_t(m)*k), b(size_t(k)*n);
    std::vector<Out> result(size_t(m)*n);
    const int active=std::min(k,1024);
    for (int col=0;col<k;++col) for (int row=0;row<m;++row)
        a[row+size_t(col)*m]=convert<In>(validate_only ? varied_input(row,col,seed)
            : col<active ? ((row+seed)%2 ? -1.0f : 1.0f)/16 : 0);
    for (int col=0;col<n;++col) for (int row=0;row<k;++row)
        b[row+size_t(col)*k]=convert<In>(validate_only ? varied_input(row,col,seed^0xa5a5a5a5u)
            : ((col+seed)%3 ? 1.0f : -1.0f)/16);
    std::vector<float> reference;
    if (validate_only) {
        reference.assign(size_t(m)*n,0);
        for(int col=0;col<n;++col) for(int inner=0;inner<k;++inner) for(int row=0;row<m;++row)
            reference[row+size_t(col)*m] += float(a[row+size_t(inner)*m])*float(b[inner+size_t(col)*k]);
    }
    In *da=nullptr,*db=nullptr; Out *dc=nullptr;
    cu(cudaMalloc(&da,a.size()*sizeof(In))); cu(cudaMalloc(&db,b.size()*sizeof(In)));
    cu(cudaMalloc(&dc,result.size()*sizeof(Out)));
    cu(cudaMemcpy(da,a.data(),a.size()*sizeof(In),cudaMemcpyHostToDevice));
    cu(cudaMemcpy(db,b.data(),b.size()*sizeof(In),cudaMemcpyHostToDevice));
    const Out alpha=convert<Out>(1),beta=convert<Out>(0);
    auto launch=[&](){ blas(cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&alpha,da,
        mode.half_in?CUDA_R_16F:CUDA_R_32F,m,db,mode.half_in?CUDA_R_16F:CUDA_R_32F,k,
        &beta,dc,mode.half_out?CUDA_R_16F:CUDA_R_32F,m,mode.compute,CUBLAS_GEMM_DEFAULT)); };
    std::vector<double> times,rates;
    if (trace_only) {
        trace_mode=mode.name;
        trace_size=n;
        cpti(cuptiActivityRegisterCallbacks(buffer_requested,buffer_completed));
        cpti(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
        launch(); cu(cudaDeviceSynchronize()); cpti(cuptiActivityFlushAll(0));
        cpti(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    } else if (validate_only) { launch(); cu(cudaDeviceSynchronize()); }
    else {
        for(int i=0;i<10;++i) launch(); cu(cudaDeviceSynchronize());
        for(int rep=0;rep<5;++rep) {
            cudaEvent_t start,stop; cu(cudaEventCreate(&start)); cu(cudaEventCreate(&stop));
            cu(cudaEventRecord(start)); for(int i=0;i<20;++i) launch();
            cu(cudaEventRecord(stop)); cu(cudaEventSynchronize(stop));
            float elapsed=0; cu(cudaEventElapsedTime(&elapsed,start,stop));
            if (!(elapsed>0) || !std::isfinite(elapsed)) std::exit(3);
            double ms=elapsed/20.0; times.push_back(ms); rates.push_back(2.0*m*n*k/(ms*1e9));
            cu(cudaEventDestroy(start)); cu(cudaEventDestroy(stop));
        }
    }
    cu(cudaMemcpy(result.data(),dc,result.size()*sizeof(Out),cudaMemcpyDeviceToHost));
    size_t failures=0;
    for(int col=0;col<n;++col) for(int row=0;row<m;++row) {
        float expected=validate_only ? reference[row+size_t(col)*m]
            : active*((row+seed)%2?-1.0f:1.0f)*((col+seed)%3?1.0f:-1.0f)/256;
        float actual=float(result[row+size_t(col)*m]);
        if (!std::isfinite(actual) || actual != expected) ++failures;
    }
    if (!trace_only) {
        std::printf("{\"mode\":\"%s\",\"compute_type\":\"%s\",\"input_type\":\"%s\","
            "\"output_type\":\"%s\",\"tensor_allowed\":%s,\"m\":%d,\"n\":%d,\"k\":%d,\"seed\":%u,"
            "\"input_pattern\":\"%s\",\"cells_checked\":%zu,\"failures\":%zu",
            mode.name,mode.compute_name,mode.half_in?"FP16":"FP32",mode.half_out?"FP16":"FP32",
            mode.tensor?"true":"false",m,n,k,seed,validate_only?"seeded-dyadic-v1":"same-dyadic-v1",result.size(),failures);
        if (!validate_only) {
            std::sort(rates.begin(),rates.end());
            std::printf(",\"warmups\":10,\"iterations\":20,\"reps\":5,\"times_ms\":[");
            for(size_t i=0;i<times.size();++i) std::printf("%s%.8f",i?",":"",times[i]);
            std::printf("],\"median_TF_s\":%.6f,\"best_TF_s\":%.6f",rates[2],rates.back());
        }
        std::printf("}\n"); std::fflush(stdout);
    }
    cu(cudaFree(da)); cu(cudaFree(db)); cu(cudaFree(dc));
    return failures==0;
}

// Dispatch storage/scalar types consistently with the cuBLAS computation contract.
static bool dispatch(cublasHandle_t handle,const Mode &mode,int m,int n,int k,unsigned seed,bool validate,bool trace) {
    if (!mode.half_in) return run<float,float>(handle,mode,m,n,k,seed,validate,trace);
    if (mode.half_out) return run<__half,__half>(handle,mode,m,n,k,seed,validate,trace);
    return run<__half,float>(handle,mode,m,n,k,seed,validate,trace);
}

// Benchmark all paths, validate three seeds per path, or trace one path without timed instrumentation.
int main(int argc,char **argv) {
    if (argc<2 || argc>4) { std::fprintf(stderr,"usage: %s --benchmark | --validate | --trace MODE [2048|4096|8192]\n",argv[0]); return 2; }
    cublasHandle_t handle; blas(cublasCreate(&handle)); bool pass=true;
    if (argc==2 && !std::strcmp(argv[1],"--benchmark")) {
        for(int n:{2048,4096,8192}) for(const auto &mode:modes)
            pass &= dispatch(handle,mode,n,n,n,1,false,false);
    } else if (argc==2 && !std::strcmp(argv[1],"--validate")) {
        const int shapes[][3]={{256,256,256},{64,96,128},{127,65,95}};
        for(const auto &mode:modes) for(int i=0;i<3;++i)
            pass &= dispatch(handle,mode,shapes[i][0],shapes[i][1],shapes[i][2],i+1,true,false);
    } else if ((argc==3 || argc==4) && !std::strcmp(argv[1],"--trace")) {
        int n=2048;
        if (argc==4) {
            if (!std::strcmp(argv[3],"4096")) n=4096;
            else if (!std::strcmp(argv[3],"8192")) n=8192;
            else if (std::strcmp(argv[3],"2048")) { blas(cublasDestroy(handle)); return 2; }
        }
        bool found=false;
        for(const auto &mode:modes) if (!std::strcmp(argv[2],mode.name)) {
            found=true; pass &= dispatch(handle,mode,n,n,n,1,false,true);
        }
        pass &= found;
    } else pass=false;
    blas(cublasDestroy(handle));
    return pass?0:1;
}
