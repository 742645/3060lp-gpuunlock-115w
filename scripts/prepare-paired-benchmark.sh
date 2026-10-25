#!/bin/bash
# Freeze the same source/binary/libraries before collecting either measurement state.
set -euo pipefail
cd /home/gpuuser/gpu-unlock-ada
base=build/paired-five-paths
test ! -e "$base/FROZEN"
mkdir -p "$base"
cp /home/gpuuser/3060lp-work/benchmarks/paired_gemm.cu "$base/paired_gemm.cu"
nvcc -ccbin g++-12 -O3 -arch=sm_86 -std=c++17 -o "$base/paired_gemm" "$base/paired_gemm.cu" -lcublas -lcupti
"$base/paired_gemm" --validate > "$base/preflight-numerical.jsonl"
: > "$base/preflight-trace.jsonl"
for n in 2048 4096 8192; do
    for mode in fp32_simt fp16_simt tf32_tensor_fp32acc fp16_tensor_fp16acc fp16_tensor_fp32acc; do
        "$base/paired_gemm" --trace "$mode" "$n" >> "$base/preflight-trace.jsonl"
    done
done
python3 /home/gpuuser/3060lp-work/tools/verify_tensor_paths.py "$base/preflight-trace.jsonl" "$base/preflight-instructions"
sha256sum "$base/paired_gemm" "$base/paired_gemm.cu" \
    /lib/x86_64-linux-gnu/libcublas.so.12 /lib/x86_64-linux-gnu/libcublasLt.so.12 \
    /lib/x86_64-linux-gnu/libcupti.so.12 > "$base/SHA256SUMS"
touch "$base/FROZEN"
printf 'PAIRED_BENCHMARK_PREPARED\n'
