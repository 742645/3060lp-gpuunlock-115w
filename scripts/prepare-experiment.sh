#!/bin/bash
# Prepare source and compile the same benchmark binaries for both measurement states.
set -euo pipefail
cd /home/gpuuser/gpu-unlock-ada
test "$(git rev-parse HEAD)" = 4e7fefcc4923f9e083d5324dc75a7d2f906af77f
if ! grep -q 'static bool ga106m_experimental;' kernel/gpu_unlock/main.c; then
    python3 /home/gpuuser/3060lp-work/tools/adapt_ga106m.py "$PWD"
fi
git apply --reverse --check /home/gpuuser/3060lp-work/patches/ga106m-experimental.patch
mkdir -p build/matched-bench
gcc -Wall -Wextra -Werror -Wno-unused-parameter -Wno-sign-compare \
    -pthread -Itests/main_shim tests/test_main.c -o build/test_main
build/test_main
nvcc -ccbin g++-12 -O3 -arch=sm_86 -std=c++17 \
    -o build/matched-bench/mma_bench diagnostics/bench/mma_bench.cu
nvcc -ccbin g++-12 -O3 -arch=sm_86 -std=c++17 \
    -I diagnostics/bench -o build/matched-bench/validate_gemm \
    /home/gpuuser/3060lp-work/benchmarks/validate_gemm_sm86.cu -lcublas
nvcc -ccbin g++-12 -O3 -arch=sm_86 -std=c++17 \
    -o build/matched-bench/gemm_bench /home/gpuuser/3060lp-work/benchmarks/gemm_bench.cu -lcublas
sha256sum build/matched-bench/{mma_bench,validate_gemm,gemm_bench} > build/matched-bench/SHA256SUMS
cat build/matched-bench/SHA256SUMS
