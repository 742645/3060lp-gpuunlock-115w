#!/bin/bash
# Collect a single stock/unlocked pass, keeping launch shape, binaries and policy identical.
set -euo pipefail
state=${1:?state must be stock or unlocked}
case "$state" in stock|unlocked) ;; *) exit 2;; esac
cd /home/gpuuser/gpu-unlock-ada
out="build/matched-bench/$state"
mkdir -p "$out"
exec > >(tee "$out/collection.log") 2>&1
date -u --iso-8601=seconds
uname -r | tee "$out/kernel.txt"
cat /proc/cmdline | tee "$out/cmdline.txt"
sha256sum -c build/matched-bench/SHA256SUMS
nvidia-smi -q > "$out/nvidia-smi-before.txt"
nvidia-smi --query-gpu=uuid,name,pci.bus_id,driver_version,power.limit,clocks.max.sm,memory.total \
    --format=csv | tee "$out/gpu.csv"
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
python3 tools/read_sm_rates.py --expect-uuid "$CUDA_VISIBLE_DEVICES" | tee "$out/sm-rates.json"
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tee "$out/processes.txt"
test ! -s "$out/processes.txt"
nvidia-smi --query-gpu=timestamp,utilization.gpu,power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem \
    --format=csv -lms 100 > "$out/telemetry.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true; wait "$monitor" 2>/dev/null || true' EXIT
build/matched-bench/validate_gemm | tee "$out/numerical.txt"
build/matched-bench/mma_bench 5000 5 | tee "$out/mma.txt"
build/matched-bench/gemm_bench | tee "$out/gemm.jsonl"
nvidia-smi -q > "$out/nvidia-smi-after.txt"
printf 'BENCHMARK_COLLECTION_PASSED %s\n' "$state"
