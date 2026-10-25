#!/bin/bash
# Compile the expanded probe into a new run directory and measure the current unlocked GPU.
set -euo pipefail
cd /home/gpuuser/gpu-unlock-ada
out="build/strict-fp16-bench/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$(dirname "$out")"
mkdir "$out"
exec > >(tee "$out/collection.log") 2>&1
printf 'RUN_DIRECTORY=%s\n' "$PWD/$out"
date -u --iso-8601=seconds
uname -r > "$out/kernel.txt"
cat /proc/cmdline > "$out/cmdline.txt"
export CUDA_VISIBLE_DEVICES=GPU-00000000-0000-4000-8000-000000000002
nvidia-smi --query-gpu=uuid,name,pci.bus_id,driver_version,power.limit,memory.total --format=csv > "$out/gpu.csv"
python3 tools/read_sm_rates.py --expect-uuid "$CUDA_VISIBLE_DEVICES" | tee "$out/sm-rates.json"
python3 - "$out" <<'VERIFY'
import json, sys
from pathlib import Path
assert json.loads((Path(sys.argv[1])/'sm-rates.json').read_text())['rates']['fmla32'] == 0
VERIFY
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader > "$out/processes.txt"
test ! -s "$out/processes.txt"
cp /home/gpuuser/3060lp-work/benchmarks/gemm_bench.cu "$out/gemm_bench.cu"
nvcc -ccbin g++-12 -O3 -arch=sm_86 -std=c++17 -o "$out/gemm_bench" "$out/gemm_bench.cu" -lcublas
sha256sum "$out/gemm_bench" "$out/gemm_bench.cu" build/matched-bench/{mma_bench,validate_gemm} > "$out/SHA256SUMS"
sha256sum -c "$out/SHA256SUMS"
nvidia-smi -q > "$out/nvidia-smi-before.txt"
nvidia-smi --query-gpu=timestamp,utilization.gpu,power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem \
    --format=csv -lms 100 > "$out/telemetry.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true; wait "$monitor" 2>/dev/null || true' EXIT
build/matched-bench/validate_gemm | tee "$out/numerical.txt"
"$out/gemm_bench" --validate-strict16 | tee "$out/strict16-numerical.jsonl"
build/matched-bench/mma_bench 5000 5 | tee "$out/mma.txt"
"$out/gemm_bench" | tee "$out/gemm.jsonl"
nvidia-smi -q > "$out/nvidia-smi-after.txt"
journalctl -k -b --no-pager > "$out/kernel-journal.txt"
if grep -E 'NVRM: Xid|GPU has fallen off|RmInitAdapter failed' "$out/kernel-journal.txt"; then exit 1; fi
sha256sum -c "$out/SHA256SUMS"
printf 'STRICT_FP16_COLLECTION_PASSED\n'
