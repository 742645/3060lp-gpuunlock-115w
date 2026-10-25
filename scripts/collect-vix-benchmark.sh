#!/bin/bash
# Measure the frozen five-path workload only after actual 115 W plus full-rate unlock passes.
set -euo pipefail
work=/home/gpuuser/3060lp-work
cd /home/gpuuser/gpu-unlock-ada
base=/home/gpuuser/gpu-unlock-ada/build/paired-five-paths
out="$work/vix-combination-results"
test ! -e "$out"
sha256sum -c "$base/SHA256SUMS"
mkdir "$out"
exec > >(tee "$out/collection.log") 2>&1
export CUDA_VISIBLE_DEVICES=GPU-00000000-0000-4000-8000-000000000002
date -u --iso-8601=seconds > "$out/date.txt"
uname -r > "$out/kernel.txt"
cat /proc/cmdline > "$out/cmdline.txt"
nvidia-smi -q > "$out/nvidia-smi.txt"
journalctl -k -b --no-pager > "$out/kernel-journal.txt"
python3 /home/gpuuser/gpu-unlock-ada/tools/read_sm_rates.py --expect-uuid "$CUDA_VISIBLE_DEVICES" > "$out/sm-rates.json"
python3 "$work/tools/vix_evidence.py" "$out" > "$out/combination-proof.json"
/usr/local/bin/nvbios system status > "$out/vix-status.json"
cp "$base/SHA256SUMS" "$out/SHA256SUMS"
cp "$work/vix-image-proof.json" "$out/vix-image-proof.json"
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader > "$out/processes.txt"
test ! -s "$out/processes.txt"
"$base/paired_gemm" --validate > "$out/numerical.jsonl"
: > "$out/kernel-trace.jsonl"
for n in 2048 4096 8192; do
    for mode in fp32_simt fp16_simt tf32_tensor_fp32acc fp16_tensor_fp16acc fp16_tensor_fp32acc; do
        "$base/paired_gemm" --trace "$mode" "$n" >> "$out/kernel-trace.jsonl"
    done
done
python3 "$work/tools/verify_tensor_paths.py" "$out/kernel-trace.jsonl" "$out/instructions"
nvidia-smi --query-gpu=timestamp,utilization.gpu,power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem \
    --format=csv -lms 100 > "$out/telemetry.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true; wait "$monitor" 2>/dev/null || true' EXIT
"$base/paired_gemm" --benchmark | tee "$out/gemm.jsonl"
nvidia-smi -q > "$out/nvidia-smi-after.txt"
journalctl -k -b --no-pager > "$out/kernel-journal.txt"
python3 /home/gpuuser/gpu-unlock-ada/tools/read_sm_rates.py --expect-uuid "$CUDA_VISIBLE_DEVICES" > "$out/sm-rates-after.json"
python3 - "$work" "$out" <<'VERIFY'
from pathlib import Path
import json,sys
work,out=map(Path,sys.argv[1:])
sys.path.insert(0,str(work/'tools'))
from vix_evidence import verify_runtime
from paired_results import read_measurements,read_rows,EXPECTED
verify_runtime((out/'nvidia-smi-after.txt').read_text(),json.loads((out/'sm-rates-after.json').read_text()),
               (out/'kernel-journal.txt').read_text(),(out/'cmdline.txt').read_text())
assert len(read_measurements(out/'gemm.jsonl'))==15
checks=read_rows(out/'numerical.jsonl')
assert len(checks)==15 and not any(row['failures'] for row in checks)
instructions=json.loads((out/'instructions/verification.json').read_text())
assert {(row['mode'],row['n']) for row in instructions['paths']}==EXPECTED
assert all(row['verified'] for row in instructions['paths'])
VERIFY
sha256sum -c "$base/SHA256SUMS"
printf 'VIX_115W_COMBINATION_COLLECTION_PASSED\n'
