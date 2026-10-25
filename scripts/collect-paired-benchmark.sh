#!/bin/bash
# Collect a clean stock or freshly unlocked run; reject an incorrect hardware state before measuring.
set -euo pipefail
state=${1:?stock or unlocked required}
case "$state" in stock|unlocked) ;; *) exit 2;; esac
cd /home/gpuuser/gpu-unlock-ada
base=build/paired-five-paths
test -e "$base/FROZEN"
sha256sum -c "$base/SHA256SUMS"
test ! -e "$base/$state"
out="$base/$state"
mkdir "$out"
exec > >(tee "$out/collection.log") 2>&1
date -u --iso-8601=seconds
uname -r > "$out/kernel.txt"
cat /proc/cmdline > "$out/cmdline.txt"
journalctl -k -b --no-pager > "$out/kernel-journal-before.txt"
export CUDA_VISIBLE_DEVICES=GPU-00000000-0000-4000-8000-000000000002
nvidia-smi --query-gpu=uuid,name,pci.bus_id,driver_version,power.limit,memory.total --format=csv > "$out/gpu.csv"
python3 tools/read_sm_rates.py --expect-uuid "$CUDA_VISIBLE_DEVICES" | tee "$out/sm-rates.json"
python3 - "$state" "$out" <<'VERIFY'
import json, sys
from pathlib import Path
state, out = sys.argv[1], Path(sys.argv[2])
rates = json.loads((out/'sm-rates.json').read_text())['rates']
assert rates['fmla32'] == (1 if state == 'stock' else 0), 'hardware rate does not match requested state'
cmdline = (out/'cmdline.txt').read_text()
journal = (out/'kernel-journal-before.txt').read_text()
if state == 'stock':
    assert 'gpu_unlock.targets=' not in cmdline, 'stock boot must not enable the unlock'
    assert 'pci 0000:00:10.0: gpu_unlock: unlocked' not in journal
else:
    assert 'gpu_unlock.targets=0000:00:10.0' in cmdline
    assert 'pci 0000:00:10.0: gpu_unlock: unlocked' in journal, 'need a fresh successful transaction'
VERIFY
cp "$base/SHA256SUMS" "$out/SHA256SUMS"
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader > "$out/processes.txt"
test ! -s "$out/processes.txt"
nvidia-smi -q > "$out/nvidia-smi-before.txt"
"$base/paired_gemm" --validate > "$out/numerical.jsonl"
: > "$out/kernel-trace.jsonl"
for n in 2048 4096 8192; do
    for mode in fp32_simt fp16_simt tf32_tensor_fp32acc fp16_tensor_fp16acc fp16_tensor_fp32acc; do
        "$base/paired_gemm" --trace "$mode" "$n" >> "$out/kernel-trace.jsonl"
    done
done
python3 /home/gpuuser/3060lp-work/tools/verify_tensor_paths.py "$out/kernel-trace.jsonl" "$out/instructions"
nvidia-smi --query-gpu=timestamp,utilization.gpu,power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem \
    --format=csv -lms 100 > "$out/telemetry.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true; wait "$monitor" 2>/dev/null || true' EXIT
"$base/paired_gemm" --benchmark | tee "$out/gemm.jsonl"
nvidia-smi -q > "$out/nvidia-smi-after.txt"
journalctl -k -b --no-pager > "$out/kernel-journal.txt"
if grep -E 'NVRM: Xid|GPU has fallen off|RmInitAdapter failed' "$out/kernel-journal.txt"; then exit 1; fi
sha256sum -c "$base/SHA256SUMS"
printf 'PAIRED_COLLECTION_PASSED %s\n' "$state"
