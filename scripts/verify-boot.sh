#!/bin/bash
# Verify the persistent boot, driver health and seeded numerical results without repeating throughput runs.
set -euo pipefail
cd /home/gpuuser/gpu-unlock-ada
out=build/matched-bench/repeat
mkdir -p "$out"
journalctl -k -b --no-pager > "$out/kernel-journal.txt"
cat /proc/cmdline | tee "$out/cmdline.txt"
uname -r > "$out/kernel.txt"
if grep -F 'pci 0000:00:10.0: gpu_unlock: unlocked' "$out/kernel-journal.txt"; then
    printf 'fresh-transaction\n' | tee "$out/boot-state.txt"
else
    # EALREADY alone is insufficient: require retained selectors plus the independent rate/numeric checks below.
    grep -F 'pci 0000:00:10.0: entry: plm=0xffffffff ss0=0x88888888 ss1=0x8' "$out/kernel-journal.txt"
    grep -F 'pci 0000:00:10.0: gpu_unlock: unlock failed (-114)' "$out/kernel-journal.txt"
    printf 'retained-unlock-no-new-transaction\n' | tee "$out/boot-state.txt"
fi
if grep -E 'NVRM: Xid|GPU has fallen off|RmInitAdapter failed' "$out/kernel-journal.txt"; then exit 1; fi
uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
python3 tools/read_sm_rates.py --expect-uuid "$uuid" | tee "$out/sm-rates.json"
python3 - <<'VERIFY'
import json
with open('build/matched-bench/repeat/sm-rates.json') as f:
    assert json.load(f)['rates']['fmla32'] == 0
VERIFY
sha256sum -c build/matched-bench/SHA256SUMS
build/matched-bench/validate_gemm | tee "$out/numerical.txt"
nvidia-smi -q > "$out/nvidia-smi.txt"
sha256sum -c /home/gpuuser/3060lp-work/backups/stock-initrd.SHA256SUMS
grub-editenv /boot/grub/grubenv list
printf 'PERSISTENT_BOOT_REPEAT_PASSED\n'
