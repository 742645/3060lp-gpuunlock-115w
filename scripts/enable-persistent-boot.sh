#!/bin/bash
# Select the verified, kernel-specific experiment by default while keeping the ordinary Ubuntu entry.
set -euo pipefail
kernel=$(uname -r)
test -s "/boot/initrd.img-$kernel-gpu-unlock"
grep -q 'gpu_unlock.ga106m_experimental=1' /proc/cmdline
journalctl -k -b --no-pager | grep -F 'pci 0000:00:10.0: gpu_unlock: unlocked' > /dev/null
if journalctl -k -b --no-pager | grep -E 'NVRM: Xid|GPU has fallen off|RmInitAdapter failed'; then exit 1; fi
uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
gpu_rates=$(mktemp /tmp/gpu-unlock-rates.XXXXXX)
trap 'rm -f "$gpu_rates"' EXIT
python3 /home/gpuuser/gpu-unlock-ada/tools/read_sm_rates.py --expect-uuid "$uuid" > "$gpu_rates"
python3 - "$gpu_rates" <<'VERIFY'
import json, sys
with open(sys.argv[1]) as f:
    assert json.load(f)['rates']['fmla32'] == 0
VERIFY
mkdir -p /etc/default/grub.d
cat > /etc/default/grub.d/99-gpu-unlock-ga106m.cfg <<'CONFIG'
# Tested on this exact GA106M VM; regenerate the experiment image/entry for a new kernel.
GRUB_DEFAULT=gpu-unlock-ga106m
GRUB_TIMEOUT_STYLE=menu
GRUB_TIMEOUT=5
CONFIG
update-grub
grub-script-check /boot/grub/grub.cfg
grep -F 'set default="gpu-unlock-ga106m"' /boot/grub/grub.cfg
grub-editenv /boot/grub/grubenv unset next_entry
printf 'PERSISTENT_EXPERIMENT_ENABLED\n'
