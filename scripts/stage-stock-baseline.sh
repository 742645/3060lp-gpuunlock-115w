#!/bin/bash
# Disable automatic unlock before physical reset; this alone does not clear retained GPU registers.
set -euo pipefail
test -e /home/gpuuser/gpu-unlock-ada/build/paired-five-paths/FROZEN
backup=/home/gpuuser/3060lp-work/backups/paired-default-unlock.cfg
if [ -f /etc/default/grub.d/99-gpu-unlock-ga106m.cfg ] && [ ! -e "$backup" ]; then
    cp -a /etc/default/grub.d/99-gpu-unlock-ga106m.cfg "$backup"
fi
bash /home/gpuuser/3060lp-work/scripts/disable-experiment.sh
test ! -e /etc/default/grub.d/99-gpu-unlock-ga106m.cfg
grep -F 'set default="0"' /boot/grub/grub.cfg
printf 'STOCK_BOOT_STAGED_REQUIRES_GPU_RESET_AND_RATE_CHECK\n'
