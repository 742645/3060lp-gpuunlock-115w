#!/bin/bash
# Restore the standard GRUB default; existing selector overrides can survive a VM warm reboot.
set -euo pipefail
rm -f /etc/default/grub.d/99-gpu-unlock-ga106m.cfg
grub-editenv /boot/grub/grubenv unset next_entry
update-grub
grub-script-check /boot/grub/grub.cfg
printf 'Standard boot selected. Future boots will not run the unlock transaction.\n'
printf 'Existing overrides survived a VM warm reboot in this experiment; clearing them may require a physical GPU power cycle, then a stock boot and rate verification.\n'
