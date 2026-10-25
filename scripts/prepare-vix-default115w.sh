#!/bin/bash
# Raise only the supplied ROM's default from 100 W to its existing 115 W maximum, in a new file.
set -euo pipefail
work=/home/gpuuser/3060lp-work
editor=/home/gpuuser/nvidia-vbios-vix-editor/nvbios
test ! -e "$work/3060-default115w.rom"
python3 "$editor" edit /home/gpuuser/3060-115w.rom "$work/3060-default115w.rom" \
  --set power_default_w=115 --expect-sha256 cda18b6b1b913d0700e20b2083f54154e7cde2974eac1497e0aec97845b67fa8 \
  > "$work/vix-default115w-edit.json"
python3 - "$work" <<'VERIFY'
from pathlib import Path
import json,sys
j=json.loads((Path(sys.argv[1])/'vix-default115w-edit.json').read_text())
changes=j['result']['affected_fields']
assert len(changes)==1 and changes[0]['field']=='power_default_w'
assert changes[0]['before']==100.0 and changes[0]['after']==115.0
VERIFY
python3 "$editor" normalize "$work/3060-default115w.rom" "$work/vix-default115w-ampere.bin" \
  --representation ampere-relocated-v1 > "$work/vix-default115w-normalize.json"
printf '%s  %s\n' e32f881baea5e74307ac14a612c3344bfcc9f8b5648245d4cb8da44870fded25 "$work/vix-default115w-ampere.bin" | sha256sum -c -
python3 /home/gpuuser/nvidia-linux-vbios-vix-loader/nvbios system prepare \
  --device 0000:00:10.0 "$work/vix-default115w-ampere.bin" \
  --driver-source /home/gpuuser/open-gpu-kernel-modules-615.71.09 \
  --userspace deb --no-auto-source --jobs 2 --boot-backend grub --initramfs initramfs-tools \
  --fallback-entry gnulinux-simple-00000000-0000-4000-8000-000000000001 \
  --output "$work/vix-default115w-profile"
/usr/local/bin/nvbios system enroll --profile "$work/vix-default115w-profile/profile.json" > "$work/vix-default115w-enroll.json"
normal=gnulinux-simple-00000000-0000-4000-8000-000000000001
trap 'grub-set-default "$normal"; grub-editenv /boot/grub/grubenv unset next_entry' EXIT
# The lifecycle requires its previous verified entry selected while replacing an existing generation.
old=$(python3 - <<'SELECT'
import json
from pathlib import Path
print(json.loads(Path('/var/lib/nvidia-vbios/lifecycle/fleet.json').read_text())['last_ready']['entry'])
SELECT
)
grub-set-default "$old"
/usr/local/bin/nvbios system reconcile --kernel "$(uname -r)" > "$work/vix-default115w-reconcile.json"
cmp /boot/initrd.img-"$(uname -r)" "$work/backups/vix115/normal-615.initrd"
printf 'VIX_DEFAULT115W_RECONCILED\n'
