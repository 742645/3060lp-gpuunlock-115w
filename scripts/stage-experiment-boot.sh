#!/bin/bash
# Install the tested module and a separate one-shot GRUB/initramfs experiment; leave stock image intact.
set -euo pipefail
kernel=$(uname -r)
work=/home/gpuuser/3060lp-work
source=/home/gpuuser/gpu-unlock-ada
test "$kernel" = 6.8.0-147-generic
test "$(cat /sys/bus/pci/devices/0000:00:10.0/device)" = 0x2520
mkdir -p "$work/backups"
sha256sum "/boot/initrd.img-$kernel" > "$work/backups/stock-initrd.SHA256SUMS"
cp -a /etc/default/grub "$work/backups/grub.default.before"
cp -a /boot/grub/grub.cfg "$work/backups/grub.cfg.before"
install -Dm644 "$source/kernel/gpu_unlock/gpu_unlock.ko" "/lib/modules/$kernel/extra/gpu_unlock.ko"
make -C "$source" ASSETS=/home/gpuuser/.cache/gpu-unlock/assets install-firmware
depmod -a "$kernel"
modinfo gpu_unlock | grep -E 'filename:|parm:|vermagic:'
conf="$work/initramfs-conf"
test ! -e "$conf"
cp -a /etc/initramfs-tools "$conf"
mkdir -p "$conf/hooks" "$conf/scripts/init-top"
cat > "$conf/hooks/gpu-unlock-experiment" <<'HOOK'
#!/bin/sh
set -e
case "${1:-}" in prereqs) exit 0;; esac
. /usr/share/initramfs-tools/hook-functions
manual_add_modules gpu_unlock
mkdir -p "$DESTDIR/lib/firmware/gpu-unlock"
cp -a /lib/firmware/gpu-unlock/610.43.03 "$DESTDIR/lib/firmware/gpu-unlock/"
HOOK
cat > "$conf/scripts/init-top/00-gpu-unlock-experiment" <<'EARLY'
#!/bin/sh
# Run before udev can bind NVIDIA; explicit parameters still require the exact GPU ID.
case "${1:-}" in prereqs) exit 0;; esac
cmdline=" $(cat /proc/cmdline) "
case "$cmdline" in *" gpu_unlock.targets=0000:00:10.0 "*) ;; *) exit 0;; esac
case "$cmdline" in *" gpu_unlock.ga106m_experimental=1 "*) ;; *) exit 0;; esac
echo '<6>gpu_unlock_experiment: begin before udev' > /dev/kmsg
modprobe gpu_unlock targets=0000:00:10.0 ga106m_experimental=1
result=$?
echo "<6>gpu_unlock_experiment: modprobe completed status=$result" > /dev/kmsg
# A loaded module alone is not success: inspect the selected card's transaction log.
exit 0
EARLY
cp /usr/share/initramfs-tools/scripts/init-top/udev "$conf/scripts/init-top/udev"
sed -i 's/PREREQS=""/PREREQS="00-gpu-unlock-experiment"/' "$conf/scripts/init-top/udev"
chmod 755 "$conf/hooks/gpu-unlock-experiment" "$conf/scripts/init-top/00-gpu-unlock-experiment" "$conf/scripts/init-top/udev"
image="/boot/initrd.img-$kernel-gpu-unlock"
mkinitramfs -d "$conf" -o "$image" "$kernel"
unpack="$work/initrd-inspection"
test ! -e "$unpack"
unmkinitramfs "$image" "$unpack"
order=$(find "$unpack" -path '*/scripts/init-top/ORDER' -print -quit)
test -n "$order"
cat "$order"
python3 - "$order" <<'VERIFY'
import sys
from pathlib import Path
s = Path(sys.argv[1]).read_text()
assert s.index('/00-gpu-unlock-experiment') < s.index('/udev'), 'unlock must precede udev'
VERIFY
lsinitramfs "$image" | grep -E 'gpu_unlock\.ko|v67-ga102|00-gpu-unlock-experiment'
boot_uuid=$(grub-probe --target=fs_uuid /boot)
cat > /etc/grub.d/41_gpu_unlock_experiment <<GRUB
#!/bin/sh
exec cat <<'ENTRY'
menuentry 'GPU Unlock GA106M experiment ($kernel)' --id gpu-unlock-ga106m {
    insmod part_gpt
    insmod ext2
    search --no-floppy --fs-uuid --set=root $boot_uuid
    linux /vmlinuz-$kernel root=/dev/mapper/ubuntu--vg-ubuntu--lv ro gpu_unlock.targets=0000:00:10.0 gpu_unlock.ga106m_experimental=1
    initrd /initrd.img-$kernel-gpu-unlock
}
ENTRY
GRUB
chmod 755 /etc/grub.d/41_gpu_unlock_experiment
update-grub
grub-script-check /boot/grub/grub.cfg
sha256sum -c "$work/backups/stock-initrd.SHA256SUMS"
grub-reboot gpu-unlock-ga106m
grub-editenv /boot/grub/grubenv list
printf 'EXPERIMENT_BOOT_STAGED\n'
