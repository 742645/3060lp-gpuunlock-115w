#!/bin/bash
# Prepare the pinned Vix lifecycle with gated GPU unlock hooks; do not reboot or enable automation.
set -euo pipefail
work=/home/gpuuser/3060lp-work
backup="$work/backups/vix115"
loader=/home/gpuuser/nvidia-linux-vbios-vix-loader
kernel=$(uname -r)
normal=gnulinux-simple-00000000-0000-4000-8000-000000000001
test "$kernel" = 6.8.0-147-generic
test "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)" = 615.71.09
grep -q 'Open Kernel Module' /proc/driver/nvidia/version
test "$(git -C "$loader" rev-parse HEAD)" = 418942a5fe40269787d4b57e97ab0d2a922011cc
test ! -e "$work/vix-profile"
cp -a /boot/initrd.img-"$kernel" "$backup/normal-615.initrd"
sha256sum "$backup/normal-615.initrd" > "$backup/normal-615.SHA256SUMS"
# The old experiment image embeds 580 and must not be offered with 615 userspace.
mv /etc/grub.d/41_gpu_unlock_experiment "$backup/41_gpu_unlock_experiment.580"
sed -i 's/^GRUB_DEFAULT=.*/GRUB_DEFAULT=saved/' /etc/default/grub
grub-set-default "$normal"
grub-editenv /boot/grub/grubenv unset next_entry
update-grub
grub-script-check /boot/grub/grub.cfg
# These hooks remain inert without both explicit kernel flags. Normal image stays unchanged.
install -m755 "$work/initramfs-conf/hooks/gpu-unlock-experiment" /etc/initramfs-tools/hooks/gpu-unlock-experiment
install -m755 "$work/initramfs-conf/scripts/init-top/00-gpu-unlock-experiment" /etc/initramfs-tools/scripts/init-top/00-gpu-unlock-experiment
install -m755 "$work/initramfs-conf/scripts/init-top/udev" /etc/initramfs-tools/scripts/init-top/udev
# Select the separate patched build explicitly alongside Ubuntu's DKMS directory.
test ! -e /etc/depmod.d/99-vix-experiment.conf
for module in nvidia nvidia-modeset nvidia-drm nvidia-uvm nvidia-peermem; do
  printf 'override %s %s updates/vbios-override\n' "$module" "$kernel"
done > /etc/depmod.d/99-vix-experiment.conf
python3 "$loader/nvbios" system prepare \
  --device 0000:00:10.0 "$work/vix-115w-ampere.bin" \
  --driver-source /home/gpuuser/open-gpu-kernel-modules-615.71.09 \
  --userspace deb --no-auto-source --jobs 1 \
  --boot-backend grub --initramfs initramfs-tools --fallback-entry "$normal" \
  --output "$work/vix-profile"
python3 "$work/tools/adapt_vix_loader.py" "$loader"
python3 "$loader/nvbios" system install --repository "$loader"
/usr/local/bin/nvbios system enroll --profile "$work/vix-profile/profile.json"
# Always preserve normal boot selection even if a subsequent build fails.
trap 'grub-set-default "$normal"; grub-editenv /boot/grub/grubenv unset next_entry' EXIT
/usr/local/bin/nvbios system reconcile --kernel "$kernel"
/usr/local/bin/nvbios system status
sha256sum -c "$backup/normal-615.SHA256SUMS"
cmp /boot/initrd.img-"$kernel" "$backup/normal-615.initrd"
printf 'VIX_COMBINATION_IMAGE_READY_FOR_INSPECTION\n'
