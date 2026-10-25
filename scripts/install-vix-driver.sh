#!/bin/bash
# Cache an exact 580 rollback set, preserve boot files, then install the user-approved R615 open stack.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
work=/home/gpuuser/3060lp-work
backup="$work/backups/vix115"
mkdir -p "$backup/debs-580" "$backup/boot-580"
python3 - "$backup" <<'PY'
from pathlib import Path
import json,subprocess,sys
root=Path(sys.argv[1])
result=subprocess.run(['dpkg-query','-W','-f=${db:Status-Status}\t${binary:Package}\t${Version}\n'],capture_output=True,text=True,check=True)
packages=[]
for line in result.stdout.splitlines():
    status,name,version=line.split('\t')
    if status=='installed' and ('nvidia' in name) and ('580' in name):
        packages.append({'package':name,'version':version})
assert len(packages)>=15, 'expected original 580 stack before migration'
manifest=root/'installed-580.json'
assert not manifest.exists(), 'rollback manifest already exists; review before retry'
manifest.write_text(json.dumps(packages,indent=2)+'\n')
subprocess.run(['apt-get','download',*[p['package']+'='+p['version'] for p in packages]],cwd=root/'debs-580',check=True)
assert len(list((root/'debs-580').glob('*.deb')))==len(packages)
print('Exact rollback packages cached:',len(packages))
PY
tar -czf "$backup/etc-before.tar.gz" /etc/default/grub /etc/default/grub.d /etc/grub.d /etc/initramfs-tools /etc/modprobe.d
cp -a /boot/initrd.img-6.8.0-147-generic "$backup/boot-580/"
cp -a /boot/initrd.img-6.8.0-147-generic-gpu-unlock "$backup/boot-580/"
cp -a /boot/grub/grub.cfg "$backup/boot-580/grub.cfg"
sha256sum "$backup"/boot-580/* "$backup"/debs-580/*.deb > "$backup/SHA256SUMS"
bash "$work/scripts/disable-experiment.sh"
apt-get install -y --no-install-recommends nvidia-driver-615-open=615.71.09-0ubuntu0.24.04.2 dracut-core mokutil
dpkg --audit
dpkg-query -W nvidia-driver-615-open nvidia-dkms-615-open nvidia-utils-615 dracut-core mokutil
modinfo -F version nvidia
modinfo -F license nvidia
printf 'R615_OPEN_INSTALLED_REBOOT_REQUIRED\n'
