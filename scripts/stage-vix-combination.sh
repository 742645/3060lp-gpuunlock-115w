#!/bin/bash
# Verify the separate Vix initrd and publish a one-shot combined entry; preserve normal default.
set -euo pipefail
work=/home/gpuuser/3060lp-work
kernel=$(uname -r)
test "$kernel" = 6.8.0-147-generic
/usr/local/bin/nvbios system status > "$work/vix-status-before-boot.json"
python3 - "$work" "$kernel" <<'SELECT'
from pathlib import Path
import json,re,shlex,sys
work=Path(sys.argv[1])
status=json.loads(Path('/var/lib/nvidia-vbios/lifecycle/status-'+sys.argv[2]+'.json').read_text())
assert status['state']=='ready' and re.fullmatch(r'vbios-[A-Za-z0-9.-]+',status['entry'])
entry=Path('/boot/grub/nvidia-vbios-entries')/(status['entry']+'.cfg')
text=entry.read_text()
initrds=[shlex.split(line)[1:] for line in text.splitlines() if line.strip().startswith('initrd ')]
assert len(initrds)==1 and len(initrds[0])==1
image=Path('/boot'+initrds[0][0])
assert image.is_file() and image.is_relative_to('/boot/nvidia-vbios')
(work/'vix-selected-image.txt').write_text(str(image)+'\n')
(work/'vix-selected-entry.txt').write_text(str(entry)+'\n')
SELECT
image=$(cat "$work/vix-selected-image.txt")
unpack="$work/vix-default115w-image-inspection"
test ! -e "$unpack"
unmkinitramfs "$image" "$unpack"
python3 - "$work" "$kernel" <<'VERIFY'
from pathlib import Path
import hashlib,json,subprocess,sys
work=Path(sys.argv[1]); kernel=sys.argv[2]
sys.path.insert(0,str(work/'tools'))
from vix_evidence import PAYLOAD_SHA256,verify_order
tree=work/'vix-default115w-image-inspection'
orders=list(tree.glob('**/scripts/init-top/ORDER'))
assert len(orders)==1
order=verify_order(orders[0].read_text())
gpu_modules=list(tree.glob('**/gpu_unlock.ko'))
assert len(gpu_modules)==1
assert gpu_modules[0].read_bytes()==Path('/lib/modules/'+kernel+'/extra/gpu_unlock.ko').read_bytes()
nvmodules=list(tree.glob('**/nvidia.ko'))
assert len(nvmodules)==1
assert subprocess.check_output(['modinfo','-F','version',str(nvmodules[0])],text=True).strip()=='615.71.09'
assert subprocess.check_output(['modinfo','-F','vbios_override_per_gpu',str(nvmodules[0])],text=True).strip()=='1'
payloads=[p for p in tree.glob('**/firmware/nvidia/**/*.bin') if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==PAYLOAD_SHA256]
assert len(payloads)==1, 'exact selected payload missing'
for original in Path('/lib/firmware/gpu-unlock/610.43.03').rglob('*'):
    if original.is_file():
        copies=list(tree.glob('**/firmware/gpu-unlock/610.43.03/'+str(original.relative_to('/lib/firmware/gpu-unlock/610.43.03'))))
        assert len(copies)==1 and copies[0].read_bytes()==original.read_bytes()
activation=list(tree.glob('**/etc/modprobe.d/zz-vbios-override-test.conf'))
assert len(activation)==1
assert 'pci=0000:00:10.0' in activation[0].read_text()
assert 'RmLinuxVbiosOverride=2' in activation[0].read_text()
proof={'order':order,'payload_sha256':PAYLOAD_SHA256,'nvidia_version':'615.71.09','nvidia_per_gpu_patch':True}
(work/'vix-image-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
entry=Path((work/'vix-selected-entry.txt').read_text().strip()).read_text().splitlines()
entry[0]="menuentry 'GPU Unlock + Vix 115W experiment (615.71.09)' --id gpu-unlock-vix-115w {"
found=0
for i,line in enumerate(entry):
    if line.strip().startswith('linux '):
        assert 'gpu_unlock.' not in line
        entry[i]+=' gpu_unlock.targets=0000:00:10.0 gpu_unlock.ga106m_experimental=1'
        found+=1
assert found==1
script="#!/bin/sh\nexec cat <<'GPU_VIX_ENTRY'\n"+'\n'.join(entry)+"\nGPU_VIX_ENTRY\n"
Path('/etc/grub.d/42_gpu_unlock_vix').write_text(script)
VERIFY
chmod 755 /etc/grub.d/42_gpu_unlock_vix
update-grub
grub-script-check /boot/grub/grub.cfg
cmp /boot/initrd.img-"$kernel" "$work/backups/vix115/normal-615.initrd"
grub-set-default gnulinux-simple-00000000-0000-4000-8000-000000000001
grub-reboot gpu-unlock-vix-115w
grub-editenv /boot/grub/grubenv list
printf 'VIX_COMBINATION_ONE_SHOT_STAGED\n'
