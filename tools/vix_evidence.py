"""Validate combined VBIOS/power/unlock evidence before allowing a benchmark label."""

import argparse
import json
from pathlib import Path
import re

GPU_UUID = 'GPU-00000000-0000-4000-8000-000000000002'
ORIGINAL_PAYLOAD_SHA256 = 'b184ef3350e144df407664160f13e0e3dfda9e4ad1b0acd73a28f9afcc683039'
PAYLOAD_SHA256 = 'e32f881baea5e74307ac14a612c3344bfcc9f8b5648245d4cb8da44870fded25'
FAULTS = r'NVRM: Xid|GPU has fallen off|RmInitAdapter failed'


def verify_order(text):
    """Require GPU unlock before both udev and the Vix NVIDIA force-load script."""
    names = re.findall(r'/scripts/init-top/([A-Za-z0-9_-]+)', text)
    positions = {}
    for name in ('00-gpu-unlock-experiment', 'udev', 'vbios_override'):
        if name not in names:
            raise ValueError('missing early script: ' + name)
        positions[name] = names.index(name)
    if not positions['00-gpu-unlock-experiment'] < min(positions['udev'], positions['vbios_override']):
        raise ValueError('NVIDIA may bind before GPU unlock')
    return positions


def verify_runtime(smi, rates, journal, cmdline, *, watts=115.0):
    """Require actual current cap, full hardware rates and the selected runtime VBIOS."""
    if GPU_UUID not in smi or '615.71.09' not in smi:
        raise ValueError('unexpected GPU or driver')
    if not re.search(r'VBIOS Version\s*:\s*94\.06\.13\.00\.63', smi, re.I):
        raise ValueError('runtime VBIOS differs from supplied ROM')
    powers = {}
    for field in ('Current Power Limit', 'Default Power Limit', 'Max Power Limit'):
        match = re.search(r'^\s*' + field + r'\s*:\s*([\d.]+) W', smi, re.M)
        if not match:
            raise ValueError('missing ' + field)
        powers[field] = float(match[1])
    if abs(powers['Current Power Limit'] - watts) > .01 or powers['Max Power Limit'] < watts:
        raise ValueError('actual power cap is not the requested ' + str(watts) + ' W')
    if any(rates['rates'][name] != 0 for name in ('fmla32', 'fmla16', 'imla4')):
        raise ValueError('GPU hardware rates are limited')
    if not all(flag in cmdline.split() for flag in ('gpu_unlock.targets=0000:00:10.0',
                                                   'gpu_unlock.ga106m_experimental=1')):
        raise ValueError('combined boot flags missing')
    if re.search(FAULTS, journal):
        raise ValueError('GPU driver fault detected')
    if 'vbios_override: GPU 0000:00:10 selected:' not in journal:
        raise ValueError('Vix payload was not staged for this GPU')
    if 'gpu_unlock: unlocked' in journal:
        provenance = 'fresh transaction'
    elif 'gpu_unlock' in journal and ('-114' in journal or 'already' in journal.lower()):
        provenance = 'retained across warm reboot; independently verified full rates'
    else:
        raise ValueError('GPU unlock boot evidence missing')
    return {'power_w': powers, 'unlock_provenance': provenance, 'combination_verified': True}


def main():
    """Verify a saved runtime snapshot without changing GPU state."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--watts', type=float, default=115.0)
    args = parser.parse_args()
    root = args.snapshot
    result = verify_runtime((root/'nvidia-smi.txt').read_text(),
                            json.loads((root/'sm-rates.json').read_text()),
                            (root/'kernel-journal.txt').read_text(),
                            (root/'cmdline.txt').read_text(), watts=args.watts)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
