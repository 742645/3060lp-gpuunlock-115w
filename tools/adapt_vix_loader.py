"""Apply a narrow legacy-BIOS mokutil compatibility fix to the pinned Vix loader."""

import argparse
from pathlib import Path

OLD = "        state = self.base._text(['mokutil', '--sb-state']).lower()"
NEW = "        state = secure_boot_state(self.base, self.root)"
SIGN_OLD = '        state = self._text(["mokutil", "--sb-state"]).lower()'
SIGN_NEW = ('        from .portable import secure_boot_state\n'
            '        state = secure_boot_state(self, getattr(self, "filesystem_root", Path("/")))')
HELPER = '''def secure_boot_state(ops, root):
    """Accept mokutil exit 1 only for its exact non-EFI diagnostic on a BIOS boot."""
    result = ops._run(['mokutil', '--sb-state'], allow_failure=True)
    state = (result.stdout + result.stderr).decode().strip().lower()
    bios_message = 'efi variables are not supported on this system'
    if result.returncode:
        if not (result.returncode == 1 and state == bios_message
                and not (Path(root) / 'sys/firmware/efi').exists()):
            raise LifecycleError('cannot determine Secure Boot state: mokutil failed')
    return state


'''


def adapt(source):
    """Return the patched source, refusing changed or already patched inputs."""
    if source.count(OLD) != 1 or 'def secure_boot_state(' in source:
        raise ValueError('unexpected Vix portable.py; inspect the pinned source')
    return source.replace(OLD, NEW).replace('class BootOperations:', HELPER + 'class BootOperations:', 1)


def adapt_signing(source):
    """Use the same strict BIOS diagnostic check at the post-build signing boundary."""
    if source.count(SIGN_OLD) != 1 or SIGN_NEW in source:
        raise ValueError('unexpected Vix system.py; inspect the pinned source')
    return source.replace(SIGN_OLD, SIGN_NEW)


def main():
    """Patch only the explicitly selected checkout's portable adapter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkout', type=Path)
    args = parser.parse_args()
    path = args.checkout / 'src/nvidia_vbios/lifecycle/portable.py'
    source = path.read_text(encoding='utf-8')
    if HELPER not in source or NEW not in source:
        path.write_bytes(adapt(source).encode('utf-8'))
    path = args.checkout / 'src/nvidia_vbios/lifecycle/system.py'
    path.write_bytes(adapt_signing(path.read_text(encoding='utf-8')).encode('utf-8'))


if __name__ == '__main__':
    main()
