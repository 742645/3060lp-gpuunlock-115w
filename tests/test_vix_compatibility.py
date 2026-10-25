"""Regression checks for the BIOS-only mokutil exception without weakening UEFI checks."""

import ast
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from tools.adapt_vix_loader import HELPER, OLD, NEW, SIGN_OLD, SIGN_NEW, adapt, adapt_signing


class VixCompatibilityTests(unittest.TestCase):
    def state(self, code, output, *, efi=False, stderr=b''):
        """Execute the exact inserted helper against a fake command and firmware tree."""
        namespace = {'Path': Path, 'LifecycleError': ValueError}
        exec(compile(ast.parse(HELPER), '<vix-bios-adapter>', 'exec'), namespace)
        calls = []
        def run(command, **kwargs):
            calls.append((command, kwargs))
            return SimpleNamespace(returncode=code, stdout=output, stderr=stderr)
        with tempfile.TemporaryDirectory() as directory:
            if efi:
                (Path(directory) / 'sys/firmware/efi').mkdir(parents=True)
            result = namespace['secure_boot_state'](SimpleNamespace(_run=run), Path(directory))
        self.assertEqual(calls, [(['mokutil', '--sb-state'], {'allow_failure': True})])
        return result

    def test_bios_diagnostic_on_stderr(self):
        self.assertIn('efi variables', self.state(1, b'', stderr=b'EFI variables are not supported on this system\n'))

    def test_enabled_uefi_state_is_preserved(self):
        self.assertEqual(self.state(0, b'SecureBoot enabled\n', efi=True), 'secureboot enabled')

    def test_disabled_uefi_state_is_preserved(self):
        self.assertEqual(self.state(0, b'SecureBoot disabled\n', efi=True), 'secureboot disabled')

    def test_efi_diagnostic_cannot_hide_uefi_failure(self):
        with self.assertRaises(ValueError):
            self.state(1, b'EFI variables are not supported on this system', efi=True)

    def test_other_failures_remain_blocking(self):
        for code, output in ((2, b'EFI variables are not supported on this system'),
                             (1, b'SecureBoot disabled'), (1, b'Permission denied')):
            with self.subTest(code=code, output=output), self.assertRaises(ValueError):
                self.state(code, output)

    def test_patch_refuses_drift_and_second_application(self):
        source = 'class BootOperations:\n    def prerequisites(self):\n' + OLD + '\n'
        patched = adapt(source)
        self.assertIn(NEW, patched)
        ast.parse(patched)
        with self.assertRaises(ValueError):
            adapt(patched)
        with self.assertRaises(ValueError):
            adapt(source.replace(OLD, '        changed()'))

    def test_signing_boundary_uses_same_strict_check(self):
        source = 'class SystemOperations:\n    def sign(self):\n' + SIGN_OLD + '\n'
        patched = adapt_signing(source)
        self.assertIn(SIGN_NEW, patched)
        ast.parse(patched)
        with self.assertRaises(ValueError):
            adapt_signing(patched)
        with self.assertRaises(ValueError):
            adapt_signing(source.replace(SIGN_OLD, '        changed()'))


if __name__ == '__main__':
    unittest.main()
