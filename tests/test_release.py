"""Regression checks for anonymized publication data and archive consistency."""

import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

from tools.verify_release import audit, inspect_bytes


class ReleasePrivacyTests(unittest.TestCase):
    def test_private_configuration_is_reported_without_values(self):
        """Report only a path and category for a leaked local deployment target."""
        private_address = '.'.join(('192', '168', '1', '42')).encode()
        result = inspect_bytes('config.txt', private_address)
        self.assertEqual(result, [{'path': 'config.txt', 'type': 'private_network_address'}])
        self.assertNotIn(private_address.decode(), str(result))

    def test_documentation_aliases_are_allowed(self):
        """The release uses reserved addresses and explicitly anonymous hardware identifiers."""
        data = b'192.0.2.10 /home/gpuuser GPU-00000000-0000-4000-8000-000000000001'
        self.assertEqual(inspect_bytes('fixture.txt', data), [])

    def test_hardware_identifier_must_be_anonymous(self):
        """An ordinary UUID cannot silently enter the public evidence again."""
        data = ('GPU-' + '-'.join(('12345678', '1234', '1234', '1234', '123456789abc'))).encode()
        self.assertEqual(inspect_bytes('gpu.txt', data)[0]['type'], 'unredacted_device_or_boot_identifier')

    def test_archive_cannot_reintroduce_old_content(self):
        """An archive and its expanded evidence must publish identical bytes."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root/'results/2026-10-25'
            (base/'raw').mkdir(parents=True)
            (base/'raw/record.txt').write_bytes(b'anonymous\n')
            archive_path = base/'benchmark-evidence.tar.gz'
            with tarfile.open(archive_path, 'w:gz') as archive:
                info = tarfile.TarInfo('record.txt')
                info.size = 4
                archive.addfile(info, io.BytesIO(b'old\n'))
            result = audit(root, [archive_path.relative_to(root).as_posix()], check_manifest=False)
            self.assertIn('archive_expanded_file_mismatch', {row['type'] for row in result['findings']})

    def test_digest_change_is_rejected(self):
        """The release manifest covers actual anonymized file bytes."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'record.txt').write_bytes(b'changed')
            (root/'SANITIZED_SHA256SUMS').write_text(hashlib.sha256(b'original').hexdigest()+'  record.txt\n')
            result = audit(root, ['record.txt', 'SANITIZED_SHA256SUMS'])
            self.assertIn('release_digest_mismatch', {row['type'] for row in result['findings']})


if __name__ == '__main__':
    unittest.main()
