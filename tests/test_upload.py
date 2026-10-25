"""Check the configurable remote workspace without opening an SSH connection."""

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('upload_under_test', ROOT/'tools/upload.py')
upload = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, remote=type('RemoteStub', (), {'invoke': None})):
    spec.loader.exec_module(upload)


class UploadConfigurationTests(unittest.TestCase):
    def test_configured_remote_workspace_is_quoted_as_python_data(self):
        """Spaces in the configured path must not become shell syntax."""
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                Path('sample.txt').write_text('sample')
                with patch.dict(os.environ, GPU_REMOTE_WORKDIR='/srv/gpu work'):
                    source = upload.upload_source([Path('sample.txt')])
                self.assertIn("root = Path('/srv/gpu work')", source)
                compile(source.split("\n", 1)[1].rsplit('\nUPLOAD_PY', 1)[0], '<upload>', 'exec')
            finally:
                os.chdir(previous)

    def test_invalid_remote_root_is_rejected(self):
        """Do not upload through a relative root or parent traversal."""
        for value in ('relative', '/srv/../other'):
            with self.subTest(root=value), patch.dict(os.environ, GPU_REMOTE_WORKDIR=value):
                with self.assertRaises(ValueError):
                    upload.upload_source([])


if __name__ == '__main__':
    unittest.main()
