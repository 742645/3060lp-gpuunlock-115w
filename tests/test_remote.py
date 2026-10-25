"""Validate remote invocation without network access or real credentials."""

import importlib.util
import io
import os
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("remote", Path(__file__).parents[1] / "tools/remote.py")
remote = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote)
SETTINGS = {"GPU_SSH_HOST": "192.0.2.10", "GPU_SSH_USER": "test-user",
            "GPU_SSH_HOST_KEY": "SHA256:" + "A" * 43, "GPU_SSH_PASSWORD": "test-only-password"}


class RemoteTests(unittest.TestCase):
    """Check credential lifetime, host pinning and remote error propagation."""

    def test_pinned_ssh_password_not_in_arguments(self):
        """Keep secrets in a temporary file and stdin; delete the file after execution."""
        observed = {}

        class Process:
            def __init__(self, args, **kwargs):
                observed["args"] = args
                observed["path"] = Path(args[args.index("-pwfile") + 1])
                self.stdin = io.BytesIO()
                self.stdout = io.BytesIO(b"verified\n")
                self.secret = observed["path"].read_text()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def wait(self):
                return 7

        with patch.dict(os.environ, SETTINGS, clear=True), \
                patch.object(remote.subprocess, "Popen", Process):
            self.assertEqual(remote.invoke("printf safe", sudo=True), 7)
        self.assertIn(SETTINGS["GPU_SSH_HOST_KEY"], observed["args"])
        self.assertIn("test-user@192.0.2.10", observed["args"])
        self.assertNotIn("test-only-password", " ".join(observed["args"]))
        self.assertFalse(observed["path"].exists())
        self.assertIn("sudo -S", observed["args"][-1])

    def test_missing_local_setting_does_not_start_ssh(self):
        """Never fall back to the original experiment's host, user or host key."""
        for name in SETTINGS:
            settings = {key: value for key, value in SETTINGS.items() if key != name}
            with self.subTest(setting=name), patch.dict(os.environ, settings, clear=True), \
                    patch.object(remote.subprocess, "Popen") as process:
                with self.assertRaises(KeyError):
                    remote.invoke("true")
                process.assert_not_called()

    def test_empty_host_key_is_rejected_before_ssh(self):
        """Require host pinning even if the key variable exists but is blank."""
        with patch.dict(os.environ, {**SETTINGS, "GPU_SSH_HOST_KEY": " "}, clear=True), \
                patch.object(remote.subprocess, "Popen") as process:
            with self.assertRaisesRegex(ValueError, "GPU_SSH_HOST_KEY"):
                remote.invoke("true")
            process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
