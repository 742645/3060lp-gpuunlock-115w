"""Run a script over pinned SSH using the existing PuTTY client, without stored credentials."""

import argparse
import base64
import os
from pathlib import Path
import subprocess
import tempfile

PLINK = r"C:\Program Files\PuTTY\plink.exe"


def required_env(name):
    """Require an explicit local setting and report only its name when it is missing."""
    value = os.environ[name]
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def invoke(script, *, sudo=False, log=None):
    """Execute UTF-8 Bash source using explicit environment settings and a pinned SSH host key."""
    host = required_env("GPU_SSH_HOST")
    user = required_env("GPU_SSH_USER")
    host_key = required_env("GPU_SSH_HOST_KEY")
    password = required_env("GPU_SSH_PASSWORD")
    encoded = base64.b64encode(script.encode()).decode("ascii")
    command = f"printf %s {encoded} | base64 -d | bash"
    if sudo:
        command = f"sudo -S -p '' bash -c '{command}'"
    with tempfile.TemporaryDirectory(prefix="gpu-ssh-") as temp:
        secret = Path(temp) / "password"
        secret.write_text(password + "\n", encoding="utf-8")
        args = [PLINK, "-ssh", "-batch", "-noagent", "-hostkey", host_key,
                "-pwfile", str(secret), f"{user}@{host}", command]
        with subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT) as process:
            process.stdin.write((password + "\n").encode() if sudo else b"")
            process.stdin.close()
            sink = open(log, "wb") if log else None
            try:
                for line in process.stdout:
                    if sink:
                        sink.write(line)
                        sink.flush()
                    print(line.decode("utf-8", errors="replace"), end="", flush=True)
            finally:
                if sink:
                    sink.close()
            return process.wait()


def main():
    """Read a Bash file or command and propagate the remote process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--script", type=Path)
    source.add_argument("--command")
    parser.add_argument("--sudo", action="store_true")
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    script = args.script.read_text(encoding="utf-8-sig") if args.script else args.command
    raise SystemExit(invoke(script, sudo=args.sudo, log=args.log))


if __name__ == "__main__":
    main()
