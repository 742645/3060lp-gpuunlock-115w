"""Upload explicit workspace files to the experiment directory over pinned SSH."""

import argparse
import base64
import os
from pathlib import Path, PurePosixPath
from remote import invoke


def upload_source(paths):
    """Generate a remote Python script containing only explicitly requested relative files."""
    remote_root = PurePosixPath(os.environ.get("GPU_REMOTE_WORKDIR", "/home/gpuuser/3060lp-work"))
    if not remote_root.is_absolute() or ".." in remote_root.parts:
        raise ValueError("GPU_REMOTE_WORKDIR must be an absolute POSIX path without parent traversal")
    lines = ["from pathlib import Path", "import base64", f"root = Path({str(remote_root)!r})"]
    for path in paths:
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("upload paths must stay within the workspace")
        content = base64.b64encode(path.read_bytes()).decode()
        lines.extend([f"p = root / {path.as_posix()!r}", "p.parent.mkdir(parents=True, exist_ok=True)",
                      f"p.write_bytes(base64.b64decode({content!r}))"])
    return "python3 - <<'UPLOAD_PY'\n" + "\n".join(lines) + "\nUPLOAD_PY\n"


def main():
    """Copy selected files without storing passwords or introducing SCP quoting ambiguities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()
    raise SystemExit(invoke(upload_source(args.paths)))


if __name__ == "__main__":
    main()
