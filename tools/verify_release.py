"""Audit only the Git publication index, including compressed evidence and file digests."""

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile

UUID = re.compile(rb'(?i)\b[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\b')
PATTERNS = {
    'private_network_address': rb'\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b',
    'windows_user_directory': rb'(?i)[A-Z]:[\\/]+Users[\\/]+[^\\/\s]+',
    'private_key': rb'-----BEGIN (?:OPENSSH |RSA |EC |DSA |ENCRYPTED )?PRIVATE KEY-----',
    'github_access_token': rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b',
    'api_key': rb'\b(?:sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}|AIza[A-Za-z0-9_-]{35}|AKIA[A-Z0-9]{16}|glpat-[A-Za-z0-9_-]{20,})\b',
    'credential_uri': rb'(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|https?)://[^\s/:@]+:[^\s/@]+@',
    'ssh_host_fingerprint': rb'SHA256:[A-Za-z0-9+/]{40,}=?',
}
ARCHIVE_ROOTS = {
    'benchmark-evidence.tar.gz': 'raw',
    'paired-five-paths-preparation.tar.gz': 'paired-five-paths-preparation',
    'paired-five-paths-results.tar.gz': 'paired-five-paths-results',
    'strict-fp16-20261025T051633Z.tar.gz': 'strict-fp16-20261025T051633Z',
    'vix-combination.tar.gz': '.',
}
MANIFEST = 'SANITIZED_SHA256SUMS'


def inspect_bytes(path, data):
    """Return paths and problem types only; never return matching values."""
    findings = [{'path': path, 'type': kind} for kind, pattern in PATTERNS.items() if re.search(pattern, data)]
    if any(not value.group().lower().startswith(b'00000000-0000-4000-8000-') for value in UUID.finditer(data)):
        findings.append({'path': path, 'type': 'unredacted_device_or_boot_identifier'})
    if any(match.group(1) != b'gpuuser' for match in re.finditer(rb'/home/([A-Za-z0-9_.-]+)', data)):
        findings.append({'path': path, 'type': 'local_account_directory'})
    if data.startswith(b'SQLite format 3\x00'):
        findings.append({'path': path, 'type': 'database'})
    return findings


def publication_files(root):
    """Read the staged/tracked file set without adding ignored or local backup files."""
    command = ['git', '-c', 'safe.directory=' + root.as_posix(), 'ls-files', '-z']
    return subprocess.check_output(command, cwd=root).decode().split('\0')[:-1]


def audit(root, files, *, check_manifest=True):
    """Check redaction, archive parity, metadata and actual published-byte SHA256 values."""
    findings = []
    members = 0
    for name in files:
        path = root / name
        if path.is_symlink() or not path.is_file():
            findings.append({'path': name, 'type': 'missing_or_linked_publication_file'})
            continue
        if name in ('agents.md', 'PREUPLOAD_CHECK.md') or name.startswith(('build/', 'upstream-')):
            findings.append({'path': name, 'type': 'local_only_file_in_index'})
        data = path.read_bytes()
        findings.extend(inspect_bytes(name, data))
        if not name.endswith('.tar.gz'):
            continue
        archive_root = ARCHIVE_ROOTS.get(path.name)
        if archive_root is None:
            findings.append({'path': name, 'type': 'unreviewed_archive'})
            continue
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
            for member in archive.getmembers():
                label = name + '::' + member.name
                relative = PurePosixPath(member.name)
                if relative.is_absolute() or '..' in relative.parts or not (member.isdir() or member.isfile()):
                    findings.append({'path': label, 'type': 'unsafe_archive_member'})
                    continue
                if member.uid or member.gid or member.uname or member.gname or member.mtime:
                    findings.append({'path': label, 'type': 'non_anonymous_archive_metadata'})
                if not member.isfile():
                    continue
                content = archive.extractfile(member).read()
                findings.extend(inspect_bytes(label, content))
                expanded = path.parent / archive_root / relative
                if not expanded.is_file() or expanded.read_bytes() != content:
                    findings.append({'path': label, 'type': 'archive_expanded_file_mismatch'})
                members += 1
    if check_manifest:
        expected = {}
        manifest_path = root / MANIFEST
        if not manifest_path.is_file():
            findings.append({'path': MANIFEST, 'type': 'missing_release_manifest'})
        else:
            for line in manifest_path.read_text().splitlines():
                digest, name = line.split('  ', 1)
                if name in expected or not re.fullmatch(r'[a-f0-9]{64}', digest):
                    findings.append({'path': MANIFEST, 'type': 'invalid_release_manifest'})
                expected[name] = digest
            if set(expected) != set(files) - {MANIFEST}:
                findings.append({'path': MANIFEST, 'type': 'release_manifest_coverage_mismatch'})
            for name, digest in expected.items():
                path = root / name
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    findings.append({'path': name, 'type': 'release_digest_mismatch'})
    return {'files': len(files), 'archive_members': members, 'findings': findings}


def main():
    """Print a value-free publication audit and fail if any issue remains."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    result = audit(root, publication_files(root))
    print(json.dumps(result, indent=2))
    return bool(result['findings'])


if __name__ == '__main__':
    raise SystemExit(main())
