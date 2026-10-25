"""Verify actual cuBLAS kernel instructions from untimed CUPTI activity records (Linux)."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

MODES = ("fp32_simt", "fp16_simt", "tf32_tensor_fp32acc", "fp16_tensor_fp16acc", "fp16_tensor_fp32acc")
SIZES = (2048, 4096, 8192)


def check_opcodes(mode, opcodes):
    """Reject silent Tensor fallback or a tensor instruction in a requested SIMT path."""
    tensor = [op for op in opcodes if re.match(r"[HID]MMA(?:\.|$)", op)]
    if mode == "fp32_simt":
        valid = not tensor and "FFMA" in opcodes
    elif mode == "fp16_simt":
        valid = not tensor and any(op.startswith("HFMA2") for op in opcodes)
    elif mode == "tf32_tensor_fp32acc":
        valid = any(op.startswith("HMMA.") and ".TF32" in op for op in tensor)
    elif mode == "fp16_tensor_fp16acc":
        valid = any(op.startswith("HMMA.") and ".F16" in op for op in tensor)
    elif mode == "fp16_tensor_fp32acc":
        valid = any(op.startswith("HMMA.") and ".F32" in op and ".TF32" not in op for op in tensor)
    else:
        raise ValueError(f"unknown mode: {mode}")
    if not valid:
        raise ValueError(f"actual kernel instructions do not match {mode}: {sorted(opcodes)}")


def extract_function(text, name):
    """Select only the exact named function from cuobjdump output."""
    blocks = re.split(r"\s*Function\s*:\s*", text)[1:]
    for block in blocks:
        function, _, body = block.partition("\n")
        if function.strip() == name:
            # Following fatbin headings carry no instructions; retain only this function's instruction lines.
            return "\n".join(line for line in body.splitlines() if re.search(r"/\*[0-9a-f]+\*/", line))
    return ""


def disassemble(name):
    """Find the traced kernel in installed cuBLAS libraries, using Ampere cubin architectures only."""
    for library in ("libcublasLt.so.12", "libcublas.so.12"):
        path = Path("/lib/x86_64-linux-gnu") / library
        for arch in ("sm_80", "sm_86"):
            result = subprocess.run(["cuobjdump", "--dump-sass", "--gpu-architecture", arch,
                                     "--function", name, str(path)], capture_output=True, text=True, check=True)
            body = extract_function(result.stdout, name)
            if body:
                opcodes = set(re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]+)", body))
                return {"kernel": name, "library": str(path), "architecture": arch,
                        "opcodes": sorted(opcodes)}, body
    raise ValueError(f"traced kernel not found in installed cuBLAS cubins: {name}")


def verify(trace_path, output):
    """Verify all five paths at all three measured sizes and save concise disassembly evidence."""
    trace = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    expected = {(mode, size) for mode in MODES for size in SIZES}
    if {(row["mode"], row["n"]) for row in trace} != expected:
        raise ValueError("incomplete CUPTI mode/size coverage")
    output.mkdir(parents=True, exist_ok=True)
    functions = {}
    for name in sorted({row["kernel"] for row in trace}):
        entry, body = disassemble(name)
        filename = hashlib.sha256(name.encode()).hexdigest()[:16] + ".sass"
        (output / filename).write_text(body + "\n")
        entry["disassembly_file"] = filename
        functions[name] = entry
    paths = []
    for mode, size in sorted(expected):
        names = sorted({row["kernel"] for row in trace if (row["mode"], row["n"]) == (mode, size)})
        opcodes = {op for name in names for op in functions[name]["opcodes"]}
        check_opcodes(mode, opcodes)
        paths.append({"mode": mode, "n": size, "kernels": names, "verified": True})
    return {"source": "untimed CUPTI kernel activity + installed cuBLAS SASS", "paths": paths,
            "functions": list(functions.values())}


def main():
    """Persist instruction verification only after every requested path passes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = verify(args.trace, args.output)
    (args.output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Verified {len(result['paths'])} paths using {len(result['functions'])} actual kernel functions")


if __name__ == "__main__":
    main()
