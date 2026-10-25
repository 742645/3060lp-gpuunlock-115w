"""Validate clean stock/unlocked same-input GEMM evidence before calculating a comparison."""

import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics

CONTRACTS = {
    "fp32_simt": ("CUBLAS_COMPUTE_32F_PEDANTIC", "FP32", "FP32", False),
    "fp16_simt": ("CUBLAS_COMPUTE_16F_PEDANTIC", "FP16", "FP16", False),
    "tf32_tensor_fp32acc": ("CUBLAS_COMPUTE_32F_FAST_TF32", "FP32", "FP32", True),
    "fp16_tensor_fp16acc": ("CUBLAS_COMPUTE_16F", "FP16", "FP16", True),
    "fp16_tensor_fp32acc": ("CUBLAS_COMPUTE_32F", "FP16", "FP32", True),
}
SIZES = (2048, 4096, 8192)
EXPECTED = {(mode, n) for mode in CONTRACTS for n in SIZES}


def read_rows(path):
    """Read newline-delimited JSON objects, preserving errors rather than silently skipping them."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_measurements(path):
    """Check precision, identical mathematical inputs, arithmetic and measured throughput."""
    result = {}
    for row in read_rows(path):
        key = row["mode"], row["n"]
        if key not in EXPECTED or key in result:
            raise ValueError("unexpected/duplicate measurement")
        contract = tuple(row[field] for field in ("compute_type", "input_type", "output_type", "tensor_allowed"))
        if contract != CONTRACTS[row["mode"]]:
            raise ValueError("precision/Tensor contract changed")
        if (row["m"], row["k"], row["seed"], row["input_pattern"]) != (row["n"], row["n"], 1, "same-dyadic-v1"):
            raise ValueError("matrix inputs changed")
        if row["failures"] or row["cells_checked"] != row["m"]*row["n"]:
            raise ValueError("incorrect/incomplete numerical output")
        times = row["times_ms"]
        if (row["warmups"], row["iterations"], row["reps"], len(times)) != (10, 20, 5, 5):
            raise ValueError("launch policy changed")
        if any(not math.isfinite(t) or t <= 0 for t in times):
            raise ValueError("invalid timing")
        rate = statistics.median(2*row["n"]**3/(t*1e9) for t in times)
        if not math.isclose(rate, row["median_TF_s"], rel_tol=1e-5):
            raise ValueError("throughput disagrees with timing")
        result[key] = row
    if set(result) != EXPECTED:
        raise ValueError("incomplete measurement coverage")
    return result


def validate_state(root, state):
    """Require independent speed readout, real kernel instruction evidence and no driver faults."""
    rate = json.loads((root/"sm-rates.json").read_text())["rates"]["fmla32"]
    if rate != (1 if state == "stock" else 0):
        raise ValueError("GPU speed state does not match measurement label")
    cmdline = (root/"cmdline.txt").read_text()
    journal = (root/"kernel-journal.txt").read_text()
    success = "pci 0000:00:10.0: gpu_unlock: unlocked"
    if state == "stock":
        if "gpu_unlock.targets=" in cmdline or success in journal:
            raise ValueError("stock boot contains an unlock request/transaction")
    elif "gpu_unlock.targets=0000:00:10.0" not in cmdline or success not in journal:
        raise ValueError("fresh unlock transaction missing")
    if re.search(r"NVRM: Xid|GPU has fallen off|RmInitAdapter failed", journal):
        raise ValueError("driver fault detected")
    if (root/"processes.txt").read_text().strip():
        raise ValueError("other GPU workloads present")
    numerical = read_rows(root/"numerical.jsonl")
    expected_checks = {(mode, seed) for mode in CONTRACTS for seed in (1, 2, 3)}
    if len(numerical) != 15 or {(row["mode"], row["seed"]) for row in numerical} != expected_checks or any(row["failures"] for row in numerical):
        raise ValueError("seeded numerical check failed/incomplete")
    instructions = json.loads((root/"instructions/verification.json").read_text())
    if {(row["mode"], row["n"]) for row in instructions["paths"]} != EXPECTED or not all(row["verified"] for row in instructions["paths"]):
        raise ValueError("actual kernel instruction coverage incomplete")
    if f"PAIRED_COLLECTION_PASSED {state}" not in (root/"collection.log").read_text():
        raise ValueError("collection did not complete")
    caps = []
    for name in ("nvidia-smi-before.txt", "nvidia-smi-after.txt"):
        match = re.search(r"^\s*Current Power Limit\s*:\s*([\d.]+) W", (root/name).read_text(), re.M)
        if not match:
            raise ValueError("power cap missing")
        caps.append(float(match.group(1)))
    if caps[0] != caps[1]:
        raise ValueError("power cap changed during measurement")
    return read_measurements(root/"gemm.jsonl"), caps[0]


def compare(root):
    """Compute gains only when GPU, driver, kernel, binaries, libraries and power cap match."""
    stock, unlocked = root/"stock", root/"unlocked"
    for name in ("gpu.csv", "kernel.txt", "SHA256SUMS"):
        if (stock/name).read_text() != (unlocked/name).read_text():
            raise ValueError(f"comparison metadata changed: {name}")
    before, cap = validate_state(stock, "stock")
    after, after_cap = validate_state(unlocked, "unlocked")
    if cap != after_cap:
        raise ValueError("power caps differ")
    rows = []
    for key in sorted(EXPECTED):
        a,b = before[key]["median_TF_s"],after[key]["median_TF_s"]
        rows.append({"mode":key[0],"n":key[1],"stock_TFLOPS":a,"unlocked_TFLOPS":b,"gain_percent":100*(b/a-1)})
    identity = next(csv.DictReader((stock/"gpu.csv").read_text().splitlines(), skipinitialspace=True))
    return {"driver":identity["driver_version"],"power_cap_W":cap,"measurements":rows,
            "numerical_failures":0,"actual_kernel_paths_verified":True,
            "tensor_fp32_precision":"TF32 multiplication with FP32 accumulation"}


def main():
    """Save a validated paired report; absence of either hardware state is an error."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    text=json.dumps(compare(args.root),indent=2)+"\n"
    if args.output: args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
