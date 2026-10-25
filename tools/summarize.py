"""Validate and compare the same-card stock/unlocked benchmark evidence."""

import argparse
import csv
import io
import json
import math
from pathlib import Path
import re
import statistics

EXPECTED = {(mode, size) for mode in ("FP32_strict", "FP16_FP32acc") for size in (2048, 4096, 8192)}
EXPECTED_WITH_STRICT16 = EXPECTED | {("FP16_strict", size) for size in (2048, 4096, 8192)}


def read_gemm(path, expected=EXPECTED):
    """Reject missing, duplicate, incorrect or invalidly timed GEMM measurements."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = {}
    for row in rows:
        key = row["mode"], row["n"]
        times = row["times_ms"]
        if key in result or key not in expected or row["failures"] != 0:
            raise ValueError("unexpected/duplicate GEMM or numerical failures")
        if row["mode"] == "FP16_strict":
            strict_fields = {"compute_type": "CUBLAS_COMPUTE_16F_PEDANTIC", "input_type": "FP16",
                             "scale_type": "FP16", "output_type": "FP16", "tensor_cores_allowed": False}
            if any(row.get(name) != value for name, value in strict_fields.items()):
                raise ValueError("strict FP16 type/compute contract missing")
        if row["cells_checked"] != row["n"] ** 2 or row["reps"] != 5 or len(times) != 5:
            raise ValueError("incomplete output/timing coverage")
        if row["iterations"] != 20 or row["warmups"] != 10:
            raise ValueError("unmatched launch policy")
        if not all(math.isfinite(t) and t > 0 for t in times):
            raise ValueError("invalid GEMM timing")
        derived = statistics.median(2 * row["n"] ** 3 / (t * 1e9) for t in times)
        if not math.isclose(derived, row["median_TF_s"], rel_tol=1e-5):
            raise ValueError("reported throughput disagrees with timing")
        result[key] = row
    if set(result) != expected:
        raise ValueError("incomplete GEMM modes/shapes")
    return result


def power_limit(directory):
    """Read the actual GPU cap from the detailed report; the laptop CSV field returns N/A."""
    text = (directory / "nvidia-smi-before.txt").read_text()
    match = re.search(r"^\s*Current Power Limit\s*:\s*([\d.]+) W", text, re.M)
    if not match:
        raise ValueError("GPU power cap missing")
    return float(match.group(1))


def numeric_check(path):
    """Require all six seeded FP16/BF16 CPU-reference cases to pass."""
    text = path.read_text()
    results = re.findall(r"failures=(\d+), max_abs=([\d.eE+-]+)", text)
    if len(results) != 6 or any(int(n) or float(error) > 1e-3 for n, error in results):
        raise ValueError("seeded numerical validation failed/incomplete")


def telemetry(path):
    """Summarize observed samples, without claiming they are electrical or thermal maxima."""
    rows = list(csv.reader(io.StringIO(path.read_text())))[1:]
    def values(index):
        return [float(row[index].strip().split()[0]) for row in rows]
    return {"samples": len(rows), "observed_max_temperature_C": max(values(4)),
            "observed_max_power_W": max(values(2)), "observed_max_sm_MHz": max(values(5))}


def summarize(root):
    """Verify matching identity/kernel/cap, successful arithmetic and full-rate retained boot."""
    stock, unlocked, repeat = (root / name for name in ("stock", "unlocked", "repeat"))
    for name in ("kernel.txt", "gpu.csv"):
        if (stock / name).read_text() != (unlocked / name).read_text():
            raise ValueError(f"comparison metadata changed: {name}")
    cap = power_limit(stock)
    if cap != power_limit(unlocked):
        raise ValueError("power caps changed")
    for folder in (stock, unlocked, repeat):
        numeric_check(folder / "numerical.txt")
    if json.loads((stock / "sm-rates.json").read_text())["rates"]["fmla32"] != 1:
        raise ValueError("stock rate is not the expected half rate")
    for folder in (unlocked, repeat):
        if json.loads((folder / "sm-rates.json").read_text())["rates"]["fmla32"] != 0:
            raise ValueError("full rate missing")
        journal = (folder / "kernel-journal.txt").read_text()
        if re.search(r"NVRM: Xid|GPU has fallen off|RmInitAdapter failed", journal):
            raise ValueError("driver fault detected")
    repeat_journal = (repeat / "kernel-journal.txt").read_text()
    boot_state = (repeat / "boot-state.txt").read_text().strip()
    if (repeat / "kernel.txt").read_text() != (stock / "kernel.txt").read_text():
        raise ValueError("repeat kernel changed")
    if boot_state == "retained-unlock-no-new-transaction":
        required = ("pci 0000:00:10.0: entry: plm=0xffffffff ss0=0x88888888 ss1=0x8",
                    "pci 0000:00:10.0: gpu_unlock: unlock failed (-114)")
        if not all(text in repeat_journal for text in required):
            raise ValueError("retained selector evidence missing")
    elif boot_state != "fresh-transaction" or "pci 0000:00:10.0: gpu_unlock: unlocked" not in repeat_journal:
        raise ValueError("repeat boot evidence missing")
    if "pci 0000:00:10.0: gpu_unlock: unlocked" not in (unlocked / "kernel-journal.txt").read_text():
        raise ValueError("initial transaction success missing")
    before, after = read_gemm(stock / "gemm.jsonl"), read_gemm(unlocked / "gemm.jsonl")
    comparison = []
    for mode, n in sorted(EXPECTED):
        a, b = before[(mode, n)]["median_TF_s"], after[(mode, n)]["median_TF_s"]
        comparison.append({"mode": mode, "n": n, "stock_TFLOPS": a,
                           "unlocked_TFLOPS": b, "gain_percent": (b/a - 1) * 100})
    identity = next(csv.DictReader(io.StringIO((stock / "gpu.csv").read_text()), skipinitialspace=True))
    return {"kernel": (stock / "kernel.txt").read_text().strip(), "power_cap_W": cap,
            "driver": identity["driver_version"], "comparison": comparison,
            "telemetry": {name: telemetry(root / name / "telemetry.csv") for name in ("stock", "unlocked")},
            "warm_boot_state": boot_state,
            "numerical_failures": 0, "Xid_errors_in_unlocked_and_repeat_journals": 0}


def summarize_single_run(root):
    """Validate the expanded nine-case unlocked run, including genuine strict FP16 metadata."""
    measurements = read_gemm(root / "gemm.jsonl", EXPECTED_WITH_STRICT16)
    numeric_check(root / "numerical.txt")
    strict = [json.loads(line) for line in (root / "strict16-numerical.jsonl").read_text().splitlines()]
    expected_shapes = {(256, 256, 256, 1), (64, 96, 128, 2), (127, 65, 95, 3)}
    shapes = {(row["m"], row["n"], row["k"], row["seed"]) for row in strict}
    if len(strict) != 3 or shapes != expected_shapes or any(
            row["mode"] != "FP16_strict" or row["failures"] or row["cells_checked"] != row["m"]*row["n"]
            for row in strict):
        raise ValueError("strict FP16 seeded validation failed/incomplete")
    if json.loads((root / "sm-rates.json").read_text())["rates"]["fmla32"] != 0:
        raise ValueError("full rate missing")
    if re.search(r"NVRM: Xid|GPU has fallen off|RmInitAdapter failed", (root / "kernel-journal.txt").read_text()):
        raise ValueError("driver fault detected")
    if "STRICT_FP16_COLLECTION_PASSED" not in (root / "collection.log").read_text():
        raise ValueError("run did not complete")
    identity = next(csv.DictReader(io.StringIO((root / "gpu.csv").read_text()), skipinitialspace=True))
    return {"state": "unlocked", "kernel": (root / "kernel.txt").read_text().strip(),
            "driver": identity["driver_version"], "power_cap_W": power_limit(root),
            "strict_fp16_compute": "CUBLAS_COMPUTE_16F_PEDANTIC (Tensor Cores disabled)",
            "measurements": [measurements[key] for key in sorted(measurements)],
            "telemetry": telemetry(root / "telemetry.csv"), "numerical_failures": 0,
            "strict16_seeded_cases": len(strict), "Xid_errors_in_current_boot_journal": 0,
            "comparison_scope": "single new unlocked run; no new stock measurement"}


def main():
    """Write a reviewable JSON comparison only after evidence validation succeeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--single-run", action="store_true", help="validate the expanded strict-FP16 run")
    args = parser.parse_args()
    result = summarize_single_run(args.root) if args.single_run else summarize(args.root)
    data = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(data, encoding="utf-8")
    print(data)


if __name__ == "__main__":
    main()
