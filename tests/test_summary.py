"""Ensure performance summaries fail closed when the underlying evidence is invalid."""

import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("summarize", ROOT / "tools/summarize.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class SummaryTests(unittest.TestCase):
    """Use the collected hardware evidence, then deliberately corrupt one property per test."""

    def setUp(self):
        """Copy immutable evidence so rejection tests never modify the real observations."""
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "raw"
        shutil.copytree(ROOT / "results/2026-10-25/raw", self.root)
        self.addCleanup(self.temp.cleanup)

    def test_real_evidence_passes(self):
        """Validate all measured shapes, unchanged cap, and warm retained state."""
        result = summary.summarize(self.root)
        self.assertEqual(result["power_cap_W"], 80)
        self.assertEqual(result["numerical_failures"], 0)
        self.assertEqual(len(result["comparison"]), 6)

    def test_changed_gpu_rejected(self):
        """Do not attribute a different GPU's speed to the unlock."""
        (self.root / "unlocked/gpu.csv").write_text("other GPU")
        with self.assertRaisesRegex(ValueError, "metadata changed"):
            summary.summarize(self.root)

    def test_changed_power_rejected(self):
        """Do not treat a raised power cap as a matched comparison."""
        path = self.root / "unlocked/nvidia-smi-before.txt"
        path.write_text(path.read_text().replace("80.00 W", "95.00 W"))
        with self.assertRaisesRegex(ValueError, "power caps changed"):
            summary.summarize(self.root)

    def test_output_mismatch_rejected(self):
        """A fast but arithmetically incorrect GEMM must not produce a result."""
        path = self.root / "unlocked/gemm.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["failures"] = 1
        path.write_text("\n".join(json.dumps(r) for r in rows))
        with self.assertRaisesRegex(ValueError, "numerical failures"):
            summary.summarize(self.root)

    def test_driver_fault_rejected(self):
        """Retained full-rate fields do not excuse a driver fault."""
        path = self.root / "repeat/kernel-journal.txt"
        path.write_text(path.read_text() + "\nNVRM: Xid 79\n")
        with self.assertRaisesRegex(ValueError, "driver fault"):
            summary.summarize(self.root)

    def test_ealready_alone_does_not_prove_retained_unlock(self):
        """Require selectors and independent full-rate fields, beyond a driver's skip status."""
        path = self.root / "repeat/kernel-journal.txt"
        path.write_text(path.read_text().replace("ss0=0x88888888", "ss0=0x0"))
        with self.assertRaisesRegex(ValueError, "selector evidence missing"):
            summary.summarize(self.root)


class StrictFp16RunTests(unittest.TestCase):
    """Require all nine measured cases and the strict FP16 calculation/output contract."""

    def setUp(self):
        """Copy the new hardware evidence for isolated corruption tests."""
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "strict16"
        shutil.copytree(ROOT / "results/2026-10-25/strict-fp16-20261025T051633Z", self.root)
        self.addCleanup(self.temp.cleanup)

    def test_real_nine_case_run_passes(self):
        """Verify all sizes, three strict CPU references and the unchanged 80 W cap."""
        result = summary.summarize_single_run(self.root)
        self.assertEqual(len(result["measurements"]), 9)
        self.assertEqual(result["strict16_seeded_cases"], 3)
        self.assertEqual(result["power_cap_W"], 80)

    def test_fp32_output_cannot_be_labelled_strict_fp16(self):
        """Reject a mixed-precision result being presented as strict FP16."""
        path = self.root / "gemm.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if row["mode"] == "FP16_strict":
                row["output_type"] = "FP32"
                break
        path.write_text("\n".join(json.dumps(row) for row in rows))
        with self.assertRaisesRegex(ValueError, "strict FP16 type/compute"):
            summary.summarize_single_run(self.root)

    def test_default_compute_cannot_be_labelled_pedantic(self):
        """Reject tensor-enabled default compute being presented as the pedantic mode."""
        path = self.root / "gemm.jsonl"
        path.write_text(path.read_text().replace("CUBLAS_COMPUTE_16F_PEDANTIC", "CUBLAS_COMPUTE_16F"))
        with self.assertRaisesRegex(ValueError, "strict FP16 type/compute"):
            summary.summarize_single_run(self.root)

    def test_strict_seeded_error_rejected(self):
        """Do not publish performance when the new FP16 CPU-reference check fails."""
        path = self.root / "strict16-numerical.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["failures"] = 1
        path.write_text("\n".join(json.dumps(row) for row in rows))
        with self.assertRaisesRegex(ValueError, "seeded validation failed"):
            summary.summarize_single_run(self.root)


if __name__ == "__main__":
    unittest.main()
