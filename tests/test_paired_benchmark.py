"""Check actual instruction evidence and reject incorrectly labelled stock/unlocked states."""

import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT=Path(__file__).parents[1]


def module(name):
    """Import a standard-library-only workspace verifier without changing sys.path."""
    spec=importlib.util.spec_from_file_location(name,ROOT/"tools"/(name+".py"))
    loaded=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


paths=module("verify_tensor_paths")
paired=module("paired_results")


class TensorPathTests(unittest.TestCase):
    """Exercise real kernel records and deliberate instruction/precision mismatches."""

    def test_all_fifteen_real_instruction_paths(self):
        """Recheck actual GPU-selected cuBLAS functions, including all measured sizes."""
        evidence=ROOT/"results/2026-10-25/paired-five-paths-preparation/preflight-instructions/verification.json"
        data=json.loads(evidence.read_text())
        functions={entry["kernel"]:entry for entry in data["functions"]}
        self.assertEqual(len(data["paths"]),15)
        for path in data["paths"]:
            opcodes={op for name in path["kernels"] for op in functions[name]["opcodes"]}
            paths.check_opcodes(path["mode"],opcodes)

    def test_tensor_instruction_rejected_for_simt(self):
        """A SIMT label must not hide a Tensor Core instruction."""
        with self.assertRaisesRegex(ValueError,"do not match"):
            paths.check_opcodes("fp32_simt",{"FFMA","HMMA.16816.F32"})

    def test_tf32_requires_tf32_instruction(self):
        """FP16 Tensor instructions cannot establish a TF32 precision measurement."""
        with self.assertRaisesRegex(ValueError,"do not match"):
            paths.check_opcodes("tf32_tensor_fp32acc",{"HMMA.16816.F32"})

    def test_exact_function_name_required(self):
        """Do not reuse a similarly named cuBLAS function's instructions."""
        text="Function : wanted_suffix\n /*0010*/ HMMA.1688.F32.TF32 R0, R1, R2;\n"
        self.assertEqual(paths.extract_function(text,"wanted"),"")


class PairedStateTests(unittest.TestCase):
    """Prevent warm-reset retained unlocks from becoming fake stock measurements."""

    def test_unlocked_gpu_cannot_be_labelled_stock(self):
        """Require the independent half-rate getter before any baseline can be accepted."""
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/"sm-rates.json").write_text(json.dumps({"rates":{"fmla32":0}}))
            with self.assertRaisesRegex(ValueError,"speed state"):
                paired.validate_state(root,"stock")

    def test_half_rate_gpu_cannot_be_labelled_unlocked(self):
        """A selected boot option alone cannot prove a successful unlock."""
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/"sm-rates.json").write_text(json.dumps({"rates":{"fmla32":1}}))
            with self.assertRaisesRegex(ValueError,"speed state"):
                paired.validate_state(root,"unlocked")

    def test_seeded_matrix_inputs_cannot_change(self):
        """Reject input changes before a performance comparison can be reported."""
        row={"mode":"fp32_simt","n":2048,"m":2048,"k":2048,"seed":2,
             "input_pattern":"same-dyadic-v1","compute_type":"CUBLAS_COMPUTE_32F_PEDANTIC",
             "input_type":"FP32","output_type":"FP32","tensor_allowed":False}
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"gemm.jsonl"
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError,"matrix inputs changed"):
                paired.read_measurements(path)


class CompletedPairTests(unittest.TestCase):
    """Validate the completed hardware pair and rejection of mismatched comparison evidence."""

    def setUp(self):
        """Copy the actual pair so mutation tests cannot alter the original observations."""
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)/"pair"
        shutil.copytree(ROOT/"results/2026-10-25/paired-five-paths-results",self.root)
        self.addCleanup(self.temp.cleanup)

    def test_actual_pair_passes(self):
        """Require all 15 pairs, the same driver, 80 W and successful numerical checks."""
        result=paired.compare(self.root)
        self.assertEqual(len(result["measurements"]),15)
        self.assertEqual(result["driver"],"580.178.04")
        self.assertEqual(result["power_cap_W"],80)
        self.assertEqual(result["numerical_failures"],0)

    def test_changed_binary_or_library_rejected(self):
        """A different executable/library set cannot be accepted as an unlock-only comparison."""
        (self.root/"unlocked/SHA256SUMS").write_text("different binary hash\n")
        with self.assertRaisesRegex(ValueError,"metadata changed: SHA256SUMS"):
            paired.compare(self.root)

    def test_changed_driver_rejected(self):
        """The requested same-driver comparison must not combine different NVIDIA versions."""
        path=self.root/"unlocked/gpu.csv"
        path.write_text(path.read_text().replace("580.178.04","570.211.01"))
        with self.assertRaisesRegex(ValueError,"metadata changed: gpu.csv"):
            paired.compare(self.root)

    def test_full_rate_without_fresh_transaction_rejected(self):
        """Retained full rate alone does not establish the fresh unlock group requested here."""
        path=self.root/"unlocked/kernel-journal.txt"
        path.write_text(path.read_text().replace("gpu_unlock: unlocked","gpu_unlock: unlock failed (-114)"))
        with self.assertRaisesRegex(ValueError,"fresh unlock transaction missing"):
            paired.compare(self.root)

    def test_both_measured_groups_have_expected_actual_instructions(self):
        """Verify real SASS opcode evidence for every mode and size in both hardware states."""
        for state in ("stock","unlocked"):
            data=json.loads((self.root/state/"instructions/verification.json").read_text())
            functions={entry["kernel"]:entry for entry in data["functions"]}
            self.assertEqual(len(data["paths"]),15)
            for path in data["paths"]:
                opcodes={op for name in path["kernels"] for op in functions[name]["opcodes"]}
                paths.check_opcodes(path["mode"],opcodes)


if __name__ == "__main__":
    unittest.main()
