"""Revalidate the recorded 115 W combined experiment and its actual arithmetic instructions."""

import json
from pathlib import Path
import re
import unittest

from tools.paired_results import CONTRACTS, EXPECTED, read_measurements, read_rows
from tools.verify_tensor_paths import check_opcodes
from tools.vix_evidence import PAYLOAD_SHA256, verify_runtime
from tools.summarize import telemetry

ROOT = Path(__file__).parents[1]
DATA = ROOT/'results/2026-10-25/vix-combination/benchmark115w'


class CompletedVixHardwareTests(unittest.TestCase):
    def test_summary_matches_raw_observations_and_retains_limits(self):
        """Prevent the report from turning a warm boot or a changed-driver comparison into stronger proof."""
        summary = json.loads((ROOT/'results/2026-10-25/vix-combination-summary.json').read_text())
        self.assertFalse(summary['cold_boot_verified'])
        self.assertTrue(summary['historical_comparison']['driver_and_vbios_changed'])
        self.assertEqual(summary['telemetry'], telemetry(DATA/'telemetry.csv'))
        raw = read_measurements(DATA/'gemm.jsonl')
        self.assertEqual({(r['mode'], r['n']) for r in summary['measurements']}, EXPECTED)
        for row in summary['measurements']:
            self.assertEqual(row['median_TFLOPS'], raw[(row['mode'], row['n'])]['median_TF_s'])
        reboot = DATA.parent/'post-default-reboot'
        verify_runtime((reboot/'nvidia-smi.txt').read_text(), json.loads((reboot/'sm-rates.json').read_text()),
                       (reboot/'kernel-journal.txt').read_text(), (reboot/'cmdline.txt').read_text())

    def test_before_and_after_are_115w_and_full_rate(self):
        """Use live-recorded power and hardware rates rather than the edited ROM's offline values."""
        for smi, rates in (('nvidia-smi.txt', 'sm-rates.json'), ('nvidia-smi-after.txt', 'sm-rates-after.json')):
            result = verify_runtime((DATA/smi).read_text(), json.loads((DATA/rates).read_text()),
                                    (DATA/'kernel-journal.txt').read_text(), (DATA/'cmdline.txt').read_text())
            self.assertTrue(result['combination_verified'])
        status = json.loads((DATA/'vix-status.json').read_text())
        self.assertTrue(status['runtime']['activation_matches'])
        self.assertTrue(all(item['matches'] for item in status['runtime']['loaded_module'].values()))
        proof = json.loads((DATA/'vix-image-proof.json').read_text())
        self.assertEqual(proof['payload_sha256'], PAYLOAD_SHA256)
        self.assertIn('VIX_115W_COMBINATION_COLLECTION_PASSED', (DATA/'collection.log').read_text())

    def test_fifteen_measurements_use_frozen_binary_and_inputs(self):
        """Require the same executable/libraries and precision contracts as the prior five-path experiment."""
        self.assertEqual(set(read_measurements(DATA/'gemm.jsonl')), EXPECTED)
        old = ROOT/'results/2026-10-25/paired-five-paths-results/unlocked/SHA256SUMS'
        self.assertEqual((DATA/'SHA256SUMS').read_text(), old.read_text())
        checks = read_rows(DATA/'numerical.jsonl')
        self.assertEqual(len(checks), 15)
        self.assertEqual({(row['mode'], row['seed']) for row in checks},
                         {(mode, seed) for mode in CONTRACTS for seed in (1, 2, 3)})
        self.assertFalse(any(row['failures'] for row in checks))

    def test_saved_sass_confirms_every_measured_path(self):
        """Reparse actual SASS bodies independently of the saved verification booleans."""
        data = json.loads((DATA/'instructions/verification.json').read_text())
        self.assertEqual({(row['mode'], row['n']) for row in data['paths']}, EXPECTED)
        functions = {}
        for row in data['functions']:
            body = (DATA/'instructions'/row['disassembly_file']).read_text()
            functions[row['kernel']] = set(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]+)', body))
        for row in data['paths']:
            opcodes = {op for kernel in row['kernels'] for op in functions[kernel]}
            check_opcodes(row['mode'], opcodes)


if __name__ == '__main__':
    unittest.main()
