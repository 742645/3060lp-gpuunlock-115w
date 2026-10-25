"""Reject power-limit and boot-order evidence that would mislabel a combined benchmark."""

import unittest

from tools.vix_evidence import GPU_UUID, verify_order, verify_runtime


class VixEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.smi = f'''GPU UUID : {GPU_UUID}
Driver Version : 615.71.09
VBIOS Version : 94.06.13.00.63
Current Power Limit : 115.00 W
Default Power Limit : 100.00 W
Max Power Limit : 115.00 W
'''
        self.rates = {'rates': {'fmla32': 0, 'fmla16': 0, 'imla4': 0}}
        self.journal = 'vbios_override: GPU 0000:00:10 selected: 484864 bytes\ngpu_unlock: already unlocked (-114)'
        self.cmdline = 'ro gpu_unlock.targets=0000:00:10.0 gpu_unlock.ga106m_experimental=1'

    def check(self, smi=None, journal=None):
        return verify_runtime(self.smi if smi is None else smi, self.rates,
                              self.journal if journal is None else journal, self.cmdline)

    def test_accept_full_rate_and_actual_115w(self):
        self.assertTrue(self.check()['combination_verified'])
        self.assertIn('retained', self.check()['unlock_provenance'])

    def test_max_115w_does_not_prove_current_115w(self):
        with self.assertRaises(ValueError):
            self.check(smi=self.smi.replace('Current Power Limit : 115.00', 'Current Power Limit : 100.00'))

    def test_stock_vbios_or_half_rates_fail(self):
        with self.assertRaises(ValueError):
            self.check(smi=self.smi.replace('94.06.13.00.63', '94.06.17.00.5F'))
        self.rates['rates']['fmla32'] = 1
        with self.assertRaises(ValueError):
            self.check()

    def test_fault_or_unstaged_override_fail(self):
        with self.assertRaises(ValueError):
            self.check(journal=self.journal + '\nNVRM: Xid 79')
        with self.assertRaises(ValueError):
            self.check(journal='gpu_unlock: unlocked')

    def test_unlock_precedes_both_nvidia_loaders(self):
        lines = ['/scripts/init-top/' + name for name in ('00-gpu-unlock-experiment', 'udev', 'vbios_override')]
        self.assertEqual(verify_order('\n'.join(lines))['00-gpu-unlock-experiment'], 0)
        with self.assertRaises(ValueError):
            verify_order('\n'.join(reversed(lines)))
        with self.assertRaises(ValueError):
            verify_order('\n'.join(lines[:2]))


if __name__ == '__main__':
    unittest.main()
