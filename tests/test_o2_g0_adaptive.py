"""CPU-only assertions for the O2 host timing adaptive protocol."""
import math
import unittest

from o2_sequence import g0_adaptive as a
from o2_sequence.catalog import PROTOCOL as V3_PROTOCOL, load
from o2_sequence.runner import code_hash


class TestO2G0Adaptive(unittest.TestCase):
    def test_provenance(self):
        self.assertEqual(a.V3_SHA, code_hash())
        self.assertEqual(a.V3_PROTOCOL, V3_PROTOCOL)
        self.assertNotEqual(a.PROTOCOL, V3_PROTOCOL)

    def test_fixed_policy(self):
        self.assertEqual(a.BOOTSTRAP_COUNT, 256)
        self.assertEqual(a.BOOTSTRAP_REPEATS, 2)
        self.assertEqual(a.FORMAL_REPEATS, 3)
        self.assertEqual(a.GROUPS, 5)
        self.assertEqual(a.TARGET_US, 125000)
        self.assertEqual(a.COUNT_QUANTUM, 64)
        self.assertEqual(a.MIN_COUNT, 64)
        self.assertEqual(a.MAX_COUNT, 4096)

    def test_full_catalog(self):
        templates, configs = a.template_config()
        self.assertEqual(len(templates), 60)
        self.assertEqual(len(set(x["id"] for x in templates)), 60)
        self.assertEqual(sum(x["topology"] == "dependent_tail" for x in templates), 30)
        self.assertGreater(len(configs), 0)

    def test_bootstrap_count_formula(self):
        for median_us in (100, 200, 250, 350, 1000):
            n = max(a.MIN_COUNT, math.ceil(
                (a.TARGET_US / median_us) / a.COUNT_QUANTUM)*a.COUNT_QUANTUM
            )
            self.assertEqual(n % a.COUNT_QUANTUM, 0)
            self.assertLessEqual(n, a.MAX_COUNT)
            self.assertGreaterEqual(n * median_us, a.TARGET_US)


if __name__ == "__main__":
    unittest.main()
