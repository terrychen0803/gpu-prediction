"""Static checks for the preregistered O2 G0 window-size ablation."""
import unittest

from o2_sequence import g0_window_ablation as ablation
from o2_sequence import g0_window_pilot as pilot
from o2_sequence.catalog import load


class TestO2WindowAblation(unittest.TestCase):
    def test_matrix(self):
        self.assertEqual(ablation.COUNTS, (8, 256, 512))
        self.assertEqual(ablation.REPEATS, 3)
        self.assertEqual(ablation.GROUPS, 5)
        self.assertEqual(60 * len(ablation.COUNTS) * ablation.REPEATS, 540)

    def test_protocol_provenance(self):
        self.assertNotEqual(ablation.DESIGN_PROTOCOL, pilot.PROTOCOL)
        self.assertEqual(set(ablation.COUNTS) - set(pilot.COUNTS), set())
        self.assertEqual(ablation.v3_hash(), pilot.REQUIRED_SOURCE_HASH)

    def test_templates(self):
        templates = load(True)
        self.assertEqual(len(templates), 60)
        self.assertEqual(len(set(x["id"] for x in templates)), 60)
        self.assertEqual(
            sum(x["topology"] == "dependent_tail" for x in templates), 30
        )


if __name__ == "__main__":
    unittest.main()
