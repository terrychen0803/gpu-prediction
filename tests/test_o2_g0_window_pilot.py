"""CPU-only invariants for the standalone O2 G0 timing-window pilot."""
import unittest

from o2_sequence import g0_window_pilot as pilot
from o2_sequence.catalog import PROTOCOL as V3_PROTOCOL, load
from o2_sequence.runner import code_hash as v3_code_hash


class TestO2G0TimingWindowPilot(unittest.TestCase):
    def test_source_protocol_unchanged(self):
        self.assertNotEqual(pilot.PROTOCOL, V3_PROTOCOL)
        self.assertEqual(pilot.SOURCE_PROTOCOL, V3_PROTOCOL)
        self.assertEqual(v3_code_hash(), pilot.REQUIRED_SOURCE_HASH)

    def test_frozen_window_matrix(self):
        self.assertEqual(pilot.GROUPS, 5)
        self.assertEqual(pilot.REPEATS, 3)
        self.assertEqual(pilot.COUNTS, (8, 64, 256, 512))
        self.assertEqual(len(pilot.TEMPLATES) * len(pilot.COUNTS) * pilot.REPEATS, 60)

    def test_selected_templates_exist_and_preserve_fold_metadata(self):
        templates = {t["id"]: t for t in load(True)}
        self.assertEqual(len(templates), 60)
        self.assertEqual(len(set(pilot.TEMPLATES)), 5)
        for name in pilot.TEMPLATES:
            self.assertIn(name, templates)
            self.assertIn(templates[name]["topology"], ("independent", "dependent_tail"))
            self.assertIn(templates[name]["fold"], range(5))
            self.assertIn(len(templates[name]["steps"]), (16, 20))

    def test_no_trace_modes_in_pilot(self):
        import inspect
        source = inspect.getsource(pilot.worker)
        self.assertIn('request["mode"] != "G0"', source)
        self.assertIn("torch.cuda.synchronize()", source)
        self.assertIn("time.perf_counter_ns()", source)
        self.assertNotIn("nsys", source)


if __name__ == "__main__":
    unittest.main()
