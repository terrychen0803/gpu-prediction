"""Offline checks for O2 split isolation and conservative R/P labels."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from o2_sequence.alignment import review_pair
from o2_sequence.catalog import load


class O2CatalogTests(unittest.TestCase):
    def test_formal_templates_remain_within_one_fold(self):
        templates = load(True)
        self.assertEqual(len(templates), 60)
        self.assertEqual({t["fold"] for t in templates}, set(range(5)))
        for fold in range(5):
            self.assertEqual(sum(t["fold"] == fold for t in templates), 12)
        self.assertTrue(any(t["topology"] == "dependent_tail" for t in templates))

    def test_pilot_separate_from_formal_catalog(self):
        self.assertTrue(all(t["fold"] is None for t in load(False)))


class O2AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.p = Path(self.temp.name) / "P"
        self.r = Path(self.temp.name) / "R"
        self.p.mkdir()
        self.r.mkdir()
        template = {"id": "fixture", "steps": [{"config_id": "gemm_fixture"}]}
        for folder in (self.p, self.r):
            (folder / "request.json").write_text(json.dumps({"template": template, "repeat": 1,
                "groups": 1, "sequences_per_group": 1}))
            (folder / "groups.csv").write_text("host_us_per_sequence\n100\n")
            (folder / "trace_audit.json").write_text(json.dumps({"warnings": []}))
        event = {"event_id": "kernel_00000000", "kernel_name": "kernel_a",
                 "grid_xyz": [1, 1, 1], "block_xyz": [32, 1, 1],
                 "dynamic_shared_memory_bytes": 0}
        (self.p / "events_allowlist.jsonl").write_text(json.dumps(event)+"\n")
        reference = {"scope": "o2.op/g01/i000/s00", "group": 1, "sequence": 0, "step": 0,
                     "config_id": "gemm_fixture", "op_type": "gemm",
                     "kernel_event_indices": [0],
                     "kernel_signatures": [["kernel_a", [1,1,1], [32,1,1], 0]]}
        (self.r / "reference_operations.jsonl").write_text(json.dumps(reference)+"\n")

    def test_exact_ordered_signature_can_join(self):
        result = review_pair(self.p, self.r, Path(self.temp.name)/"review"/"labels.jsonl")
        self.assertEqual(result["verified_step_count"], 1)
        labels = json.loads((Path(self.temp.name)/"review"/"labels.jsonl").read_text())
        self.assertEqual(labels["label_provenance"], "reference_join_unique")
        self.assertEqual(labels["p_event_ids"], ["kernel_00000000"])

    def test_warning_or_signature_change_prevents_verified_label(self):
        (self.r / "trace_audit.json").write_text(json.dumps({"warnings": ["NVTX incomplete"]}))
        result = review_pair(self.p, self.r, Path(self.temp.name)/"warning"/"labels.jsonl")
        self.assertEqual(result["verified_step_count"], 0)
        self.assertTrue(result["exact_ordered_signature"])
        event_path = self.p / "events_allowlist.jsonl"
        event = json.loads(event_path.read_text())
        event["grid_xyz"] = [2, 1, 1]
        event_path.write_text(json.dumps(event)+"\n")
        result = review_pair(self.p, self.r, Path(self.temp.name)/"changed"/"labels.jsonl")
        self.assertFalse(result["exact_ordered_signature"])
        self.assertEqual(result["verified_step_count"], 0)


if __name__ == "__main__":
    unittest.main()
