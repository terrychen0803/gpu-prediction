"""Offline contract tests for the seven-A 40x3 engineering collector."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from o4_research.audit import ScopeLookup, contains
from o4_research.common import read, write
from o4_research.formal_protocol import (
    PROTOCOL, expected_scopes, process_spread, validate_protocol, warmup_stable,
)
from o4_research.run_formal import code_hashes, plan, summarize


class FormalProtocolTests(unittest.TestCase):
    def test_plan_all_seven_is_exactly_63_distinct_cells(self):
        names = ["RN-A", "MV-A", "SW-A", "UN-A", "YO-A", "QP-A", "QD-A"]
        jobs = plan(names)
        self.assertEqual(len(jobs), 63)
        self.assertEqual(
            len({(j["workload"], j["mode"], j["repeat"]) for j in jobs}), 63
        )
        self.assertEqual(jobs, plan(names))
        self.assertEqual({j["repeat"] for j in jobs}, {1, 2, 3})

    def test_plan_rejects_duplicate_and_substitute(self):
        for jobs in (["RN-A", "RN-A"], ["RN-B"], []):
            with self.assertRaises(ValueError):
                plan(jobs)

    def test_protocol_requires_explicit_new_version(self):
        self.assertEqual(PROTOCOL["measured_iterations"], 40)
        self.assertEqual(PROTOCOL["transition_iterations"], 5)
        self.assertEqual(PROTOCOL["independent_process_repeats"], 3)
        self.assertEqual(PROTOCOL["warmup_min"], 50)
        self.assertEqual(PROTOCOL["warmup_max"], 500)
        self.assertEqual(len(expected_scopes()), 45)
        self.assertEqual(validate_protocol(dict(PROTOCOL)), PROTOCOL)
        other = dict(PROTOCOL, measured_iterations=3)
        with self.assertRaises(ValueError):
            validate_protocol(other)
        self.assertFalse(PROTOCOL["training_eligible"])
        self.assertEqual(PROTOCOL["strict_verified"], 0)

    def test_warmup_and_process_stability(self):
        self.assertFalse(warmup_stable([20.0] * 59))
        self.assertTrue(warmup_stable([20.0] * 60))
        self.assertAlmostEqual(process_spread([10, 10.2, 9.9]), 0.03)
        self.assertIsNone(process_spread([10]))
        self.assertTrue(all(len(hash_value) == 64 for hash_value in code_hashes().values()))


class ScopeLookupTests(unittest.TestCase):
    def test_index_matches_linear_single_owner(self):
        scopes = [
            {"globalTid": 7, "start": 100, "end": 200, "name": "a"},
            {"globalTid": 7, "start": 250, "end": 300, "name": "b"},
            {"globalTid": 8, "start": 120, "end": 180, "name": "c"},
        ]
        index = ScopeLookup(scopes)
        for tid in (7, 8, 9):
            for start, end in ((90, 95), (100, 110), (150, 170),
                               (190, 201), (220, 230), (250, 260), (290, 310)):
                api = {"globalTid": tid, "start": start, "end": end}
                self.assertEqual(contains(api, scopes), contains(api, index))
    def test_rejects_nested_or_overlapping_unknown_scope_ownership(self):
        scopes = [
            {"globalTid": 7, "start": 100, "end": 300},
            {"globalTid": 7, "start": 120, "end": 200},
        ]
        with self.assertRaisesRegex(ValueError, "Overlapping/nested"):
            ScopeLookup(scopes)


class SummarizeTests(unittest.TestCase):
    def fixtures(self, root, workload="RN-A", slower=False):
        jobs = plan([workload])
        for j in jobs:
            j["status"] = "completed"
            j["folder"] = f'{j["workload"]}_{j["mode"]}_r{j["repeat"]}'
            folder = root / j["folder"]
            folder.mkdir()
            write(folder / "quality.json", {
                "status": "completed", "correctness": {"passed": True}, "warmup": {"passed": True}
            })
            speed = 13 if slower and j["mode"] == "R" and j["repeat"] == 1 else 10
            with (folder / "iterations.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["stage", "iteration", "elapsed_us"])
                writer.writeheader()
                for iteration in range(40):
                    writer.writerow({"stage": "measure", "iteration": iteration, "elapsed_us": speed})
        return jobs

    def test_complete_with_warning_is_still_not_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = self.fixtures(root)
            fake_pair = [
                {"scope": f"o4.iter/{stage}/{index:06d}",
                 "ordered_signature_exact": True}
                for stage, count in (("transition", 5), ("measure", 40))
                for index in range(count)
            ]
            with patch("o4_research.audit.pair", return_value=fake_pair):
                summary = summarize(root, jobs, ["RN-A"])
            self.assertEqual(summary["status"], "completed_engineering")
            self.assertEqual(summary["completed_cells"], 9)
            self.assertEqual(summary["measured_valid_iterations"], 360)
            self.assertEqual(summary["P_R_pairs_exact"], 3)
            self.assertTrue(summary["all_three_process_spreads_pass"])
            self.assertFalse(summary["strict_reference_verified"])
            self.assertFalse(summary["training_eligible"])
            self.assertEqual(len(read(root / "aggregate.json")), 3)

    def test_spread_failure_is_not_silently_filtered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = self.fixtures(root, slower=True)
            fake_pair = [
                {"scope": f"o4.iter/{stage}/{index:06d}",
                 "ordered_signature_exact": True}
                for stage, count in (("transition", 5), ("measure", 40))
                for index in range(count)
            ]
            with patch("o4_research.audit.pair", return_value=fake_pair):
                summary = summarize(root, jobs, ["RN-A"])
            self.assertEqual(summary["status"], "completed_engineering")
            self.assertFalse(summary["all_three_process_spreads_pass"])
            self.assertEqual(summary["stability_failed_groups"][0]["mode"], "R")
            self.assertFalse(summary["strict_reference_verified"])
            medians = read(root / "aggregate.json")
            self.assertEqual(next(x for x in medians if x["mode"] == "R")["medians_us"],
                             [13, 10, 10])


if __name__ == "__main__":
    unittest.main()
