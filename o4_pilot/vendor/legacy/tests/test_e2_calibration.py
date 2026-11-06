import csv
import json
import tempfile
import unittest
from pathlib import Path

from e2_calibration.__main__ import HEADERS, cells
from e2_calibration.summarize import summarize


class E2SummaryTests(unittest.TestCase):
    def make_fixture(self, root):
        plan = {'status': 'completed', 'smoke': False, 'profile_id': 'synthetic-fixture',
                'protocol': 'e2_calibration_v3', 'groups_per_cell': 5, 'processes': 3,
                'target_ms_per_group': 100}
        (root/'plan.json').write_text(json.dumps(plan))
        for repeat in range(1, 4):
            run = root/f'run{repeat:02d}'; run.mkdir()
            env = {'status': 'completed', 'profile_id': 'synthetic-fixture', 'gpu_uuid': 'fixture-only',
                   'gpu_name': 'fixture-only', 'torch': 'fixture', 'cuda_runtime': 'fixture',
                   'source_sha256': 'fixture', 'code_sha256': 'fixture',
                   'software_hash': 'fixture', 'library_sha256': 'fixture', 'smoke': False,
                   'groups': 5, 'target_ms': 100, 'nvcc': 'fixture'}
            (run/'environment.json').write_text(json.dumps(env))
            with (run/'microbench_samples.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=HEADERS); writer.writeheader()
                for category, spec in cells():
                    cell = category + '/' + '/'.join(map(str, spec))
                    for group in range(1, 6):
                        row = dict.fromkeys(HEADERS, '')
                        row.update(profile_id='synthetic-fixture', gpu_uuid='fixture-only',
                                   category=category, cell=cell, process_repeat=repeat, group=group,
                                   repetitions=10, inner_n=1, gpu_event_us=100000,
                                   host_complete_us=100000, work_bytes=1000 if category=='memory' else '')
                        writer.writerow(row)

    def test_grouped_median_and_complete_matrix(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.make_fixture(root)
            result = summarize(root)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['cells'], 100)
            self.assertEqual(result['sample_rows'], 1500)
            self.assertEqual(result['anchors']['memory/copy/16777216/same_buffer']['median_us_per_operation'], 10000)

    def test_missing_worker_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.make_fixture(root)
            (root/'run03/microbench_samples.csv').unlink()
            with self.assertRaises(FileNotFoundError):
                summarize(root)


if __name__ == '__main__':
    unittest.main()
