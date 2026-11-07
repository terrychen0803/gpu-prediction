import importlib.util
import tempfile
import unittest
from pathlib import Path

from inference_pilot.common import read_json, write_csv, write_json
from inference_pilot.trace import parse_trace
from test_inference_pilot import fake_result, fixture


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "optional matplotlib not installed")
class PlotTests(unittest.TestCase):
    def test_all_plot_types_from_synthetic_data(self):
        from inference_pilot.plots import plot
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for label in ("source", "target"):
                for repeat in range(3):
                    run = root/"runs"/f"{label}_{repeat}"
                    fake_result(run, label, 1, 100 if label=="source" else 80, repeat)
                    write_csv(run/"warmup.csv", [{"latency_us": 200}, {"latency_us": 150}])
            run = root/"runs"/"p1"
            fixture(run)
            fake_result(run, "source", 1, 110, 1)
            manifest = read_json(run/"manifest.json")
            manifest["mode"] = "P1"
            write_json(run/"manifest.json", manifest)
            env = read_json(run/"environment.json")
            env.update({"pid": 23, "native_thread_id": 42})
            write_json(run/"environment.json", env)
            self.assertEqual(parse_trace(run)["status"], "passed")
            plot([root/"runs"], root/"plots")
            for suffix in ("latency", "modes", "timeline", "phases"):
                paths = list((root/"plots").glob(f"*_{suffix}.png"))
                self.assertTrue(paths, suffix)
                self.assertGreater(paths[0].stat().st_size, 1000)
            self.assertTrue((root/"plots"/"device_ratios.png").is_file())
