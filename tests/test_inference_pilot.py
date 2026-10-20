"""Synthetic fixtures validate analysis logic; they are not GPU measurements."""
import argparse
import contextlib
import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

from inference_pilot.common import DEFAULTS, PHASES, load_settings, read_csv, read_json, union_length, write_csv, write_json
from inference_pilot.input_data import BYTES_PER_IMAGE, FORMAT, read_fixed_input, sidecar_path
from inference_pilot.report import summarize
from inference_pilot.runner import build_command, make_suite, run_one
from inference_pilot.trace import API_GPU_CLOCK_SKEW_NS, correlate, parse_trace


@contextlib.contextmanager
def edit_db(path):
    db = sqlite3.connect(path)
    try:
        yield db
        db.commit()
    finally:
        db.close()


def fixture(path, mode="P1", optional_columns=True):
    path.mkdir(parents=True)
    pid, tid = 23, 42
    cpu_id = (1 << 48) | (pid << 24) | tid
    gpu_id = (1 << 56) | (pid << 24)
    write_json(path / "manifest.json", {"mode": mode, "measurement_count": 2, "transition_count": 1})
    write_json(path / "environment.json", {"pid": pid, "native_thread_id": tid})
    db = sqlite3.connect(path / "trace.sqlite")
    db.executescript('''
        CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
        INSERT INTO StringIds VALUES(1, 'cudaLaunchKernel_v7000');
        INSERT INTO StringIds VALUES(2, 'cudaDeviceSynchronize');
        INSERT INTO StringIds VALUES(3, 'kernel_for_testing');
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER, nameId INTEGER);
        CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, globalTid INTEGER, eventType INTEGER, text TEXT, textId INTEGER);
    ''')
    extra = ", gridX INTEGER, gridY INTEGER, gridZ INTEGER, blockX INTEGER, blockY INTEGER, blockZ INTEGER" if optional_columns else ""
    db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, globalPid INTEGER, correlationId INTEGER, streamId INTEGER, deviceId INTEGER, contextId INTEGER, shortName INTEGER" + extra + ")")
    for iteration in range(3):
        start = 1000 + iteration * 100000
        stage = "transition" if iteration == 0 else "measure"
        index = 0 if iteration == 0 else iteration - 1
        db.execute("INSERT INTO NVTX_EVENTS VALUES(?,?,?,?,?,?)", (start, start+90000, cpu_id, 59, f"pilot.iter/{stage}/{index:06d}", None))
        for phase_index, phase in enumerate(PHASES):
            pstart = start + 100 + phase_index * 1000
            string_id = 100 + iteration * 10 + phase_index
            db.execute("INSERT INTO StringIds VALUES(?,?)", (string_id, "pilot.phase/" + phase))
            # Registered-string ranges; kernels execute long after the CPU phase ends.
            db.execute("INSERT INTO NVTX_EVENTS VALUES(?,?,?,?,?,?)", (pstart, pstart+400, cpu_id, 59, None, string_id))
            corr = iteration * 10 + phase_index + 1
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?,?)", (pstart+100, pstart+120, cpu_id, corr, 1))
            kstart = start + 10000 + phase_index * 3000
            values = [kstart, kstart+4000, gpu_id, corr, 7, 0, 1, 3]
            if optional_columns:
                values += [4, 1, 1, 128, 1, 1]
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES("+",".join("?" for _ in values)+")", values)
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?,?)", (start+80000, start+85000, cpu_id, 1000+iteration, 2))
    # Unrelated process reuses all IDs. Joining solely on correlationId would be wrong.
    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?,?)", (111100, 111120, cpu_id+(1<<24), 11, 1))
    db.commit()
    db.close()
    return cpu_id, gpu_id


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_async_nvtx_mapping_process_scope_and_overlap(self):
        fixture(self.root / "p1")
        result = parse_trace(self.root / "p1")
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(result["phase_duration_coverage"], 1)
        rows = read_csv(self.root / "p1" / "features.csv")
        self.assertEqual(len(rows), 2)
        self.assertEqual(float(rows[0]["kernel_duration_sum_us"]), 24)
        self.assertEqual(float(rows[0]["kernel_busy_union_us"]), 19)
        self.assertNotIn("phase", rows[0])
        self.assertNotIn("batch", rows[0])

    def test_bounded_api_gpu_clock_skew(self):
        api = {"start": 10000, "source": "runtime", "contextId": 1}
        index = {(23, 11): [api]}
        event = {"globalPid": (1 << 56) | (23 << 24), "correlationId": 11,
                 "contextId": 1, "start": 10000 - API_GPU_CLOCK_SKEW_NS}
        self.assertEqual(correlate(event, index), (api, "matched"))
        event["start"] -= 1
        self.assertEqual(correlate(event, index), (None, "no_correlated_api"))

    def test_explicit_dropped_cupti_events_rejected(self):
        fixture(self.root / "p1")
        with edit_db(self.root / "p1" / "trace.sqlite") as db:
            db.execute("CREATE TABLE DIAGNOSTIC_EVENT(severity INTEGER, text TEXT)")
            db.execute("INSERT INTO DIAGNOSTIC_EVENT VALUES(2, 'Number of incomplete CUPTI events dropped: 3.')")
        result = parse_trace(self.root / "p1")
        self.assertEqual(result["status"], "failed")
        self.assertIn("dropped: 3", result["errors"][0])

    def test_p0_sync_segmentation_excludes_transition(self):
        fixture(self.root / "p0", mode="P0")
        with edit_db(self.root / "p0" / "trace.sqlite") as db:
            db.execute("DROP TABLE NVTX_EVENTS")
        result = parse_trace(self.root / "p0")
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(result["segmentation"], "synchronization_assisted_not_blind")
        self.assertEqual([r["iteration"] for r in read_csv(self.root / "p0" / "features.csv")], ["0", "1"])

    def test_missing_sync_is_rejected(self):
        fixture(self.root / "p0", mode="P0")
        with edit_db(self.root / "p0" / "trace.sqlite") as db:
            db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId=1001")
        result = parse_trace(self.root / "p0")
        self.assertEqual(result["status"], "failed")
        self.assertIn("synchronization count", result["errors"][0])

    def test_ambiguous_correlation_is_rejected(self):
        cpu, _ = fixture(self.root / "p1")
        with edit_db(self.root / "p1" / "trace.sqlite") as db:
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?,?)", (101210, 101215, cpu, 11, 1))
        result = parse_trace(self.root / "p1")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["unassigned_kernels"][0]["reason"], "ambiguous_correlation_id")

    def test_missing_optional_fields_remain_missing(self):
        fixture(self.root / "p1", optional_columns=False)
        self.assertEqual(parse_trace(self.root / "p1")["status"], "passed")
        self.assertEqual(read_csv(self.root / "p1" / "features.csv")[0]["grid_blocks_p50"], "")

    def test_incomplete_nvtx_is_rejected(self):
        fixture(self.root / "p1")
        with edit_db(self.root / "p1" / "trace.sqlite") as db:
            db.execute("UPDATE NVTX_EVENTS SET end=NULL WHERE text='pilot.iter/measure/000001'")
        result = parse_trace(self.root / "p1")
        self.assertEqual(result["status"], "failed")
        self.assertIn("Incomplete NVTX", result["errors"][0])

    def test_unknown_schema_is_reported(self):
        fixture(self.root / "p1")
        with edit_db(self.root / "p1" / "trace.sqlite") as db:
            db.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL RENAME COLUMN globalPid TO unknownProcessField")
        result = parse_trace(self.root / "p1")
        self.assertEqual(result["status"], "failed")
        self.assertTrue((self.root / "p1" / "trace_schema.json").exists())

    def test_kernel_crossing_completion_rejected(self):
        fixture(self.root / "p1")
        with edit_db(self.root / "p1" / "trace.sqlite") as db:
            db.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end=200000 WHERE correlationId=11")
        result = parse_trace(self.root / "p1")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["unassigned_kernels"][0]["reason"], "gpu_work_crosses_completed_iteration")


class PlanningTests(unittest.TestCase):
    def test_union_does_not_double_count(self):
        self.assertEqual(union_length([(0, 10), (5, 15), (20, 22)]), 17)

    def test_fixed_input_rejects_corruption_and_wrong_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.bin"
            data = b"\0" * BYTES_PER_IMAGE + b"\1" * BYTES_PER_IMAGE
            path.write_bytes(data)
            write_json(sidecar_path(path), {
                "format": FORMAT, "shape": [2, 3, 224, 224], "dtype": "float16",
                "layout": "NCHW_contiguous", "byte_order": "little",
                "bytes_per_image": BYTES_PER_IMAGE,
                "file_sha256": hashlib.sha256(data).hexdigest(),
                "prefix_sha256": {str(batch): hashlib.sha256(data[:batch * BYTES_PER_IMAGE]).hexdigest()
                                  for batch in (1, 2)},
            })
            prefix, provenance = read_fixed_input(path, 1)
            self.assertEqual(prefix, data[:BYTES_PER_IMAGE])
            self.assertEqual(provenance["input_prefix_sha256"], hashlib.sha256(prefix).hexdigest())
            with self.assertRaisesRegex(ValueError, "metadata or batch"):
                read_fixed_input(path, 3)
            path.write_bytes(data[:-1] + b"x")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                read_fixed_input(path, 1)

    def test_invalid_warmup_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "settings.json"
            write_json(path, {"warmup_max": 55})
            with self.assertRaises(ValueError):
                load_settings(path)

    def test_nsys_command_is_argument_list_with_capture_control(self):
        command = build_command({"mode": "P1", "out": "/a folder/results"}, "/a folder/request.json")
        self.assertIn("--trace=cuda,nvtx", command)
        self.assertIn("--capture-range-end=stop", command)
        self.assertIn("/a folder/request.json", command)
        self.assertNotIn("shell=True", command)
        fallback = build_command({"mode": "P1", "out": "/results", "nsys_cuda_trace": "cuda-sw"}, "/results/request.json")
        self.assertIn("--trace=cuda-sw,nvtx", fallback)

    def test_plan_has_12_unique_runs_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            args = argparse.Namespace(out=str(Path(temp)/"session"), label="rtx4090", settings=None,
                                      order_seed=2026, repeats=3, batches=[1], modes=["G", "A", "P0", "P1"],
                                      device="cuda:0", expect_gpu="4090", weights="missing.pth", dry_run=True)
            plan = make_suite(args)
            self.assertEqual(len(plan), 12)
            self.assertEqual(len({r["out"] for r in plan}), 12)
            for request in plan:
                self.assertTrue(run_one(request, dry_run=True))
                self.assertEqual(read_json(Path(request["out"])/"execution.json")["status"], "planned_only")
            with self.assertRaises(FileExistsError):
                make_suite(args)


def fake_result(path, label, batch, value, repeat, version="test-only"):
    path.mkdir(parents=True, exist_ok=True)
    m = {"mode": "G", "measurement_count": 2, "transition_count": 1, "label": label, "batch": batch,
         "repeat": repeat, "settings": dict(DEFAULTS), "input_sha256": f"input-b{batch}", "model": "fixture",
         "weights_sha256": "test", "dtype": "float16", "layout": "NCHW", "code_sha256": "test",
         "backend": "eager", "cuda_graph": False, "queue_depth": 1, "timer_scope": "test",
         "annotation_timer": "test", "forward_path": "test", "precision_policy": {}}
    write_json(path/"request.json", m)
    write_json(path/"manifest.json", m)
    write_json(path/"environment.json", {"gpu_name": label, "gpu_uuid": label, "torch": version, "torchvision": version, "cuda_runtime": "test", "cudnn": "test"})
    write_json(path/"quality.json", {"status": "passed"})
    write_json(path/"execution.json", {"status": "completed"})
    write_csv(path/"latency.csv", [{"stage": "measure", "latency_us": value} for _ in range(2)])


class SummaryTests(unittest.TestCase):
    def test_held_out_target_does_not_enter_multiplier(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for label in ("source", "target"):
                for batch, ratio in ((1, 1), (8, 2), (32, 9)):
                    for repeat in range(3):
                        fake_result(root/"runs"/f"{label}_{batch}_{repeat}", label, batch, 100 if label=="source" else 100*ratio, repeat)
            summarize([root/"runs"], root/"report")
            rows = read_csv(root/"report"/"leave_one_batch_out_sanity.csv")
            held = next(r for r in rows if r["source"]=="source" and r["held_out_batch"]=="32")
            self.assertEqual(float(held["estimated_ratio"]), 1.5)
            self.assertEqual(float(held["prediction_us"]), 150)
            self.assertEqual(float(held["actual_us"]), 900)

    def test_protocol_mismatch_does_not_produce_ratio(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for label in ("source", "target"):
                for repeat in range(3):
                    fake_result(root/"runs"/f"{label}_{repeat}", label, 1, 100, repeat, version=label)
            summarize([root/"runs"], root/"report")
            self.assertEqual(read_csv(root/"report"/"device_ratios.csv"), [])

    def test_fixed_input_protocol_is_separate_from_legacy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for label in ("legacy", "fixed"):
                for repeat in range(3):
                    path = root / "runs" / f"{label}_{repeat}"
                    fake_result(path, label, 1, 100, repeat)
                    if label == "fixed":
                        manifest = read_json(path / "manifest.json")
                        manifest["input_scheme"] = FORMAT
                        write_json(path / "manifest.json", manifest)
            summarize([root / "runs"], root / "report")
            self.assertEqual(read_csv(root / "report" / "device_ratios.csv"), [])
            self.assertEqual(len(read_json(root / "report" / "summary_quality.json")["protocols"]), 2)

    def test_planned_run_is_not_a_measurement(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root/"planned"/"request.json", {})
            write_json(root/"planned"/"execution.json", {"status": "planned_only"})
            with self.assertRaises(ValueError):
                summarize([root/"planned"], root/"report")
            self.assertEqual(read_json(root/"report"/"summary_quality.json")["status"], "no_valid_runs")


if __name__ == "__main__":
    unittest.main()
