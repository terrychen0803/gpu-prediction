import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from o1_operator.runner import execute, selected_jobs, worker_command
from o1_operator.spec import BY_ID, CONFIGS, PROTOCOL, input_bytes, operand_shapes
from o1_operator.summarize import summarize
from o1_operator.trace_audit import audit


class O1PlanTests(unittest.TestCase):
    def test_published_input_manifest_matches_all_pilot_operands(self):
        manifest = json.loads((Path(__file__).resolve().parent.parent /
                               "o1_operator/pilot_input_sha256.json").read_text())
        self.assertEqual(manifest["input_format"], "o1_shake256_lut_fp16_le_v1")
        self.assertEqual(manifest["seed"], 42)
        self.assertEqual(set(manifest["operands"]), set(BY_ID))
        for config_id, operands in manifest["operands"].items():
            shapes = operand_shapes(BY_ID[config_id])
            self.assertEqual(set(operands), set(shapes))
            for name, expected in operands.items():
                data, _ = input_bytes(config_id, name, shapes[name])
                self.assertEqual(hashlib.sha256(data).hexdigest(), expected)

    def test_nine_configs_and_portable_nonconstant_bytes(self):
        self.assertEqual(len(CONFIGS), 9)
        self.assertEqual(len(BY_ID), 9)
        self.assertIn("fixed_repetitions", PROTOCOL)
        shape = operand_shapes(CONFIGS[0])["a"]
        one, material = input_bytes(CONFIGS[0]["id"], "a", shape)
        two, _ = input_bytes(CONFIGS[0]["id"], "a", shape)
        self.assertEqual(one, two)
        self.assertEqual(len(one), 2 * shape[0] * shape[1])
        self.assertGreater(len(set(one)), 16)
        self.assertIn("seed=42", material)

    def test_unique_cells_and_no_metrics_in_p_r(self):
        jobs = selected_jobs(list(BY_ID), ("G0", "GE", "P", "R"), 3)
        self.assertEqual(len(jobs), 108)
        self.assertEqual(len(set(jobs)), 108)
        with self.assertRaises(ValueError):
            selected_jobs([CONFIGS[0]["id"]] * 2, ("G0",), 1)
        p = worker_command(Path("/tmp/request.json"), "P", Path("/tmp/new_run"))
        r = worker_command(Path("/tmp/request.json"), "R", Path("/tmp/new_run"))
        self.assertIn("--trace=cuda-sw", p)
        self.assertIn("--trace=cuda-sw,nvtx", r)
        self.assertIn("--gpu-metrics-devices=none", p)

    def test_dry_run_fixes_same_call_count_for_p_and_r(self):
        with tempfile.TemporaryDirectory() as folder, patch("o1_operator.runner.command_text", return_value={}):
            root=Path(folder)/"plan"
            self.assertEqual(execute(root,[CONFIGS[0]["id"]],("P","R"),1,dry_run=True),0)
            requests=[json.loads(p.read_text()) for p in root.glob("runs/*/*/run01/request.json")]
            self.assertEqual({r["mode"] for r in requests},{"P","R"})
            self.assertEqual({r["measurement_repetitions"] for r in requests},{8192})
            self.assertEqual({r["protocol"] for r in requests},{PROTOCOL})

    def test_summary_rejects_old_p_r_call_count_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/"raw"; root.mkdir()
            (root/"plan.json").write_text(json.dumps({"protocol":"old-pilot","status":"completed",
                "jobs":[{"config_id":CONFIGS[0]["id"],"mode":m,"repeat":1} for m in ("P","R")],
                "configs":[CONFIGS[0]],"modes":["P","R"],"repeats":1,"code_sha256":"fixture",
                "input_format":"fixture"}))
            for mode,calls in (("P",100),("R",101)):
                run=root/"runs"/CONFIGS[0]["id"]/mode/"run01"
                run.mkdir(parents=True)
                (run/"execution.json").write_text(json.dumps({"status":"completed"}))
                (run/"quality.json").write_text(json.dumps({"correctness":{"passed":True}}))
                (run/"trace_audit.json").write_text(json.dumps({"status":"manual_review_required",
                    "expected_calls":calls,"kernel_count":calls,"warnings":["event-loss"]}))
            result=summarize(root,Path(folder)/"analysis")
            self.assertEqual(result["rp_status_counts"],{"unverifiable_policy_mismatch":1})


def make_trace_fixture(run, mode="P", duplicate=False, warning=False, missing_kernel=False):
    run.mkdir()
    pid, tid = 123, 5
    global_tid = (pid << 24) | tid
    (run / "request.json").write_text(json.dumps({"config_id":"gemm_64_64_64","mode":mode}))
    (run / "environment.json").write_text(json.dumps({"pid":pid}))
    (run / "quality.json").write_text(json.dumps({"repetitions_per_group":1,"group_count":2}))
    (run / "trace.nsys-rep").write_bytes(b"synthetic-only")
    conn = sqlite3.connect(run / "trace.sqlite")
    conn.executescript("""
        CREATE TABLE StringIds(id INTEGER,value TEXT);
        CREATE TABLE DIAGNOSTIC_EVENT(severity INTEGER,text TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,globalPid INTEGER,correlationId INTEGER,
            deviceId INTEGER,contextId INTEGER,streamId INTEGER,demangledName INTEGER,
            gridX INTEGER,gridY INTEGER,gridZ INTEGER,blockX INTEGER,blockY INTEGER,blockZ INTEGER,dynamicSharedMemory INTEGER);
        CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,textId INTEGER,globalTid INTEGER);
    """)
    conn.executemany("INSERT INTO StringIds VALUES (?,?)", [(1,"cuLaunchKernelEx"),(2,"kernel_name")])
    if warning:
        conn.execute("INSERT INTO DIAGNOSTIC_EVENT VALUES (2,'Not all CUDA events might have been collected')")
    for index in range(2):
        base=1000 + index*1000
        conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)",
                     (base+100,base+200,global_tid,index+1,1))
        if not missing_kernel or index==0:
            conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (base+300,base+400,pid<<24,index+1,0,0,1,2,1,1,1,32,1,1,0))
        if mode=="R":
            conn.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,?)",
                         (base,base+250,f'o1.op/g{index+1:02d}/i000000',None,global_tid))
    if duplicate:
        conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)",
                     (1101,1199,global_tid,1,1))
    conn.commit(); conn.close()


class O1TraceAuditTests(unittest.TestCase):
    def test_p_records_predictor_view_without_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)/"P"; make_trace_fixture(run)
            result=audit(run,"P")
            self.assertEqual(result["status"],"recorded_structure_passed")
            events=[json.loads(line) for line in (run/"events_allowlist.jsonl").read_text().splitlines()]
            self.assertEqual(len(events),2)
            self.assertNotIn("op_type",events[0])
            self.assertNotIn("scope",events[0])
            self.assertEqual(events[0]["process_id"],"p0")

    def test_r_maps_by_api_containment_and_warning_blocks_claim(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)/"R"; make_trace_fixture(run,"R",warning=True)
            result=audit(run,"R")
            self.assertEqual(result["status"],"manual_review_required")
            self.assertEqual(result["reference_scope_count"],2)
            self.assertEqual(len((run/"reference_operations.jsonl").read_text().splitlines()),2)

    def test_ambiguous_launch_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)/"P"; make_trace_fixture(run,duplicate=True)
            result=audit(run,"P")
            self.assertEqual(result["status"],"failed")
            self.assertIn("launch API candidates",result["errors"][0])

    def test_missing_kernel_rejected_for_r(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)/"R"; make_trace_fixture(run,"R",missing_kernel=True)
            result=audit(run,"R")
            self.assertEqual(result["status"],"failed")
            self.assertIn("scope without recorded GPU work",result["errors"][0])


if __name__ == "__main__":
    unittest.main()

