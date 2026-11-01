"""Independently calibrated RTX3090 O2 G0 whole-sequence timing experiment.

NEW protocol, not O2 V3; no trace labels or automatic deletion.
Usage: python -m o2_sequence.g0_adaptive {bootstrap,calibrate,formal,review}
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

from o1_operator.full_matrix import BY_ID as FORMAL_CONFIGS
from o1_operator.spec import INPUT_FORMAT, write_inputs
from o2_sequence.catalog import FROZEN_PATH, PROTOCOL as V3_PROTOCOL, SPLIT_PATH, load
from o2_sequence.runner import code_hash as v3_hash, hash_file
from o2_sequence.worker import build_steps, check_dependent_tail, execute_sequence
from o2_sequence.g0_window_pilot import now, short_device_snapshot

PROTOCOL = "o2_g0_adaptive_host_125ms_v1_20261101"
V3_SHA = "d9ca6fdd0442f386629c2b5ab459dd6f4162a72277907ed205ad16fb63524f1f"
TARGET_US = 125_000
BOOTSTRAP_COUNT = 256
BOOTSTRAP_REPEATS = 2
FORMAL_REPEATS = 3
GROUPS = 5
COUNT_QUANTUM = 64
MIN_COUNT = 64
MAX_COUNT = 4096
SEED = 20261101
CODE_ROOT = Path(__file__).resolve().parent.parent


def put(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_source():
    if v3_hash() != V3_SHA:
        raise RuntimeError("Frozen O2 V3 source hash changed; cannot mix protocols")


def template_config():
    templates = load(True)
    if len(templates) != 60:
        raise RuntimeError("Expected exactly 60 formal templates")
    ids = sorted({s["config_id"] for t in templates for s in t["steps"]})
    return templates, {x: FORMAL_CONFIGS[x] for x in ids}


def expected_correctness(template):
    return "passed" if template["topology"] == "dependent_tail" else "not_applicable"


def worker(path):
    req = read(path)
    output = Path(req["output"])
    status = {"status": "started", "utc_start": now()}
    put(output / "quality.json", status)
    rows = []
    try:
        import torch
        from inference_pilot.benchmark import precision_policy

        checked_source()
        if req["protocol"] != PROTOCOL or req["source_protocol"] != V3_PROTOCOL:
            raise RuntimeError("Adaptive G0 protocol mismatch")
        if (req.get("matrix"), req.get("mode"), req.get("input_format"), req.get("groups")) != (
                "formal", "G0", INPUT_FORMAT, GROUPS):
            raise RuntimeError("Adaptive G0 request matrix/mode/inputs/group policy mismatch")
        if req["code_sha256"] != sha_file(__file__):
            raise RuntimeError("Adaptive worker source changed")
        n = req["sequence_count"]
        if n < MIN_COUNT or n > MAX_COUNT or n % COUNT_QUANTUM:
            raise RuntimeError(f"Invalid calibrated count {n}")
        if req["stage"] == "bootstrap":
            if n != BOOTSTRAP_COUNT:
                raise RuntimeError("Bootstrap count must be 256")
        elif req["stage"] == "formal":
            cal_path = Path(req["calibration_path"])
            if sha_file(cal_path) != req["calibration_sha256"]:
                raise RuntimeError("Calibration SHA-256 mismatch")
            calibration = read(cal_path)
            if calibration["protocol"] != PROTOCOL:
                raise RuntimeError("Unexpected calibration protocol")
            if calibration["counts"][req["template"]["id"]]["sequences_per_group"] != n:
                raise RuntimeError("Formal run count differs from frozen calibration")
        else:
            raise RuntimeError("Unknown collection stage")

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Requires one visible CUDA GPU")
        torch.cuda.set_device(0)
        torch.set_num_threads(1)
        precision_policy(torch)
        device = torch.cuda.get_device_name(0)
        if req["expect_gpu"].casefold() not in device.casefold():
            raise RuntimeError(f"Expected {req['expect_gpu']}, found {device}")
        put(output / "environment.json", {
            "utc_start": now(), "device": device, "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "python": sys.version.split()[0],
            "pre_gpu": short_device_snapshot(),
        })
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            operations, loaded = build_steps(torch, req)
            correctness = check_dependent_tail(torch, req, operations, loaded)
            if correctness["status"] != expected_correctness(req["template"]):
                raise RuntimeError("Dependent correctness differs from topology")
            for i in range(10):
                output_tensor = execute_sequence(torch, req, operations, 0, i)
                del output_tensor
            torch.cuda.synchronize()
            for group in range(1, GROUPS + 1):
                torch.cuda.synchronize()
                utc_start = now()
                t0 = time.perf_counter_ns()
                for sequence in range(n):
                    output_tensor = execute_sequence(torch, req, operations, group, sequence)
                    del output_tensor
                torch.cuda.synchronize()
                t1 = time.perf_counter_ns()
                rows.append({
                    "group": group,
                    "utc_start": utc_start, "utc_end": now(),
                    "sequence_count": n,
                    "host_sequence_batch_us": (t1 - t0) / 1000,
                    "host_us_per_sequence": (t1 - t0) / 1000 / n,
                })
            del loaded
        status.update({
            "status": "completed", "utc_end": now(),
            "dependent_correctness": correctness,
            "post_gpu": short_device_snapshot(),
        })
    except Exception as exc:
        status.update({
            "status": "failed", "utc_end": now(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
        print(status["error"], file=sys.stderr)
    finally:
        with (output / "groups.csv").open("w", newline="") as stream:
            fields = ("group", "utc_start", "utc_end", "sequence_count",
                      "host_sequence_batch_us", "host_us_per_sequence")
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        put(output / "quality.json", status)
    return 0 if status["status"] == "completed" else 1


def group_measurements(folder, expected_n):
    if read(folder / "execution.json")["status"] != "completed":
        raise RuntimeError(f"Incomplete execution: {folder}")
    quality = read(folder / "quality.json")
    if quality["status"] != "completed":
        raise RuntimeError(f"Incomplete worker: {folder}")
    with (folder / "groups.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != GROUPS or {int(x["group"]) for x in rows} != set(range(1, GROUPS + 1)):
        raise RuntimeError(f"Expected five groups: {folder}")
    if any(int(x["sequence_count"]) != expected_n for x in rows):
        raise RuntimeError(f"Wrong sequences/group: {folder}")
    times = [float(x["host_us_per_sequence"]) for x in rows]
    windows = [float(x["host_sequence_batch_us"])/1000 for x in rows]
    if any(not math.isfinite(x) or x <= 0 for x in times + windows):
        raise RuntimeError(f"Invalid timing: {folder}")
    return times, windows


def bootstrap(root, gpu):
    checked_source()
    root = Path(root).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"Refusing existing session: {root}")
    templates, configs = template_config()
    root.mkdir(parents=True)
    inventory = write_inputs(root / "inputs", configs.values())
    put(root / "input_inventory.json", inventory)
    plan = {
        "protocol": PROTOCOL, "source_protocol": V3_PROTOCOL,
        "v3_code_sha256": V3_SHA, "adaptive_code_sha256": sha_file(__file__),
        "catalog_sha256": hash_file(FROZEN_PATH),
        "o1_split_sha256": hash_file(SPLIT_PATH),
        "input_format": INPUT_FORMAT, "expect_gpu": gpu, "groups": GROUPS,
        "target_group_us": TARGET_US, "bootstrap_sequences": BOOTSTRAP_COUNT,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "formal_repeats": FORMAL_REPEATS, "sequence_quantum": COUNT_QUANTUM,
        "min_sequences": MIN_COUNT, "max_sequences": MAX_COUNT,
        "templates": templates, "configs": configs, "source": "new independent bootstrap",
        "status": "running", "failures": [], "preflight_gpu": short_device_snapshot(),
    }
    jobs = [(t["id"], rep) for t in templates for rep in range(1, BOOTSTRAP_REPEATS+1)]
    random.Random(SEED).shuffle(jobs)
    plan["jobs"] = [{"template": tid, "repeat": rep} for tid, rep in jobs]
    put(root / "bootstrap_plan.json", plan)
    for i, (tid, rep) in enumerate(jobs, 1):
        folder = root / "bootstrap" / tid / f"run{rep:02d}"
        folder.mkdir(parents=True)
        req = {
            "protocol": PROTOCOL, "source_protocol": V3_PROTOCOL,
            "code_sha256": sha_file(__file__), "stage": "bootstrap",
            "matrix": "formal", "mode": "G0", "input_format": INPUT_FORMAT,
            "template": next(t for t in templates if t["id"] == tid),
            "configs": configs, "inputs": inventory,
            "expect_gpu": gpu, "groups": GROUPS,
            "sequence_count": BOOTSTRAP_COUNT, "output": str(folder),
        }
        put(folder / "request.json", req)
        print(f"BOOTSTRAP {i}/{len(jobs)} {tid} run{rep:02d}", flush=True)
        _run_worker(folder, plan, {"template": tid, "repeat": rep})
        if plan["failures"]:
            print("Stopping bootstrap at first failure; inspect the failed worker.log", file=sys.stderr)
            break
    plan["status"] = "completed" if not plan["failures"] else "completed_with_failures"
    plan["failure_count"] = len(plan["failures"])
    put(root / "bootstrap_plan.json", plan)
    print(f"BOOTSTRAP: {len(jobs)-len(plan['failures'])}/{len(jobs)} completed")
    return bool(plan["failures"])


def _run_worker(folder, plan, job):
    argv = [sys.executable, "-m", "o2_sequence.g0_adaptive", "worker",
            "--request", str(folder / "request.json")]
    put(folder / "command.json", {"argv": argv, "cwd": str(CODE_ROOT)})
    try:
        with (folder / "worker.log").open("w") as stream:
            code = subprocess.run(
                argv, cwd=CODE_ROOT, stdout=stream, stderr=subprocess.STDOUT,
                timeout=600, check=False,
            ).returncode
        if code or read(folder / "quality.json")["status"] != "completed":
            raise RuntimeError(f"worker exit={code}, see worker.log")
        put(folder / "execution.json", {"status": "completed"})
    except Exception as exc:
        failure = {**job, "error": f"{type(exc).__name__}: {exc}"}
        plan["failures"].append(failure)
        put(folder / "execution.json", {"status": "failed", "error": failure["error"]})
        print("FAILED:", failure, file=sys.stderr)


def calibrate(root):
    root = Path(root).expanduser().resolve()
    if (root / "calibration.json").exists():
        raise FileExistsError("Calibration already frozen")
    plan = read(root / "bootstrap_plan.json")
    if plan["protocol"] != PROTOCOL or plan["status"] != "completed":
        raise RuntimeError("Bootstrap is not complete")
    if sha_file(__file__) != plan["adaptive_code_sha256"]:
        raise RuntimeError("Adaptive implementation differs from bootstrap version")
    counts = {}
    for template in plan["templates"]:
        tid = template["id"]
        samples = []
        for repeat in range(1, BOOTSTRAP_REPEATS+1):
            folder = root / "bootstrap" / tid / f"run{repeat:02d}"
            if read(folder / "quality.json")["dependent_correctness"]["status"] != expected_correctness(template):
                raise RuntimeError(f"Bootstrap correctness failed: {tid}")
            t, _ = group_measurements(folder, BOOTSTRAP_COUNT)
            samples.extend(t)
        median_us = statistics.median(samples)
        desired = math.ceil((TARGET_US / median_us) / COUNT_QUANTUM)*COUNT_QUANTUM
        if desired > MAX_COUNT:
            raise RuntimeError(f"Calibration count exceeds maximum: {tid} n={desired}")
        n = max(MIN_COUNT, desired)
        counts[tid] = {
            "bootstrap_median_us_per_sequence": median_us,
            "sequences_per_group": n,
            "predicted_group_us": median_us*n,
        }
    frozen = {
        "protocol": PROTOCOL, "source_protocol": V3_PROTOCOL,
        "adaptive_code_sha256": sha_file(__file__),
        "bootstrap_plan_sha256": sha_file(root / "bootstrap_plan.json"),
        "target_group_us": TARGET_US, "count_quantum": COUNT_QUANTUM,
        "max_count": MAX_COUNT, "estimator": "median over 2x5 bootstrap groups",
        "counts": counts,
    }
    put(root / "calibration.json", frozen)
    sha = sha_file(root / "calibration.json")
    (root / "calibration.sha256").write_text(f"{sha}  calibration.json\n")
    print(f"CALIBRATION: 60/60 frozen; SHA256 {sha}")
    print(f"Count range: {min(x['sequences_per_group'] for x in counts.values())}-"
          f"{max(x['sequences_per_group'] for x in counts.values())}")


def formal(root):
    checked_source()
    root = Path(root).expanduser().resolve()
    if (root / "formal").exists():
        raise FileExistsError("Formal session already exists")
    bootstrap_plan = read(root / "bootstrap_plan.json")
    cal_path = root / "calibration.json"
    cal = read(cal_path)
    if bootstrap_plan["status"] != "completed" or cal["protocol"] != PROTOCOL:
        raise RuntimeError("Bootstrap/calibration incomplete")
    if cal["bootstrap_plan_sha256"] != sha_file(root / "bootstrap_plan.json"):
        raise RuntimeError("Bootstrap plan modified after calibration")
    if cal["adaptive_code_sha256"] != sha_file(__file__):
        raise RuntimeError("Adaptive source modified after calibration")
    templates, configs = template_config()
    if bootstrap_plan["catalog_sha256"] != hash_file(FROZEN_PATH):
        raise RuntimeError("Frozen O2 template catalog changed since bootstrap")
    if bootstrap_plan["o1_split_sha256"] != hash_file(SPLIT_PATH):
        raise RuntimeError("O1 split changed since bootstrap")
    if set(cal["counts"]) != {t["id"] for t in templates}:
        raise RuntimeError("Calibration template set mismatch")
    inventory = read(root / "input_inventory.json")
    plan = {
        "protocol": PROTOCOL, "calibration_sha256": sha_file(cal_path),
        "adaptive_code_sha256": sha_file(__file__),
        "source_code_sha256": V3_SHA, "status": "running",
        "failures": [], "expect_gpu": bootstrap_plan["expect_gpu"],
        "preflight_gpu": short_device_snapshot(),
    }
    jobs = [(t["id"], rep) for t in templates for rep in range(1, FORMAL_REPEATS + 1)]
    random.Random(SEED + 1).shuffle(jobs)
    plan["jobs"] = [{"template": tid, "repeat": rep} for tid, rep in jobs]
    (root / "formal").mkdir()
    put(root / "formal_plan.json", plan)
    by_id = {t["id"]: t for t in templates}
    for i, (tid, rep) in enumerate(jobs, 1):
        folder = root / "formal" / tid / f"run{rep:02d}"
        folder.mkdir(parents=True)
        request = {
            "protocol": PROTOCOL, "source_protocol": V3_PROTOCOL,
            "code_sha256": sha_file(__file__), "stage": "formal",
            "matrix": "formal", "mode": "G0", "input_format": INPUT_FORMAT,
            "template": by_id[tid], "configs": configs, "inputs": inventory,
            "expect_gpu": bootstrap_plan["expect_gpu"], "groups": GROUPS,
            "sequence_count": cal["counts"][tid]["sequences_per_group"],
            "output": str(folder), "calibration_path": str(cal_path),
            "calibration_sha256": sha_file(cal_path),
        }
        put(folder / "request.json", request)
        print(f"FORMAL {i}/{len(jobs)} {tid} run{rep:02d} "
              f"n={request['sequence_count']}", flush=True)
        _run_worker(folder, plan, {"template": tid, "repeat": rep})
        if plan["failures"]:
            print("Stopping formal at first failure; inspect the failed worker.log", file=sys.stderr)
            break
    plan["status"] = "completed" if not plan["failures"] else "completed_with_failures"
    plan["failure_count"] = len(plan["failures"])
    put(root / "formal_plan.json", plan)
    print(f"FORMAL: {len(jobs)-len(plan['failures'])}/{len(jobs)} completed")
    return bool(plan["failures"])


def review(root):
    root = Path(root).expanduser().resolve()
    plan = read(root / "formal_plan.json")
    cal = read(root / "calibration.json")
    if plan["status"] != "completed" or plan["failure_count"] != 0:
        raise RuntimeError("Formal collection is incomplete")
    if sha_file(root / "calibration.json") != plan["calibration_sha256"]:
        raise RuntimeError("Calibration manifest changed")
    if len(plan["jobs"]) != 60*FORMAL_REPEATS:
        raise RuntimeError("Incorrect formal job count")
    folder = root / "analysis"
    folder.mkdir(exist_ok=False)
    rows = []
    for template in load(True):
        tid = template["id"]
        n = cal["counts"][tid]["sequences_per_group"]
        repeat_medians, windows, intra_spreads = [], [], []
        for rep in range(1, FORMAL_REPEATS + 1):
            run = root / "formal" / tid / f"run{rep:02d}"
            if read(run / "quality.json")["dependent_correctness"]["status"] != expected_correctness(template):
                raise RuntimeError(f"Incorrect dependent tail: {run}")
            samples, group_windows = group_measurements(run, n)
            repeat_medians.append(statistics.median(samples))
            windows.extend(group_windows)
            intra_spreads.append((max(samples)-min(samples))/statistics.median(samples)*100)
        spread = (max(repeat_medians)-min(repeat_medians))/statistics.median(repeat_medians)*100
        rows.append({
            "template": tid, "topology": template["topology"],
            "sequences_per_group": n,
            "median_sequence_us": statistics.median(repeat_medians),
            "repeat_spread_pct": spread, "stable_5pct": spread <= 5,
            "min_group_ms": min(windows), "median_group_ms": statistics.median(windows),
            "all_groups_ge_100ms": min(windows) >= 100,
            "median_intra_repeat_spread_pct": statistics.median(intra_spreads),
            "run1_us": repeat_medians[0], "run2_us": repeat_medians[1],
            "run3_us": repeat_medians[2],
        })
    with (folder / "timing_quality.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    stable = sum(x["stable_5pct"] for x in rows)
    windows = sum(x["all_groups_ge_100ms"] for x in rows)
    result = {
        "protocol": PROTOCOL, "planned_runs": 180, "completed_runs": 180,
        "bootstrap_runs": 120, "calibration_sha256": sha_file(root / "calibration.json"),
        "stable_5pct": stable, "total_templates": 60,
        "all_groups_ge_100ms_templates": windows,
        "median_group_ms": statistics.median(x["median_group_ms"] for x in rows),
        "min_group_ms": min(x["min_group_ms"] for x in rows),
        "failed_stability_templates": [x["template"] for x in rows if not x["stable_5pct"]],
        "failed_window_templates": [x["template"] for x in rows if not x["all_groups_ge_100ms"]],
        "timing_quality": "PASS" if stable == 60 else "PARTIAL",
    }
    put(folder / "summary.json", result)
    print("===== RTX3090 O2 G0 INDEPENDENT ADAPTIVE VALIDATION =====")
    print("BOOTSTRAP: 120/120 (separate)")
    print("FORMAL: 180/180 PASS")
    print(f"STABLE <=5%: {stable}/60")
    print(f"ALL GROUPS >=100ms: {windows}/60 templates")
    print(f"MIN GROUP: {result['min_group_ms']:.3f}ms")
    print("STABILITY FAILURES:", result["failed_stability_templates"])
    print("WINDOW FAILURES:", result["failed_window_templates"])
    print("TIMING QUALITY:", result["timing_quality"])
    print("REPORT:", folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("worker")
    p.add_argument("--request", required=True)
    p = sub.add_parser("bootstrap")
    p.add_argument("--out", required=True)
    p.add_argument("--expect-gpu", default="3090")
    for cmd in ("calibrate", "formal", "review"):
        p = sub.add_parser(cmd)
        p.add_argument("--root", required=True)
    args = parser.parse_args()
    if args.cmd == "worker":
        return worker(args.request)
    if args.cmd == "bootstrap":
        return int(bootstrap(args.out, args.expect_gpu))
    if args.cmd == "calibrate":
        calibrate(args.root)
    elif args.cmd == "formal":
        return int(formal(args.root))
    else:
        review(args.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
