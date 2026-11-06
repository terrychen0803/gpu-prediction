"""Exploratory O2 G0 timing-window pilot; never modifies the frozen V3 protocol.

CLI:
    python -m o2_sequence.g0_window_pilot run --out /path/to/new/session --expect-gpu 3090
    python -m o2_sequence.g0_window_pilot review --root /path/to/session

One fresh CUDA process per (template, sequence_count, repeat); five timed groups.
Uses the O2 V3 frozen formal templates, inputs, operations and sequence semantics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from o1_operator.full_matrix import BY_ID as FORMAL
from o1_operator.spec import INPUT_FORMAT, write_inputs
from o2_sequence.catalog import FROZEN_PATH, PROTOCOL as SOURCE_PROTOCOL, SPLIT_PATH, load
from o2_sequence.runner import code_hash as source_code_hash, hash_file
from o2_sequence.worker import build_steps, check_dependent_tail, execute_sequence

PROTOCOL = "o2_g0_window_pilot_v4_exploratory_20261101"
REQUIRED_SOURCE_HASH = "d9ca6fdd0442f386629c2b5ab459dd6f4162a72277907ed205ad16fb63524f1f"
GROUPS = 5
COUNTS = (8, 64, 256, 512)
REPEATS = 3
TEMPLATES = (
    "fold0_v00_dependent_tail",
    "fold0_v04_dependent_tail",
    "fold0_v04_independent",
    "fold3_v03_independent",
    "fold4_v01_dependent_tail",
)
ROOT = Path(__file__).resolve().parent.parent


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def short_device_snapshot():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,pstate,temperature.gpu,"
             "clocks.sm,clocks.mem,power.draw,utilization.gpu",
             "--format=csv,noheader"], text=True, capture_output=True, timeout=10, check=True,
        )
        return result.stdout.strip()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


def worker(request_path):
    request = json.loads(Path(request_path).read_text())
    output = Path(request["output"])
    status = {"status": "started", "start_utc": now()}
    dump(output / "quality.json", status)
    rows = []
    try:
        import torch
        from inference_pilot.benchmark import precision_policy

        if request["protocol"] != PROTOCOL or request["source_protocol"] != SOURCE_PROTOCOL:
            raise ValueError("Wrong O2 timing protocol")
        if request["source_code_sha256"] != REQUIRED_SOURCE_HASH:
            raise ValueError("Wrong O2 V3 base code hash")
        if request["mode"] != "G0" or request["matrix"] != "formal":
            raise ValueError("Pilot is G0 formal only")
        if request["groups"] != GROUPS or request["sequence_count"] not in COUNTS:
            raise ValueError("Unregistered timing-window configuration")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one visible CUDA GPU is required")
        torch.cuda.set_device(0)
        torch.set_num_threads(1)
        precision_policy(torch)
        device = torch.cuda.get_device_name(0)
        if request["expect_gpu"].casefold() not in device.casefold():
            raise RuntimeError(f"Expected {request['expect_gpu']}, got {device}")
        dump(output / "environment.json", {
            "utc": now(), "device": device, "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "python": sys.version.split()[0],
            "start_gpu": short_device_snapshot(), "pid": os.getpid(),
        })
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            operations, loaded = build_steps(torch, request)
            dependent = check_dependent_tail(torch, request, operations, loaded)
            for i in range(10):
                result = execute_sequence(torch, request, operations, 0, i)
                del result
            torch.cuda.synchronize()
            for g in range(1, GROUPS + 1):
                torch.cuda.synchronize()
                started = now()
                t0 = time.perf_counter_ns()
                for s in range(request["sequence_count"]):
                    result = execute_sequence(torch, request, operations, g, s)
                    del result
                torch.cuda.synchronize()
                t1 = time.perf_counter_ns()
                rows.append({
                    "group": g, "utc_start": started, "utc_end": now(),
                    "sequence_count": request["sequence_count"],
                    "host_sequence_batch_us": (t1 - t0) / 1000,
                    "host_us_per_sequence": (t1 - t0) / 1000 / request["sequence_count"],
                })
            del loaded
        status.update({
            "status": "completed", "end_utc": now(),
            "groups": GROUPS, "sequence_count": request["sequence_count"],
            "dependent_correctness": dependent,
            "end_gpu": short_device_snapshot(),
        })
    except Exception as exc:
        status.update({
            "status": "failed", "end_utc": now(), "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
        print(status["error"], file=sys.stderr)
    finally:
        with (output / "groups.csv").open("w", newline="") as f:
            columns = ("group", "utc_start", "utc_end", "sequence_count",
                       "host_sequence_batch_us", "host_us_per_sequence")
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        dump(output / "quality.json", status)
    return 0 if status["status"] == "completed" else 1


def run(out, expect_gpu, dry_run):
    out = Path(out).expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"Refusing existing output: {out}")
    if source_code_hash() != REQUIRED_SOURCE_HASH:
        raise RuntimeError("O2 V3 base changed; do not start until version is reviewed")
    templates = [x for x in load(True) if x["id"] in TEMPLATES]
    if len(templates) != len(TEMPLATES):
        raise RuntimeError("Pilot templates missing from frozen O2 catalog")
    configs = {key: FORMAL[key] for key in sorted(
        {step["config_id"] for t in templates for step in t["steps"]}
    )}
    out.mkdir(parents=True)
    inventory = write_inputs(out / "inputs", configs.values())
    dump(out / "input_inventory.json", inventory)
    jobs = [{"template_id": t["id"], "sequence_count": n, "repeat": r}
            for t in templates for n in COUNTS for r in range(1, REPEATS + 1)]
    random.Random(20261101).shuffle(jobs)
    plan = {
        "protocol": PROTOCOL, "source_protocol": SOURCE_PROTOCOL,
        "source_code_sha256": source_code_hash(),
        "pilot_code_sha256": file_sha(__file__),
        "catalog_sha256": hash_file(FROZEN_PATH),
        "split_sha256": hash_file(SPLIT_PATH),
        "matrix": "formal", "mode": "G0", "expect_gpu": expect_gpu,
        "groups": GROUPS, "sequence_counts": list(COUNTS), "repeats": REPEATS,
        "templates": templates, "jobs": jobs, "status": "planned_only" if dry_run else "running",
        "failures": [], "start_utc": now(), "dry_run": dry_run,
        "measurement": "per-group synchronized batch host time / number of sequences",
        "preflight_gpu": short_device_snapshot(),
    }
    dump(out / "plan.json", plan)
    by_id = {x["id"]: x for x in templates}
    for k, job in enumerate(jobs, 1):
        tid, count, repeat = job["template_id"], job["sequence_count"], job["repeat"]
        folder = out / "runs" / tid / f"n{count:04d}" / f"run{repeat:02d}"
        folder.mkdir(parents=True, exist_ok=False)
        request = {
            "protocol": PROTOCOL, "source_protocol": SOURCE_PROTOCOL,
            "source_code_sha256": source_code_hash(), "matrix": "formal", "mode": "G0",
            "input_format": INPUT_FORMAT, "template": by_id[tid], "configs": configs,
            "inputs": inventory, "groups": GROUPS, "sequence_count": count,
            "output": str(folder), "expect_gpu": expect_gpu,
        }
        dump(folder / "request.json", request)
        argv = [sys.executable, "-m", "o2_sequence.g0_window_pilot",
                "worker", "--request", str(folder / "request.json")]
        dump(folder / "command.json", {"argv": argv})
        print(f"TIMING PILOT {k}/{len(jobs)} {tid} n={count} repeat={repeat}", flush=True)
        if dry_run:
            dump(folder / "execution.json", {"status": "planned_only"})
            continue
        try:
            with (folder / "worker.log").open("w") as f:
                result = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT,
                                        cwd=ROOT, timeout=600, check=False)
            quality = json.loads((folder / "quality.json").read_text())
            if result.returncode or quality.get("status") != "completed":
                raise RuntimeError(f"worker exit {result.returncode}; see worker.log")
            dump(folder / "execution.json", {"status": "completed"})
        except Exception as exc:
            detail = {**job, "error": f"{type(exc).__name__}: {exc}"}
            plan["failures"].append(detail)
            dump(folder / "execution.json", {"status": "failed", "error": detail["error"]})
            print("FAILED:", detail, file=sys.stderr)
    plan["status"] = "planned_only" if dry_run else ("completed_with_failures"
                      if plan["failures"] else "completed")
    plan["failure_count"] = len(plan["failures"])
    plan["end_utc"] = now()
    dump(out / "plan.json", plan)
    print(f"FINISHED: {len(jobs)-len(plan['failures'])}/{len(jobs)} completed; "
          f"failures={len(plan['failures'])}")
    return 1 if plan["failures"] else 0


def review(root):
    root = Path(root).expanduser().resolve()
    plan = json.loads((root / "plan.json").read_text())
    if plan["protocol"] != PROTOCOL or plan["status"] != "completed":
        raise RuntimeError("Pilot is incomplete; refusing quality PASS")
    report = root / "analysis"
    report.mkdir(exist_ok=False)
    results = []
    for template in plan["templates"]:
        for count in COUNTS:
            repeat_medians, group_windows, group_spreads = [], [], []
            for repeat in range(1, REPEATS + 1):
                folder = root / "runs" / template["id"] / f"n{count:04d}" / f"run{repeat:02d}"
                quality = json.loads((folder / "quality.json").read_text())
                if quality["status"] != "completed":
                    raise RuntimeError(f"Failed worker: {folder}")
                if quality["dependent_correctness"]["status"] != (
                        "passed" if template["topology"] == "dependent_tail" else "not_applicable"):
                    raise RuntimeError(f"Correctness mismatch: {folder}")
                with (folder / "groups.csv").open() as f:
                    group_rows = list(csv.DictReader(f))
                if len(group_rows) != GROUPS or any(
                        int(row["sequence_count"]) != count for row in group_rows):
                    raise RuntimeError(f"Wrong group count: {folder}")
                times = [float(x["host_us_per_sequence"]) for x in group_rows]
                windows = [float(x["host_sequence_batch_us"]) / 1000 for x in group_rows]
                repeat_medians.append(statistics.median(times))
                group_windows.extend(windows)
                group_spreads.append((max(times)-min(times))/statistics.median(times))
            spread = (max(repeat_medians) - min(repeat_medians))/statistics.median(repeat_medians)
            results.append({
                "template": template["id"], "topology": template["topology"],
                "sequence_count": count, "repeat_spread_pct": round(spread*100, 4),
                "stable_le_5pct": spread <= 0.05,
                "median_sequence_us": round(statistics.median(repeat_medians), 4),
                "min_group_ms": round(min(group_windows), 4),
                "median_group_ms": round(statistics.median(group_windows), 4),
                "group_spread_pct_median": round(statistics.median(group_spreads)*100, 4),
                "run1_us": round(repeat_medians[0], 4),
                "run2_us": round(repeat_medians[1], 4),
                "run3_us": round(repeat_medians[2], 4),
            })
    with (report / "window_quality.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = {
        "protocol": PROTOCOL, "jobs_completed": len(plan["jobs"]),
        "source_code_sha256": plan["source_code_sha256"],
        "pilot_code_sha256": plan["pilot_code_sha256"],
        "by_count": [
            {"sequences": count,
             "stable": sum(x["stable_le_5pct"] for x in results if x["sequence_count"] == count),
             "total": len(TEMPLATES),
             "median_group_ms": statistics.median(
                 x["median_group_ms"] for x in results if x["sequence_count"] == count),
             "min_group_ms": min(x["min_group_ms"] for x in results if x["sequence_count"] == count)}
            for count in COUNTS
        ],
        "data_quality_note": "5% repeat spread is exploratory; not O2 V3 acceptance",
    }
    dump(report / "summary.json", summary)
    print("===== O2 G0 TIMING WINDOW PILOT =====")
    for x in summary["by_count"]:
        print(f"n={x['sequences']:<4} stable={x['stable']}/{x['total']} "
              f"median_group_ms={x['median_group_ms']:.3f} "
              f"min_group_ms={x['min_group_ms']:.3f}")
    print("===== PER TEMPLATE =====")
    for x in results:
        print(x["template"], f"n={x['sequence_count']}",
              f"spread={x['repeat_spread_pct']:.2f}%",
              f"group_median={x['median_group_ms']:.2f}ms")
    print("Review:", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--out", required=True)
    p.add_argument("--expect-gpu", default="3090")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("worker")
    p.add_argument("--request", required=True)
    p = sub.add_parser("review")
    p.add_argument("--root", required=True)
    args = parser.parse_args()
    if args.cmd == "run":
        return run(args.out, args.expect_gpu, args.dry_run)
    if args.cmd == "worker":
        return worker(args.request)
    review(args.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
