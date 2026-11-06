"""Versioned seven-A × GI/P/R × three-process 40-measured collection runner.

No training, no resumptions/overwrites, no silent substitutions and no strict gate promotion.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import statistics
import sys
from pathlib import Path

from .adapters import SPECS, require_a
from .common import command, gpu_guard, now, read, sha, write
from .formal_protocol import PROTOCOL, process_spread


from .provenance import source_hashes, snapshot, verify_release


def code_hashes():
    return source_hashes()


def plan(workloads, *, plan_only=False):
    if not workloads or len(workloads) > len(SPECS) or len(set(workloads)) != len(workloads):
        raise ValueError("Specify between one and seven unique A workloads")
    for name in workloads:
        require_a(name)
    jobs = [
        {"workload": name, "mode": mode, "repeat": repeat,
         "status": "planned_only" if plan_only else "not_started"}
        for name in workloads
        for mode in PROTOCOL["modes"]
        for repeat in range(1, PROTOCOL["independent_process_repeats"] + 1)
    ]
    random.Random(PROTOCOL["schedule_seed"]).shuffle(jobs)
    return jobs


def validate_sources(manifest, requested, expected_digest):
    root = manifest.parent
    if sha(manifest) != expected_digest:
        raise ValueError("Frozen manifest SHA mismatch; refusing GPU execution")
    data = read(manifest)
    for name in requested:
        entry = data["workloads"].get(name)
        if entry is None or entry["status"] != "ready":
            raise ValueError("Frozen A asset unavailable: " + name)
        for key, item in entry.items():
            if isinstance(item, dict) and "sha256" in item and "path" in item:
                path = root / item["path"]
                if not path.is_file() or sha(path) != item["sha256"]:
                    raise ValueError(f"Asset mismatch for {name}/{key}: {path}")
    return data


def measured_median(folder, measured_count):
    q = read(folder / "quality.json")
    if q["status"] != "completed" or not q["correctness"]["passed"] or not q.get("warmup", {}).get("passed"):
        raise ValueError("Worker or correctness incomplete")
    with (folder / "iterations.csv").open() as f:
        records = [r for r in csv.DictReader(f) if r["stage"] == "measure"]
    if len(records) != measured_count:
        raise ValueError(f"Measured count {len(records)} != {measured_count}")
    if {int(r["iteration"]) for r in records} != set(range(measured_count)):
        raise ValueError("Measured iteration IDs missing or duplicated")
    values = [float(r["elapsed_us"]) for r in records]
    if not all(0 < v < float("inf") for v in values):
        raise ValueError("Invalid CUDA timing")
    return statistics.median(values)


def summarize(root, jobs, workloads):
    by_id = {(j["workload"], j["mode"], j["repeat"]): j for j in jobs}
    groups = []
    pairs = []
    paired = {}
    for name in workloads:
        for rep in range(1, 4):
            p = by_id[name, "P", rep]
            r = by_id[name, "R", rep]
            row = {"workload": name, "repeat": rep,
                   "status": "not_attempted", "matched_measured": 0}
            if p["status"] == "completed" and r["status"] == "completed":
                dest = root / f"{name}_r{rep}_pair"
                try:
                    from .audit import pair
                    result = pair(root / p["folder"] / "audit",
                                  root / r["folder"] / "audit", dest)
                    matched = sum(bool(x["ordered_signature_exact"]) for x in result
                                  if x["scope"].startswith("o4.iter/measure/"))
                    exact = len(result) == (
                        PROTOCOL["measured_iterations"] + PROTOCOL["transition_iterations"]
                    ) and all(x["ordered_signature_exact"] for x in result)
                    row.update(status="exact" if exact else "mismatch",
                               matched_measured=matched,
                               expected_measured=PROTOCOL["measured_iterations"],
                               exact_all_windows=exact, folder=dest.name,
                               strict_reference_verified=False)
                except Exception as e:
                    row.update(status="failed", error=f"{type(e).__name__}: {e}")
                    write(root / f"{name}_r{rep}_pair_failure.json", row)
            pairs.append(row)
            paired[name, rep] = row

        for mode in PROTOCOL["modes"]:
            values = []
            for rep in range(1, 4):
                job = by_id[name, mode, rep]
                if job["status"] == "completed":
                    try:
                        values.append(measured_median(root / job["folder"],
                                                      PROTOCOL["measured_iterations"]))
                    except Exception as e:
                        job["measurement_validation_error"] = str(e)
            spread = process_spread(values)
            groups.append({
                "workload": name, "mode": mode,
                "completed_processes_with_40_samples": len(values),
                "medians_us": values,
                "median_of_process_medians_us": (
                    statistics.median(values) if values else None
                ),
                "spread_fraction": spread,
                "repeat_stability_passed": (
                    spread is not None and spread <= PROTOCOL["repeat_spread_threshold"]
                ),
                "target_kind": ("iteration_device_event_us" if mode == "GI"
                                else "profiled_iteration_event_us"),
                "training_eligible": False,
            })
    complete = all(j["status"] == "completed" and
                   "measurement_validation_error" not in j for j in jobs)
    pairs_exact = all(p["status"] == "exact" for p in pairs)
    summary = {
        "status": "completed_engineering" if complete and pairs_exact
                  else "review_required",
        "protocol": PROTOCOL["protocol_id"], "utc_end": now(),
        "planned_cells": len(jobs),
        "completed_cells": sum(j["status"] == "completed" for j in jobs),
        "failed_cells": sum(j["status"] == "failed" for j in jobs),
        "measured_valid_iterations": (
            sum(PROTOCOL["measured_iterations"] for j in jobs
                if j["status"] == "completed"
                and "measurement_validation_error" not in j)
        ),
        "P_R_pairs_exact": sum(p["status"] == "exact" for p in pairs),
        "P_R_pairs_planned": len(pairs),
        "trace_structural_passes": sum(
            j.get("trace_structure") == "recorded_structure_complete" for j in jobs
        ),
        "NVTX_warning_captures": [
            {"workload": j["workload"], "mode": j["mode"],
             "repeat": j["repeat"], "warnings": j["NVTX_warnings"]}
            for j in jobs if j.get("NVTX_warnings")
        ],
        "unmatched_work_API_captures": [
            {"workload": j["workload"], "mode": j["mode"],
             "repeat": j["repeat"], "count": j["unmatched_work_APIs"]}
            for j in jobs if j.get("unmatched_work_APIs", 0) > 0
        ],
        "all_three_process_spreads_pass": all(g["repeat_stability_passed"] for g in groups),
        "stability_failed_groups": [
            {"workload": g["workload"], "mode": g["mode"],
             "spread_fraction": g["spread_fraction"]}
            for g in groups if not g["repeat_stability_passed"]
        ],
        "NVTX_warning_policy": "engineering_accept_only_keep_original_warning",
        "strict_reference_verified": False,
        "training_eligible": False,
        "models_trained": 0,
        "new_confirmation_dataset": False,
    }
    summary["quality_status"] = "engineering_accepted" if complete and pairs_exact and summary["all_three_process_spreads_pass"] else "review_required"
    summary["diagnostic_warning_captures"] = [{"workload": j["workload"], "mode": j["mode"], "repeat": j["repeat"], "warnings": j["diagnostic_warnings"]} for j in jobs if j.get("diagnostic_warnings")]
    write(root / "pair_summary.json", pairs)
    write(root / "aggregate.json", groups)
    write(root / "SUMMARY.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--assets-manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--workload", nargs="+", choices=sorted(SPECS), required=True)
    parser.add_argument("--expect-gpu", choices=["3090", "4090", "GB10"], required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--cell-timeout-seconds", type=int, default=3600)
    args = parser.parse_args()
    if args.cell_timeout_seconds < 0:
        parser.error("Timeout must be nonnegative; zero means no process time cap")
    root = Path(args.out).resolve()
    if root.exists():
        raise FileExistsError("Run output already exists: " + str(root))
    manifest = Path(args.assets_manifest).resolve()
    jobs = plan(args.workload, plan_only=args.plan_only)
    validate_sources(manifest, args.workload, args.expected_manifest_sha256)
    release = verify_release()
    hashes = code_hashes()
    root.mkdir(parents=True, exist_ok=False)
    inventory = {
        "utc_start": now(), "protocol": PROTOCOL, "status":
            "planned_only" if args.plan_only else "running",
        "jobs": jobs, "workloads": args.workload,
        "expected_gpu": args.expect_gpu,
        "assets_manifest": str(manifest),
        "assets_manifest_sha256": args.expected_manifest_sha256,
        "code_hashes": hashes,
        "trace_backend_requested": PROTOCOL["trace_backend"],
        "strict_verified": 0, "training_eligible": False,
        "models_trained": 0,
    }
    inventory["source_release"] = release
    write(root / "environment_snapshot.json", snapshot())
    write(root / "inventory.json", inventory)
    if args.plan_only:
        print("PLAN_ONLY:", len(jobs), "cells; GPU executions = 0")
        return 0
    if shutil.which("nsys") is None:
        inventory["status"] = "blocked_missing_nsys"
        write(root / "inventory.json", inventory)
        raise RuntimeError("Nsight Systems required for P/R modes")
    python = sys.executable
    for j in jobs:
        folder = root / (f"{j['workload']}_{j['mode']}_r{j['repeat']}")
        folder.mkdir()
        j["folder"] = folder.name
        req = {
            "workload": j["workload"], "mode": j["mode"],
            "repeat": j["repeat"], "output": str(folder),
            "expect_gpu": args.expect_gpu, "protocol": PROTOCOL,
            "assets_manifest": str(manifest),
            "assets_manifest_sha256": args.expected_manifest_sha256,
            "trace_backend": "cuda-sw",
        }
        write(folder / "request.json", req)
        j["status"] = "started"
        write(root / "inventory.json", inventory)
        try:
            if code_hashes() != hashes:
                raise RuntimeError("Collector source changed during active session")
            if sha(manifest) != args.expected_manifest_sha256:
                raise RuntimeError("Asset manifest changed during session")
            gpu_guard(folder, args.expect_gpu)
            cmd = [python, "-m", "o4_research.formal_worker",
                   "--request", str(folder / "request.json")]
            if j["mode"] != "GI":
                cmd = ["nsys", "profile", "--trace=cuda-sw,nvtx",
                       "--sample=none", "--cpuctxsw=none",
                       "--gpu-metrics-devices=none", "--discard-environment=true",
                       "--capture-range=cudaProfilerApi",
                       "--capture-range-end=stop",
                       "--output=" + str(folder / "trace")] + cmd
            env = os.environ.copy()
            env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       YOLO_CONFIG_DIR=str(root / "ultralytics_settings"),
                       TOKENIZERS_PARALLELISM="false")
            command(cmd, folder, "collect",
                    timeout=args.cell_timeout_seconds or None,
                    env=env, cwd=Path(__file__).resolve().parents[1])
            q = read(folder / "quality.json")
            if q["status"] != "completed" or not q["correctness"]["passed"]:
                raise RuntimeError("Worker/correctness incomplete")
            if j["mode"] != "GI":
                command(["nsys", "export", "--type=sqlite",
                         "--output=" + str(folder / "trace.sqlite"),
                         str(folder / "trace.nsys-rep")],
                        folder, "export", timeout=args.cell_timeout_seconds or None)
                from .audit_v5 import audit
                audit_summary = audit(folder, folder / "audit")
                j["trace_structure"] = audit_summary["status"]
                j["schema"] = audit_summary["schema"]
                j["GPU_records"] = audit_summary["GPU_records"]
                j["preparation_GPU_records"] = audit_summary["preparation_GPU_records"]
                j["diagnostic_warnings"] = audit_summary["warnings"]
                j["NVTX_warnings"] = [x["text"] for x in audit_summary["engineering_acceptance"]["accepted_warnings"]]
                j["engineering_acceptance"] = audit_summary["engineering_acceptance"]
                if not j["engineering_acceptance"]["accepted"]:
                    raise RuntimeError("Engineering acceptance blocked; see audit/engineering_acceptance.json")
                j["unmatched_work_APIs"] = audit_summary["unmatched_work_APIs"]
                if j["trace_structure"] != "recorded_structure_complete":
                    raise RuntimeError("Recorded structure integrity failed")
            j["status"] = "completed"
            j["GPU_execution_completed"] = True
        except Exception as e:
            j["status"] = "failed"
            j["error"] = f"{type(e).__name__}: {e}"
            write(folder / "failure.json", j)
        finally:
            write(root / "inventory.json", inventory)
        print(j, flush=True)
    summary = summarize(root, jobs, args.workload)
    inventory["status"] = summary["status"]
    inventory["utc_end"] = now()
    write(root / "inventory.json", inventory)
    print(summary, flush=True)
    if inventory["status"] != "completed_engineering" or summary["quality_status"] != "engineering_accepted":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
