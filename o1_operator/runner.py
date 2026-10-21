"""Project-scoped O1 pilot runner; every cell uses a fresh process and directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from .spec import BY_ID, CONFIGS, INPUT_FORMAT, MODES, PROTOCOL, write_inputs, write_json_exclusive
from .worker import MAX_CALLS_PER_GROUP

ROOT = Path(__file__).resolve().parent.parent
SOURCES = (Path(__file__).with_name("spec.py"), Path(__file__).with_name("worker.py"),
           Path(__file__).with_name("runner.py"), Path(__file__).with_name("trace_audit.py"),
           Path(__file__).with_name("summarize.py"), ROOT / "inference_pilot" / "benchmark.py")


def code_hash():
    h = hashlib.sha256()
    for path in SOURCES:
        h.update(str(path.relative_to(ROOT)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def command_text(argv):
    try:
        done = subprocess.run(argv, text=True, capture_output=True, timeout=20, check=False)
        return {"argv": argv, "returncode": done.returncode, "stdout": done.stdout.strip(),
                "stderr": done.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "error": f"{type(exc).__name__}: {exc}"}


def run_logged(argv, log_path, timeout=240):
    with Path(log_path).open("x") as stream:
        process = subprocess.Popen(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise TimeoutError(f"Command exceeded {timeout}s; stopped its process group; see {log_path}")


def worker_command(request_path, mode, out):
    argv = [sys.executable, "-m", "o1_operator", "worker", "--request", str(request_path)]
    if mode in ("P", "R"):
        trace = "cuda-sw,nvtx" if mode == "R" else "cuda-sw"
        argv = ["nsys", "profile", f"--trace={trace}", "--sample=none", "--cpuctxsw=none",
                "--gpu-metrics-devices=none", "--capture-range=cudaProfilerApi",
                "--capture-range-end=stop", "--output=" + str(out / "trace"), *argv]
    return argv


def selected_jobs(config_ids, modes, repeats, order_seed=42):
    if len(config_ids) != len(set(config_ids)) or any(c not in BY_ID for c in config_ids):
        raise ValueError("Config IDs must be unique O1 pilot IDs")
    if len(modes) != len(set(modes)) or any(m not in MODES for m in modes):
        raise ValueError("Modes must be unique among G0 GE P R")
    if repeats < 1 or repeats > 3:
        raise ValueError("O1 pilot repeats must be 1 to 3")
    rng = random.Random(order_seed)
    jobs = []
    for repeat in range(1, repeats + 1):
        block = [(c, m, repeat) for c in config_ids for m in modes]
        rng.shuffle(block)
        jobs.extend(block)
    return jobs


def execute(out_root, config_ids, modes, repeats, expect_gpu="GB10", dry_run=False):
    out_root = Path(out_root).resolve()
    if out_root.exists():
        raise FileExistsError(f"O1 output directory already exists; use a new path: {out_root}")
    jobs = selected_jobs(config_ids, modes, repeats)
    if not jobs:
        raise ValueError("No O1 jobs selected")
    if not dry_run and any(m in ("P", "R") for m in modes) and not shutil.which("nsys"):
        raise FileNotFoundError("Nsight Systems CLI nsys is required for P/R")
    out_root.mkdir(parents=True)
    configs = [BY_ID[c] for c in config_ids]
    inputs = write_inputs(out_root / "inputs", configs)
    write_json_exclusive(out_root / "input_inventory.json", inputs)
    plan = {"protocol": PROTOCOL, "input_format": INPUT_FORMAT, "code_sha256": code_hash(),
            "configs": configs, "modes": list(modes), "repeats": repeats,
            "order_seed": 42, "jobs": [{"config_id": c, "mode": m, "repeat": r} for c,m,r in jobs],
            "dry_run": dry_run, "status": "planned_only" if dry_run else "running",
            "timing_policy": {"warmup_min_calls": 100, "warmup_min_ms": 100,
                              "groups": 5, "min_calls_per_group": 100, "target_group_ms": 100,
                              "max_calls_per_group": MAX_CALLS_PER_GROUP,
                              "fixed_calls_per_group_all_modes": MAX_CALLS_PER_GROUP,
                              "GE_main": "single_call_event",
                              "GE_auxiliary": "batch_amortized_event"},
            "execution_policy": {"dtype": "float16", "backend": "eager", "autocast": False,
                                 "autograd": False, "cuda_graph": False, "stream": "default",
                                 "operator_output": "out_of_place", "conv_layout": "NCHW_contiguous",
                                 "conv_bias": False, "conv_groups": 1},
            "environment_preflight": {"nsys": command_text(["nsys", "--version"]),
                                      "nvidia_smi": command_text(["nvidia-smi", "--query-gpu=name,driver_version,pstate,temperature.gpu,power.draw,clocks.sm,utilization.gpu", "--format=csv"]),
                                      "compute_apps": command_text(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv"])} }
    write_json_exclusive(out_root / "plan.json", plan)
    failures = []
    for index, (config_id, mode, repeat) in enumerate(jobs, 1):
        run = out_root / "runs" / config_id / mode / f"run{repeat:02d}"
        run.mkdir(parents=True, exist_ok=False)
        request = {"protocol": PROTOCOL, "input_format": INPUT_FORMAT,
                   "config_id": config_id, "mode": mode, "repeat": repeat,
                   "output": str(run), "inputs": inputs[config_id], "expect_gpu": expect_gpu,
                   "measurement_repetitions": MAX_CALLS_PER_GROUP,
                   "code_sha256": plan["code_sha256"],
                   "driver": plan["environment_preflight"]["nvidia_smi"].get("stdout"),
                   "nsys": plan["environment_preflight"]["nsys"].get("stdout")}
        request_path = run / "request.json"
        write_json_exclusive(request_path, request)
        argv = worker_command(request_path, mode, run)
        write_json_exclusive(run / "command.json", {"argv": argv, "cwd": str(ROOT), "dry_run": dry_run})
        print(f"O1 {index}/{len(jobs)} {config_id} {mode} repeat={repeat}", flush=True)
        if dry_run:
            write_json_exclusive(run / "execution.json", {"status": "planned_only", "gpu_executed": False})
            continue
        try:
            exit_code = run_logged(argv, run / ("profile.log" if mode in ("P", "R") else "worker.log"))
            worker_quality = json.loads((run / "quality.json").read_text()) if (run / "quality.json").exists() else {}
            if exit_code or worker_quality.get("status") != "completed":
                raise RuntimeError(f"O1 worker/profiler exit={exit_code}; inspect log and worker_error.txt")
            if mode in ("P", "R"):
                report = run / "trace.nsys-rep"
                if not report.is_file():
                    raise FileNotFoundError("Nsight did not produce trace.nsys-rep")
                export = ["nsys", "export", "--type=sqlite", "--quiet=true",
                          "--output=" + str(run / "trace.sqlite"), str(report)]
                write_json_exclusive(run / "export_command.json", {"argv": export})
                if run_logged(export, run / "export.log"):
                    raise RuntimeError("Nsight SQLite export failed; inspect export.log")
                from .trace_audit import audit
                trace_result = audit(run, mode)
                if trace_result["status"] == "failed":
                    raise RuntimeError("O1 Nsight trace structural audit failed; inspect trace_audit.json")
            write_json_exclusive(run / "execution.json", {"status": "completed", "gpu_executed": True,
                                                       "manual_trace_review_required": mode in ("P", "R")})
        except Exception as exc:
            failures.append({"config_id": config_id, "mode": mode, "repeat": repeat,
                             "error": f"{type(exc).__name__}: {exc}"})
            write_json_exclusive(run / "execution.json", {"status": "failed", "gpu_executed": True,
                                                       "error": failures[-1]["error"]})
            print(f"O1 failed {config_id}/{mode}/run{repeat:02d}: {exc}", file=sys.stderr, flush=True)
    plan["status"] = "planned_only" if dry_run else "completed_with_failures" if failures else "completed"
    plan["failures"] = failures
    (out_root / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    return 0 if not failures else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="O1 nine-configuration controlled operator pilot")
    parser.add_argument("--out", required=True)
    parser.add_argument("--configs", nargs="+", choices=list(BY_ID), default=list(BY_ID))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--expect-gpu", default="GB10")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    return execute(args.out, args.configs, args.modes, args.repeats, args.expect_gpu, args.dry_run)

