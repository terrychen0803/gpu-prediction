"""Run the frozen O2 GB10 template/mode/repeat matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from o1_operator.full_matrix import BY_ID as FORMAL
from o1_operator.runner import command_text, run_logged
from o1_operator.spec import BY_ID as PILOT, INPUT_FORMAT, write_inputs, write_json_exclusive
from .catalog import FROZEN_PATH, PROTOCOL, SPLIT_PATH, load
from .trace import audit
from .worker import GROUPS, SEQUENCES_PER_GROUP

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [Path(__file__).with_name(x) for x in
           ("catalog.py", "worker.py", "trace.py", "runner.py", "alignment.py", "formal_templates_v1.json")]


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hash():
    digest = hashlib.sha256()
    for path in SOURCES:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def command(request_path, mode, folder):
    argv = [sys.executable, "-m", "o2_sequence", "worker", "--request", str(request_path)]
    if mode in ("P", "R"):
        argv = ["nsys", "profile", "--trace=" + ("cuda-sw,nvtx" if mode == "R" else "cuda-sw"),
                "--sample=none", "--cpuctxsw=none", "--gpu-metrics-devices=none",
                "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
                "--output=" + str(folder / "trace"), *argv]
    return argv


def execute(out_root, matrix="pilot", template_ids=None, modes=("G0", "P", "R"),
            repeats=1, expect_gpu="GB10", dry_run=False):
    out_root = Path(out_root).resolve()
    if out_root.exists():
        raise FileExistsError(f"O2 output already exists: {out_root}")
    if matrix not in ("pilot", "formal"):
        raise ValueError("O2 matrix must be pilot or formal")
    if repeats < 1 or repeats > 3 or not modes or len(modes) != len(set(modes)) or any(
            m not in ("G0", "P", "R") for m in modes):
        raise ValueError("O2 modes/repeats invalid")
    if not dry_run and any(m in ("P", "R") for m in modes) and not shutil.which("nsys"):
        raise FileNotFoundError("O2 P/R requires nsys on PATH")
    templates = load(matrix == "formal")
    if template_ids:
        unknown = set(template_ids)-{t["id"] for t in templates}
        if unknown or len(template_ids) != len(set(template_ids)):
            raise ValueError(f"Unknown/duplicate O2 templates: {sorted(unknown)}")
        templates = [t for t in templates if t["id"] in template_ids]
    if not templates:
        raise ValueError("O2 has no selected templates")
    catalog = FORMAL if matrix == "formal" else PILOT
    ids = sorted({s["config_id"] for t in templates for s in t["steps"]})
    configs = {x: catalog[x] for x in ids}
    jobs = [{"template_id": t["id"], "mode": mode, "repeat": rep}
            for t in templates for rep in range(1, repeats+1) for mode in modes]
    out_root.mkdir(parents=True)
    inputs = write_inputs(out_root / "inputs", configs.values())
    write_json_exclusive(out_root / "input_inventory.json", inputs)
    plan = {"protocol": PROTOCOL, "matrix": matrix, "expect_gpu": expect_gpu,
            "input_format": INPUT_FORMAT, "code_sha256": code_hash(),
            "catalog_sha256": hash_file(FROZEN_PATH), "o1_split_sha256": hash_file(SPLIT_PATH),
            "templates": templates, "configs": configs, "modes": list(modes),
            "repeats": repeats, "groups": GROUPS,
            "sequences_per_group": SEQUENCES_PER_GROUP,
            "jobs": jobs, "dry_run": dry_run, "status": "planned_only" if dry_run else "running",
            "execution_policy": {"dtype": "float16", "backend": "eager", "autocast": False,
                "autograd": False, "cuda_graph": False, "stream": "default",
                "inputs": "preallocated", "per_step_synchronization": False,
                "P_per_step_markers": False, "R_per_step_nvtx_markers": True,
                "G0_target": "whole_sequence_host_batch_amortized",
                "P_R_label_policy": "R reference plus exact ordered trace-signature review"},
            "preflight": {"nsys": command_text(["nsys", "--version"]),
                "nvidia_smi": command_text(["nvidia-smi", "--query-gpu=name,driver_version,pstate,temperature.gpu,power.draw,clocks.sm,utilization.gpu", "--format=csv"]),
                "compute_apps": command_text(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv"])}}
    write_json_exclusive(out_root / "plan.json", plan)
    failures = []
    by_id = {t["id"]: t for t in templates}
    for number, job in enumerate(jobs, 1):
        template_id, mode, repeat = job["template_id"], job["mode"], job["repeat"]
        folder = out_root / "runs" / template_id / mode / f"run{repeat:02d}"
        folder.mkdir(parents=True, exist_ok=False)
        request = {"protocol": PROTOCOL, "input_format": INPUT_FORMAT,
                   "matrix": matrix, "template": by_id[template_id], "configs": configs,
                   "inputs": inputs, "mode": mode, "repeat": repeat, "groups": GROUPS,
                   "sequences_per_group": SEQUENCES_PER_GROUP,
                   "output": str(folder), "expect_gpu": expect_gpu,
                   "code_sha256": plan["code_sha256"]}
        request_path = folder / "request.json"
        write_json_exclusive(request_path, request)
        argv = command(request_path, mode, folder)
        write_json_exclusive(folder / "command.json", {"argv": argv, "cwd": str(ROOT)})
        print(f"O2 {number}/{len(jobs)} {template_id} {mode} run{repeat:02d}", flush=True)
        if dry_run:
            write_json_exclusive(folder / "execution.json", {"status": "planned_only"})
            continue
        try:
            exit_code = run_logged(argv, folder / "profile.log" if mode in ("P", "R")
                                   else folder / "worker.log", timeout=600)
            quality = json.loads((folder / "quality.json").read_text()) if (folder / "quality.json").exists() else {}
            if exit_code or quality.get("status") != "completed":
                raise RuntimeError(f"O2 worker/profiler failed (exit {exit_code}); inspect run log")
            if mode in ("P", "R"):
                report = folder / "trace.nsys-rep"
                if not report.is_file():
                    raise FileNotFoundError("O2 Nsight report missing")
                export = ["nsys", "export", "--type=sqlite", "--quiet=true",
                          "--output=" + str(folder / "trace.sqlite"), str(report)]
                write_json_exclusive(folder / "export_command.json", {"argv": export})
                if run_logged(export, folder / "export.log", timeout=600):
                    raise RuntimeError("O2 Nsight SQLite export failed")
                result = audit(folder, mode)
                if result["status"] == "failed":
                    raise RuntimeError("O2 structural trace audit failed")
            write_json_exclusive(folder / "execution.json", {"status": "completed",
                "manual_review_required": mode == "R" or
                 (mode == "P" and result["status"] == "manual_review_required")})
        except Exception as exc:
            failures.append({**job, "error": f"{type(exc).__name__}: {exc}"})
            write_json_exclusive(folder / "execution.json", {"status": "failed",
                "error": failures[-1]["error"]})
            print(f"O2 ERROR {failures[-1]}", file=sys.stderr, flush=True)
    plan["status"] = "planned_only" if dry_run else "completed_with_failures" if failures else "completed"
    plan["failure_count"] = len(failures)
    (out_root / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False)+"\n")
    write_json_exclusive(out_root / "failures.json", failures)
    return len(failures)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Collect O2 mixed sequences on one GPU")
    parser.add_argument("--out", required=True)
    parser.add_argument("--matrix", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--template", action="append", dest="templates")
    parser.add_argument("--mode", action="append", dest="modes", choices=("G0", "P", "R"))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--expect-gpu", default="GB10")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    return 1 if execute(args.out, args.matrix, args.templates,
                        args.modes or ("G0", "P", "R"), args.repeats,
                        args.expect_gpu, args.dry_run) else 0
