from __future__ import annotations

import importlib.util
import os
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path

from .common import MODES, command_info, load_settings, read_json, write_json
from .trace import parse_trace

ROOT = Path(__file__).resolve().parent.parent


def doctor(out=None):
    info = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "nsys_path": shutil.which("nsys"), "nsys_version": command_info(["nsys", "--version"]),
            "nvidia_smi": command_info(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv"])}
    try:
        import torch
        import torchvision
        info.update({"torch": torch.__version__, "torchvision": torchvision.__version__, "cuda_runtime": torch.version.cuda,
                     "cuda_available": torch.cuda.is_available(),
                     "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
    except Exception as exc:
        info["pytorch_error"] = f"{type(exc).__name__}: {exc}"
    info["ready_for_G_A"] = bool(info.get("cuda_available"))
    info["ready_for_P0_P1"] = info["ready_for_G_A"] and bool(info["nsys_path"])
    if out:
        write_json(out, info)
    return info


def build_command(request, request_path, nsys="nsys"):
    command = [sys.executable, "-m", "inference_pilot", "_worker", "--request", str(request_path)]
    if request["mode"] in ("P0", "P1"):
        backend = request.get("nsys_cuda_trace", "cuda")
        if backend not in ("cuda", "cuda-sw"):
            raise ValueError(f"Unsupported Nsight CUDA trace backend: {backend}")
        trace = f"{backend},nvtx" if request["mode"] == "P1" else backend
        command = [nsys, "profile", f"--trace={trace}", "--sample=none", "--cpuctxsw=none",
                   "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
                   "--output=" + str(Path(request["out"]) / "trace"), *command]
    return command


def execute_logged(command, logfile):
    with Path(logfile).open("w", encoding="utf-8") as stream:
        return subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, shell=False).returncode


def run_one(request, dry_run=False, nsys="nsys"):
    out = Path(request["out"]).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a nonempty run directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    request = dict(request, out=str(out))
    request["weights"] = str(Path(request["weights"]).resolve())
    if not dry_run:
        if not Path(request["weights"]).is_file():
            raise FileNotFoundError("Prepare the weights first with the prepare command")
        if not Path(request["weights"] + ".json").is_file():
            raise FileNotFoundError("Missing weights metadata sidecar; copy both prepared files")
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torchvision") is None:
            raise RuntimeError("Install the GPU-compatible PyTorch and TorchVision pair in this Python environment")
        if request["mode"] in ("P0", "P1") and not shutil.which(nsys):
            raise RuntimeError(f"Nsight Systems CLI not found: {nsys}")
        request["nsys_version"] = command_info([nsys, "--version"])
    path = out / "request.json"
    write_json(path, request)
    command = build_command(request, path, nsys)
    write_json(out / "command.json", {"argv": command, "cwd": str(ROOT), "dry_run": dry_run})
    if dry_run:
        write_json(out / "execution.json", {"status": "planned_only", "gpu_executed": False})
        return True
    try:
        code = execute_logged(command, out / ("profile.log" if request["mode"].startswith("P") else "worker.log"))
        quality = read_json(out / "quality.json") if (out / "quality.json").exists() else {}
        if code != 0 or quality.get("status") != "passed":
            raise RuntimeError(f"Worker/profiler failed (exit={code}); inspect worker_error.txt and logs")
        if request["mode"].startswith("P"):
            report = out / "trace.nsys-rep"
            if not report.is_file():
                raise RuntimeError("Profiler exited without trace.nsys-rep")
            export_command = [nsys, "export", "--type=sqlite", "--output=" + str(out / "trace.sqlite"), str(report)]
            write_json(out / "export_command.json", {"argv": export_command})
            if execute_logged(export_command, out / "export.log") != 0:
                raise RuntimeError("SQLite export failed; inspect export.log")
            tq = parse_trace(out)
            if tq["status"] != "passed":
                raise RuntimeError("Trace parsing failed: " + "; ".join(tq["errors"]))
        write_json(out / "execution.json", {"status": "completed", "gpu_executed": True,
                                             "manual_trace_review_required": request["mode"].startswith("P")})
        return True
    except Exception as exc:
        write_json(out / "execution.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        print(f"FAILED {out}: {exc}", file=sys.stderr)
        return False


def make_suite(args):
    root = Path(args.out).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use a new session output directory; results are never silently overwritten")
    if not args.label or Path(args.label).name != args.label or any(c in args.label for c in '/\\:') or args.label in ('.', '..'):
        raise ValueError("label must be a simple directory name, for example rtx4090")
    if len(set(args.modes)) != len(args.modes) or len(set(args.batches)) != len(args.batches):
        raise ValueError("Duplicate modes/batches are not allowed")
    settings = load_settings(args.settings)
    rng = random.Random(args.order_seed)
    requests = []
    for repeat in range(args.repeats):
        jobs = [(b, m) for b in args.batches for m in args.modes]
        rng.shuffle(jobs)
        for batch, mode in jobs:
            requests.append({"mode": mode, "batch": batch, "device": args.device, "label": args.label,
                             "expect_gpu": args.expect_gpu, "weights": str(Path(args.weights).resolve()), "settings": settings,
                             "nsys_cuda_trace": getattr(args, "nsys_cuda_trace", "cuda"),
                             "repeat": repeat + 1, "order_index": len(requests),
                             "out": str(root / args.label / f"b{batch:02d}" / mode / f"run{repeat + 1:02d}")})
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "schedule.json", {"order_seed": args.order_seed, "dry_run": args.dry_run, "requests": requests})
    return requests
