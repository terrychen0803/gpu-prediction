"""One O2 template/mode/repeat in a fresh CUDA process."""
from __future__ import annotations

import csv
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from o1_operator.full_matrix import PROTOCOL as O1_FORMAL
from o1_operator.spec import INPUT_FORMAT, PROTOCOL as O1_PILOT
from o1_operator.worker import correctness, load_operation, reset_cuda_context_for_trace
from .catalog import PROTOCOL

GROUPS = 5
SEQUENCES_PER_GROUP = 8


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def build_steps(torch, request):
    import torch.nn.functional as F

    template = request["template"]
    configs = request["configs"]
    loaded = {}
    for config_id in dict.fromkeys(s["config_id"] for s in template["steps"]):
        sub = {"protocol": O1_FORMAL if request["matrix"] == "formal" else O1_PILOT,
               "config_id": config_id, "inputs": request["inputs"][config_id]}
        op, ref, operands = load_operation(torch, sub)
        correctness(torch, op, ref)
        loaded[config_id] = (op, operands, ref)
    operations = []
    for index, step in enumerate(template["steps"]):
        config_id = step["config_id"]
        config = configs[config_id]
        independent, operands, _ = loaded[config_id]
        if step["input"] == "preallocated":
            operations.append(independent)
        elif config["op"] == "gemm":
            right = operands["b"]
            operations.append(lambda previous, right=right: torch.mm(previous, right))
        elif config["op"] == "bmm":
            right = operands["b"]
            operations.append(lambda previous, right=right: torch.bmm(previous, right))
        else:
            raise ValueError(f"Unsupported O2 dependent step: {index}")
    return operations, loaded


def check_dependent_tail(torch, request, operations, loaded):
    """Compare the tensor-dependent tail with repeated CPU FP32 then FP16 casts."""
    steps = request["template"]["steps"]
    first = next((i for i, step in enumerate(steps)
                  if step["input"] == "previous_output"), None)
    if first is None:
        return {"status": "not_applicable"}
    chain_id = steps[first-1]["config_id"]
    config = request["configs"][chain_id]
    expected = loaded[chain_id][2]
    right = loaded[chain_id][1]["b"].cpu()
    for step in steps[first:]:
        if step["config_id"] != chain_id:
            raise ValueError("O2 dependent tail changes config unexpectedly")
        if config["op"] == "gemm":
            expected = torch.mm(expected.float(), right.float()).half()
        else:
            expected = torch.bmm(expected.float(), right.float()).half()
    actual = execute_sequence(torch, {**request, "mode": "P"}, operations, 0, 0)
    torch.cuda.synchronize()
    actual = actual.float().cpu()
    expected = expected.float()
    if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(expected).all()):
        raise RuntimeError("O2 dependent tail has non-finite GPU or CPU values")
    delta = actual-expected
    max_abs = float(delta.abs().max())
    relative_l2 = float(torch.linalg.vector_norm(delta) /
                        torch.linalg.vector_norm(expected).clamp_min(1e-12))
    scale = max(float(expected.abs().max()), 1.0)
    max_scaled = max_abs / scale
    # Long FP16 chains can differ at cancellation points even when the
    # aggregate numerical error is tiny. Check both norm and worst-case
    # magnitude against the output scale; record the actual values.
    if relative_l2 > 1e-2 or max_scaled > 1e-2:
        raise RuntimeError(f"O2 dependent tail accuracy failed: relative_l2={relative_l2:.6g}, "
                           f"max_abs_over_output_scale={max_scaled:.6g}")
    return {"status": "passed", "reference": "CPU FP32 op then FP16 cast at each edge",
            "relative_l2_error": relative_l2, "max_absolute_error": max_abs,
            "max_abs_over_output_scale": max_scaled,
            "threshold_relative_l2": 1e-2, "threshold_max_abs_over_output_scale": 1e-2}


def execute_sequence(torch, request, operations, group, sequence):
    previous = None
    for index, (step, op) in enumerate(zip(request["template"]["steps"], operations)):
        if request["mode"] == "R":
            torch.cuda.nvtx.range_push(f"o2.op/g{group:02d}/i{sequence:03d}/s{index:02d}")
        try:
            previous = op(previous) if step["input"] == "previous_output" else op()
        finally:
            if request["mode"] == "R":
                torch.cuda.nvtx.range_pop()
    return previous


def run(request_path):
    request = json.loads(Path(request_path).read_text())
    output = Path(request["output"])
    if request["protocol"] != PROTOCOL or request["input_format"] != INPUT_FORMAT:
        raise ValueError("O2 request protocol/input format mismatch")
    if request["mode"] not in ("G0", "P", "R"):
        raise ValueError("O2 invalid mode")
    if request["groups"] != GROUPS or request["sequences_per_group"] != SEQUENCES_PER_GROUP:
        raise ValueError("O2 measurement count mismatch")
    status = {"status": "started", "errors": [], "utc_start": now()}
    write(output / "quality.json", status)
    rows = []
    capturing = False
    try:
        import torch
        from inference_pilot.benchmark import precision_policy

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("O2 needs exactly one CUDA-visible GPU")
        torch.cuda.set_device(0)
        torch.set_num_threads(1)
        policy = precision_policy(torch)
        props = torch.cuda.get_device_properties(0)
        if request["expect_gpu"].casefold() not in props.name.casefold():
            raise RuntimeError(f"O2 expected {request['expect_gpu']}, found {props.name}")
        write(output / "environment.json", {"utc_start": now(), "pid": os.getpid(),
              "machine": platform.machine(), "python": sys.version.split()[0],
              "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "cudnn": torch.backends.cudnn.version(), "gpu_name": props.name,
              "gpu_uuid": str(getattr(props, "uuid", "unavailable")),
              "precision_policy": policy, "stream_policy": "default_one_stream"})
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            ops, loaded = build_steps(torch, request)
            status["dependent_correctness"] = check_dependent_tail(torch, request, ops, loaded)
            # Warm up the complete topology before capture. There are no
            # per-step events or synchronization in the measurement loop.
            for _ in range(10):
                out = execute_sequence(torch, {**request, "mode": "P"}, ops, 0, 0)
                del out
            torch.cuda.synchronize()
            if request["mode"] in ("P", "R"):
                torch.cuda.profiler.start()
                capturing = True
            for group in range(1, GROUPS + 1):
                torch.cuda.synchronize()
                utc_start = now()
                start = time.perf_counter_ns()
                for sequence in range(SEQUENCES_PER_GROUP):
                    out = execute_sequence(torch, request, ops, group, sequence)
                    del out
                torch.cuda.synchronize()
                end = time.perf_counter_ns()
                rows.append({"group": group, "utc_start": utc_start, "utc_end": now(),
                             "monotonic_start_ns": start, "monotonic_end_ns": end,
                             "host_sequence_batch_us": (end-start)/1000,
                             "sequence_count": SEQUENCES_PER_GROUP,
                             "host_us_per_sequence": (end-start)/1000/SEQUENCES_PER_GROUP})
            if capturing:
                time.sleep(1.0)
                status["profile_finalization"] = {
                    "cuda_device_reset_before_profiler_stop": True,
                    "cuda_device_reset_code": reset_cuda_context_for_trace(torch.version.cuda)}
                torch.cuda.profiler.stop()
                capturing = False
            status.update({"status": "completed", "group_count": GROUPS,
                           "sequences_per_group": SEQUENCES_PER_GROUP,
                           "steps_per_sequence": len(ops),
                           "expected_scopes": GROUPS * SEQUENCES_PER_GROUP * len(ops),
                           "utc_end": now(), "correctness": "independent_steps_passed",
                           "trace_integrity": "pending_sqlite_review" if request["mode"] != "G0" else "not_applicable"})
            del loaded
    except Exception as exc:
        status["status"] = "failed"
        status["errors"].append(f"{type(exc).__name__}: {exc}")
        (output / "worker_error.txt").write_text(traceback.format_exc())
        print(status["errors"][-1], file=sys.stderr)
    finally:
        if capturing:
            try:
                torch.cuda.profiler.stop()
            except Exception:
                pass
        with (output / "groups.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("group", "utc_start", "utc_end",
                "monotonic_start_ns", "monotonic_end_ns", "host_sequence_batch_us",
                "sequence_count", "host_us_per_sequence"))
            writer.writeheader()
            writer.writerows(rows)
        write(output / "quality.json", status)
    return 0 if status["status"] == "completed" else 1
