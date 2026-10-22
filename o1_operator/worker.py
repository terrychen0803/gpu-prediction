"""One independent O1 configuration/mode/repeat GPU worker."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .calibration import FORMAL_MAX_CALLS_PER_GROUP, MAX_CALLS_PER_GROUP
from .spec import BY_ID, INPUT_FORMAT, PROTOCOL, operand_shapes
from .full_matrix import BY_ID as FORMAL_BY_ID
from .full_matrix import BOOTSTRAP_PROTOCOL, PROTOCOL as FORMAL_PROTOCOL

WARMUP_MIN_CALLS = 100
WARMUP_MIN_NS = 100_000_000
WARMUP_MAX_CALLS = 20_000
FORMAL_WARMUP_MAX_CALLS = 100_000
WARMUP_TIMEOUT_SECONDS = 45
GROUPS = 5
MIN_CALLS_PER_GROUP = 100
TARGET_GROUP_NS = 100_000_000


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def reset_cuda_context_for_trace(cuda_version):
    """Flush this worker process's CUDA trace data before cudaProfilerStop."""
    import ctypes

    major = str(cuda_version).split(".", 1)[0]
    if not major.isdecimal():
        raise RuntimeError(f"Cannot locate CUDA runtime for version {cuda_version!r}")
    try:
        runtime = ctypes.CDLL(f"libcudart.so.{major}")
    except OSError as exc:
        raise RuntimeError(f"CUDA runtime libcudart.so.{major} unavailable") from exc
    reset = runtime.cudaDeviceReset
    reset.restype = ctypes.c_int
    reset.argtypes = []
    result = int(reset())
    if result != 0:
        raise RuntimeError(f"cudaDeviceReset failed with CUDA error {result}")
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def check_input_file(path, expected):
    data = Path(path).read_bytes()
    if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
        raise ValueError(f"O1 fixed input byte count/hash mismatch: {path}")
    return data


def load_operation(torch, request):
    import torch.nn.functional as F

    catalog = BY_ID if request["protocol"] == PROTOCOL else FORMAL_BY_ID
    config = catalog[request["config_id"]]
    files = request["inputs"]
    shapes = operand_shapes(config)
    if set(files) != set(shapes):
        raise ValueError("O1 operand inventory does not match configuration")
    cpu = {}
    gpu = {}
    for name, shape in shapes.items():
        if files[name]["shape"] != shape:
            raise ValueError(f"O1 operand shape mismatch: {name}")
        data = check_input_file(files[name]["path"], files[name])
        cpu[name] = torch.frombuffer(bytearray(data), dtype=torch.float16).reshape(shape).clone().contiguous()
        gpu[name] = cpu[name].to("cuda", non_blocking=False)
    if config["op"] == "gemm":
        def op(): return torch.mm(gpu["a"], gpu["b"])
        reference = torch.mm(cpu["a"].float(), cpu["b"].float()).half()
    elif config["op"] == "bmm":
        def op(): return torch.bmm(gpu["a"], gpu["b"])
        reference = torch.bmm(cpu["a"].float(), cpu["b"].float()).half()
    elif config["op"] == "conv":
        def op():
            return F.conv2d(gpu["input"], gpu["weight"], bias=None,
                            stride=config["stride"], padding=config["padding"], groups=1)
        reference = F.conv2d(cpu["input"].float(), cpu["weight"].float(), bias=None,
                             stride=config["stride"], padding=config["padding"], groups=1).half()
    elif config["op"] == "add":
        def op(): return torch.add(gpu["a"], gpu["b"])
        reference = torch.add(cpu["a"].float(), cpu["b"].float()).half()
    elif config["op"] == "layernorm":
        shape = (config["d"],)
        def op(): return F.layer_norm(gpu["input"], shape, weight=None, bias=None, eps=config["eps"])
        reference = F.layer_norm(cpu["input"].float(), shape, weight=None,
                                 bias=None, eps=config["eps"]).half()
    else:
        raise ValueError(f"Unsupported O1 operation: {config['op']}")
    return op, reference, gpu


def correctness(torch, op, reference):
    output = op()
    torch.cuda.synchronize()
    if output.shape != reference.shape:
        raise RuntimeError("O1 output shape differs from FP32 reference")
    finite = bool(torch.isfinite(output).all().item()) and bool(torch.isfinite(reference).all().item())
    if not finite:
        raise RuntimeError("O1 output/reference contains non-finite values")
    actual = output.float().cpu()
    expected = reference.float()
    delta = actual - expected
    max_abs = float(delta.abs().max().item())
    rel_l2 = float(torch.linalg.vector_norm(delta).item() /
                   max(torch.linalg.vector_norm(expected).item(), 1e-12))
    passed = bool(torch.allclose(actual, expected, rtol=1e-2, atol=1e-2))
    result = {"passed": passed, "finite": finite, "rtol": 1e-2, "atol": 1e-2,
              "max_absolute_error": max_abs, "relative_l2_error": rel_l2,
              "output_shape": list(output.shape), "reference": "CPU FP32 op then FP16 cast"}
    if not passed:
        raise RuntimeError(f"O1 correctness failed: max_abs={max_abs:.6g}, relative_l2={rel_l2:.6g}")
    return result


def warmup(torch, op, max_calls=WARMUP_MAX_CALLS):
    started = time.perf_counter_ns()
    calls = 0
    while calls < WARMUP_MIN_CALLS or time.perf_counter_ns() - started < WARMUP_MIN_NS:
        if calls >= max_calls or (time.perf_counter_ns() - started) / 1e9 > WARMUP_TIMEOUT_SECONDS:
            raise RuntimeError(f"O1 warm-up reached cap ({calls}/{max_calls} calls or "
                               f"{WARMUP_TIMEOUT_SECONDS}s) before 100 calls and 100 ms; preserve failed run")
        step = min(100, max_calls - calls)
        for _ in range(step):
            result = op()
            del result
        calls += step
        torch.cuda.synchronize()
    elapsed = time.perf_counter_ns() - started
    return {"calls": calls, "host_elapsed_ns": elapsed, "utc_end": utc_now()}


def run_worker(request_path):
    request = json.loads(Path(request_path).read_text())
    output = Path(request["output"])
    if request["protocol"] not in (PROTOCOL, BOOTSTRAP_PROTOCOL, FORMAL_PROTOCOL) or request["input_format"] != INPUT_FORMAT:
        raise ValueError("O1 worker request protocol/input format mismatch")
    catalog = BY_ID if request["protocol"] == PROTOCOL else FORMAL_BY_ID
    if request["config_id"] not in catalog or request["mode"] not in ("G0", "GE", "P", "R"):
        raise ValueError("O1 worker config or mode is invalid")
    status = {"status": "started", "errors": [], "trace_integrity": "not_applicable"}
    write_json(output / "quality.json", status)
    group_rows = []
    call_rows = []
    capturing = False
    try:
        import torch
        from inference_pilot.benchmark import precision_policy

        if not torch.cuda.is_available():
            raise RuntimeError("O1 requires a CUDA-visible host execution; CPU fallback is prohibited")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("O1 pilot requires exactly one CUDA-visible device")
        torch.cuda.set_device(0)
        torch.set_num_threads(1)
        policy = precision_policy(torch)
        props = torch.cuda.get_device_properties(0)
        if request["expect_gpu"].lower() not in props.name.lower():
            raise RuntimeError(f"Unexpected GPU {props.name!r}; expected {request['expect_gpu']!r}")
        env = {"utc_start": utc_now(), "machine": platform.machine(), "platform": platform.platform(),
               "python": sys.version.split()[0], "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
               "cudnn": torch.backends.cudnn.version(), "gpu_name": props.name,
               "gpu_uuid": str(getattr(props, "uuid", "unavailable")),
               "sm_count": props.multi_processor_count, "driver": request.get("driver"),
               "nsys": request.get("nsys"), "precision_policy": policy,
               "pid": os.getpid(), "stream_policy": "default_stream_one_visible_device"}
        write_json(output / "environment.json", env)
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            op, reference, operands = load_operation(torch, request)
            status["correctness"] = correctness(torch, op, reference)
            warm = warmup(torch, op,
                          FORMAL_WARMUP_MAX_CALLS if request["protocol"] != PROTOCOL
                          else WARMUP_MAX_CALLS)
            status["warmup"] = warm
            reps = request["measurement_repetitions"]
            max_reps = MAX_CALLS_PER_GROUP if request["protocol"] == PROTOCOL else FORMAL_MAX_CALLS_PER_GROUP
            if not isinstance(reps, int) or not (MIN_CALLS_PER_GROUP <= reps <= max_reps):
                raise ValueError(f"O1 measurement_repetitions must be a fixed integer in [100,{max_reps}]")
            status["repetitions_per_group"] = reps
            if request["mode"] in ("P", "R"):
                torch.cuda.profiler.start()
                capturing = True
            for group in range(1, GROUPS + 1):
                torch.cuda.synchronize()
                utc_start = utc_now()
                start_ns = time.perf_counter_ns()
                single_latencies = []
                if request["mode"] == "G0":
                    for index in range(reps):
                        t0 = time.perf_counter_ns()
                        result = op()
                        torch.cuda.synchronize()
                        t1 = time.perf_counter_ns()
                        del result
                        single_latencies.append((t1 - t0) / 1000)
                        call_rows.append((group, index, t0, t1, (t1 - t0) / 1000, "host_single_call"))
                elif request["mode"] == "GE":
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    for index in range(reps):
                        t0 = time.perf_counter_ns()
                        begin.record()
                        result = op()
                        end.record()
                        end.synchronize()
                        t1 = time.perf_counter_ns()
                        del result
                        latency = begin.elapsed_time(end) * 1000
                        single_latencies.append(latency)
                        call_rows.append((group, index, t0, t1, latency, "device_event_single_call"))
                else:
                    for index in range(reps):
                        if request["mode"] == "R":
                            torch.cuda.nvtx.range_push(f"o1.op/g{group:02d}/i{index:06d}")
                        try:
                            result = op()
                        finally:
                            if request["mode"] == "R":
                                torch.cuda.nvtx.range_pop()
                        del result
                    torch.cuda.synchronize()
                end_ns = time.perf_counter_ns()
                batch_event_us = None
                if request["mode"] == "GE":
                    # Auxiliary batch-amortized event label; never mixed into single-call main label.
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(reps):
                        result = op()
                        del result
                    end.record()
                    end.synchronize()
                    batch_event_us = begin.elapsed_time(end) * 1000 / reps
                group_rows.append({"group": group, "mode": request["mode"], "repetitions": reps,
                                   "utc_start": utc_start, "utc_end": utc_now(),
                                   "monotonic_start_ns": start_ns, "monotonic_end_ns": end_ns,
                                   "host_group_us": (end_ns - start_ns) / 1000,
                                   "main_single_call_median_us": statistics.median(single_latencies) if single_latencies else "",
                                   "batch_event_us_per_call": batch_event_us if batch_event_us is not None else "",
                                   "group_target_100ms_met": end_ns - start_ns >= TARGET_GROUP_NS})
            if capturing:
                # Nsight on GB10 needs time to drain after the final synchronization.
                time.sleep(1.0)
                status["profile_finalization"] = {
                    "scope": "worker_process_cuda_context_only",
                    "cuda_device_reset_before_profiler_stop": True,
                    "cuda_device_reset_code": reset_cuda_context_for_trace(torch.version.cuda),
                    "utc": utc_now(),
                }
                torch.cuda.profiler.stop()
                capturing = False
            status.update({"status": "completed", "group_count": len(group_rows),
                           "main_target_kind": "request_host_service_us" if request["mode"] == "G0" else
                           "isolated_device_event_us" if request["mode"] == "GE" else "trace_only",
                           "trace_integrity": "pending_sqlite_review" if request["mode"] in ("P", "R") else "not_applicable"})
            env["utc_end"] = utc_now()
            write_json(output / "environment.json", env)
            del operands
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
            writer = csv.DictWriter(stream, fieldnames=("group", "mode", "repetitions", "utc_start", "utc_end",
                                                           "monotonic_start_ns", "monotonic_end_ns", "host_group_us",
                                                           "main_single_call_median_us", "batch_event_us_per_call",
                                                           "group_target_100ms_met"))
            writer.writeheader(); writer.writerows(group_rows)
        with (output / "call_samples.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("group", "call_index", "monotonic_start_ns", "monotonic_end_ns", "latency_us", "target_kind"))
            writer.writerows(call_rows)
        write_json(output / "quality.json", status)
    return 0 if status["status"] == "completed" else 1

