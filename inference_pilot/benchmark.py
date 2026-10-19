"""CUDA worker. Torch is deliberately imported only inside GPU entry points."""
from __future__ import annotations

import contextlib
import hashlib
import os
import platform
import statistics
import sys
import threading
import time
import traceback
from pathlib import Path

from .common import code_hash, command_info, read_json, sha256, stats, write_csv, write_json


def prepare_weights(path):
    import torch
    import torchvision
    from torchvision.models import ResNet50_Weights, resnet50
    path = Path(path).resolve()
    if path.exists() or path.with_suffix(path.suffix + ".json").exists():
        raise FileExistsError(f"Refusing to overwrite weights: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    torch.save(model.state_dict(), path)
    write_json(path.with_suffix(path.suffix + ".json"), {
        "architecture": "torchvision.resnet50", "weights": "IMAGENET1K_V2",
        "sha256": sha256(path), "torch": torch.__version__, "torchvision": torchvision.__version__,
    })
    print(f"Prepared {path}")


def phased_forward(torch, model, x, annotate):
    def phase(name):
        return torch.cuda.nvtx.range("pilot.phase/" + name) if annotate else contextlib.nullcontext()
    # The same decomposition runs in every mode; no synchronization inside a phase.
    with phase("stem"):
        x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
    with phase("layer1"):
        x = model.layer1(x)
    with phase("layer2"):
        x = model.layer2(x)
    with phase("layer3"):
        x = model.layer3(x)
    with phase("layer4"):
        x = model.layer4(x)
    with phase("head"):
        x = model.fc(torch.flatten(model.avgpool(x), 1))
    return x


def precision_policy(torch):
    # Do not mix the old allow_tf32 API with the newer fp32_precision API.
    if hasattr(torch.backends, "fp32_precision"):
        torch.backends.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
        policy = "fp32_precision=ieee"
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        policy = "allow_tf32=False"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    return {"tf32": policy, "cudnn_benchmark": False, "cudnn_deterministic": False,
            "fp16_reduced_precision_reduction": False, "autocast": False}


def gpu_snapshot():
    # All visible physical GPUs, kept as raw diagnostics; CUDA device index is recorded separately.
    return command_info(["nvidia-smi", "--query-gpu=uuid,name,driver_version,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu,memory.used", "--format=csv"])


def worker(request_path):
    request = read_json(request_path)
    out = Path(request["out"])
    quality = {"status": "started", "errors": [], "trace_integrity": "not_checked"}
    write_json(out / "quality.json", quality)
    latency, warmup = [], []
    capturing = False
    try:
        import torch
        import torchvision
        from torchvision.models import resnet50
        if not torch.cuda.is_available():
            raise RuntimeError("A working CUDA PyTorch installation is required; no CPU timing fallback is used.")
        device = torch.device(request["device"])
        if device.type != "cuda":
            raise ValueError("Only CUDA devices are supported")
        torch.cuda.set_device(device)
        device = torch.device("cuda", torch.cuda.current_device())
        props = torch.cuda.get_device_properties(device)
        if request.get("expect_gpu") and request["expect_gpu"].lower() not in props.name.lower():
            raise RuntimeError(f"GPU name {props.name!r} does not match --expect-gpu {request['expect_gpu']!r}")
        cfg = request["settings"]
        torch.set_num_threads(cfg["cpu_threads"])
        torch.set_num_interop_threads(1)
        torch.manual_seed(cfg["seed"])
        policy = precision_policy(torch)
        weights = Path(request["weights"])
        sidecar = read_json(weights.with_suffix(weights.suffix + ".json"))
        checksum = sha256(weights)
        if sidecar.get("weights") != "IMAGENET1K_V2" or sidecar.get("sha256") != checksum:
            raise ValueError("Weights checksum/version does not match the prepare sidecar")
        model = resnet50(weights=None)
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
        model.eval().to(device=device, dtype=torch.float16)
        generator = torch.Generator(device="cpu").manual_seed(cfg["seed"])
        cpu_input = torch.randn(request["batch"], 3, 224, 224, generator=generator).to(torch.float16).contiguous()
        input_checksum = hashlib.sha256(cpu_input.numpy().tobytes()).hexdigest()
        x = cpu_input.to(device)
        environment = {
            "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "cpu": platform.processor(), "cpu_threads": cfg["cpu_threads"], "pid": os.getpid(),
            "native_thread_id": threading.get_native_id(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "device": str(device), "gpu_name": props.name,
            "compute_capability": [props.major, props.minor], "sm_count": props.multi_processor_count,
            "total_memory_bytes": props.total_memory, "gpu_uuid": str(getattr(props, "uuid", "unavailable")),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "precision_policy": policy,
            "nsys_version": request.get("nsys_version"), "gpu_snapshot_before": gpu_snapshot(),
        }
        write_json(out / "environment.json", environment)
        manifest = dict(request)
        manifest.update({"schema_version": 1, "model": "torchvision.resnet50", "weights_name": "IMAGENET1K_V2",
                         "weights_sha256": checksum, "input_sha256": input_checksum, "shape": list(x.shape),
                         "dtype": "float16", "layout": "NCHW_contiguous", "code_sha256": code_hash(),
                         "backend": "eager", "cuda_graph": False, "queue_depth": 1,
                         "timer_scope": "host_forward_plus_device_synchronize", "annotation_timer": "push_and_pop_included",
                         "forward_path": "same_six_phase_decomposition_all_modes", "stream_policy": "default",
                         "precision_policy": policy})
        write_json(out / "manifest.json", manifest)
        annotate = request["mode"] in ("A", "P1")
        profile = request["mode"] in ("P0", "P1")

        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            reference = model(x).float()
            plain = phased_forward(torch, model, x, False).float()
            marked = phased_forward(torch, model, x, True).float()
            torch.cuda.synchronize(device)
            finite = all(bool(torch.isfinite(v).all()) for v in (reference, plain, marked))
            correct = finite and all(v.shape == (request["batch"], 1000) for v in (reference, plain, marked))
            correct = correct and all(torch.allclose(reference, v, rtol=cfg["rtol"], atol=cfg["atol"]) for v in (plain, marked))
            diffs = [(reference - v).abs() for v in (plain, marked)]
            quality["correctness"] = {"passed": bool(correct), "rtol": cfg["rtol"], "atol": cfg["atol"],
                                      "finite": finite,
                                      "max_absolute_difference": max(float(v.max()) for v in diffs) if finite else None,
                                      "max_relative_difference_denominator_floor_1e_minus6": max(float((v / reference.abs().clamp_min(1e-6)).max()) for v in diffs) if finite else None}
            if not correct:
                raise RuntimeError("Native and phased ResNet50 outputs failed correctness validation")
            del reference, plain, marked, diffs
            torch.cuda.synchronize(device)

            def one(stage, index):
                label = f"pilot.iter/{stage}/{index:06d}"
                t0 = time.perf_counter_ns()
                if annotate:
                    torch.cuda.nvtx.range_push(label)
                try:
                    result = phased_forward(torch, model, x, annotate)
                    torch.cuda.synchronize(device)
                finally:
                    if annotate:
                        torch.cuda.nvtx.range_pop()
                t1 = time.perf_counter_ns()
                # Keep tensor destruction and row creation outside measured interval.
                del result
                return {"stage": stage, "iteration": index, "host_start_ns": t0, "host_end_ns": t1,
                        "latency_us": (t1 - t0) / 1000}

            for index in range(cfg["warmup_min"]):
                warmup.append(one("warmup", index))
            windows = []
            while len(warmup) + cfg["warmup_window"] <= cfg["warmup_max"]:
                start = len(warmup)
                for index in range(start, start + cfg["warmup_window"]):
                    warmup.append(one("warmup", index))
                windows.append(statistics.median(v["latency_us"] for v in warmup[-cfg["warmup_window"]:]))
                if len(windows) >= 3 and all(abs(b / a - 1) < cfg["stability_threshold"] for a, b in zip(windows[-3:-1], windows[-2:])):
                    break
            else:
                raise RuntimeError("Warm-up did not reach the preregistered stability threshold")
            quality["warmup"] = {"passed": True, "iterations": len(warmup), "window_medians_us": windows}
            torch.cuda.reset_peak_memory_stats(device)
            if profile:
                torch.cuda.profiler.start()
                capturing = True
            for index in range(cfg["transition_iterations"]):
                latency.append(one("transition", index))
            elapsed = 0.0
            count = 0
            wall_start = time.perf_counter()
            while count < cfg["min_iterations"] or elapsed < cfg["min_seconds"]:
                if time.perf_counter() - wall_start >= cfg["max_seconds"]:
                    raise RuntimeError("Measurement timeout before minimum iteration/time requirements")
                row = one("measure", count)
                latency.append(row)
                elapsed += row["latency_us"] / 1e6
                count += 1
            if capturing:
                # On GB10, stopping immediately after the final synchronize can
                # discard GPU timestamps still being drained by Nsight. This wait
                # is outside the measured iterations and their latency records.
                time.sleep(1.0)
                torch.cuda.profiler.stop()
                capturing = False
            manifest.update({"measurement_count": count, "transition_count": cfg["transition_iterations"],
                             "warmup_count": len(warmup), "measured_seconds": elapsed})
            write_json(out / "manifest.json", manifest)
            quality.update({"status": "passed", "measurement_count": count,
                            "latency_us": stats([r["latency_us"] for r in latency if r["stage"] == "measure"]),
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                            "trace_integrity": "pending_parser_and_log_review" if profile else "not_applicable"})
        environment["gpu_snapshot_after"] = gpu_snapshot()
        write_json(out / "environment.json", environment)
    except Exception as exc:
        quality["status"] = "failed"
        quality["errors"].append(f"{type(exc).__name__}: {exc}")
        (out / "worker_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(quality["errors"][-1], file=sys.stderr)
    finally:
        if capturing:
            try:
                torch.cuda.profiler.stop()
            except Exception:
                pass
        write_csv(out / "latency.csv", latency, ["stage", "iteration", "host_start_ns", "host_end_ns", "latency_us"])
        write_csv(out / "warmup.csv", warmup, ["stage", "iteration", "host_start_ns", "host_end_ns", "latency_us"])
        write_json(out / "quality.json", quality)
    return 0 if quality["status"] == "passed" else 1
