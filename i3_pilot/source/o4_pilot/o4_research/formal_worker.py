"""Seven-A 40x3 single-process worker; must be launched by run_formal.

New version, separate from the 8/1/3 inventory smoke and frozen small-op RN-A worker.
No machine process termination or global GPU reset.
"""
import argparse
import contextlib
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path

from .adapters import Adapter, require_a
from .common import csvwrite, now, read, sha, write
from .formal_protocol import PROTOCOL, validate_protocol, warmup_stable
from .provenance import snapshot


def run(req_path):
    req = read(req_path)
    validate_protocol(req["protocol"])
    require_a(req["workload"])
    if req["mode"] not in PROTOCOL["modes"]:
        raise ValueError("Unsupported formal mode")
    if req["repeat"] not in (1, 2, 3):
        raise ValueError("Only three independent-process repeat slots")
    manifest = Path(req["assets_manifest"]).resolve()
    if sha(manifest) != req["assets_manifest_sha256"]:
        raise ValueError("Asset manifest changed since planning")
    folder = Path(req["output"]).resolve()
    quality = {"status": "started", "protocol_id": PROTOCOL["protocol_id"],
               "repeat": req["repeat"], "utc_start": now(),
               "formal_reference_verified": False, "strict_verified": 0,
               "training_eligible": False, "errors": []}
    rows, warm = [], []
    inventory, capturing = None, False
    write(folder / "quality.json", quality)
    try:
        import torch
        import torchvision
        from o4_iteration import common as legacy_bootstrap
        from inference_pilot.benchmark import precision_policy

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one visible CUDA GPU required")
        torch.cuda.set_device(0)
        props = torch.cuda.get_device_properties(0)
        if req["expect_gpu"].lower() not in props.name.lower():
            raise RuntimeError("Expected GPU identity does not match actual device")
        precision = precision_policy(torch)
        stream = torch.cuda.current_stream()
        write(folder / "toolchain_provenance.json", snapshot())
        write(folder / "environment.json", {
            "utc": now(), "pid": os.getpid(), "python": sys.version,
            "machine": platform.machine(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "gpu": props.name,
            "precision": precision, "protocol_id": PROTOCOL["protocol_id"],
            "trace_backend_requested": req["trace_backend"],
            "assets_manifest_sha256": req["assets_manifest_sha256"],
        })
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            adapter = Adapter(torch, req["workload"], manifest)
            quality["correctness"] = adapter.correctness()
            if not quality["correctness"]["passed"]:
                raise RuntimeError("Adapter correctness not passed")
            write(folder / "adapter.json", adapter.metadata)
            if req["mode"] == "R":
                from o4_iteration.adapters import dispatch_inventory
                inventory = dispatch_inventory(torch)
                adapter.prepare()
                normal = adapter.forward().detach()
                adapter.prepare()
                with inventory:
                    instrumented = adapter.forward().detach()
                torch.cuda.synchronize()
                if not torch.equal(normal, instrumented):
                    raise RuntimeError("TorchDispatch altered native FP16 output")
                quality["instrumented_output_exact"] = True
                del normal, instrumented

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            end.record(stream)
            end.synchronize()
            torch.cuda.nvtx.range_push("o4.setup_nvtx")
            torch.cuda.nvtx.range_pop()

            for i in range(PROTOCOL["warmup_max"]):
                adapter.prepare()
                torch.cuda.synchronize()
                if torch.cuda.current_stream() != stream:
                    raise RuntimeError("Warmup changed CUDA stream")
                start.record(stream)
                output = adapter.forward()
                end.record(stream)
                end.synchronize()
                duration = float(start.elapsed_time(end)) * 1000
                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError("Nonfinite or nonpositive warmup event duration")
                del output
                warm.append({"iteration": i, "elapsed_us": duration})
                if (i + 1) % PROTOCOL["warmup_window"] == 0 and warmup_stable(
                    [x["elapsed_us"] for x in warm]
                ):
                    break
            else:
                raise RuntimeError("Warmup failed last-three-window stability before cap")
            quality["warmup"] = {"passed": True, "iterations": len(warm)}

            if req["mode"] != "GI":
                torch.cuda.profiler.start()
                capturing = True
            with inventory if inventory is not None else contextlib.nullcontext():
                for stage, count in (
                    ("transition", PROTOCOL["transition_iterations"]),
                    ("measure", PROTOCOL["measured_iterations"]),
                ):
                    for i in range(count):
                        marked = req["mode"] != "GI"
                        if marked:
                            torch.cuda.nvtx.range_push(f"o4.prepare/{stage}/{i:06d}")
                        adapter.prepare()
                        torch.cuda.synchronize()
                        if marked:
                            torch.cuda.nvtx.range_pop()
                        if torch.cuda.current_stream() != stream:
                            raise RuntimeError("Forward changed current CUDA stream")
                        if marked:
                            torch.cuda.nvtx.range_push(f"o4.iter/{stage}/{i:06d}")
                        if inventory is not None:
                            inventory.begin(stage, i)
                        utc_start = now()
                        tick = time.perf_counter_ns()
                        start.record(stream)
                        output = adapter.forward()
                        end.record(stream)
                        end.synchronize()
                        duration = float(start.elapsed_time(end)) * 1000
                        tick_end = time.perf_counter_ns()
                        utc_end = now()
                        if inventory is not None:
                            inventory.end()
                        if marked:
                            torch.cuda.nvtx.range_pop()
                        if torch.cuda.current_stream() != stream:
                            raise RuntimeError("Forward stream changed")
                        if not math.isfinite(duration) or duration <= 0:
                            raise ValueError("Nonfinite or nonpositive CUDA event duration")
                        rows.append({
                            "stage": stage, "iteration": i, "utc_start": utc_start,
                            "utc_end": utc_end, "start_monotonic_ns": tick,
                            "end_monotonic_ns": tick_end, "elapsed_us": duration,
                            "target_kind": "iteration_device_event_us" if req["mode"] == "GI"
                                else "profiled_iteration_event_us",
                            "cache_before": adapter.last_cache_before,
                            "cache_after": adapter.last_cache_after,
                            "status": "completed",
                        })
                        del output

            if capturing:
                del start, end, adapter
                torch.cuda.synchronize()
                time.sleep(1)
                from o1_operator.worker import reset_cuda_context_for_trace
                quality["own_context_reset_returncode"] = reset_cuda_context_for_trace(torch.version.cuda)
                torch.cuda.profiler.stop()
                capturing = False

            quality.update({
                "status": "completed", "utc_end": now(),
                "measured_iterations": PROTOCOL["measured_iterations"],
                "trace_integrity": "not_applicable" if req["mode"] == "GI" else "pending_audit",
                "reference_status": "engineering_warning_policy_strict_unverified",
            })
    except Exception as exc:
        quality["status"] = "failed"
        quality["utc_end"] = now()
        quality["errors"].append(f"{type(exc).__name__}: {exc}")
        (folder / "worker_error.txt").write_text(traceback.format_exc())
        print(quality["errors"][-1], file=sys.stderr)
    finally:
        if capturing:
            try:
                import torch
                torch.cuda.profiler.stop()
            except Exception:
                pass
        if rows:
            csvwrite(folder / "iterations.csv", rows)
        if warm:
            csvwrite(folder / "warmup.csv", warm)
        if inventory is not None:
            with (folder / "reference_dispatch.jsonl").open("w") as stream_file:
                for row in inventory.rows:
                    stream_file.write(json.dumps(row) + "\n")
        write(folder / "quality.json", quality)
    return 0 if quality["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.request))


if __name__ == "__main__":
    main()
