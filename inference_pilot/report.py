from __future__ import annotations

import collections
import hashlib
import itertools
import json
import statistics
from pathlib import Path

from .common import read_csv, read_json, stats, write_csv, write_json


def discover(roots):
    found = {}
    for root in roots:
        path = Path(root).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        for request in path.rglob("request.json"):
            found[str(request.parent)] = request.parent
    return sorted(found.values())


def protocol(manifest, env):
    # Batch is a grouped workload configuration; hardware is deliberately excluded.
    values = {key: manifest[key] for key in ("model", "weights_sha256", "dtype", "layout", "code_sha256",
              "backend", "cuda_graph", "queue_depth", "timer_scope", "annotation_timer", "forward_path", "precision_policy", "settings")}
    values["software"] = {key: env[key] for key in ("torch", "torchvision", "cuda_runtime", "cudnn")}
    if manifest.get("input_scheme"):
        values["input_scheme"] = manifest["input_scheme"]
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:16]


def collect(roots):
    rows, excluded = [], []
    for directory in discover(roots):
        try:
            execution = read_json(directory / "execution.json")
            if execution.get("status") != "completed":
                raise ValueError(f"execution status: {execution.get('status')}")
            manifest, env = read_json(directory / "manifest.json"), read_json(directory / "environment.json")
            quality = read_json(directory / "quality.json")
            if quality.get("status") != "passed":
                raise ValueError("worker quality failed")
            if manifest["mode"].startswith("P"):
                trace_quality = read_json(directory / "trace_quality.json")
                if trace_quality.get("status") != "passed":
                    raise ValueError("trace parsing failed")
            samples = [float(r["latency_us"]) for r in read_csv(directory / "latency.csv") if r["stage"] == "measure"]
            if len(samples) != manifest["measurement_count"] or not samples:
                raise ValueError("latency count mismatch")
            row = {"run_dir": str(directory), "label": manifest["label"], "gpu_name": env["gpu_name"],
                   "gpu_uuid": env.get("gpu_uuid", "unavailable"), "batch": manifest["batch"], "mode": manifest["mode"],
                   "repeat": manifest["repeat"], "protocol": protocol(manifest, env), "input_sha256": manifest["input_sha256"],
                   "count": len(samples), "stability_threshold": manifest["settings"]["stability_threshold"],
                   "manual_trace_review_required": manifest["mode"].startswith("P"),
                   **{key + "_us": value for key, value in stats(samples).items()}}
            if manifest["mode"].startswith("P"):
                features = read_csv(directory / "features.csv")
                if len(features) != len(samples):
                    raise ValueError("feature count mismatch")
                row.update({"trace_kernel_count_median": statistics.median(int(f["kernel_count"]) for f in features),
                            "trace_gpu_span_median_us": statistics.median(float(f["gpu_span_us"]) for f in features),
                            "trace_kernel_busy_median_us": statistics.median(float(f["kernel_busy_union_us"]) for f in features)})
            rows.append(row)
        except (OSError, ValueError, KeyError) as exc:
            excluded.append({"run_dir": str(directory), "reason": str(exc)})
    return rows, excluded


def summarize(roots, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows, excluded = collect(roots)
    write_csv(out / "summary.csv", rows)
    write_csv(out / "excluded_runs.csv", excluded, ["run_dir", "reason"])
    if not rows:
        write_json(out / "summary_quality.json", {"status": "no_valid_runs", "excluded": len(excluded)})
        raise ValueError("No completed, valid GPU runs; planned/synthetic outputs are not experiment results")
    buckets = collections.defaultdict(list)
    for row in rows:
        buckets[(row["protocol"], row["label"], row["batch"], row["mode"])].append(row)
    aggregates = []
    for (signature, label, batch, mode), group in sorted(buckets.items()):
        medians = [r["median_us"] for r in group]
        median = statistics.median(medians)
        stable = (max(medians) - min(medians)) / median <= group[0]["stability_threshold"]
        compatible = len({r["input_sha256"] for r in group}) == 1 and len({r["gpu_name"] for r in group}) == 1 and len({r["gpu_uuid"] for r in group}) == 1
        aggregates.append({"protocol": signature, "label": label, "batch": batch, "mode": mode,
                           "runs": len(group), "median_of_run_medians_us": median,
                           "run_median_spread_fraction": (max(medians) - min(medians)) / median,
                           "stable": stable, "compatible": compatible, "at_least_3_runs": len(group) >= 3,
                           "input_sha256": group[0]["input_sha256"]})
    write_csv(out / "aggregate.csv", aggregates)
    eligible = [a for a in aggregates if a["compatible"] and a["at_least_3_runs"]]
    by_mode = {(a["protocol"], a["label"], a["batch"], a["mode"]): a for a in eligible}
    overheads = []
    for a in eligible:
        if a["mode"] != "G" or not a["stable"]:
            continue
        base_key = (a["protocol"], a["label"], a["batch"])
        for name, numerator, denominator in (("annotation", "A", "G"), ("profiling_unmarked", "P0", "G"),
                                              ("profiling_marked", "P1", "A"), ("total_marked_profile", "P1", "G")):
            n, d = by_mode.get((*base_key, numerator)), by_mode.get((*base_key, denominator))
            if n and d and n["input_sha256"] == d["input_sha256"]:
                nt, dt = n["median_of_run_medians_us"], d["median_of_run_medians_us"]
                overheads.append({"protocol": a["protocol"], "label": a["label"], "batch": a["batch"],
                                  "comparison": name, "fraction": nt / dt - 1, "difference_us": nt - dt,
                                  "both_groups_stable": n["stable"] and d["stable"]})
    write_csv(out / "overheads.csv", overheads)
    gt = [a for a in eligible if a["mode"] == "G" and a["stable"]]
    ratios = []
    for source, target in itertools.permutations(gt, 2):
        if source["label"] == target["label"] or source["protocol"] != target["protocol"] or source["batch"] != target["batch"]:
            continue
        if source["input_sha256"] != target["input_sha256"]:
            continue
        st, tt = source["median_of_run_medians_us"], target["median_of_run_medians_us"]
        ratios.append({"protocol": source["protocol"], "source": source["label"], "target": target["label"],
                       "batch": source["batch"], "source_us": st, "target_us": tt, "target_over_source": tt / st})
    write_csv(out / "device_ratios.csv", ratios)
    sanity = []
    for held in ratios:
        training = [r for r in ratios if r["protocol"] == held["protocol"] and r["source"] == held["source"] and
                    r["target"] == held["target"] and r["batch"] != held["batch"]]
        if len(training) >= 2:
            factor = statistics.median(r["target_over_source"] for r in training)
            prediction = held["source_us"] * factor
            sanity.append({"protocol": held["protocol"], "source": held["source"], "target": held["target"],
                           "held_out_batch": held["batch"], "training_batches": ",".join(str(r["batch"]) for r in training),
                           "estimated_ratio": factor, "prediction_us": prediction, "actual_us": held["target_us"],
                           "absolute_relative_error": abs(prediction / held["target_us"] - 1)})
    write_csv(out / "leave_one_batch_out_sanity.csv", sanity)
    quality = {"status": "completed", "included_runs": len(rows), "excluded_runs": len(excluded),
               "protocols": sorted({r["protocol"] for r in rows}),
               "insufficient_repeats": sum(not a["at_least_3_runs"] for a in aggregates),
               "unstable_G_groups": sum(a["mode"] == "G" and not a["stable"] for a in aggregates),
               "note": "Ratios require matched protocol/input and stable G groups with >=3 runs. Different protocols are never pooled. Trace diagnostics require human review."}
    write_json(out / "summary_quality.json", quality)
    (out / "README.md").write_text(
        "# Pilot 結果\n\n"
        f"納入 {len(rows)} runs；排除 {len(excluded)} runs，原因見 excluded_runs.csv。\n\n"
        "summary.csv 是每個 run 的統計，aggregate.csv 是 run 中位數的彙整。"
        "overheads.csv 記錄標記與 profiler 的差異；device_ratios.csv 是無 profiler 時間的目標／來源倍率。\n\n"
        "leave_one_batch_out_sanity.csv 只用其他 batch 的倍率預測保留配置，屬小樣本流程檢查。"
        "請另外檢查 trace_diagnostics.json 與 Nsight 診斷；解析通過不等於已證明無事件遺失。\n",
        encoding="utf-8")
    print(f"Summary: {out.resolve()} ({len(rows)} valid runs)")
    return quality
