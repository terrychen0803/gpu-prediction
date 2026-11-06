"""Summarize O1 repeats without treating calls as independent configurations."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows, columns):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def signatures(run):
    """Hash the recorded GPU path without duration or absolute timestamps."""
    import sqlite3
    from .challenge_matrix import PROTOCOL as CHALLENGE_PROTOCOL, ZERO_GPU_VIEW_ID
    request = read_json(Path(run) / "request.json")
    if request["protocol"] == CHALLENGE_PROTOCOL and request["config_id"] == ZERO_GPU_VIEW_ID:
        trace = read_json(Path(run) / "trace_audit.json")
        if trace["status"] not in ("recorded_zero_gpu_work", "manual_review_required") or trace["kernel_count"] != 0:
            raise ValueError(f"Metadata-only view has no verified zero-kernel trace: {run}")
        return 0, hashlib.sha256(b"[]").hexdigest()
    path = Path(run) / "trace.sqlite"
    env = read_json(Path(run) / "environment.json")
    from .trace_audit import pid_of
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    strings = {r["id"]: r["value"] for r in conn.execute("SELECT id,value FROM StringIds")}
    candidates = [n for n in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL")
                  if conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (n,)).fetchone()]
    if not candidates and read_json(Path(run) / "request.json")["config_id"] == "challenge_transpose_view_256_256":
        conn.close()
        return 0, hashlib.sha256(b"[]").hexdigest()
    if len(candidates) != 1:
        raise ValueError(f"O1 signature audit requires one kernel table: {path}")
    rows = [dict(r) for r in conn.execute(f'SELECT * FROM "{candidates[0]}"') if pid_of(r["globalPid"]) == env["pid"]]
    rows.sort(key=lambda r: (r["start"], r["end"]))
    path_signature = [(strings.get(r["demangledName"], ""),
                       r["gridX"],r["gridY"],r["gridZ"],r["blockX"],r["blockY"],r["blockZ"],
                       r["dynamicSharedMemory"]) for r in rows]
    conn.close()
    return len(rows), hashlib.sha256(json.dumps(path_signature, separators=(",", ":")).encode()).hexdigest()


def summarize(root, out=None):
    root = Path(root).resolve()
    plan = read_json(root / "plan.json")
    if plan["status"] == "planned_only":
        raise ValueError("O1 dry-run plan has no GPU measurements")
    out = Path(out).resolve() if out else root / "analysis"
    if out.exists():
        raise FileExistsError(f"O1 analysis path exists; use a new path: {out}")
    out.mkdir(parents=True)
    run_rows = []
    for job in plan["jobs"]:
        config_id, mode, repeat = job["config_id"], job["mode"], job["repeat"]
        run = root / "runs" / config_id / mode / f"run{repeat:02d}"
        execution = read_json(run / "execution.json") if (run / "execution.json").exists() else {"status":"missing"}
        quality = read_json(run / "quality.json") if (run / "quality.json").exists() else {}
        groups = list(csv.DictReader((run / "groups.csv").open())) if (run / "groups.csv").exists() else []
        trace = read_json(run / "trace_audit.json") if (run / "trace_audit.json").exists() else {}
        if len(groups) == 5 and mode in ("G0", "GE") and execution["status"] == "completed":
            group_medians = [float(g["main_single_call_median_us"]) for g in groups]
            run_median = statistics.median(group_medians)
        else:
            run_median = ""
        run_rows.append({"config_id": config_id, "mode": mode, "repeat": repeat,
                         "execution_status": execution["status"],
                         "correctness": quality.get("correctness", {}).get("passed", ""),
                         "warmup_calls": quality.get("warmup", {}).get("calls", ""),
                         "groups": len(groups), "calls_per_group": quality.get("repetitions_per_group", ""),
                         "groups_ge_100ms": sum(g["group_target_100ms_met"] == "True" for g in groups),
                         "run_median_us": run_median,
                         "trace_status": trace.get("status", ""),
                         "kernel_count": trace.get("kernel_count", ""),
                         "warning_count": len(trace.get("warnings", [])),
                         "run_path": str(run)})
    write_csv(out / "run_inventory.csv", run_rows, list(run_rows[0]))
    aggregates = []
    for config in plan["configs"]:
        for mode in plan["modes"]:
            subset = [r for r in run_rows if r["config_id"] == config["id"] and r["mode"] == mode]
            medians = [r["run_median_us"] for r in subset if isinstance(r["run_median_us"], float)]
            spread = (max(medians)-min(medians))/statistics.median(medians) if len(medians)==3 else None
            aggregates.append({"config_id": config["id"], "op": config["op"], "mode": mode,
                               "expected_runs": plan["repeats"],
                               "completed_runs": sum(r["execution_status"] == "completed" for r in subset),
                               "valid_time_runs": len(medians),
                               "median_of_run_medians_us": statistics.median(medians) if len(medians)==3 else "",
                               "run_spread": spread if spread is not None else "",
                               "spread_le_5pct": spread <= 0.05 if spread is not None else "",
                               "all_groups_ge_100ms": all(r["groups_ge_100ms"] == 5 for r in subset) if subset else False,
                               "trace_warning_runs": sum(r["warning_count"] > 0 for r in subset),
                               "trace_failed_runs": sum(r["trace_status"] == "failed" for r in subset)})
    write_csv(out / "aggregate.csv", aggregates, list(aggregates[0]))
    pairs = []
    if "P" in plan["modes"] and "R" in plan["modes"]:
        for config in plan["configs"]:
            for repeat in range(1, plan["repeats"]+1):
                p = root / "runs" / config["id"] / "P" / f"run{repeat:02d}"
                r = root / "runs" / config["id"] / "R" / f"run{repeat:02d}"
                pa = read_json(p / "trace_audit.json") if (p / "trace_audit.json").exists() else {}
                ra = read_json(r / "trace_audit.json") if (r / "trace_audit.json").exists() else {}
                p_calls = pa.get("expected_calls")
                r_calls = ra.get("expected_calls")
                same = False
                error = ""
                if p_calls is not None and r_calls is not None and p_calls != r_calls:
                    error = f"P/R call counts differ: {p_calls} vs {r_calls}"
                elif pa.get("status") not in (None, "failed") and ra.get("status") not in (None, "failed"):
                    try:
                        pn, ph = signatures(p); rn, rh = signatures(r)
                        same = pn == rn and ph == rh
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                else:
                    error = "P/R trace audit unavailable or failed"
                if error.startswith("P/R call counts differ"):
                    status = "unverifiable_policy_mismatch"
                elif error:
                    status = "unverifiable_audit_failed"
                elif pa.get("warnings") or ra.get("warnings"):
                    status = "manual_review_required_event_loss"
                elif not same:
                    status = "unverifiable_path_changed"
                else:
                    status = "recorded_path_equal_candidate_only"
                pairs.append({"config_id": config["id"], "repeat": repeat,
                              "P_expected_calls": p_calls if p_calls is not None else "",
                              "R_expected_calls": r_calls if r_calls is not None else "",
                              "P_kernel_count": pa.get("kernel_count", ""),
                              "R_kernel_count": ra.get("kernel_count", ""),
                              "recorded_sequence_equal": same,
                              "P_warning_count": len(pa.get("warnings", [])),
                              "R_warning_count": len(ra.get("warnings", [])),
                              "R_scope_inventory_exact": ra.get("reference_scope_inventory_exact", False),
                              "status": status, "error": error})
        write_csv(out / "rp_alignment.csv", pairs, list(pairs[0]))
    summary = {"protocol": plan["protocol"], "source_root": str(root),
               "expected_run_count": len(plan["jobs"]),
               "completed_run_count": sum(r["execution_status"] == "completed" for r in run_rows),
               "failed_run_count": sum(r["execution_status"] != "completed" for r in run_rows),
               "time_aggregate_count": sum(a["median_of_run_medians_us"] != "" for a in aggregates),
               "trace_warning_run_count": sum(r["warning_count"] > 0 for r in run_rows),
               "rp_status_counts": {status: sum(p["status"] == status for p in pairs)
                                    for status in sorted({p["status"] for p in pairs})},
               "provenance": {"code_sha256": plan["code_sha256"], "input_format": plan["input_format"]}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary

