"""Evaluator-only exact-order R to P review; never writes labels into P runs."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def event_signature(event):
    return [event["kernel_name"], event["grid_xyz"], event["block_xyz"],
            event["dynamic_shared_memory_bytes"]]


def review_pair(p_folder, r_folder, output):
    """Require an exact full ordered event signature before projecting R boundaries."""
    p_folder, r_folder, output = map(Path, (p_folder, r_folder, output))
    p_request = json.loads((p_folder / "request.json").read_text())
    r_request = json.loads((r_folder / "request.json").read_text())
    if (p_request["template"] != r_request["template"] or
            p_request["repeat"] != r_request["repeat"] or
            p_request["groups"] != r_request["groups"] or
            p_request["sequences_per_group"] != r_request["sequences_per_group"]):
        raise ValueError("O2 P/R pair has different workload or repetition policy")
    p_audit = json.loads((p_folder / "trace_audit.json").read_text())
    r_audit = json.loads((r_folder / "trace_audit.json").read_text())
    references = read_jsonl(r_folder / "reference_operations.jsonl")
    events = read_jsonl(p_folder / "events_allowlist.jsonl")
    total = len(references)
    expected = p_request["groups"]*p_request["sequences_per_group"]*len(p_request["template"]["steps"])
    if total != expected or len({x["scope"] for x in references}) != expected:
        raise ValueError("R reference inventory differs from expected O2 steps")
    references.sort(key=lambda x: (x["group"], x["sequence"], x["step"]))
    p_signature = [event_signature(x) for x in events]
    r_signature = [signature for ref in references for signature in ref["kernel_signatures"]]
    r_indices = [index for ref in references for index in ref["kernel_event_indices"]]
    exact_order = (p_signature == r_signature and r_indices == list(range(len(r_indices))))
    warnings = p_audit.get("warnings", []) + r_audit.get("warnings", [])
    reason = ("exact_ordered_signature" if exact_order else
              "event_count_mismatch" if len(p_signature) != len(r_signature) else
              "R_scope_GPU_order_not_monotone" if r_indices != list(range(len(r_indices))) else
              "kernel_signature_or_launch_geometry_path_change")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        offset = 0
        for ref in references:
            count = len(ref["kernel_signatures"])
            p_ids = [x["event_id"] for x in events[offset:offset+count]] if exact_order else []
            row = {"scope": ref["scope"], "group": ref["group"], "sequence": ref["sequence"],
                   "step": ref["step"], "config_id": ref["config_id"], "op_type": ref["op_type"],
                   "r_kernel_event_indices": ref["kernel_event_indices"],
                   "p_event_ids": p_ids, "r_to_p_reason": reason,
                   "label_provenance": "reference_join_unique" if exact_order and not warnings
                                       else "unverifiable",
                   "manual_review_required": bool(warnings),
                   "warning_text": warnings}
            stream.write(json.dumps(row) + "\n")
            offset += count
    # Whole-run times are descriptive. They do not validate labels or match
    # events by approximate duration.
    def host_us(folder):
        with (folder / "groups.csv").open() as stream:
            return [float(row["host_us_per_sequence"]) for row in csv.DictReader(stream)]
    p_times, r_times = host_us(p_folder), host_us(r_folder)
    result = {"template_id": p_request["template"]["id"], "repeat": p_request["repeat"],
              "step_count": total, "r_kernel_count": len(r_signature),
              "p_kernel_count": len(p_signature), "exact_ordered_signature": exact_order,
              "reason": reason, "warnings": warnings,
              "verified_step_count": total if exact_order and not warnings else 0,
              "manual_review_step_count": total if warnings else 0,
              "p_host_us_per_sequence_median": statistics.median(p_times),
              "r_host_us_per_sequence_median": statistics.median(r_times),
              "r_over_p_host_ratio": statistics.median(r_times)/statistics.median(p_times)
              if statistics.median(p_times) else None}
    (output.parent / "pair_summary.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def review_root(root, out):
    root, out = map(Path, (root, out))
    plan = json.loads((root / "plan.json").read_text())
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for template in plan["templates"]:
        for repeat in range(1, plan["repeats"]+1):
            base = root / "runs" / template["id"]
            p = base / "P" / f"run{repeat:02d}"
            r = base / "R" / f"run{repeat:02d}"
            if not p.exists() or not r.exists() or not (p / "events_allowlist.jsonl").exists() or not (r / "reference_operations.jsonl").exists():
                rows.append({"template_id": template["id"], "repeat": repeat,
                             "step_count": plan["groups"]*plan["sequences_per_group"]*len(template["steps"]),
                             "verified_step_count": 0, "reason": "missing_P_or_R_capture"})
                continue
            pair_out = out / template["id"] / f"run{repeat:02d}" / "rp_alignment.jsonl"
            rows.append(review_pair(p, r, pair_out))
    with (out / "pair_summary.csv").open("x", newline="") as stream:
        keys = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    counts = {"pairs": len(rows), "steps": sum(x["step_count"] for x in rows),
              "verified_steps": sum(x["verified_step_count"] for x in rows),
              "exact_signature_pairs": sum(bool(x.get("exact_ordered_signature")) for x in rows),
              "warning_pairs": sum(bool(x.get("warnings")) for x in rows),
              "missing_pairs": sum(x["reason"] == "missing_P_or_R_capture" for x in rows)}
    (out / "summary.json").write_text(json.dumps(counts, indent=2)+"\n")
    return counts
