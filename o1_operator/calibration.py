"""Freeze per-mode O1 call counts from completed, independent pilot cells."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from .spec import BY_ID, INPUT_FORMAT, MODES, PROTOCOL
from .full_matrix import BY_ID as FORMAL_BY_ID
from .full_matrix import BOOTSTRAP_PROTOCOL, PROTOCOL as FORMAL_PROTOCOL

BASELINE_REPETITIONS = 8192
GROUP_COUNT = 5
TARGET_GROUP_MS = 125
QUANTUM = 256
MAX_CALLS_PER_GROUP = 16384
FORMAL_MAX_CALLS_PER_GROUP = 65536


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calls_for_target(min_group_ms, baseline_repetitions=BASELINE_REPETITIONS,
                     max_calls=MAX_CALLS_PER_GROUP):
    if not math.isfinite(min_group_ms) or min_group_ms <= 0:
        raise ValueError("Observed group duration must be finite and positive")
    calls = math.ceil((baseline_repetitions * TARGET_GROUP_MS / min_group_ms) / QUANTUM) * QUANTUM
    if not 100 <= calls <= max_calls:
        raise ValueError(f"Calibrated calls {calls} outside [100,{max_calls}]")
    return calls


def load_manifest(path, catalog=BY_ID, expected_protocol=PROTOCOL):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get("protocol") != expected_protocol or manifest.get("input_format") != INPUT_FORMAT:
        raise ValueError("O1 calibration manifest protocol/input format mismatch")
    if not isinstance(manifest.get("expect_gpu"), str) or not manifest["expect_gpu"].strip():
        raise ValueError("O1 calibration manifest must identify its GPU")
    policy = manifest.get("policy", {})
    max_calls = FORMAL_MAX_CALLS_PER_GROUP if expected_protocol == FORMAL_PROTOCOL else MAX_CALLS_PER_GROUP
    expected_policy = {"baseline_repetitions": BASELINE_REPETITIONS, "groups": GROUP_COUNT,
                       "target_group_ms": TARGET_GROUP_MS, "quantum": QUANTUM,
                       "max_calls_per_group": max_calls}
    if policy != expected_policy:
        raise ValueError("O1 calibration manifest policy mismatch")
    mapping = manifest.get("config_repetitions")
    if not isinstance(mapping, dict) or set(mapping) != set(catalog):
        raise ValueError(f"O1 calibration manifest must cover all {len(catalog)} configurations")
    for config_id, modes in mapping.items():
        if not isinstance(modes, dict) or set(modes) != set(MODES):
            raise ValueError(f"O1 calibration modes incomplete: {config_id}")
        for mode, count in modes.items():
            if type(count) is not int or not (100 <= count <= max_calls) or count % QUANTUM:
                raise ValueError(f"Invalid fixed call count for {config_id}/{mode}: {count!r}")
        if modes["P"] != modes["R"]:
            raise ValueError(f"P/R fixed call count mismatch: {config_id}")
    return manifest


def read_completed_group_times(root, modes, allow_p_warnings=False):
    root = Path(root).resolve()
    plan = json.loads((root / "plan.json").read_text())
    if plan["status"] != "completed" or plan["input_format"] != INPUT_FORMAT:
        raise ValueError(f"Source plan incomplete or input format differs: {root}")
    observed = []
    for job in plan["jobs"]:
        if job["mode"] not in modes:
            continue
        config_id, mode, repeat = job["config_id"], job["mode"], job["repeat"]
        run = root / "runs" / config_id / mode / f"run{repeat:02d}"
        request = json.loads((run / "request.json").read_text())
        quality = json.loads((run / "quality.json").read_text())
        execution = json.loads((run / "execution.json").read_text())
        if (request["protocol"] != plan["protocol"] or
                quality["status"] != "completed" or not quality["correctness"]["passed"] or
                execution["status"] != "completed" or
                quality["repetitions_per_group"] != BASELINE_REPETITIONS):
            raise ValueError(f"Invalid source cell: {run}")
        warning_count = 0
        if mode == "P":
            trace = json.loads((run / "trace_audit.json").read_text())
            warning_count = len(trace["warnings"])
            allowed_status = ("recorded_structure_passed", "manual_review_required") if allow_p_warnings else ("recorded_structure_passed",)
            if trace["status"] not in allowed_status or (warning_count and not allow_p_warnings):
                raise ValueError(f"P calibration trace has warnings: {run}")
        group_path = run / "groups.csv"
        with group_path.open(newline="") as stream:
            groups = list(csv.DictReader(stream))
        if len(groups) != GROUP_COUNT or any(int(row["repetitions"]) != BASELINE_REPETITIONS for row in groups):
            raise ValueError(f"Source group inventory mismatch: {run}")
        durations = [float(row["host_group_us"]) / 1000 for row in groups]
        if any(not math.isfinite(value) or value <= 0 for value in durations):
            raise ValueError(f"Source group duration invalid: {run}")
        observed.append({"config_id": config_id, "mode": mode, "repeat": repeat,
                         "expect_gpu": request["expect_gpu"],
                         "source_protocol": plan["protocol"], "session": root.name,
                         "source_trace_warning_count": warning_count,
                         "group_csv_sha256": sha256(group_path),
                         "minimum_group_ms": min(durations)})
    return plan, observed


def build_manifest(g0ge_root, p_roots, catalog=BY_ID, protocol=PROTOCOL,
                   source_protocol=None, allow_p_warnings=False):
    if not p_roots:
        raise ValueError("At least one P calibration root is required")
    roots = [(g0ge_root, ("G0", "GE")), *((root, ("P",)) for root in p_roots)]
    sources = []
    observations = []
    for root, modes in roots:
        plan, rows = read_completed_group_times(root, modes, allow_p_warnings)
        if source_protocol is not None and plan["protocol"] != source_protocol:
            raise ValueError(f"Calibration source protocol differs: {root}")
        sources.append({"session": Path(root).resolve().name, "plan_sha256": sha256(Path(root) / "plan.json"),
                        "source_protocol": plan["protocol"], "observed_modes": list(modes),
                        "measurement_code_sha256": plan["code_sha256"]})
        observations.extend(rows)
    by_cell = {}
    for row in observations:
        by_cell.setdefault((row["config_id"], row["mode"]), []).append(row)
    expected = {(config_id, mode) for config_id in catalog for mode in ("G0", "GE", "P")}
    if set(by_cell) != expected:
        raise ValueError(f"Calibration cells missing/extra: {sorted(expected - set(by_cell))}, {sorted(set(by_cell) - expected)}")
    expected_gpus = {row["expect_gpu"] for row in observations}
    if len(expected_gpus) != 1:
        raise ValueError(f"Calibration source GPU expectations differ: {sorted(expected_gpus)}")
    mapping = {}
    calibration = []
    max_calls = FORMAL_MAX_CALLS_PER_GROUP if protocol == FORMAL_PROTOCOL else MAX_CALLS_PER_GROUP
    for config_id in catalog:
        mapping[config_id] = {}
        for mode in ("G0", "GE", "P"):
            minimum = min(row["minimum_group_ms"] for row in by_cell[(config_id, mode)])
            count = calls_for_target(minimum, max_calls=max_calls)
            mapping[config_id][mode] = count
            calibration.append({"config_id": config_id, "mode": mode,
                                "source_run_count": len(by_cell[(config_id, mode)]),
                                "minimum_observed_group_ms": minimum,
                                "fixed_calls_per_group": count})
        mapping[config_id]["R"] = mapping[config_id]["P"]
    return {"protocol": protocol, "input_format": INPUT_FORMAT,
            "expect_gpu": expected_gpus.pop(),
            "source_trace_warning_policy": "allow_and_record" if allow_p_warnings else "reject",
            "policy": {"baseline_repetitions": BASELINE_REPETITIONS, "groups": GROUP_COUNT,
                       "target_group_ms": TARGET_GROUP_MS, "quantum": QUANTUM,
                       "max_calls_per_group": max_calls},
            "config_repetitions": mapping, "calibration_cells": calibration,
            "source_sessions": sources, "source_runs": observations,
            "R_calibration_basis": "P group minimum; P/R share count; R must be validated independently"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Freeze O1 per-mode calls from completed pilot sessions")
    parser.add_argument("--g0ge-root", required=True)
    parser.add_argument("--p-root", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--matrix", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--allow-p-warnings", action="store_true")
    args = parser.parse_args(argv)
    if args.allow_p_warnings and args.matrix != "formal":
        raise ValueError("Ignoring P source warnings is only available for formal calibration")
    formal = args.matrix == "formal"
    catalog = FORMAL_BY_ID if formal else BY_ID
    protocol = FORMAL_PROTOCOL if formal else PROTOCOL
    manifest = build_manifest(args.g0ge_root, args.p_root, catalog, protocol,
                              BOOTSTRAP_PROTOCOL if formal else None, args.allow_p_warnings)
    out = Path(args.out).resolve()
    with out.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    load_manifest(out, catalog, protocol)
    print(f"O1 calibration manifest: {out} sha256={sha256(out)}")
    return 0
