"""Conservative O2 CUDA runtime/kernel/NVTX inventory from Nsight SQLite."""
from __future__ import annotations

import bisect
import collections
import json
import re
import sqlite3
from pathlib import Path

from o1_operator.trace_audit import pid_of, require, rows, sha256, table_columns

SCOPE = re.compile(r"^o2\.op/g(\d{2})/i(\d{3})/s(\d{2})$")


def signature(item):
    return [item["kernel_name"], item["grid_xyz"], item["block_xyz"],
            item["dynamic_shared_memory_bytes"]]


def audit(run, mode):
    run = Path(run)
    request = json.loads((run / "request.json").read_text())
    env = json.loads((run / "environment.json").read_text())
    quality = json.loads((run / "quality.json").read_text())
    result = {"status": "failed", "mode": mode, "errors": [], "warnings": [],
              "report_sha256": sha256(run / "trace.nsys-rep"),
              "sqlite_sha256": sha256(run / "trace.sqlite")}
    conn = sqlite3.connect(f"file:{run / 'trace.sqlite'}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("Nsight SQLite quick_check failed")
        require(conn, "StringIds", ("id", "value"))
        require(conn, "CUPTI_ACTIVITY_KIND_RUNTIME", ("start", "end", "globalTid", "correlationId", "nameId"))
        tables = [t for t in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL")
                  if table_columns(conn, t) and conn.execute(f'SELECT 1 FROM "{t}" LIMIT 1').fetchone()]
        if len(tables) != 1:
            raise ValueError(f"O2 expects one populated kernel table, found {tables}")
        table = tables[0]
        require(conn, table, ("start", "end", "globalPid", "correlationId", "deviceId",
              "contextId", "streamId", "demangledName", "gridX", "gridY", "gridZ",
              "blockX", "blockY", "blockZ", "dynamicSharedMemory"))
        strings = {r["id"]: r["value"] for r in rows(conn, "StringIds")}
        diagnostics = rows(conn, "DIAGNOSTIC_EVENT") if table_columns(conn, "DIAGNOSTIC_EVENT") else []
        result["warnings"] = [x["text"] for x in diagnostics if x["severity"] >= 2]
        pid = env["pid"]
        apis = [r for r in rows(conn, "CUPTI_ACTIVITY_KIND_RUNTIME") if pid_of(r["globalTid"]) == pid]
        launches = collections.defaultdict(list)
        for api in apis:
            if api["end"] < api["start"]:
                raise ValueError("Invalid CUDA API interval")
            if "launch" in strings.get(api["nameId"], "").lower():
                launches[api["correlationId"]].append(api)
        kernels = [r for r in rows(conn, table) if pid_of(r["globalPid"]) == pid]
        kernels.sort(key=lambda x: (x["start"], x["end"]))
        if not kernels:
            raise ValueError("No kernels for O2 worker PID")
        if any(len({x[field] for x in kernels}) != 1 for field in ("deviceId", "contextId", "streamId")):
            raise ValueError("O2 expected one GPU device/context/stream")
        mapped = []
        for index, kernel in enumerate(kernels):
            if kernel["end"] <= kernel["start"]:
                raise ValueError(f"Invalid kernel duration at {index}")
            candidates = [x for x in launches[kernel["correlationId"]]
                          if x["start"] <= kernel["start"] + 5000]
            if len(candidates) != 1:
                raise ValueError(f"Kernel {index} has {len(candidates)} matching CUDA launches")
            api = candidates[0]
            item = {"index": index, "kernel_name": strings.get(kernel["demangledName"], ""),
                    "grid_xyz": [kernel[f"grid{x}"] for x in "XYZ"],
                    "block_xyz": [kernel[f"block{x}"] for x in "XYZ"],
                    "dynamic_shared_memory_bytes": kernel["dynamicSharedMemory"],
                    "kernel_start_ns": kernel["start"], "kernel_end_ns": kernel["end"],
                    "api_start_ns": api["start"], "api_end_ns": api["end"],
                    "api_tid": api["globalTid"], "correlation_id": kernel["correlationId"]}
            mapped.append(item)
        origin = min(min(x["api_start_ns"], x["kernel_start_ns"]) for x in mapped)
        result.update({"kernel_count": len(mapped), "device_count": 1,
                       "context_count": 1, "stream_count": 1,
                       "launch_count": len({(x["correlation_id"], x["api_start_ns"]) for x in mapped})})
        scopes = []
        if table_columns(conn, "NVTX_EVENTS"):
            for raw in rows(conn, "NVTX_EVENTS"):
                name = raw["text"] or strings.get(raw["textId"], "")
                if isinstance(name, str) and name.startswith("o2.op/"):
                    if pid_of(raw["globalTid"]) != pid or not SCOPE.fullmatch(name):
                        raise ValueError(f"Unexpected O2 NVTX scope {name}")
                    raw["name"] = name
                    scopes.append(raw)
        if mode == "P":
            if scopes:
                raise ValueError("P trace has O2 operator markers")
            with (run / "events_allowlist.jsonl").open("x") as stream:
                for item in mapped:
                    index = item["index"]
                    safe = {"event_id": f"kernel_{index:08d}", "event_kind": "kernel",
                            "start_ns": item["kernel_start_ns"]-origin,
                            "end_ns": item["kernel_end_ns"]-origin,
                            "api_start_ns": item["api_start_ns"]-origin,
                            "api_end_ns": item["api_end_ns"]-origin,
                            "process_id": "p0", "context_id": "c0", "stream_id": "s0",
                            "correlation_id": f"corr_{index:08d}", "api_event_id": f"api_{index:08d}",
                            "launch_order": index,
                            "kernel_name": item["kernel_name"], "grid_xyz": item["grid_xyz"],
                            "block_xyz": item["block_xyz"],
                            "dynamic_shared_memory_bytes": item["dynamic_shared_memory_bytes"]}
                    stream.write(json.dumps(safe) + "\n")
        elif mode == "R":
            expected = {f"o2.op/g{g:02d}/i{i:03d}/s{s:02d}"
                        for g in range(1, quality["group_count"]+1)
                        for i in range(quality["sequences_per_group"])
                        for s in range(quality["steps_per_sequence"])}
            names = {x["name"] for x in scopes}
            if len(scopes) != len(expected) or names != expected:
                raise ValueError(f"R scope inventory differs: {len(scopes)}/{len(expected)}")
            scopes_by_tid = collections.defaultdict(list)
            for scope in scopes:
                scopes_by_tid[scope["globalTid"]].append(scope)
            starts = {}
            for tid, seq in scopes_by_tid.items():
                seq.sort(key=lambda x: x["start"])
                if any(a["end"] is None or a["end"] > b["start"] for a,b in zip(seq,seq[1:])):
                    raise ValueError("Overlapping R scopes")
                starts[tid] = [x["start"] for x in seq]
            by_scope = collections.defaultdict(list)
            for item in mapped:
                seq = scopes_by_tid.get(item["api_tid"], [])
                pos = bisect.bisect_right(starts.get(item["api_tid"], []), item["api_start_ns"])-1
                scope = seq[pos] if pos >= 0 else None
                if scope is None or scope["end"] is None or item["api_end_ns"] > scope["end"]:
                    raise ValueError(f"R kernel {item['index']} has no containing CUDA API/NVTX scope")
                by_scope[scope["name"]].append(item)
            if any(not by_scope[name] for name in expected):
                raise ValueError("At least one R step has no GPU kernel")
            scopes.sort(key=lambda x: x["start"])
            with (run / "reference_operations.jsonl").open("x") as stream:
                for scope in scopes:
                    match = SCOPE.fullmatch(scope["name"])
                    group, sequence, step = map(int, match.groups())
                    config_id = request["template"]["steps"][step]["config_id"]
                    item = {"scope": scope["name"], "group": group, "sequence": sequence,
                            "step": step, "config_id": config_id,
                            "op_type": request["configs"][config_id]["op"],
                            "input_source": request["template"]["steps"][step]["input"],
                            "api_start_ns": scope["start"], "api_end_ns": scope["end"],
                            "kernel_event_indices": [x["index"] for x in by_scope[scope["name"]]],
                            "kernel_signatures": [signature(x) for x in by_scope[scope["name"]]],
                            "label_provenance": "reference_direct"}
                    stream.write(json.dumps(item) + "\n")
            result["reference_scope_count"] = len(scopes)
            result["reference_mapped_kernel_count"] = sum(map(len, by_scope.values()))
        else:
            raise ValueError(f"Unsupported mode {mode}")
        result["status"] = "manual_review_required" if result["warnings"] else "recorded_structure_passed"
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        conn.close()
        (run / "trace_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    return result
