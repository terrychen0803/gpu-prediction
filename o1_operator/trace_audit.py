"""Conservative Nsight SQLite audit for O1 P/R captures."""
from __future__ import annotations

import collections
import bisect
import hashlib
import json
import sqlite3
from pathlib import Path


def pid_of(global_id):
    return (int(global_id) >> 24) & ((1 << 24) - 1)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def table_columns(conn, table):
    return {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}


def require(conn, table, columns):
    if not table_columns(conn, table):
        raise ValueError(f"Nsight SQLite lacks required table: {table}")
    missing = set(columns) - table_columns(conn, table)
    if missing:
        raise ValueError(f"Nsight SQLite {table} lacks columns: {sorted(missing)}")


def rows(conn, table):
    return [dict(row) for row in conn.execute(f'SELECT * FROM "{table}"')]


def audit(run, mode):
    run = Path(run)
    if mode not in ("P", "R"):
        raise ValueError("O1 trace audit only supports P/R")
    request = json.loads((run / "request.json").read_text())
    env = json.loads((run / "environment.json").read_text())
    quality = json.loads((run / "quality.json").read_text())
    pid = env["pid"]
    path = run / "trace.sqlite"
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    result = {"mode": mode, "status": "failed", "errors": [], "warnings": [],
              "sqlite_sha256": sha256(path), "report_sha256": sha256(run / "trace.nsys-rep")}
    try:
        if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("Nsight SQLite quick_check failed")
        require(conn, "CUPTI_ACTIVITY_KIND_RUNTIME", ("start", "end", "globalTid", "correlationId", "nameId"))
        require(conn, "StringIds", ("id", "value"))
        kernels_tables = [t for t in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL")
                          if table_columns(conn, t) and conn.execute(f'SELECT 1 FROM "{t}" LIMIT 1').fetchone()]
        if len(kernels_tables) != 1:
            raise ValueError(f"Expected one populated Nsight kernel table; found {kernels_tables}")
        kernel_table = kernels_tables[0]
        require(conn, kernel_table, ("start", "end", "globalPid", "correlationId", "deviceId",
                                     "contextId", "streamId", "demangledName", "gridX", "gridY", "gridZ",
                                     "blockX", "blockY", "blockZ", "dynamicSharedMemory"))
        strings = {r["id"]: r["value"] for r in rows(conn, "StringIds")}
        diagnostics = rows(conn, "DIAGNOSTIC_EVENT") if table_columns(conn, "DIAGNOSTIC_EVENT") else []
        result["warnings"] = [d["text"] for d in diagnostics if d["severity"] >= 2]
        result["diagnostic_count"] = len(diagnostics)
        apis = [r for r in rows(conn, "CUPTI_ACTIVITY_KIND_RUNTIME") if pid_of(r["globalTid"]) == pid]
        launches = collections.defaultdict(list)
        for row in apis:
            if row["end"] < row["start"]:
                raise ValueError("CUDA runtime API has invalid interval")
            name = strings.get(row["nameId"], "")
            if "launch" in name.lower():
                launches[row["correlationId"]].append(row)
        kernels = [r for r in rows(conn, kernel_table) if pid_of(r["globalPid"]) == pid]
        kernels.sort(key=lambda r: (r["start"], r["end"]))
        if not kernels:
            raise ValueError("No CUDA kernels attributed to O1 worker PID")
        if len({r["deviceId"] for r in kernels}) != 1 or len({r["contextId"] for r in kernels}) != 1 or len({r["streamId"] for r in kernels}) != 1:
            raise ValueError("O1 pilot expects exactly one recorded GPU device, context and stream")
        mapped = []
        for index, kernel in enumerate(kernels):
            if kernel["end"] <= kernel["start"]:
                raise ValueError(f"CUDA kernel {index} has invalid duration")
            candidates = launches.get(kernel["correlationId"], [])
            candidates = [a for a in candidates if a["start"] <= kernel["start"] + 5000]
            if len(candidates) != 1:
                raise ValueError(f"CUDA kernel {index} has {len(candidates)} launch API candidates")
            api = candidates[0]
            name = strings.get(kernel["demangledName"], "")
            mapped.append({"kernel": kernel, "api": api, "name": name, "index": index})
        # The capture range is the whole measurement. These counts are structural, not proof of no event loss.
        expected_calls = quality["repetitions_per_group"] * quality["group_count"]
        result.update({"kernel_count": len(kernels), "expected_calls": expected_calls,
                       "launch_api_count": len({(x["api"]["correlationId"],x["api"]["start"]) for x in mapped}),
                       "device_count": 1, "context_count": 1, "stream_count": 1,
                       "recorded_event_consistency": "passed"})
        if mode == "R":
            require(conn, "NVTX_EVENTS", ("start", "end", "text", "textId", "globalTid"))
            scopes = []
            for row in rows(conn, "NVTX_EVENTS"):
                name = row["text"] or strings.get(row["textId"], "")
                if isinstance(name, str) and name.startswith("o1.op/") and pid_of(row["globalTid"]) == pid:
                    row["name"] = name
                    scopes.append(row)
            scopes.sort(key=lambda r: (r["start"], r["end"]))
            if len(scopes) != expected_calls:
                raise ValueError(f"R scope count {len(scopes)} != expected calls {expected_calls}")
            expected_scope_names = {f"o1.op/g{group:02d}/i{index:06d}"
                                    for group in range(1, quality["group_count"] + 1)
                                    for index in range(quality["repetitions_per_group"])}
            actual_scope_names = {scope["name"] for scope in scopes}
            if actual_scope_names != expected_scope_names:
                raise ValueError("R reference scope ID inventory has missing, duplicate or unexpected IDs")
            result["reference_scope_inventory_exact"] = True
            scopes_by_tid = collections.defaultdict(list)
            for scope in scopes:
                scopes_by_tid[scope["globalTid"]].append(scope)
            starts_by_tid = {}
            for tid, seq in scopes_by_tid.items():
                seq.sort(key=lambda r: r["start"])
                if any(left["end"] is None or left["end"] > right["start"]
                       for left, right in zip(seq, seq[1:])):
                    raise ValueError("Nested or overlapping O1 operator scopes require explicit review")
                starts_by_tid[tid] = [scope["start"] for scope in seq]
            by_scope = collections.defaultdict(list)
            for x in mapped:
                tid = x["api"]["globalTid"]
                seq = scopes_by_tid.get(tid, [])
                pos = bisect.bisect_right(starts_by_tid.get(tid, []), x["api"]["start"]) - 1
                scope = seq[pos] if pos >= 0 else None
                if scope is None or scope["end"] is None or x["api"]["end"] > scope["end"]:
                    raise ValueError(f"R kernel {x['index']} has no containing NVTX scope")
                by_scope[scope["name"]].append(x["index"])
            if len(by_scope) != expected_calls or any(not by_scope[s["name"]] for s in scopes):
                raise ValueError("R contains an operator scope without recorded GPU work")
            with (run / "reference_operations.jsonl").open("x") as stream:
                for scope in scopes:
                    item = {"scope": scope["name"], "kernel_event_indices": by_scope[scope["name"]],
                            "api_start_ns": scope["start"], "api_end_ns": scope["end"],
                            "label_provenance": "reference_direct"}
                    stream.write(json.dumps(item) + "\n")
            result["reference_scope_count"] = len(scopes)
            result["reference_mapped_kernel_count"] = sum(map(len, by_scope.values()))
        else:
            if table_columns(conn, "NVTX_EVENTS"):
                unexpected = [r for r in rows(conn, "NVTX_EVENTS") if
                              (r["text"] or strings.get(r["textId"], "")).startswith("o1.op/")]
                if unexpected:
                    raise ValueError("P capture unexpectedly contains O1 per-operator NVTX scopes")
            # Predictor-safe event view. IDs are capture-local opaque references, never features.
            origin = min(min(x["api"]["start"], x["kernel"]["start"]) for x in mapped)
            with (run / "events_allowlist.jsonl").open("x") as stream:
                for x in mapped:
                    k, api, index = x["kernel"], x["api"], x["index"]
                    item = {"event_id": f"kernel_{index:08d}", "event_kind": "kernel",
                            "start_ns": k["start"]-origin, "end_ns": k["end"]-origin,
                            "process_id": "p0", "context_id": "c0", "stream_id": "s0",
                            "correlation_id": f"corr_{index:08d}", "api_event_id": f"api_{index:08d}",
                            "launch_order": index, "kernel_name": x["name"],
                            "grid_xyz": [k[f"grid{axis}"] for axis in "XYZ"],
                            "block_xyz": [k[f"block{axis}"] for axis in "XYZ"],
                            "dynamic_shared_memory_bytes": k["dynamicSharedMemory"],
                            "api_start_ns": api["start"]-origin, "api_end_ns": api["end"]-origin}
                    stream.write(json.dumps(item) + "\n")
        result["status"] = "manual_review_required" if result["warnings"] else "recorded_structure_passed"
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        conn.close()
        (run / "trace_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result

