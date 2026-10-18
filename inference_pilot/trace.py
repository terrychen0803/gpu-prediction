"""Conservative adapter for Nsight Systems SQLite exports.

Supported: one CUDA worker, runtime/driver launch correlation, NVTX push/pop ranges.
Unsupported schemas or ambiguous joins fail explicitly; SQLite is always read-only.
"""
from __future__ import annotations

import bisect
import collections
import re
import sqlite3
from pathlib import Path

from .common import PHASES, quantile, read_json, union_length, write_csv, write_json

ITERATION = re.compile(r"^pilot\.iter/(transition|measure)/(\d+)$")
MASK = (1 << 24) - 1


class TraceError(ValueError):
    pass


def process_id(global_id):
    return (int(global_id) >> 24) & MASK


def quote(identifier):
    return '"' + identifier.replace('"', '""') + '"'


class Export:
    def __init__(self, path):
        self.db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
        self.db.row_factory = sqlite3.Row
        self.schema = {row[0]: [r[1] for r in self.db.execute(f"PRAGMA table_info({quote(row[0])})")]
                       for row in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        self.strings = {r["id"]: r["value"] for r in self.rows("StringIds")} if "StringIds" in self.schema else {}

    def rows(self, name):
        return [dict(row) for row in self.db.execute(f"SELECT * FROM {quote(name)}")]

    def require(self, name, columns):
        missing = set(columns) - set(self.schema.get(name, []))
        if missing:
            raise TraceError(f"Unsupported schema: {name} missing {sorted(missing)}; inspect trace_schema.json")

    def text(self, row, direct, ids):
        for key in direct:
            if isinstance(row.get(key), str) and row[key]:
                return row[key]
        for key in ids:
            if row.get(key) in self.strings:
                return self.strings[row[key]]
        return ""


def containing(ranges, api):
    return [r for r in ranges if r["globalTid"] == api["globalTid"] and
            r["start"] <= api["start"] and api["end"] <= r["end"]]


def correlate(event, api_index):
    if event.get("correlationId") is None:
        return None, "missing_correlation_id"
    key = (process_id(event["globalPid"]), event["correlationId"])
    candidates = [a for a in api_index.get(key, []) if a["start"] <= event["start"] and
                  (a.get("contextId") is None or event.get("contextId") is None or a["contextId"] == event["contextId"])]
    # Runtime and nested driver records can both exist; prefer the runtime source.
    for kind in ("runtime", "driver"):
        subset = [a for a in candidates if a["source"] == kind]
        if len(subset) == 1:
            return subset[0], "matched"
        if len(subset) > 1:
            return None, "ambiguous_correlation_id"
    return None, "no_correlated_api"


def kernel_features(kernels):
    durations = [(k["end"] - k["start"]) / 1000 for k in kernels]
    busy = union_length((k["start"], k["end"]) for k in kernels) / 1000
    span = (max(k["end"] for k in kernels) - min(k["start"] for k in kernels)) / 1000 if kernels else 0
    def products(prefix):
        keys = [prefix + axis for axis in ("X", "Y", "Z")]
        return [k[keys[0]] * k[keys[1]] * k[keys[2]] for k in kernels if all(k.get(key) is not None for key in keys)]
    grids, blocks = products("grid"), products("block")
    return {
        "kernel_count": len(kernels), "kernel_duration_sum_us": sum(durations),
        "kernel_duration_p10_us": quantile(durations, .1), "kernel_duration_p50_us": quantile(durations, .5),
        "kernel_duration_p90_us": quantile(durations, .9), "kernel_busy_union_us": busy,
        "gpu_span_us": span, "kernel_gap_fraction": 1 - busy / span if span else None,
        "short_kernel_fraction_lt10us": sum(t < 10 for t in durations) / len(durations) if durations else None,
        "stream_count": len({k["streamId"] for k in kernels}),
        "grid_blocks_p50": quantile(grids, .5) if len(grids) == len(kernels) else None,
        "block_threads_p50": quantile(blocks, .5) if len(blocks) == len(kernels) else None,
        "launch_count": len({k["api_id"] for k in kernels if k.get("api_id")}),
    }


def parse_trace(run_dir, sqlite_path=None):
    out = Path(run_dir).resolve()
    quality = {"status": "failed", "errors": [], "warnings": [], "adapter": "nsys_sqlite_v1"}
    export = None
    try:
        manifest = read_json(out / "manifest.json")
        environment = read_json(out / "environment.json")
        mode = manifest["mode"]
        if mode not in ("P0", "P1"):
            raise TraceError("Only P0 and P1 have GPU trace data")
        export = Export(sqlite_path or out / "trace.sqlite")
        write_json(out / "trace_schema.json", export.schema)
        pid, tid = environment["pid"], environment["native_thread_id"]
        expected = manifest["measurement_count"]
        transition = manifest["transition_count"]
        apis = []
        for table, kind in (("CUPTI_ACTIVITY_KIND_RUNTIME", "runtime"), ("CUPTI_ACTIVITY_KIND_DRIVER", "driver")):
            if table not in export.schema:
                continue
            export.require(table, ("start", "end", "globalTid", "correlationId"))
            for idx, row in enumerate(export.rows(table)):
                if process_id(row["globalTid"]) != pid:
                    continue
                row.update({"source": kind, "api_id": f"{kind}:{idx}", "name": export.text(row, ["name"], ["nameId"])})
                if row["end"] is None or row["end"] < row["start"]:
                    raise TraceError("Incomplete CUDA API interval")
                apis.append(row)
        if not apis:
            raise TraceError("No CUDA API records for manifest PID; check global ID encoding or schema")
        api_index = collections.defaultdict(list)
        for a in apis:
            api_index[(pid, a["correlationId"])].append(a)
        main_apis = [a for a in apis if (a["globalTid"] & MASK) == tid]
        if not main_apis:
            raise TraceError("No CUDA APIs on the worker's recorded native thread")

        phases, windows = [], []
        phase_by_window = {}
        if mode == "P1":
            export.require("NVTX_EVENTS", ("start", "end", "globalTid"))
            for row in export.rows("NVTX_EVENTS"):
                if row.get("globalTid") is None or process_id(row["globalTid"]) != pid or (row["globalTid"] & MASK) != tid:
                    continue
                name = export.text(row, ["text"], ["textId"])
                if not (name.startswith("pilot.iter/") or name.startswith("pilot.phase/")):
                    continue
                if row["end"] is None or row["end"] <= row["start"]:
                    raise TraceError(f"Incomplete NVTX range: {name}")
                if row.get("eventType", 59) != 59:
                    raise TraceError("This adapter expects NVTX push/pop ranges (eventType 59)")
                match = ITERATION.match(name)
                if match:
                    row.update({"stage": match[1], "iteration": int(match[2]), "name": name})
                    windows.append(row)
                elif name.startswith("pilot.phase/"):
                    phase = name.removeprefix("pilot.phase/")
                    if phase not in PHASES:
                        raise TraceError(f"Unknown phase: {phase}")
                    row["phase"] = phase
                    phases.append(row)
            windows.sort(key=lambda w: w["start"])
            for stage, count in (("transition", transition), ("measure", expected)):
                ids = sorted(w["iteration"] for w in windows if w["stage"] == stage)
                if ids != list(range(count)):
                    raise TraceError(f"NVTX {stage} count/IDs differ from manifest; expected {count}, found {len(ids)}")
            phase_starts = sorted(phases, key=lambda p: p["start"])
            phase_times = [p["start"] for p in phase_starts]
            for wi, w in enumerate(windows):
                left = bisect.bisect_left(phase_times, w["start"])
                right = bisect.bisect_right(phase_times, w["end"])
                contained = [p for p in phase_starts[left:right] if p["globalTid"] == w["globalTid"] and p["end"] <= w["end"]]
                if sorted(p["phase"] for p in contained) != sorted(PHASES):
                    raise TraceError("Each iteration must contain exactly the six expected phase ranges")
                if any(b["start"] < a["end"] for a, b in zip(contained, contained[1:])):
                    raise TraceError("The six reference phase ranges must not overlap")
                phase_by_window[wi] = contained
            quality["segmentation"] = "NVTX_reference_boundaries"
        else:
            syncs = [a for a in main_apis if a["source"] == "runtime" and a["name"].startswith("cudaDeviceSynchronize")]
            if not syncs:
                syncs = [a for a in main_apis if a["source"] == "driver" and a["name"].startswith("cuCtxSynchronize")]
            syncs.sort(key=lambda a: a["start"])
            if len(syncs) != expected + transition:
                raise TraceError(f"P0 synchronization count differs: expected {expected + transition}, found {len(syncs)}. No guessed truncation is allowed.")
            previous = min(a["start"] for a in main_apis)
            for idx, sync in enumerate(syncs):
                is_transition = idx < transition
                windows.append({"start": previous, "end": sync["end"], "globalTid": sync["globalTid"],
                                "stage": "transition" if is_transition else "measure", "iteration": idx if is_transition else idx - transition})
                previous = sync["end"]
            quality["segmentation"] = "synchronization_assisted_not_blind"
        for a, b in zip(windows, windows[1:]):
            if b["start"] < a["end"]:
                raise TraceError("Overlapping iteration boundaries are not supported")
        starts = [w["start"] for w in windows]

        def window_for(api):
            if api is None:
                return None
            idx = bisect.bisect_right(starts, api["start"]) - 1
            if idx >= 0:
                w = windows[idx]
                if api["globalTid"] == w["globalTid"] and api["end"] <= w["end"]:
                    return idx
            return None

        # Detect the export table; do not concatenate duplicate kernel table variants.
        candidates = [name for name in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL")
                      if name in export.schema and export.db.execute(f"SELECT 1 FROM {quote(name)} LIMIT 1").fetchone()]
        if len(candidates) != 1:
            raise TraceError(f"Expected one populated kernel table, found {candidates}")
        table = candidates[0]
        export.require(table, ("start", "end", "globalPid", "correlationId", "streamId"))
        grouped = collections.defaultdict(list)
        mappings, unmapped = [], []
        devices, contexts = set(), set()
        for idx, k in enumerate(export.rows(table)):
            if process_id(k["globalPid"]) != pid:
                continue
            if k["end"] is None or k["end"] <= k["start"]:
                raise TraceError("Incomplete kernel interval")
            devices.add(k.get("deviceId"))
            contexts.add(k.get("contextId"))
            api, reason = correlate(k, api_index)
            if api is not None and "launch" not in api["name"].lower():
                api, reason = None, "correlated_api_is_not_a_kernel_launch"
            wi = window_for(api)
            phase = None
            if wi is not None and mode == "P1":
                matches = containing(phase_by_window[wi], api)
                if matches:
                    phase = max(matches, key=lambda p: p["start"])["phase"]
            if api is not None and wi is None:
                reason = "api_outside_iteration"
            if wi is not None and not (windows[wi]["start"] <= k["start"] < k["end"] <= windows[wi]["end"]):
                reason = "gpu_work_crosses_completed_iteration"
                wi = None
            if wi is None:
                unmapped.append({"kernel_id": idx, "reason": reason})
            if wi is not None:
                k.update({"api_id": api["api_id"], "phase": phase})
                grouped[wi].append(k)
            w = windows[wi] if wi is not None else {}
            mappings.append({"kernel_id": idx, "stage": w.get("stage"), "iteration": w.get("iteration"),
                             "phase": phase, "mapping_reason": reason if wi is None else ("matched" if phase or mode == "P0" else "no_phase"),
                             "kernel_name": export.text(k, ["name"], ["demangledName", "shortName", "mangledName"]),
                             "kernel_start_ns": k["start"], "kernel_end_ns": k["end"],
                             "duration_us": (k["end"] - k["start"]) / 1000, "stream_id": k["streamId"],
                             "correlation_id": k["correlationId"], "api_id": api["api_id"] if api else None,
                             "api_start_ns": api["start"] if api else None, "api_end_ns": api["end"] if api else None})
        write_csv(out / "phase_mapping.csv", mappings)
        if len(devices) > 1 or len(contexts) > 1:
            raise TraceError("This single-worker adapter requires one CUDA device and context")
        if unmapped:
            quality["unassigned_kernels"] = unmapped
            raise TraceError(f"{len(unmapped)} kernels cannot be assigned to complete iterations; inspect phase_mapping.csv")
        copies = collections.defaultdict(list)
        copy_table = "CUPTI_ACTIVITY_KIND_MEMCPY"
        if copy_table in export.schema:
            export.require(copy_table, ("start", "end", "globalPid", "correlationId", "bytes"))
            for c in export.rows(copy_table):
                if process_id(c["globalPid"]) != pid:
                    continue
                api, reason = correlate(c, api_index)
                wi = window_for(api)
                if wi is None:
                    raise TraceError(f"Unassigned memory copy: {reason}")
                copies[wi].append(c)
        features, phase_rows = [], []
        measurement = []
        for wi, w in enumerate(windows):
            kernels = grouped[wi]
            if not kernels:
                raise TraceError("An iteration contains no kernels")
            if w["stage"] != "measure":
                continue
            measurement.extend(kernels)
            if mode == "P1":
                assigned_here = [k for k in kernels if k.get("phase")]
                count_coverage = len(assigned_here) / len(kernels)
                duration_coverage = sum(k["end"] - k["start"] for k in assigned_here) / sum(k["end"] - k["start"] for k in kernels)
                if count_coverage < .99 or duration_coverage < .99:
                    raise TraceError(f"Iteration {w['iteration']} phase mapping coverage below 99%")
            row = {"iteration": w["iteration"], "boundary_method": quality["segmentation"], **kernel_features(kernels),
                   "memcpy_count": len(copies[wi]), "memcpy_bytes": sum(c["bytes"] for c in copies[wi])}
            # No batch, model name, phase labels, target time, or NVTX text in the main feature table.
            features.append(row)
            if mode == "P1":
                for name in (*PHASES, "unassigned"):
                    subset = [k for k in kernels if (k.get("phase") or "unassigned") == name]
                    phase_rows.append({"iteration": w["iteration"], "phase": name, **kernel_features(subset)})
        assigned = [k for k in measurement if k.get("phase")]
        if mode == "P1":
            count_fraction = len(assigned) / len(measurement)
            duration_fraction = sum(k["end"] - k["start"] for k in assigned) / sum(k["end"] - k["start"] for k in measurement)
            quality.update({"phase_count_coverage": count_fraction, "phase_duration_coverage": duration_fraction})
            if count_fraction < .99 or duration_fraction < .99:
                raise TraceError("Phase mapping coverage is below 99%")
        write_csv(out / "features.csv", features)
        write_csv(out / "phase_summary.csv", phase_rows, ["iteration", "phase", *kernel_features([]).keys()])
        write_csv(out / "reference_windows.csv", [{k: w.get(k) for k in ("stage", "iteration", "start", "end", "globalTid")} for w in windows])
        write_csv(out / "phase_ranges.csv", [
            {"stage": windows[wi]["stage"], "iteration": windows[wi]["iteration"], "phase": p["phase"],
             "start_ns": p["start"], "end_ns": p["end"]}
            for wi, ps in phase_by_window.items() for p in ps
        ], ["stage", "iteration", "phase", "start_ns", "end_ns"])
        log_flags = []
        for path in (out / "profile.log", out / "export.log"):
            if path.exists():
                log_flags += [line for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
                              if re.search(r"\b(lost|dropped|overflow|warning|error)\b", line, re.I)]
        diagnostic_tables = [t for t in export.schema if "DIAGNOSTIC" in t.upper()]
        write_json(out / "trace_diagnostics.json", {"log_flags": log_flags,
                   "diagnostic_tables": {t: export.rows(t)[:100] for t in diagnostic_tables}})
        quality.update({"status": "passed", "measurement_count": len(features), "kernel_table": table,
                        "trace_integrity": "manual_review_required", "log_flags": log_flags,
                        "warnings": ["Parser checks do not establish that the profiler lost no events; inspect trace_diagnostics.json and Nsight report diagnostics."]})
    except Exception as exc:
        quality["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        if export:
            export.db.close()
        write_json(out / "trace_quality.json", quality)
    return quality
