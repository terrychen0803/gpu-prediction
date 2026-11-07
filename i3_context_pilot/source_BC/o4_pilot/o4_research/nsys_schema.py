"""Explicit Nsight SQLite structural gate for O4 development-only audits.

Allowlist is based on observed 3.28.0 (GB10), 3.29.0 (RTX4090),
and 3.31.0 (RTX3090) layouts. 4090 3.29.0 was checked against the
required/optional/feature columns and SQLite quick_check on RN-A P/R smoke.
This is structural acceptance only; event/reference completeness remains unverified.
Passing column checks is NOT a proof of CUDA event completeness, timestamp
semantics, R-label correctness, or strict Reference eligibility.
"""
SUPPORTED_SCHEMAS = frozenset(("3.28.0", "3.29.0", "3.31.0"))

REQUIRED = {
    "META_DATA_EXPORT": {"name", "value"},
    "StringIds": {"id", "value"},
    "DIAGNOSTIC_EVENT": {"severity", "text"},
    "NVTX_EVENTS": {"globalTid", "text", "textId", "start", "end"},
    "CUPTI_ACTIVITY_KIND_RUNTIME": {"globalTid", "nameId", "correlationId", "start", "end"},
}

OPTIONAL_ACTIVITY = {
    "CUPTI_ACTIVITY_KIND_DRIVER": {"globalTid", "nameId", "correlationId", "start", "end"},
    "CUPTI_ACTIVITY_KIND_KERNEL": {"globalPid", "correlationId", "start", "end", "deviceId", "contextId", "streamId"},
    "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL": {"globalPid", "correlationId", "start", "end", "deviceId", "contextId", "streamId"},
    "CUPTI_ACTIVITY_KIND_MEMSET": {"globalPid", "correlationId", "start", "end", "deviceId", "contextId", "streamId", "bytes"},
    "CUPTI_ACTIVITY_KIND_MEMCPY": {"globalPid", "correlationId", "start", "end", "deviceId", "contextId", "streamId"},
}


def validate_export_schema(version, schema):
    """Check expected tables and columns before any semantic audit.

    `schema` maps table name to PRAGMA table_info rows (dicts containing name).
    No transformation or repair of raw event data is performed.
    """
    if version not in SUPPORTED_SCHEMAS:
        raise ValueError(f"Unknown Nsight export schema {version}; investigate first")

    present = {table: {column["name"] for column in cols}
               for table, cols in schema.items()}
    for table, fields in REQUIRED.items():
        if table not in present:
            raise ValueError(f"Nsight schema {version}: missing required table {table}")
        missing = fields - present[table]
        if missing:
            raise ValueError(f"Nsight schema {version}: {table} missing columns {sorted(missing)}")

    if not any(table in present for table in (
        "CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL"
    )):
        raise ValueError(f"Nsight schema {version}: no CUDA kernel activity table")

    for table, fields in OPTIONAL_ACTIVITY.items():
        if table in present:
            missing = fields - present[table]
            if missing:
                raise ValueError(f"Nsight schema {version}: {table} missing columns {sorted(missing)}")

    return {
        "export_schema": version,
        "structural_contract": "passed",
        "semantic_equivalence": "unverified",
        "gpu_event_completeness": "unverified",
        "strict_reference_verified": False,
        "no_timestamp_or_event_mutations": True,
    }
