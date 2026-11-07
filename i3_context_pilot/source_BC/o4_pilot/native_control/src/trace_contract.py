"""Additive engineering acceptance, independent of strict reference qualification.

No trace repair. Only the exact known NVTX warning is provisionally accepted.
This module also works in the standalone native package.
"""
import math

VERSION = 'o4_engineering_acceptance_v1_20261106'
KNOWN_NVTX = 'Not all NVTX events might have been collected.'
FEATURE_COLUMNS = {
    'CUPTI_ACTIVITY_KIND_KERNEL': {'demangledName', 'gridX', 'gridY', 'gridZ', 'blockX', 'blockY', 'blockZ', 'dynamicSharedMemory'},
    'CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL': {'demangledName', 'gridX', 'gridY', 'gridZ', 'blockX', 'blockY', 'blockZ', 'dynamicSharedMemory'},
    'CUPTI_ACTIVITY_KIND_MEMCPY': {'bytes', 'copyKind', 'srcKind', 'dstKind'},
    'CUPTI_ACTIVITY_KIND_MEMSET': {'bytes', 'memKind', 'value'},
}

def diagnostic_policy(diagnostics, requested='cuda-sw'):
    accepted, blocked = [], []
    software = hardware = False
    for row in diagnostics:
        text = ' '.join(str(row.get('text', '')).split())
        low = text.lower()
        software |= 'cuda legacy/software instrumented trace was collected' in low
        hardware |= 'hardware tracing used for cuda tracing' in low
        severity = row.get('severity')
        if not isinstance(severity, int):
            blocked.append({'kind': 'unknown_severity', 'text': text})
            continue
        if any(k in low for k in ('incomplete cupti', 'events dropped', 'not all cuda events', 'buffer overflow')):
            blocked.append({'kind': 'cuda_completeness', 'text': text})
        elif severity >= 2:
            if text == KNOWN_NVTX and severity == 2:
                accepted.append({'kind': 'known_nvtx', 'text': text})
            else:
                blocked.append({'kind': 'unreviewed_diagnostic', 'text': text})
    actual = 'conflicting' if software and hardware else 'cuda-sw' if software else 'cuda' if hardware else 'unverified'
    if actual != requested:
        blocked.append({'kind': 'backend_mismatch_or_unverified', 'actual': actual, 'requested': requested})
    return {'accepted_warnings': accepted, 'blockers': blocked, 'actual_backend': actual}

def feature_contract(schema, events):
    failures = []
    for table, required in FEATURE_COLUMNS.items():
        if table in schema:
            present = {x['name'] for x in schema[table]}
            missing = required - present
            if missing:
                failures.append({'table': table, 'missing_columns': sorted(missing)})
    def valid_int(x, minimum=0):
        return isinstance(x, int) and not isinstance(x, bool) and x >= minimum
    for e in events:
        kind = e['event_kind']
        bad = []
        if kind == 'kernel':
            for key in ('grid_xyz', 'block_xyz'):
                v = e.get(key)
                if not isinstance(v, list) or len(v) != 3 or not all(valid_int(x, 1) for x in v):bad.append(key)
            if not valid_int(e.get('dynamic_shared_memory_bytes')):bad.append('dynamic_shared_memory_bytes')
            if not e.get('kernel_name'):bad.append('kernel_name_for_evaluator_signature')
        elif kind in ('copy', 'memset'):
            if not valid_int(e.get('bytes'), 1):bad.append('bytes')
            if kind == 'copy':
                if not valid_int(e.get('copy_kind')):bad.append('copy_kind')
            else:
                for key in ('memory_kind', 'memset_value'):
                    if not valid_int(e.get(key)):bad.append(key)
        else:bad.append('event_kind')
        if bad:failures.append({'uid': e.get('uid'), 'invalid_fields': bad})
    return failures

def api_return_audit(apis):
    failures = []
    for a in apis:
        # Runtime event queries may legitimately return not-ready (34).
        name = a.get('name', '').lower()
        value = a.get('returnValue')
        if value == 0:continue
        if value in (34, 600) and ('eventquery' in name or 'streamquery' in name):continue
        failures.append({'api_key': a.get('key'), 'name': a.get('name'), 'returnValue': value,
                         'reason': 'missing_return_value' if value is None else 'nonzero_return'})
    return failures

def acceptance(diagnostics, schema, events, apis, recorded_ok, requested='cuda-sw'):
    policy = diagnostic_policy(diagnostics, requested)
    feature_failures = feature_contract(schema, events)
    api_failures = api_return_audit(apis)
    blockers = list(policy['blockers'])
    if not recorded_ok:blockers.append({'kind': 'recorded_structure_failed'})
    if not events:blockers.append({'kind': 'no_recorded_GPU_work'})
    if feature_failures:blockers.append({'kind': 'feature_contract_failed', 'count': len(feature_failures)})
    if api_failures:blockers.append({'kind': 'api_return_failed', 'count': len(api_failures)})
    return {'version': VERSION, 'status': 'blocked' if blockers else 'warning_accepted' if policy['accepted_warnings'] else 'passed',
            'accepted': not blockers, 'blockers': blockers, 'accepted_warnings': policy['accepted_warnings'],
            'actual_backend': policy['actual_backend'], 'feature_failures': feature_failures,
            'api_return_failures': api_failures, 'strict_verified': 0, 'training_eligible': False,
            'scope': 'recorded engineering evidence only; unmatched work, clock and reference gates remain independent'}
