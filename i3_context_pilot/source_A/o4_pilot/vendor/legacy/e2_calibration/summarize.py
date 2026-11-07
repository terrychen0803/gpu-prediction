"""Summarize completed E2 runs without modifying raw sample files."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .__main__ import HEADERS, cells, digest


def summarize(root):
    root = Path(root).resolve()
    plan = json.loads((root/'plan.json').read_text())
    if plan.get('status') != 'completed' or plan.get('smoke'):
        raise ValueError('E2 formal profile requires completed plan with smoke=false')
    expected_cells = {cat + '/' + '/'.join(map(str, spec)) for cat, spec in cells()}
    groups = int(plan['groups_per_cell'])
    repeats = int(plan['processes'])
    if groups != 5 or repeats != 3 or float(plan['target_ms_per_group']) != 100:
        raise ValueError('formal E2 protocol requires 3 processes, 5 groups and 100 ms target')
    raw, environments = [], []
    for repeat in range(1, repeats+1):
        run = root/f'run{repeat:02d}'
        env = json.loads((run/'environment.json').read_text())
        if env.get('status') != 'completed' or env.get('profile_id') != plan['profile_id']:
            raise ValueError(f'incomplete or mismatched E2 process: {run}')
        environments.append(env)
        with (run/'microbench_samples.csv').open(newline='') as f:
            rows = list(csv.DictReader(f))
        if len(rows) != len(expected_cells)*groups:
            raise ValueError(f'wrong sample count in {run}: {len(rows)}')
        seen = {(row['cell'], int(row['group'])) for row in rows}
        if seen != {(cell, group) for cell in expected_cells for group in range(1, groups+1)}:
            raise ValueError(f'missing or duplicate cell/group in {run}')
        if any(int(row['process_repeat']) != repeat or row['profile_id'] != plan['profile_id'] for row in rows):
            raise ValueError(f'wrong repeat/profile_id in {run}')
        raw.extend(rows)
    contract = ('gpu_uuid', 'gpu_name', 'torch', 'cuda_runtime', 'source_sha256',
                'library_sha256', 'smoke', 'groups', 'target_ms')
    if any(any(env[k] != environments[0][k] for k in contract) for env in environments[1:]):
        raise ValueError('hardware/software contract differs across E2 processes')
    by_cell = defaultdict(list)
    for row in raw:
        if int(row['repetitions']) <= 0 or float(row['host_complete_us']) <= 0:
            raise ValueError('invalid timing sample')
        if row['category'] != 'host' and float(row['gpu_event_us']) <= 0:
            raise ValueError('invalid CUDA event sample')
        by_cell[row['cell']].append(row)
    summary = []
    short_groups = []
    for cell in sorted(by_cell):
        samples = by_cell[cell]
        per_process = []
        for repeat in range(1, repeats+1):
            subset = [r for r in samples if int(r['process_repeat']) == repeat]
            # Host completion governs launch/sync; CUDA event governs device work.
            field = 'host_complete_us' if subset[0]['category'] in ('launch', 'host') else 'gpu_event_us'
            per_process.append(statistics.median(float(r[field])/int(r['repetitions'])/int(r['inner_n']) for r in subset))
            for r in subset:
                if float(r[field]) < 80_000:
                    short_groups.append({'cell': cell, 'repeat': repeat, 'group': int(r['group']),
                                         'duration_us': float(r[field]), 'timing_field': field})
        base = samples[0]
        value = statistics.median(per_process)
        work = float(base['flops'] or base['work_bytes'] or 0)
        summary.append({'category': base['category'], 'cell': cell,
                        'precision': base['precision'], 'memory_policy': base['memory_policy'],
                        'timing_metric': 'host_complete_us' if base['category'] in ('launch','host') else 'gpu_event_us',
                        'median_us_per_operation': value,
                        'process_spread_fraction': (max(per_process)-min(per_process))/value,
                        'work_per_operation': work,
                        'effective_work_per_s': work*1e6/value if work else '',
                        'unit': 'FLOP/s' if base['category']=='compute' else
                                'logical_byte/s' if base['category'] in ('memory','transfer') else 'operation/s',
                        'samples': len(samples)})
    with (root/'microbench_samples.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS); writer.writeheader(); writer.writerows(raw)
    with (root/'curves.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    profile = {'status': 'completed_with_duration_warnings' if short_groups else 'completed',
               'profile_id': plan['profile_id'], 'protocol': plan['protocol'],
               'gpu_name': environments[0]['gpu_name'], 'gpu_uuid': environments[0]['gpu_uuid'],
               'software_hash': environments[0]['software_hash'], 'source_sha256': environments[0]['source_sha256'],
               'library_sha256': environments[0]['library_sha256'], 'torch': environments[0]['torch'],
               'cuda_runtime': environments[0]['cuda_runtime'], 'nvcc': environments[0]['nvcc'],
               'processes': repeats, 'groups_per_cell': groups, 'target_ms_per_group': 100,
               'cells': len(summary), 'sample_rows': len(raw), 'short_duration_groups': short_groups,
               'anchors': {row['cell']: row for row in summary if row['cell'] in (
                   'compute/2048/2048/2048', 'compute/1/1024/1024',
                   'memory/copy/16777216/same_buffer', 'memory/copy/268435456/rotating_4_buffers',
                   'launch/short/100/batched', 'host/synchronize')},
               'notes': ['Effective logical FLOP/s and byte/s are not hardware peak or measured DRAM transactions.',
                         'Host completion for launch includes Python/ctypes dispatch and synchronization.',
                         'Device event spans for pageable transfer may include host blocking gaps.',
                         'Three devices are not enough to identify many independent hardware coefficients.']}
    (root/'hardware_profile.json').write_text(json.dumps(profile, indent=2, ensure_ascii=False)+'\n')
    return profile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root')
    args = p.parse_args()
    profile = summarize(args.root)
    print(json.dumps({k:profile[k] for k in ('status','gpu_name','cells','sample_rows')},indent=2))


if __name__ == '__main__':
    main()
