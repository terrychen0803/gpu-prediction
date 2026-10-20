"""Read-only E3 pilot: group G labels and P0 source features by workload."""
from __future__ import annotations

import argparse
import math
import statistics
from collections import defaultdict
from pathlib import Path

from inference_pilot.common import read_csv, read_json, sha256, write_csv, write_json
from inference_pilot.report import discover, protocol

FEATURES = ('gpu_span_us', 'kernel_count', 'kernel_gap_fraction',
            'short_kernel_fraction_lt10us', 'launch_count')
REQUIRED_MODES = ('G', 'P0')
WEIGHTS_SHA256 = '72f27a9ddb2f3c57258816e525b12701cb0216f0e17192b780111c45ca8a8597'


def config_id(resolution, batch):
    return f'r{resolution}_b{batch}'


def read_run(directory):
    execution = read_json(directory / 'execution.json')
    if execution.get('status') != 'completed' or execution.get('gpu_executed') is not True:
        raise ValueError(f'incomplete/non-GPU run: {directory}')
    manifest = read_json(directory / 'manifest.json')
    env = read_json(directory / 'environment.json')
    quality = read_json(directory / 'quality.json')
    if quality.get('status') != 'passed' or not quality.get('correctness', {}).get('passed'):
        raise ValueError(f'failed quality/correctness: {directory}')
    mode = manifest['mode']
    if mode not in REQUIRED_MODES:
        return None
    resolution = manifest.get('resolution', manifest['shape'][2])
    batch = manifest['batch']
    shape = manifest['shape']
    if shape != [batch, 3, resolution, resolution]:
        raise ValueError(f'inconsistent shape: {directory}')
    if manifest.get('weights_sha256') != WEIGHTS_SHA256 or not manifest.get('input_file_sha256'):
        raise ValueError(f'wrong/missing fixed weights or input: {directory}')
    measured = [r for r in read_csv(directory / 'latency.csv') if r['stage'] == 'measure']
    if len(measured) != manifest['measurement_count'] or not measured:
        raise ValueError(f'measured iteration mismatch: {directory}')
    latencies = [float(r['latency_us']) for r in measured]
    if any(not math.isfinite(x) or x <= 0 for x in latencies):
        raise ValueError(f'invalid host latency: {directory}')
    feature_medians = {}
    trace_warnings = []
    if mode == 'P0':
        trace = read_json(directory / 'trace_quality.json')
        if trace.get('status') != 'passed' or trace.get('segmentation') != 'synchronization_assisted_not_blind':
            raise ValueError(f'P0 trace parse/segmentation failed: {directory}')
        rows = read_csv(directory / 'features.csv')
        if len(rows) != len(measured) or {int(r['iteration']) for r in rows} != set(range(len(measured))):
            raise ValueError(f'P0 trace feature count/iteration mismatch: {directory}')
        for key in FEATURES:
            values = [float(r[key]) for r in rows]
            if any(not math.isfinite(v) for v in values) or (key == 'gpu_span_us' and min(values) <= 0):
                raise ValueError(f'invalid P0 feature {key}: {directory}')
            feature_medians[key] = statistics.median(values)
        trace_warnings = trace.get('profiler_diagnostic_warnings', [])
    return {
        'run_dir': str(directory), 'label': manifest['label'], 'gpu_uuid': env['gpu_uuid'],
        'resolution': resolution, 'batch': batch, 'config_id': config_id(resolution, batch),
        'mode': mode, 'repeat': manifest['repeat'], 'protocol': protocol(manifest, env),
        'code_sha256': manifest['code_sha256'], 'input_sha256': manifest['input_sha256'],
        'input_file_sha256': manifest['input_file_sha256'], 'weights_sha256': manifest['weights_sha256'],
        'run_median_us': statistics.median(latencies), 'feature_medians': feature_medians,
        'trace_warnings': trace_warnings,
        'manifest_sha256': sha256(directory / 'manifest.json'),
        'latency_sha256': sha256(directory / 'latency.csv'),
        'features_sha256': sha256(directory / 'features.csv') if mode == 'P0' else '',
    }


def collect(roots):
    inventory = []
    for directory in discover(roots):
        run = read_run(directory)
        if run is not None:
            inventory.append(run)
    buckets = defaultdict(list)
    for run in inventory:
        buckets[(run['label'], run['config_id'], run['mode'])].append(run)
    devices = {r['label'] for r in inventory}
    configs = {r['config_id'] for r in inventory}
    if not devices or not configs:
        raise ValueError('No completed G/P0 GPU runs')
    if {(label, cfg, mode) for label in devices for cfg in configs for mode in REQUIRED_MODES} != set(buckets):
        raise ValueError('G/P0 matrix incomplete across devices/configurations')
    feature_rows, label_rows = [], []
    contracts = {}
    for (label, cfg, mode), runs in sorted(buckets.items()):
        if len(runs) != 3 or {r['repeat'] for r in runs} != {1, 2, 3}:
            raise ValueError(f'Need exactly three independent repeats: {label} {cfg} {mode}')
        for key in ('gpu_uuid', 'protocol', 'code_sha256', 'input_sha256',
                    'input_file_sha256', 'weights_sha256', 'resolution', 'batch'):
            if len({r[key] for r in runs}) != 1:
                raise ValueError(f'Inconsistent {key}: {label} {cfg} {mode}')
        contract = {key: runs[0][key] for key in ('gpu_uuid', 'protocol', 'code_sha256',
                    'input_sha256', 'input_file_sha256', 'weights_sha256', 'resolution', 'batch')}
        if (label, cfg) in contracts and contract != contracts[(label, cfg)]:
            raise ValueError(f'G/P0 contract mismatch: {label} {cfg}')
        contracts[(label, cfg)] = contract
        base = {'label': label, 'config_id': cfg, 'resolution': runs[0]['resolution'],
                'batch': runs[0]['batch'], 'gpu_uuid': runs[0]['gpu_uuid'],
                'protocol': runs[0]['protocol'], 'input_sha256': runs[0]['input_sha256']}
        if mode == 'G':
            medians = [r['run_median_us'] for r in runs]
            value = statistics.median(medians)
            spread = (max(medians) - min(medians)) / value
            if spread > 0.05:
                raise ValueError(f'Unstable G run medians: {label} {cfg}: {spread:.2%}')
            label_rows.append({**base, 'g_us': value, 'g_repeat_spread': spread})
        else:
            feature_rows.append({**base, **{key: statistics.median(r['feature_medians'][key] for r in runs)
                                             for key in FEATURES},
                                 'trace_warning_runs': sum(bool(r['trace_warnings']) for r in runs)})
    for cfg in configs:
        for key in ('protocol', 'input_sha256', 'input_file_sha256', 'weights_sha256', 'code_sha256'):
            if len({contracts[(label, cfg)][key] for label in devices}) != 1:
                raise ValueError(f'Cross-device contract mismatch: {cfg} {key}')
    return inventory, feature_rows, label_rows


def fit_c1(train_features, train_labels, ridge=1.0):
    """Log-linear calibration, ridge toward slope 1; never sees held-out labels."""
    label_map = {(r['label'], r['config_id']): r['g_us'] for r in train_labels}
    pairs = [(math.log(float(r['gpu_span_us'])), math.log(float(label_map[(r['label'], r['config_id'])])))
             for r in train_features]
    if len(pairs) < 2:
        raise ValueError('C1 requires at least two training configurations')
    xs, ys = zip(*pairs)
    xm, ym = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - xm) ** 2 for x in xs)
    sxy = sum((x - xm) * (y - ym) for x, y in pairs)
    slope = (sxy + ridge) / (sxx + ridge)
    return ym - slope * xm, slope


def evaluate(features, labels):
    by_label = {(r['label'], r['config_id']): r for r in labels}
    configs = sorted({r['config_id'] for r in features})
    splits, predictions = [], []
    for held in configs:
        train_configs = [cfg for cfg in configs if cfg != held]
        splits.append({'held_out_config': held, 'training_configs': train_configs})
        for device in sorted({r['label'] for r in features}):
            training_features = [r for r in features if r['label'] == device and r['config_id'] != held]
            training_labels = [r for r in labels if r['label'] == device and r['config_id'] != held]
            a, b = fit_c1(training_features, training_labels)
            test = next(r for r in features if r['label'] == device and r['config_id'] == held)
            actual = float(by_label[(device, held)]['g_us'])
            for model, estimate in (('C0_gpu_span', float(test['gpu_span_us'])),
                                    ('C1_log_linear_ridge1', math.exp(a + b * math.log(float(test['gpu_span_us']))))):
                predictions.append({'held_out_config': held, 'label': device, 'model': model,
                                    'prediction_us': estimate, 'actual_g_us': actual,
                                    'absolute_percentage_error': abs(estimate / actual - 1) * 100,
                                    'trace_warning_runs': test['trace_warning_runs']})
    return splits, predictions


def main(argv=None):
    p = argparse.ArgumentParser(description='E3 G/P0 source calibration, grouped pilot only')
    p.add_argument('--out', required=True)
    p.add_argument('roots', nargs='+', help='Exact formal E1 session roots; no smoke/plan directories')
    args = p.parse_args(argv)
    out = Path(args.out).resolve()
    if out.exists():
        p.error(f'Output exists; use a new directory: {out}')
    inventory, features, labels = collect(args.roots)
    splits, predictions = evaluate(features, labels)
    out.mkdir(parents=True)
    write_csv(out / 'source_features.csv', features)
    write_csv(out / 'g_labels_evaluator_only.csv', labels)
    write_csv(out / 'predictions_evaluator_only.csv', predictions)
    write_json(out / 'split_manifest.json', splits)
    write_json(out / 'run_inventory.json', inventory)
    write_json(out / 'quality.json', {
        'status': 'pilot_only_trace_review_pending' if any(r['trace_warning_runs'] for r in features) else 'pilot_only',
        'devices': sorted({r['label'] for r in features}),
        'configurations': sorted({r['config_id'] for r in features}),
        'formal_run_count': len(inventory), 'source_feature_columns': list(FEATURES),
        'models': ['C0_gpu_span', 'C1_log_linear_ridge1'],
        'note': 'Three configurations per device are insufficient for a formal generalization claim. All P0 event warnings remain visible.'})
    print(f'E3 pilot: {len(features)} device-configurations, {len(predictions)} held-out predictions -> {out}')


if __name__ == '__main__':
    main()
