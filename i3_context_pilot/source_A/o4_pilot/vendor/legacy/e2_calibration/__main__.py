"""Portable E2 Compute/Memory/Launch/Host calibration.

The parent process starts one independent worker per repeat. All GPU timing uses
CUDA events; host submission/completion use perf_counter_ns. No profiler is
enabled for formal samples.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROTOCOL = 'e2_calibration_v3'
MAX_REPETITIONS = 100000
SIZES = (64 << 10, 256 << 10, 1 << 20, 4 << 20, 16 << 20, 64 << 20, 256 << 20)
TRANSFER_SIZES = (4 << 10, 16 << 10, 64 << 10, 256 << 10, 1 << 20, 4 << 20, 16 << 20, 64 << 20)
SQUARE = (128, 512, 2048)
SKINNY_M = (1, 8, 32)
SKINNY_NK = (1024, 4096)
LAUNCH_N = (1, 10, 100, 1000)
SOURCE = Path(__file__).with_name('kernels.cu')
HEADERS = ('profile_id', 'gpu_uuid', 'gpu_name', 'software_hash', 'code_sha256', 'category',
           'cell', 'shape_or_bytes', 'precision', 'layout', 'memory_policy', 'timing_scope',
           'process_repeat', 'group', 'repetitions', 'inner_n', 'gpu_event_us',
           'host_enqueue_us', 'host_complete_us', 'work_bytes', 'flops',
           'utc_start', 'utc_end', 'monotonic_start_ns', 'monotonic_end_ns', 'notes')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def code_digest():
    h = hashlib.sha256()
    for path in (SOURCE, Path(__file__)):
        h.update(path.read_bytes())
    return h.hexdigest()


def utc_now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def cells(smoke=False):
    if smoke:
        return [('compute', (128, 128, 128)), ('memory', ('copy', 1 << 20, 'same_buffer')),
                ('launch', ('empty', 10, 'batched')), ('transfer', ('H2D', 1 << 20, 'pinned')),
                ('host', ('synchronize',))]
    result = [('compute', (n, n, n)) for n in SQUARE]
    result += [('compute', (m, nk, nk)) for m in SKINNY_M for nk in SKINNY_NK]
    result += [('memory', (op, size, policy)) for op in ('copy', 'read', 'write')
               for size in SIZES for policy in ('same_buffer', 'rotating_4_buffers')]
    result += [('launch', (kernel, n, policy)) for kernel in ('empty', 'short')
               for n in LAUNCH_N for policy in ('batched', 'sync_each')]
    result += [('transfer', (direction, size, alloc)) for direction in ('H2D', 'D2H')
               for size in TRANSFER_SIZES for alloc in ('pinned', 'pageable')]
    result += [('host', ('synchronize',))]
    return result


def compile_library(path, capability):
    if path.exists():
        return path
    arch = ''.join(str(x) for x in capability)
    cmd = ['nvcc', '-O3', '-std=c++17', '-shared', '-Xcompiler', '-fPIC',
           '-gencode', f'arch=compute_{arch},code=sm_{arch}', str(SOURCE), '-o', str(path)]
    done = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if done.returncode:
        raise RuntimeError(f'nvcc compilation failed: {done.stderr[-3000:]}')
    return path


def gpu_library(path):
    lib = ctypes.CDLL(str(path))
    ptr = ctypes.c_void_p
    lib.e2_launch_empty.argtypes = [ptr]
    lib.e2_launch_short.argtypes = [ptr, ptr]
    lib.e2_launch_memory.argtypes = [ctypes.c_int, ptr, ptr, ptr, ctypes.c_size_t, ptr]
    for name in ('e2_launch_empty', 'e2_launch_short', 'e2_launch_memory'):
        getattr(lib, name).restype = ctypes.c_int
    return lib


def checked(code, op):
    if code:
        raise RuntimeError(f'CUDA {op} failed with error code {code}')


def make_operation(torch, lib, category, args):
    """Return operation, active tensors and metadata; allocations stay outside timing."""
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    if category == 'compute':
        m, n, k = args
        a = torch.ones((m, k), dtype=torch.float16, device='cuda')
        b = torch.ones((k, n), dtype=torch.float16, device='cuda')
        out = torch.empty((m, n), dtype=torch.float16, device='cuda')
        def op(_):
            torch.mm(a, b, out=out)
        op(0); torch.cuda.synchronize()
        if not bool(torch.all(out == k)):
            raise RuntimeError(f'GEMM validation failed: {args}')
        return op, (a, b, out), {'flops': 2*m*n*k, 'work_bytes': '', 'precision': 'FP16',
                                 'layout': 'row_major', 'memory_policy': 'resident', 'inner_n': 1,
                                 'notes': 'torch.mm; FP16 input/output; accumulation selected by backend'}
    if category == 'memory':
        kind, size, policy = args
        count = 1 if policy == 'same_buffer' else 4
        srcs = [torch.full((size,), 1, dtype=torch.uint8, device='cuda') for _ in range(count)]
        dsts = [torch.empty((size,), dtype=torch.uint8, device='cuda') for _ in range(count)]
        accum = torch.zeros((1,), dtype=torch.int32, device='cuda')
        def op(i):
            j = i % count
            checked(lib.e2_launch_memory({'copy': 0, 'read': 1, 'write': 2}[kind],
                    srcs[j].data_ptr(), dsts[j].data_ptr(), accum.data_ptr(), size, stream), kind)
        op(0); torch.cuda.synchronize()
        if kind == 'copy' and not bool(torch.equal(srcs[0], dsts[0])):
            raise RuntimeError('copy kernel failed byte validation')
        if kind == 'write' and not bool(torch.all(dsts[0] == 0xA5)):
            raise RuntimeError('write kernel failed byte validation')
        if kind == 'read' and int(accum.item()) != ((size//4)*0x01010101) % (1 << 32) - (1 << 32) * (((size//4)*0x01010101) % (1 << 32) >= (1 << 31)):
            raise RuntimeError('read kernel failed checksum validation')
        return op, (srcs, dsts, accum), {'flops': '', 'work_bytes': size*(2 if kind == 'copy' else 1),
                                          'precision': 'uint32', 'layout': 'contiguous',
                                          'memory_policy': policy, 'inner_n': 1,
                                          'notes': 'logical bytes; no DRAM/cache-hit inference'}
    if category == 'launch':
        kernel, n, policy = args
        counter = torch.zeros((1,), dtype=torch.int32, device='cuda')
        def one():
            if kernel == 'empty':
                checked(lib.e2_launch_empty(stream), 'empty launch')
            else:
                checked(lib.e2_launch_short(counter.data_ptr(), stream), 'short launch')
        def op(_):
            for _ in range(n):
                one()
                if policy == 'sync_each': torch.cuda.synchronize()
        op(0); torch.cuda.synchronize()
        if kernel == 'short' and int(counter.item()) != n:
            raise RuntimeError('short kernel launch count failed')
        return op, (counter,), {'flops': '', 'work_bytes': '', 'precision': 'uint32',
                                'layout': 'single_counter', 'memory_policy': policy, 'inner_n': n,
                                'notes': 'empty=CUDA kernel with no output; short=atomicAdd; sync_each event span includes host gaps'}
    if category == 'transfer':
        direction, size, allocation = args
        cpu = torch.full((size,), 0x5A, dtype=torch.uint8, pin_memory=allocation == 'pinned')
        gpu = torch.empty((size,), dtype=torch.uint8, device='cuda')
        if direction == 'D2H': gpu.fill_(0x5A)
        def op(_):
            if direction == 'H2D': gpu.copy_(cpu, non_blocking=True)
            else: cpu.copy_(gpu, non_blocking=True)
        op(0); torch.cuda.synchronize()
        if direction == 'H2D' and not bool(torch.all(gpu == 0x5A)):
            raise RuntimeError('H2D validation failed')
        if direction == 'D2H' and not bool(torch.all(cpu == 0x5A)):
            raise RuntimeError('D2H validation failed')
        return op, (cpu, gpu), {'flops': '', 'work_bytes': size, 'precision': 'uint8',
                                'layout': 'contiguous', 'memory_policy': allocation, 'inner_n': 1,
                                'notes': 'logical transfer bytes; GB10 memory path differs from discrete PCIe'}
    if category == 'host':
        def op(_): torch.cuda.synchronize()
        return op, (), {'flops': '', 'work_bytes': '', 'precision': 'none', 'layout': 'none',
                        'memory_policy': 'fixed_python_call', 'inner_n': 1,
                        'notes': 'host-only synchronization with no queued work; CUDA event duration is not interpreted'}
    raise ValueError(category)


def timing_group(torch, op, repetitions, category):
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    utc_start = utc_now()
    monotonic_start_ns = time.perf_counter_ns()
    start.record()
    for i in range(repetitions): op(i)
    host_enqueue_ns = time.perf_counter_ns()
    end.record(); end.synchronize()
    monotonic_end_ns = time.perf_counter_ns()
    return {'gpu_event_us': '' if category == 'host' else start.elapsed_time(end)*1000,
            'host_enqueue_us': (host_enqueue_ns-monotonic_start_ns)/1000,
            'host_complete_us': (monotonic_end_ns-monotonic_start_ns)/1000,
            'utc_start': utc_start, 'utc_end': utc_now(),
            'monotonic_start_ns': monotonic_start_ns, 'monotonic_end_ns': monotonic_end_ns}


def worker(args):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; E2 cannot use CPU fallback')
    torch.cuda.set_device(args.device)
    props = torch.cuda.get_device_properties(args.device)
    if args.expect_gpu and args.expect_gpu.lower() not in props.name.lower():
        raise RuntimeError(f'Expected GPU {args.expect_gpu!r}, found {props.name!r}')
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    lib = gpu_library(args.library)
    output = Path(args.out) / f'run{args.repeat:02d}'
    output.mkdir(parents=True, exist_ok=False)
    env = {'profile_id': args.profile_id, 'protocol': PROTOCOL, 'status': 'started',
           'utc_start': utc_now(), 'label': args.label, 'device': args.device,
           'gpu_name': props.name, 'gpu_uuid': str(getattr(props, 'uuid', 'unavailable')),
           'compute_capability': list(torch.cuda.get_device_capability(args.device)),
           'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
           'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
           'nvcc': args.nvcc_version, 'source_sha256': digest(SOURCE),
           'code_sha256': code_digest(), 'software_hash': args.software_hash,
           'library_sha256': digest(args.library), 'process_repeat': args.repeat,
           'groups': args.groups, 'target_ms': args.target_ms, 'smoke': args.smoke,
           'cells': len(cells(args.smoke))}
    write_json(output/'environment.json', env)
    rows = []
    try:
        for category, spec in cells(args.smoke):
            cell = category + '/' + '/'.join(map(str, spec))
            op, keepalive, meta = make_operation(torch, lib, category, spec)
            # Short probe sizes each cell without mixing allocator or JIT setup into formal groups.
            probe = timing_group(torch, op, 1, category)
            scope = 'host_complete_us' if category == 'host' else 'gpu_event_us'
            probe_ms = max(float(probe[scope])/1000, 0.01)
            repetitions = max(1, min(MAX_REPETITIONS, math.ceil(args.target_ms / probe_ms)))
            # A second probe accounts for warm-up and Python submission limits.
            probe2 = timing_group(torch, op, repetitions, category)
            observed_ms = max(float(probe2[scope])/1000, 0.01)
            repetitions = max(1, min(MAX_REPETITIONS, math.ceil(repetitions * args.target_ms / observed_ms)))
            for group in range(1, args.groups+1):
                for attempt in range(6):
                    measured = timing_group(torch, op, repetitions, category)
                    duration_ms = float(measured[scope])/1000
                    if duration_ms >= 0.8*args.target_ms:
                        break
                    next_repetitions = min(MAX_REPETITIONS, max(repetitions+1,
                                           math.ceil(repetitions*args.target_ms/max(duration_ms, 0.01))))
                    if next_repetitions == repetitions or attempt == 5:
                        raise RuntimeError(f'{cell}: cannot reach 80% of target group time with {repetitions} repetitions')
                    repetitions = next_repetitions
                row = dict.fromkeys(HEADERS, '')
                row.update(profile_id=args.profile_id, gpu_uuid=env['gpu_uuid'], gpu_name=props.name,
                           software_hash=args.software_hash, code_sha256=code_digest(),
                           category=category, cell=cell, shape_or_bytes='/'.join(map(str, spec)),
                           process_repeat=args.repeat, group=group, repetitions=repetitions,
                           timing_scope='cuda_event_and_host' if category != 'host' else 'host_only')
                row.update(meta); row.update(measured)
                rows.append(row)
            del op, keepalive
            print(f'run{args.repeat:02d} {cell} groups={args.groups} reps={repetitions}', flush=True)
        with (output/'microbench_samples.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=HEADERS); writer.writeheader(); writer.writerows(rows)
        env['status'] = 'completed'; env['utc_end'] = utc_now(); env['sample_rows'] = len(rows)
        write_json(output/'environment.json', env)
    except Exception as exc:
        env['status'] = 'failed'; env['utc_end'] = utc_now(); env['error'] = f'{type(exc).__name__}: {exc}'
        write_json(output/'environment.json', env)
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description='E2 hardware calibration; real CUDA only')
    p.add_argument('--out', required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--expect-gpu')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--smoke', action='store_true', help='Separate short functional validation, never a formal E2 profile')
    p.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--repeat', type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument('--profile-id', help=argparse.SUPPRESS)
    p.add_argument('--library', help=argparse.SUPPRESS)
    p.add_argument('--software-hash', help=argparse.SUPPRESS)
    p.add_argument('--nvcc-version', help=argparse.SUPPRESS)
    p.add_argument('--groups', type=int, default=5, help=argparse.SUPPRESS)
    p.add_argument('--target-ms', type=float, default=100, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args._worker:
        worker(args); return 0
    out = Path(args.out).resolve()
    if out.exists():
        p.error(f'output exists; use a new directory: {out}')
    import torch
    if not torch.cuda.is_available():
        p.error('CUDA is unavailable; verify GPU/device permissions')
    props = torch.cuda.get_device_properties(args.device)
    if args.expect_gpu and args.expect_gpu.lower() not in props.name.lower():
        p.error(f'Expected {args.expect_gpu!r}, found {props.name!r}')
    nvcc = subprocess.run(['nvcc','--version'], capture_output=True, text=True, check=False)
    if nvcc.returncode:
        p.error('nvcc unavailable; install the CUDA toolkit matching the target GPU')
    out.mkdir(parents=True)
    try:
        lib = compile_library(out/'libe2.so', torch.cuda.get_device_capability(args.device))
        source_hash = digest(SOURCE)
        software_hash = hashlib.sha256(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda,
                       'nvcc': nvcc.stdout, 'code': code_digest()}, sort_keys=True).encode()).hexdigest()
        profile_id = hashlib.sha256(json.dumps({'protocol': PROTOCOL, 'gpu_uuid': str(getattr(props,'uuid','unavailable')),
                     'software_hash': software_hash, 'smoke': args.smoke}, sort_keys=True).encode()).hexdigest()[:16]
        write_json(out/'plan.json', {'protocol': PROTOCOL, 'status': 'started', 'profile_id': profile_id,
                   'label': args.label, 'gpu': props.name, 'gpu_uuid': str(getattr(props,'uuid','unavailable')),
                   'source_sha256': source_hash, 'code_sha256': code_digest(),
                   'library_sha256': digest(lib), 'software_hash': software_hash,
                   'cells': [(cat, list(spec)) for cat,spec in cells(args.smoke)],
                   'processes': 1 if args.smoke else 3, 'groups_per_cell': 1 if args.smoke else 5,
                   'target_ms_per_group': 10 if args.smoke else 100, 'smoke': args.smoke, 'utc_start': utc_now()})
        for repeat in range(1, 2 if args.smoke else 4):
            cmd = [sys.executable, '-m', 'e2_calibration', '--_worker', '--out', str(out), '--label', args.label,
                   '--device', args.device, '--repeat', str(repeat), '--profile-id', profile_id,
                   '--library', str(lib), '--software-hash', software_hash,
                   '--nvcc-version', nvcc.stdout.strip(), '--groups', str(1 if args.smoke else 5),
                   '--target-ms', str(10 if args.smoke else 100)]
            if args.expect_gpu: cmd += ['--expect-gpu', args.expect_gpu]
            if args.smoke: cmd.append('--smoke')
            with (out/f'run{repeat:02d}.log').open('w') as log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                raise RuntimeError(f'worker run{repeat:02d} failed; inspect {out/f"run{repeat:02d}.log"}')
        plan = json.loads((out/'plan.json').read_text()); plan['status']='completed'; plan['utc_end']=utc_now()
        write_json(out/'plan.json', plan)
        print(f'E2 {"smoke" if args.smoke else "formal"} completed: {out}')
        return 0
    except Exception as exc:
        plan_path = out/'plan.json'
        plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}
        plan.update(status='failed', error=f'{type(exc).__name__}: {exc}', utc_end=utc_now())
        write_json(plan_path, plan)
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
