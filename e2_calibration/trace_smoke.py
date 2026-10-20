"""Tiny explicit CUDA workload to verify the native E2 kernels in Nsight."""
from __future__ import annotations

import argparse
import ctypes

from .__main__ import checked, gpu_library


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--library', required=True)
    args = p.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    lib = gpu_library(args.library)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    counter = torch.zeros((1,), dtype=torch.int32, device='cuda')
    src = torch.ones((1 << 20,), dtype=torch.uint8, device='cuda')
    dst = torch.empty_like(src)
    for _ in range(10):
        checked(lib.e2_launch_empty(stream), 'trace empty')
        checked(lib.e2_launch_short(counter.data_ptr(), stream), 'trace short')
        checked(lib.e2_launch_memory(0, src.data_ptr(), dst.data_ptr(), counter.data_ptr(), src.numel(), stream), 'trace copy')
    torch.cuda.synchronize()
    if int(counter.item()) != 10 or not bool(torch.equal(src, dst)):
        raise RuntimeError('Trace smoke validation failed')
    print('trace_smoke_passed empty=10 short=10 copy=10')


if __name__ == '__main__':
    main()
