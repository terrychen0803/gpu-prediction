#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 || $1 != /* || -e $1 ]]; then
  echo "Usage: $0 /absolute/path/to/NEW_libmemset_observer.so" >&2
  exit 2
fi
diag_output=$1
diag_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
diag_cuda_root=${CUDATOOLKIT_ROOT:-}
if [[ -z $diag_cuda_root ]]; then
  diag_nvcc=$(command -v nvcc) || { echo "nvcc not found; set CUDATOOLKIT_ROOT" >&2; exit 2; }
  diag_cuda_root=$(dirname -- "$(dirname -- "$(readlink -f -- "$diag_nvcc")")")
fi
mkdir -p -- "$(dirname -- "$diag_output")"
"${CXX:-c++}" -std=c++17 -shared -fPIC -O2 -pthread \
  -I"$diag_cuda_root/include" "$diag_root/o4_research/native/memset_observer.cpp" \
  -L"$diag_cuda_root/lib64" -lcupti -Wl,-rpath,"$diag_cuda_root/lib64" \
  -o "$diag_output" > "$diag_output.build.log" 2>&1
sha256sum "$diag_root/o4_research/native/memset_observer.cpp" "$diag_output"
