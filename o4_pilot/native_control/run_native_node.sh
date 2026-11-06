#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 /absolute/path/to/NEW_run GB10|4090|3090" >&2
  exit 2
fi
diag_out=$1
diag_gpu=$2
diag_package=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
diag_python=${PYTHON:-python3}
diag_nvcc=${NVCC:-nvcc}
if [[ $diag_out != /* || -e $diag_out ]]; then
  echo "Output must be an absolute path that does not exist: $diag_out" >&2
  exit 2
fi
mkdir -p -- "$diag_out"/{native,src,tests,preflight,captures,reports}
cp -- "$diag_package/native/serial_mre.cu" "$diag_out/native/"
cp -- "$diag_package/src/quality.py" "$diag_package/src/collect_controls.py" "$diag_package/src/analyze_native.py" "$diag_package/src/nsys_schema.py" "$diag_package/src/trace_contract.py" "$diag_package/src/runtime_guard.py" "$diag_out/src/"
cp -- "$diag_package/tests/test_quality.py" "$diag_package/tests/test_collector.py" "$diag_out/tests/"
"$diag_nvcc" --version > "$diag_out/preflight/nvcc_version.txt"
nsys --version > "$diag_out/preflight/nsys_version.txt"
nsys profile --help > "$diag_out/preflight/nsys_profile_help.txt" 2>&1
"$diag_nvcc" -std=c++17 -O2 -lineinfo -arch=compute_80 -code=compute_80 "$diag_out/native/serial_mre.cu" -o "$diag_out/native/serial_mre" -ldl > "$diag_out/native/build.log" 2>&1
"$diag_python" -m unittest discover -s "$diag_out/tests" -v > "$diag_out/tests.log" 2>&1
"$diag_python" "$diag_out/src/collect_controls.py" --cell correctness --expect-gpu "$diag_gpu"
"$diag_python" "$diag_out/src/collect_controls.py" --cell A1 --expect-gpu "$diag_gpu"
"$diag_python" "$diag_out/src/analyze_native.py" --cells A1 --out "$diag_out/reports/A1_check"
"$diag_python" - "$diag_out/reports/A1_check/A1/summary.json" <<'PY'
import json,sys
from pathlib import Path
s=json.load(open(sys.argv[1]))
d=json.loads((Path(sys.argv[1]).parent/'engineering_diagnostics.json').read_text())
if d['blockers']:raise SystemExit('Native engineering diagnostic blocked; inspect engineering_diagnostics.json')
if not s['all_known_submitted_work_accounted'] or not s['NVTX_exact']:
    raise SystemExit('A1 known work/scope failed; preserve run and inspect before collecting more')
PY
for diag_cell in A2; do
  "$diag_python" "$diag_out/src/collect_controls.py" --cell "$diag_cell" --expect-gpu "$diag_gpu"
done
"$diag_python" "$diag_out/src/analyze_native.py" --cells A1 A2 --out "$diag_out/reports/final_v4"
"$diag_python" - "$diag_out/reports/final_v4" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
for cell in ('A1','A2'):
    d=json.loads((root/cell/'engineering_diagnostics.json').read_text())
    s=json.loads((root/cell/'summary.json').read_text())
    if d['blockers'] or not s['all_known_submitted_work_accounted'] or not s['NVTX_exact']:
        raise SystemExit('Native engineering review required: '+cell)
PY
echo "Engineering controls complete. Review every gate; no automatic strict upgrade: $diag_out/reports/final_v4/SUMMARY.json"
