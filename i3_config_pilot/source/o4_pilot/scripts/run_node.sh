#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -ne 2 ]]; then
  echo "Usage: PYTHON=/path/to/python bash scripts/run_node.sh <GB10|4090|3090> <single|three|resnet>" >&2
  exit 2
fi
python_bin="${PYTHON:-python3}"
expect_gpu="$1"
phase="$2"
case "$expect_gpu" in GB10|4090|3090) ;; *) echo "Unsupported expected GPU" >&2; exit 2;; esac
"$python_bin" scripts/verify_assets.py
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_path="analysis_runs/o4_${expect_gpu}_${phase}_${stamp}"
case "$phase" in
 single) args=(--workload gemm_16_64_64 --repeats 1) ;;
 three) args=(--repeats 3 --keep-going) ;;
 resnet) args=(--workload RN-A --repeats 3 --keep-going) ;;
 *) echo "Unknown phase: $phase" >&2; exit 2 ;;
esac
set +e
"$python_bin" -m o4_iteration run --out "$run_path" --expect-gpu "$expect_gpu" "${args[@]}"
run_code=$?
set -e
"$python_bin" -m o4_iteration summarize --run "$run_path" --out "$run_path/reports"
if [[ $run_code -ne 0 ]]; then
 echo "Run failed or incomplete; preserved evidence: $run_path" >&2
 exit "$run_code"
fi
echo "Finished engineering pilot; reference gate is reported separately: $run_path/reports/REPORT.md"
