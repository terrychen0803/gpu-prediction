#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; printf "I3 entry failed (exit %s) at line %s; preserve the output directory and inspect its logs.\n" "$rc" "$LINENO" >&2; exit "$rc"' ERR

I3_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${I3_PYTHON:?Set I3_PYTHON to an existing compatible CUDA Python executable}"
action="${1:-}"
if [[ $# -eq 0 ]]; then
  printf 'Usage: %s verify|test|plan|bn|collect [collector arguments]\n' "$0" >&2
  exit 2
fi
shift
export PYTHONPATH="$I3_ROOT/source:$I3_ROOT/source/o4_pilot"
"$I3_PYTHON" "$I3_ROOT/scripts/verify_delivery.py"
case "$action" in
  test)
    export CUDA_VISIBLE_DEVICES=''
    "$I3_PYTHON" -m unittest discover -s "$I3_ROOT/source/tests" -v
    "$I3_PYTHON" -m unittest discover -s "$I3_ROOT/tests" -v
    "$I3_PYTHON" -m unittest discover -s "$I3_ROOT/source/o4_pilot/tests" -v
    (cd "$I3_ROOT/source/o4_pilot/vendor/legacy" && "$I3_PYTHON" -m unittest discover -s tests -v)
    ;;
  verify|plan|bn|collect)
    : "${I3_ASSETS:?Set I3_ASSETS to the verified assets_A0/ASSETS.json path}"
    "$I3_PYTHON" -m o4_i3.controlled verify --path "$I3_ASSETS"
    if [[ "$action" == verify ]]; then exit 0; fi
    if [[ "$action" == plan ]]; then
      exec "$I3_PYTHON" -m o4_i3.collect --assets-manifest "$I3_ASSETS" --plan-only "$@"
    elif [[ "$action" == bn ]]; then
      exec "$I3_PYTHON" -m o4_i3.collect --assets-manifest "$I3_ASSETS" --config A0_batchnorm_eval_a "$@"
    else
      exec "$I3_PYTHON" -m o4_i3.collect --assets-manifest "$I3_ASSETS" "$@"
    fi
    ;;
  *) printf 'Unknown I3 action: %s\n' "$action" >&2; exit 2 ;;
esac
