#!/usr/bin/env bash
set -euo pipefail
node_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
node_python=${PYTHON:-python3}
cd "$node_root"
exec "$node_python" -m o4_research.node "$@"
