"""Predeclared O1 ambiguity probes, separate from the frozen 139-cell matrix."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .spec import input_bytes, operand_shapes, write_inputs

BOOTSTRAP_PROTOCOL = "o1_challenge_bootstrap_8192_v2_20261022"
PROTOCOL = "o1_challenge_calibrated_v2_20261022"
ZERO_GPU_VIEW_ID = "challenge_transpose_view_256_256"
ZERO_GPU_VIEW_PR_CALLS = 8192

CONFIGS = (
    {"id": "challenge_gemm_64_64_64", "op": "gemm", "m": 64, "n": 64, "k": 64,
     "pair_group": "gemm_bmm_b1_64_64_64"},
    {"id": "challenge_bmm_1_64_64_64", "op": "bmm", "b": 1, "m": 64, "n": 64, "k": 64,
     "pair_group": "gemm_bmm_b1_64_64_64"},
    {"id": "challenge_conv_1_16_32_32_k1_s1_p0", "op": "conv", "b": 1,
     "cin": 16, "cout": 32, "h": 32, "k": 1, "stride": 1, "padding": 0},
    {"id": "challenge_relu_65536", "op": "relu", "elements": 65536},
    {"id": "challenge_softmax_32_256", "op": "softmax", "rows": 32, "d": 256},
    {"id": "challenge_transpose_view_256_256", "op": "transpose_view", "rows": 256, "d": 256},
    {"id": "challenge_transpose_copy_256_256", "op": "transpose_copy", "rows": 256, "d": 256},
)
BY_ID = {config["id"]: config for config in CONFIGS}
assert len(CONFIGS) == len(BY_ID) == 7


def write_challenge_inputs(root, configs=CONFIGS):
    """B=1 BMM and GEMM use byte-identical matrices with different tensor ranks."""
    root = Path(root)
    inventory = {}
    for config in configs:
        if "pair_group" not in config:
            inventory.update(write_inputs(root, (config,)))
            continue
        folder = root / config["id"]
        folder.mkdir(parents=True, exist_ok=True)
        files = {}
        for operand, shape in operand_shapes(config).items():
            count = 1
            for extent in shape:
                count *= extent
            data, material = input_bytes(config["pair_group"], operand, [count])
            path = folder / f"{operand}.bin"
            with path.open("xb") as stream:
                stream.write(data)
            files[operand] = {"path": str(path.resolve()), "shape": shape,
                              "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                              "seed_material": material}
        inventory[config["id"]] = files
    return inventory
