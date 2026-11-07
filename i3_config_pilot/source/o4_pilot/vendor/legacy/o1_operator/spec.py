"""Frozen candidate cells and portable FP16 input bytes for the O1 pilot."""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from array import array
from pathlib import Path

PROTOCOL = "o1_operator_pilot_v4_calibrated_repetitions_cuda_reset_20261022"
INPUT_FORMAT = "o1_shake256_lut_fp16_le_v1"
MODES = ("G0", "GE", "P", "R")
CONFIGS = (
    {"id": "gemm_64_64_64", "op": "gemm", "m": 64, "n": 64, "k": 64},
    {"id": "gemm_256_1024_256", "op": "gemm", "m": 256, "n": 1024, "k": 256},
    {"id": "gemm_1024_1024_1024", "op": "gemm", "m": 1024, "n": 1024, "k": 1024},
    {"id": "bmm_2_32_64_64", "op": "bmm", "b": 2, "m": 32, "n": 64, "k": 64},
    {"id": "bmm_8_64_128_64", "op": "bmm", "b": 8, "m": 64, "n": 128, "k": 64},
    {"id": "bmm_4_256_256_256", "op": "bmm", "b": 4, "m": 256, "n": 256, "k": 256},
    {"id": "conv_1_16_32_32_k3_s1_p1", "op": "conv", "b": 1, "cin": 16, "cout": 32, "h": 32, "k": 3, "stride": 1, "padding": 1},
    {"id": "conv_4_64_64_56_k1_s1_p0", "op": "conv", "b": 4, "cin": 64, "cout": 64, "h": 56, "k": 1, "stride": 1, "padding": 0},
    {"id": "conv_1_64_128_56_k3_s2_p1", "op": "conv", "b": 1, "cin": 64, "cout": 128, "h": 56, "k": 3, "stride": 2, "padding": 1},
)
BY_ID = {item["id"]: item for item in CONFIGS}


def operand_shapes(config):
    op = config["op"]
    if op == "gemm":
        return {"a": [config["m"], config["k"]], "b": [config["k"], config["n"]]}
    if op == "bmm":
        return {"a": [config["b"], config["m"], config["k"]],
                "b": [config["b"], config["k"], config["n"]]}
    if op == "conv":
        return {"input": [config["b"], config["cin"], config["h"], config["h"]],
                "weight": [config["cout"], config["cin"], config["k"], config["k"]]}
    if op == "add":
        return {"a": [config["elements"]], "b": [config["elements"]]}
    if op == "layernorm":
        return {"input": [config["rows"], config["d"]]}
    if op == "relu":
        return {"input": [config["elements"]]}
    if op in ("softmax", "transpose_view", "transpose_copy"):
        return {"input": [config["rows"], config["d"]]}
    raise ValueError(f"Unsupported O1 op: {op}")


def input_bytes(config_id, operand, shape, seed=42):
    """SHAKE256 byte i selects an exact little-endian FP16 LUT entry."""
    if sys.byteorder != "little":
        raise RuntimeError("O1 FP16 input requires a little-endian host")
    if not shape or any(not isinstance(n, int) or n < 1 for n in shape):
        raise ValueError("Invalid operand shape")
    count = 1
    for n in shape:
        count *= n
    material = f"{INPUT_FORMAT}|seed={seed}|config={config_id}|operand={operand}|shape={','.join(map(str, shape))}"
    entropy = hashlib.shake_256(material.encode("ascii")).digest(count)
    lut = tuple(struct.unpack("<H", struct.pack("<e", (i - 127.5) / 512))[0] for i in range(256))
    data = array("H", (lut[v] for v in entropy)).tobytes()
    if len(data) != 2 * count:
        raise RuntimeError("O1 input byte count mismatch")
    return data, material


def write_inputs(root, configs=CONFIGS, seed=42):
    root = Path(root)
    inventory = {}
    for config in configs:
        folder = root / config["id"]
        folder.mkdir(parents=True, exist_ok=True)
        files = {}
        for operand, shape in operand_shapes(config).items():
            data, material = input_bytes(config["id"], operand, shape, seed)
            path = folder / f"{operand}.bin"
            if path.exists():
                if path.read_bytes() != data:
                    raise ValueError(f"Existing O1 input differs: {path}")
            else:
                with path.open("xb") as stream:
                    stream.write(data)
            files[operand] = {"path": str(path.resolve()), "shape": shape, "bytes": len(data),
                              "sha256": hashlib.sha256(data).hexdigest(), "seed_material": material}
        inventory[config["id"]] = files
    return inventory


def write_json_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

