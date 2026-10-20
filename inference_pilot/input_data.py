"""Portable, checksum-verified FP16 input for cross-platform measurements."""
from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path

from .common import read_json, sha256, write_json


CHANNELS = 3
HEIGHT = WIDTH = 224
BYTES_PER_IMAGE = CHANNELS * HEIGHT * WIDTH * 2
FORMAT = "e1_fp16_le_nchw_v1"


def sidecar_path(path):
    path = Path(path)
    return path.with_suffix(path.suffix + ".json")


def prepare_input(path, max_batch=32, seed=42):
    """Materialize bytes once; all devices subsequently load this exact file."""
    if sys.byteorder != "little":
        raise RuntimeError("Fixed FP16 input requires a little-endian host")
    if max_batch < 1:
        raise ValueError("max_batch must be positive")
    path = Path(path).resolve()
    sidecar = sidecar_path(path)
    if path.exists() or sidecar.exists():
        raise FileExistsError(f"Refusing to overwrite fixed input or sidecar: {path}")
    import torch
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tensor = torch.randn(max_batch, CHANNELS, HEIGHT, WIDTH, generator=generator).to(torch.float16).contiguous()
    data = tensor.numpy().tobytes(order="C")
    if len(data) != max_batch * BYTES_PER_IMAGE:
        raise RuntimeError("Unexpected serialized input length")
    metadata = {
        "format": FORMAT, "shape": [max_batch, CHANNELS, HEIGHT, WIDTH],
        "dtype": "float16", "layout": "NCHW_contiguous", "byte_order": "little",
        "bytes_per_image": BYTES_PER_IMAGE, "file_sha256": hashlib.sha256(data).hexdigest(),
        "prefix_sha256": {str(batch): hashlib.sha256(data[:batch * BYTES_PER_IMAGE]).hexdigest()
                          for batch in range(1, max_batch + 1)},
        "generator": {"method": "torch.randn CPU then float16", "seed": seed,
                      "torch": torch.__version__, "machine": platform.machine()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(data)
    temporary.replace(path)
    write_json(sidecar, metadata)
    return metadata


def read_fixed_input(path, batch):
    """Validate the whole file and requested prefix before Torch touches the data."""
    if sys.byteorder != "little":
        raise RuntimeError("Fixed FP16 input requires a little-endian host")
    if not isinstance(batch, int) or batch < 1:
        raise ValueError("batch must be a positive integer")
    path = Path(path).resolve()
    metadata = read_json(sidecar_path(path))
    shape = metadata.get("shape")
    if (metadata.get("format") != FORMAT or metadata.get("dtype") != "float16" or
            metadata.get("layout") != "NCHW_contiguous" or metadata.get("byte_order") != "little" or
            metadata.get("bytes_per_image") != BYTES_PER_IMAGE or
            not isinstance(shape, list) or len(shape) != 4 or shape[1:] != [CHANNELS, HEIGHT, WIDTH] or
            not isinstance(shape[0], int) or batch > shape[0]):
        raise ValueError(f"Unsupported fixed input metadata or batch {batch}: {path}")
    data = path.read_bytes()
    if len(data) != shape[0] * BYTES_PER_IMAGE:
        raise ValueError(f"Fixed input byte count mismatch: {path}")
    if hashlib.sha256(data).hexdigest() != metadata.get("file_sha256"):
        raise ValueError(f"Fixed input SHA-256 mismatch: {path}")
    prefix = data[:batch * BYTES_PER_IMAGE]
    digest = hashlib.sha256(prefix).hexdigest()
    if digest != metadata.get("prefix_sha256", {}).get(str(batch)):
        raise ValueError(f"Fixed input batch {batch} prefix SHA-256 mismatch: {path}")
    return prefix, {"input_scheme": FORMAT, "input_file_sha256": sha256(path),
                    "input_prefix_sha256": digest, "input_max_batch": shape[0],
                    "input_file": str(path)}
