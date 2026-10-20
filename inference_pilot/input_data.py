"""Portable, checksum-verified FP16 input for cross-platform measurements."""
from __future__ import annotations

import hashlib
import platform
import struct
import sys
from array import array
from pathlib import Path

from .common import read_json, sha256, write_json


CHANNELS = 3
HEIGHT = WIDTH = 224
BYTES_PER_IMAGE = CHANNELS * HEIGHT * WIDTH * 2
FORMAT = "e1_fp16_le_nchw_v1"
GRID_FORMAT = "e3_shake256_lut_fp16_le_nchw_v1"


def sidecar_path(path):
    path = Path(path)
    return path.with_suffix(path.suffix + ".json")


def prepare_grid_input(path, resolution, max_batch=32, seed=42):
    """Portable E3 tensor: SHAKE256 bytes mapped through a fixed finite FP16 LUT."""
    if sys.byteorder != "little":
        raise RuntimeError("Fixed FP16 input requires a little-endian host")
    if not isinstance(resolution, int) or resolution < 32 or resolution % 32:
        raise ValueError("resolution must be a multiple of 32 and at least 32")
    if not isinstance(max_batch, int) or max_batch < 1:
        raise ValueError("max_batch must be a positive integer")
    path = Path(path).resolve()
    sidecar = sidecar_path(path)
    if path.exists() or sidecar.exists():
        raise FileExistsError(f"Refusing to overwrite fixed input or sidecar: {path}")
    bytes_per_image = CHANNELS * resolution * resolution * 2
    count = max_batch * CHANNELS * resolution * resolution
    seed_material = f"{GRID_FORMAT}|seed={seed}|resolution={resolution}|max_batch={max_batch}".encode("ascii")
    random_bytes = hashlib.shake_256(seed_material).digest(count)
    lut = tuple(struct.unpack("<H", struct.pack("<e", (i - 127.5) / 64))[0] for i in range(256))
    data = array("H", (lut[value] for value in random_bytes)).tobytes()
    if len(data) != max_batch * bytes_per_image:
        raise RuntimeError("Unexpected serialized E3 input length")
    metadata = {
        "format": GRID_FORMAT, "shape": [max_batch, CHANNELS, resolution, resolution],
        "dtype": "float16", "layout": "NCHW_contiguous", "byte_order": "little",
        "bytes_per_image": bytes_per_image, "file_sha256": hashlib.sha256(data).hexdigest(),
        "prefix_sha256": {str(batch): hashlib.sha256(data[:batch * bytes_per_image]).hexdigest()
                          for batch in range(1, max_batch + 1)},
        "generator": {"method": "SHAKE256 byte stream; byte i -> FP16((i-127.5)/64)",
                      "seed": seed, "seed_material": seed_material.decode("ascii")},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(data)
    temporary.replace(path)
    write_json(sidecar, metadata)
    return metadata


def prepare_grid_set(root, manifest_path):
    """Generate missing E3 inputs and check every node against committed hashes."""
    manifest = read_json(manifest_path)
    if manifest.get('protocol') != 'e3_grid_input_v1' or manifest.get('generator_format') != GRID_FORMAT:
        raise ValueError('Unsupported E3 grid manifest')
    root = Path(root).resolve()
    verified = []
    for item in manifest['resolutions']:
        path = root / item['file_name']
        if path.name != item['file_name']:
            raise ValueError('Unsafe E3 input filename')
        sidecar = sidecar_path(path)
        if not path.exists() and not sidecar.exists():
            prepare_grid_input(path, item['resolution'], manifest['max_batch'], manifest['seed'])
        if not path.is_file() or not sidecar.is_file():
            raise FileNotFoundError(f'Incomplete E3 input pair: {path}')
        if sha256(path) != item['file_sha256']:
            raise ValueError(f'E3 input hash mismatch: {path}')
        metadata = read_json(sidecar)
        if (metadata.get('format') != GRID_FORMAT or metadata.get('file_sha256') != item['file_sha256'] or
                metadata.get('bytes_per_image') != item['bytes_per_image']):
            raise ValueError(f'E3 sidecar mismatch: {sidecar}')
        for batch, expected in item['prefix_sha256'].items():
            _, provenance = read_fixed_input(path, int(batch), item['resolution'])
            if provenance['input_prefix_sha256'] != expected:
                raise ValueError(f'E3 input prefix mismatch: {path} batch {batch}')
        verified.append({'resolution': item['resolution'], 'file': str(path), 'sha256': item['file_sha256']})
    return verified


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


def read_fixed_input(path, batch, resolution=224):
    """Validate the whole file and requested prefix before Torch touches the data."""
    if sys.byteorder != "little":
        raise RuntimeError("Fixed FP16 input requires a little-endian host")
    if not isinstance(batch, int) or batch < 1:
        raise ValueError("batch must be a positive integer")
    path = Path(path).resolve()
    metadata = read_json(sidecar_path(path))
    shape = metadata.get("shape")
    valid_format = metadata.get("format") in (FORMAT, GRID_FORMAT)
    if (not valid_format or metadata.get("dtype") != "float16" or
            metadata.get("layout") != "NCHW_contiguous" or metadata.get("byte_order") != "little" or
            metadata.get("bytes_per_image") != CHANNELS * resolution * resolution * 2 or
            not isinstance(shape, list) or len(shape) != 4 or shape[1:] != [CHANNELS, resolution, resolution] or
            (metadata.get("format") == FORMAT and resolution != 224) or
            not isinstance(shape[0], int) or batch > shape[0]):
        raise ValueError(f"Unsupported fixed input metadata or batch {batch}: {path}")
    bytes_per_image = CHANNELS * resolution * resolution * 2
    data = path.read_bytes()
    if len(data) != shape[0] * bytes_per_image:
        raise ValueError(f"Fixed input byte count mismatch: {path}")
    if hashlib.sha256(data).hexdigest() != metadata.get("file_sha256"):
        raise ValueError(f"Fixed input SHA-256 mismatch: {path}")
    prefix = data[:batch * bytes_per_image]
    digest = hashlib.sha256(prefix).hexdigest()
    if digest != metadata.get("prefix_sha256", {}).get(str(batch)):
        raise ValueError(f"Fixed input batch {batch} prefix SHA-256 mismatch: {path}")
    return prefix, {"input_scheme": metadata["format"], "input_file_sha256": sha256(path),
                    "input_prefix_sha256": digest, "input_max_batch": shape[0],
                    "input_file": str(path)}
