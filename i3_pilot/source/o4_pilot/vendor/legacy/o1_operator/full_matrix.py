"""Frozen O1 candidate workload catalog; no GPU work or timing policy here."""
from __future__ import annotations

import itertools

BOOTSTRAP_PROTOCOL = "o1_formal_139_bootstrap_8192_v2_20261022"
PROTOCOL = "o1_formal_139_calibrated_v2_20261022"


def make_configs():
    configs = []
    for m, n, k in itertools.product((16, 128, 512), (64, 256, 1024), (64, 256, 1024)):
        configs.append({"id": f"gemm_{m}_{n}_{k}", "op": "gemm", "m": m, "n": n, "k": k})
    for b, m, n, k in itertools.product((2, 8, 32), (16, 64), (64, 256), (64, 256)):
        configs.append({"id": f"bmm_{b}_{m}_{n}_{k}", "op": "bmm", "b": b, "m": m, "n": n, "k": k})
    for b, cin, cout, h, k, stride in itertools.product(
            (1, 4), (16, 64), (32, 128), (32, 64), (1, 3), (1, 2)):
        padding = k // 2
        configs.append({"id": f"conv_{b}_{cin}_{cout}_{h}_k{k}_s{stride}_p{padding}",
                        "op": "conv", "b": b, "cin": cin, "cout": cout,
                        "h": h, "k": k, "stride": stride, "padding": padding})
    for power in range(10, 22):
        elements = 1 << power
        configs.append({"id": f"add_{elements}", "op": "add", "elements": elements})
    for rows, d in itertools.product((1, 32, 256), (64, 256, 1024, 4096)):
        configs.append({"id": f"layernorm_{rows}_{d}", "op": "layernorm",
                        "rows": rows, "d": d, "eps": 1e-5})
    ids = [config["id"] for config in configs]
    if len(configs) != 139 or len(set(ids)) != len(ids):
        raise RuntimeError("O1 formal matrix count or ID uniqueness changed")
    return tuple(configs)


CONFIGS = make_configs()
BY_ID = {config["id"]: config for config in CONFIGS}
CORE_CONFIGS = tuple(config for config in CONFIGS if config["op"] in ("gemm", "bmm", "conv"))
PILOT_OVERLAP = {"conv_1_16_32_32_k3_s1_p1"}


def family_counts():
    return {op: sum(config["op"] == op for config in CONFIGS)
            for op in ("gemm", "bmm", "conv", "add", "layernorm")}
