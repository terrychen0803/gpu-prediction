"""Deterministic O2 pilot and five-fold formal mixed-sequence catalogs."""
from __future__ import annotations

import json
from pathlib import Path

from o1_operator.full_matrix import BY_ID as FORMAL
from o1_operator.spec import BY_ID as PILOT

PROTOCOL = "o2_mixed_sequence_gb10_v3_chain_norm_checked_20261024"
ROOT = Path(__file__).resolve().parent.parent
SPLIT_PATH = ROOT / "o1_operator/formal_split_seed42_20261022.json"
FROZEN_PATH = Path(__file__).with_name("formal_templates_v1.json")


def _square(config):
    return config["op"] in ("gemm", "bmm") and config["n"] == config["k"]


def make_templates():
    folds = json.loads(SPLIT_PATH.read_text())["folds"]
    result = []
    for fold, ids in enumerate(folds):
        choices = {kind: sorted(x for x in ids if FORMAL[x]["op"] == kind)
                   for kind in ("gemm", "bmm", "conv")}
        square = sorted(x for x in ids if _square(FORMAL[x]))
        if any(len(choices[k]) < 2 for k in choices) or not square:
            raise ValueError(f"Fold {fold} lacks O2 candidate diversity")
        for variant in range(6):
            g0, g1 = (choices["gemm"][(variant + x) % len(choices["gemm"])] for x in (0, 1))
            b0 = choices["bmm"][variant % len(choices["bmm"])]
            c0 = choices["conv"][variant % len(choices["conv"])]
            c1 = choices["conv"][(variant + 5) % len(choices["conv"])]
            # Fixed topology before any GB10 timing is observed. The first 16
            # steps include same-type runs, shape changes and cross-family runs.
            steps = [g0, g0, g1, b0, c0, c1, b0, g1,
                     c0, c0, b0, g0, g1, c1, b0, g0]
            for topology in ("independent", "dependent_tail"):
                entries = [{"config_id": x, "input": "preallocated"} for x in steps]
                if topology == "dependent_tail":
                    chain = square[variant % len(square)]
                    # First chain call uses the preallocated input; each later
                    # call receives the preceding output on the same stream.
                    entries += [{"config_id": chain,
                                 "input": "preallocated" if i == 0 else "previous_output"}
                                for i in range(4)]
                result.append({"id": f"fold{fold}_v{variant:02d}_{topology}",
                               "fold": fold, "topology": topology, "steps": entries})
    return result


def pilot_templates():
    ids = ("gemm_64_64_64", "gemm_256_1024_256", "bmm_2_32_64_64",
           "conv_1_16_32_32_k3_s1_p1", "conv_4_64_64_56_k1_s1_p0")
    if any(x not in PILOT for x in ids):
        raise RuntimeError("O2 pilot differs from frozen O1 pilot configurations")
    steps = [ids[i] for i in (0, 0, 1, 2, 3, 4, 2, 1, 3, 3, 0, 2, 4, 1, 0, 2)]
    return [{"id": "pilot_independent", "fold": None, "topology": "independent",
             "steps": [{"config_id": x, "input": "preallocated"} for x in steps]},
            {"id": "pilot_dependent_tail", "fold": None, "topology": "dependent_tail",
             "steps": ([{"config_id": x, "input": "preallocated"} for x in steps] +
                       [{"config_id": ids[0],
                         "input": "preallocated" if i == 0 else "previous_output"}
                        for i in range(4)])}]


def validate(templates, catalog, formal):
    ids = [t["id"] for t in templates]
    if len(ids) != len(set(ids)) or (formal and len(ids) != 60):
        raise ValueError("O2 template IDs/count invalid")
    for t in templates:
        steps = t["steps"]
        if not 16 <= len(steps) <= 48:
            raise ValueError(f"O2 step count invalid: {t['id']}")
        if formal and any(s["config_id"] not in json.loads(SPLIT_PATH.read_text())["folds"][t["fold"]]
                          for s in steps):
            raise ValueError(f"O2 template leaks across folds: {t['id']}")
        for i, step in enumerate(steps):
            config = catalog[step["config_id"]]
            if config["op"] not in ("gemm", "bmm", "conv"):
                raise ValueError("O2 core catalog contains unsupported type")
            if step["input"] == "previous_output":
                if i == 0 or step["config_id"] != steps[i-1]["config_id"] or not _square(config):
                    raise ValueError("O2 dependent edge is not shape compatible")
            elif step["input"] != "preallocated":
                raise ValueError("O2 unknown input source")


def load(formal):
    templates = json.loads(FROZEN_PATH.read_text())["templates"] if formal else pilot_templates()
    validate(templates, FORMAL if formal else PILOT, formal)
    if formal and templates != make_templates():
        raise ValueError("Frozen O2 catalog differs from deterministic generator")
    return templates
