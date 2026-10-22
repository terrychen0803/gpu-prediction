"""Label-free, device-independent formal O1 workload split assignment."""
from __future__ import annotations

import hashlib

from .full_matrix import CONFIGS, PILOT_OVERLAP

SPLIT_ID = "o1_formal139_grouped_fivefold_seed42_v1"
FOLD_COUNT = 5


def priority(config_id):
    return hashlib.sha256(f"{SPLIT_ID}|{config_id}".encode("ascii")).hexdigest()


def make_split():
    folds = [[] for _ in range(FOLD_COUNT)]
    for op in ("gemm", "bmm", "conv", "add", "layernorm"):
        eligible = sorted((c["id"] for c in CONFIGS
                           if c["op"] == op and c["id"] not in PILOT_OVERLAP),
                          key=lambda config_id: (priority(config_id), config_id))
        for index, config_id in enumerate(eligible):
            folds[index % FOLD_COUNT].append(config_id)
    folds = [sorted(fold) for fold in folds]
    all_ids = {c["id"] for c in CONFIGS}
    assigned = [config_id for fold in folds for config_id in fold]
    if len(assigned) != len(all_ids - PILOT_OVERLAP) or set(assigned) != all_ids - PILOT_OVERLAP:
        raise RuntimeError("O1 formal split coverage is incomplete or duplicated")
    evaluations = []
    for test_fold in range(FOLD_COUNT):
        validation_fold = (test_fold + 1) % FOLD_COUNT
        train = sorted(set(assigned) - set(folds[test_fold]) - set(folds[validation_fold]))
        evaluations.append({"test_fold": test_fold, "validation_fold": validation_fold,
                            "train_ids": train + sorted(PILOT_OVERLAP),
                            "validation_ids": folds[validation_fold],
                            "test_ids": folds[test_fold]})
    return {"split_id": SPLIT_ID, "seed": 42,
            "unit": "canonical workload config across devices, modes, repeats, dates and captures",
            "development_only_ids": sorted(PILOT_OVERLAP),
            "folds": folds, "evaluations": evaluations,
            "assignment_uses_timing_or_trace_labels": False}
