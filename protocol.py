"""Filename-group validation splitting and reproducibility fingerprints."""

import hashlib
import json
from pathlib import Path
import re

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def filename_group(path, label):
    stem = Path(path).stem
    match = re.match(r"(person\d+)_", stem)
    if not match:
        match = re.match(r"((?:NORMAL2-)?IM-\d+)-", stem)
    if not match:
        raise ValueError(f"Cannot infer a group from filename: {path}")
    return f"{label}:{match.group(1)}"


def group_split(samples, seed=820):
    labels = np.array([label for _, label in samples])
    groups = np.array([filename_group(path, label) for path, label in samples])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train, val = next(splitter.split(np.zeros(len(labels)), labels, groups))
    if set(groups[train]) & set(groups[val]):
        raise RuntimeError("Filename groups overlap across the holdout.")
    if len(set(labels[train])) != 2 or len(set(labels[val])) != 2:
        raise RuntimeError("Both classes are required in both partitions.")
    return sorted(train.tolist()), sorted(val.tolist())


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def split_hash(record):
    return hashlib.sha256(json.dumps({k: record[k] for k in [
        "train_indices", "internal_val_indices"]}, sort_keys=True).encode()).hexdigest()
