"""Small shared helpers: hashing, seeded generators, atomic JSON/torch writes, fixed identities."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

# Frozen inputs (all on the GPU servers; never modified by this code).
VAE_SHA256 = "e89b8078b7abb0ca7b0c2c44f9f20382ec5b742f2aee948f209d642852714d15"      # OwnAE-v2 vae-1000.pt (step 36220)
CACHE_MANIFEST_SHA256 = "f14f796c421cab6b9d5206047f927ea7c3490ff3c2435b2ebd38904a749c27e3"  # fixed-50 latent cache
NORMALIZATION_SHA256 = "a144fa69f836d15282acd1173bfee4ca3027fd832828d4ab9303a74f8ca44f01"   # per-channel mean/std in that manifest
S0_SHA256 = "3919d924ad7c48a3848ef5b4d1247f8ec9296e7b068343bfc7ffbbf1012e4935"       # vertex S0 checkpoint-034000.pt (Run B only)
EXPECTED_TOTALS = {"meshes": 50, "vertices": 52820, "edges": 154978, "faces": 102890}
SIZE_BINS = ((0, 300), (300, 700), (700, 1200), (1200, 10**9))


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_from(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") & ((1 << 63) - 1)


def generator(key: str, device="cpu") -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed_from(key))


def write_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def append_jsonl(path, row) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()


def save_torch(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def micro_f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / max(2 * tp + fp + fn, 1)


def size_bin(vertices: int) -> str:
    for low, high in SIZE_BINS:
        if low <= vertices < high:
            return f"{low}-{high}" if high < 10**9 else f"{low}+"
    raise ValueError(vertices)
