"""The fixed 50 objects: verified latent cache -> packed tensors.

Source: the frozen OwnAE-v2 cache built on 2026-10-02 (manifest SHA f14f796c...). Each object file holds
the user's own Nexus2K D9 vertices [N,3] (local order unchanged), GT faces and edges, the 8,192-point
XYZ+normal condition, and the VAE posterior mu/logvar [N,512]. Nothing here is regenerated; every file
hash, the normalization hash and the dataset totals are checked before use.

Run `python data.py --cache DIR` for a CPU-only check of the cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from common import (CACHE_MANIFEST_SHA256, EXPECTED_TOTALS, NORMALIZATION_SHA256, VAE_SHA256,
                    sha256_file)


@dataclass
class Segments:
    """Variable-length objects packed back to back in one token axis."""
    lengths: list
    cu: Tensor        # int32 [K+1] cumulative offsets (flash-attn varlen layout)
    index: Tensor     # long [T]: object index of every token
    max_len: int

    @classmethod
    def build(cls, lengths, device) -> "Segments":
        lengths = [int(n) for n in lengths]
        if not lengths or min(lengths) < 1:
            raise ValueError("every object needs at least one token")
        cu = torch.tensor([0] + list(np.cumsum(lengths)), dtype=torch.int32, device=device)
        index = torch.repeat_interleave(torch.arange(len(lengths), device=device),
                                        torch.tensor(lengths, device=device))
        return cls(lengths=lengths, cu=cu, index=index, max_len=max(lengths))

    @property
    def bounds(self) -> list:
        starts = np.cumsum([0] + self.lengths[:-1])
        return [(int(s), int(s + n)) for s, n in zip(starts, self.lengths)]

    @property
    def tokens(self) -> int:
        return int(sum(self.lengths))


def normalization_sha256(stats: dict) -> str:
    return hashlib.sha256(json.dumps(stats, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


class TopologyData:
    """Packed GPU tensors for a list of objects (all 50 by default, or a selection)."""

    def __init__(self, records: list, mean: Tensor, std: Tensor, device):
        self.device = torch.device(device)
        self.records = records
        self.uids = [r["uid"] for r in records]
        self.lengths = [len(r["vertices"]) for r in records]
        self.count = len(records)
        self.seg = Segments.build(self.lengths, self.device)
        cat = lambda key: torch.from_numpy(np.concatenate([r[key] for r in records])).to(self.device)  # noqa: E731
        self.vertices = cat("vertices").float()                       # [T,3] in [-1,1]
        self.mu = cat("mu").float()                                    # [T,512]
        self.sigma = (0.5 * cat("logvar").float()).exp()               # [T,512]
        self.mean = mean.to(self.device).float()
        self.std = std.to(self.device).float()
        self.mu_normalized = (self.mu - self.mean) / self.std
        points = max(len(r["points"]) for r in records)
        self.points = torch.zeros((self.count, points, 6), device=self.device)
        self.point_valid = torch.zeros((self.count, points), dtype=torch.bool, device=self.device)
        for i, r in enumerate(records):
            n = len(r["points"])
            self.points[i, :n] = torch.from_numpy(r["points"]).to(self.device).float()
            self.point_valid[i, :n] = torch.from_numpy(r["point_mask"]).to(self.device)
        self.gt_edges = [np.unique(np.sort(r["edges"], axis=1), axis=0) for r in records]
        self.gt_faces = [np.unique(np.sort(r["faces"], axis=1), axis=0) for r in records]
        self.fps_indices = None

    def compute_fps(self, num_tokens: int) -> None:
        """Deterministic FPS query indices for the VecSet encoder (fixed inputs, computed once)."""
        from vendor.vertex_s0 import farthest_point_sample
        self.fps_indices = farthest_point_sample(self.points[..., :3], num_tokens, self.point_valid)

    def select(self, objects: list) -> "TopologyData":
        sub = TopologyData([self.records[i] for i in objects], self.mean, self.std, self.device)
        if self.fps_indices is not None:
            sub.fps_indices = self.fps_indices[torch.tensor(objects, device=self.device)]
        return sub

    def token_slice(self, i: int) -> slice:
        start = int(self.seg.cu[i])
        return slice(start, start + self.lengths[i])


def load_cache(cache_dir, device, check_manifest_sha: bool = True) -> TopologyData:
    root = Path(cache_dir)
    manifest_path = root / "manifest.json"
    if check_manifest_sha and sha256_file(manifest_path) != CACHE_MANIFEST_SHA256:
        raise ValueError("cache manifest SHA256 differs from the fixed-50 cache (f14f796c...)")
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("complete") or manifest["source_checkpoint_sha256"] != VAE_SHA256:
        raise ValueError("incomplete cache, or built from a different VAE")
    stats = manifest["normalization"]
    if normalization_sha256(stats) != NORMALIZATION_SHA256 or manifest.get("normalization_sha256") != NORMALIZATION_SHA256:
        raise ValueError("latent normalization differs from the fixed statistics (a144fa69...)")
    uids = manifest["uids"]
    by_uid = {r["uid"]: r for r in manifest["records"]}
    if len(uids) != 50 or len(set(uids)) != 50 or list(by_uid) != uids:
        raise ValueError("cache must hold the 50 fixed objects in manifest order")
    records = []
    totals = dict(meshes=0, vertices=0, edges=0, faces=0)
    for uid in uids:
        record = by_uid[uid]
        path = (root / record["file"]).resolve()
        if not path.is_relative_to(root.resolve()) or sha256_file(path) != record["sha256"]:
            raise ValueError(f"cache file hash mismatch: {uid}")
        with np.load(path, allow_pickle=False) as arrays:
            item = {key: arrays[key].copy() for key in ("vertices", "faces", "edges", "points",
                                                         "point_mask", "mu", "logvar")}
        n = len(item["vertices"])
        if item["vertices"].dtype != np.float32 or item["vertices"].shape != (n, 3):
            raise ValueError(f"bad vertices: {uid}")
        if item["mu"].shape != (n, 512) or item["logvar"].shape != (n, 512):
            raise ValueError(f"bad posterior: {uid}")
        item["uid"] = uid
        records.append(item)
        totals["meshes"] += 1
        totals["vertices"] += n
        totals["edges"] += len(np.unique(np.sort(item["edges"], axis=1), axis=0))
        totals["faces"] += len(np.unique(np.sort(item["faces"], axis=1), axis=0))
    if totals != EXPECTED_TOTALS:
        raise ValueError(f"dataset totals {totals} differ from {EXPECTED_TOTALS}")
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std = torch.tensor(stats["std"], dtype=torch.float32)
    return TopologyData(records, mean, std, device)


def main():
    parser = argparse.ArgumentParser(description="CPU check of the fixed-50 latent cache")
    parser.add_argument("--cache", required=True)
    args = parser.parse_args()
    data = load_cache(args.cache, "cpu")
    report = dict(objects=data.count, tokens=data.seg.tokens, largest=data.seg.max_len,
                  smallest=min(data.lengths), points_per_object=int(data.points.shape[1]),
                  valid_points_min=int(data.point_valid.sum(1).min()),
                  posterior_sigma_over_std_median=float((data.sigma / data.std).median()),
                  mu_normalized_rms=float(data.mu_normalized.square().mean().sqrt()),
                  vertex_abs_max=float(data.vertices.abs().max()),
                  # D9 cell centres satisfy 256*v - 0.5 = integer; this is what the RoPE unit assumes.
                  d9_center_max_offset=float(((256 * data.vertices.double() - 0.5)
                                              - (256 * data.vertices.double() - 0.5).round()).abs().max()))
    print(json.dumps(report, indent=2))
    print("CACHE_OK")


if __name__ == "__main__":
    main()
