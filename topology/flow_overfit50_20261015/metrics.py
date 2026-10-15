"""Frozen OwnAE-v2 decoding and the strict topology metric (the gate).

Protocol (same as the archived evaluator in archive/topology-flow-20261008, evaluate_flow.py):
  latent [N,512] -> frozen decoder (FP32, no TF32) -> per-mesh centred edge/face embeddings
  -> edge logit for EVERY vertex pair, edge if logit > 0
  -> candidate faces = every triangle of the predicted edge graph -> face if face logit > 0.
No GT edge is added and nothing is repaired. A GT face whose three edges are not all predicted is a miss
("fn_missing"); a candidate GT face rejected by the face logit is "fn_present".
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from common import SIZE_BINS, VAE_SHA256, micro_f1, sha256_file, size_bin

# OwnAE-v2 indicator constants, copied from the archived _fixed100_reference.py (SHA 3e52edf6...).
EDGE_SCALE = 0.9306077080970389
FACE_SCALE = 0.39804385828730726
FACE_FACTOR = 0.25


def edge_logits(embedding: Tensor, pairs: Tensor) -> Tensor:
    delta = embedding[pairs[:, 0]] - embedding[pairs[:, 1]]
    space, time = delta.chunk(2, dim=-1)
    return (space.square().sum(-1) - time.square().sum(-1)) * EDGE_SCALE


def face_logits(embedding: Tensor, triples: Tensor) -> Tensor:
    first, second, third = [embedding[triples[:, i]] for i in range(3)]
    areas = []
    for a, b, c in zip(first.chunk(2, -1), second.chunk(2, -1), third.chunk(2, -1)):
        u, v = b - a, c - a
        areas.append((u.square().sum(-1) * v.square().sum(-1) - (u * v).sum(-1).square()).clamp_min(0))
    return FACE_SCALE * (FACE_FACTOR * (areas[0] - areas[1]))


def load_frozen_vae(path, device, check_sha: bool = True):
    """OwnAE-v2 step 36220 (B_v2_teacher_blocks, latent 512), eval mode, no gradients."""
    from vendor.ownae_v2_native_models import Config, NativeTopologyAE
    if check_sha and sha256_file(path) != VAE_SHA256:
        raise ValueError("VAE checkpoint SHA256 differs from OwnAE-v2 step 36220 (e89b8078...)")
    try:
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError:  # legacy (non-zip) serialization cannot be memory-mapped
        state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config(**state["model_config"])
    if cfg.model_variant != "B_v2_teacher_blocks" or cfg.latent_width != 512:
        raise ValueError("not the OwnAE-v2 latent-512 model")
    model = NativeTopologyAE(cfg)
    model.load_state_dict(state["model"], strict=True)
    del state
    return model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)


@torch.no_grad()
def decode(vae, latent: Tensor) -> tuple[Tensor, Tensor]:
    """Decoder half of NativeTopologyAE.forward (as vae_codec.decode_latents in the archive), FP32."""
    if latent.ndim != 2 or latent.shape[1] != 512:
        raise ValueError("decode one mesh: [N,512]")
    with torch.autocast(device_type=latent.device.type, enabled=False):
        h = vae.latent_input(latent.float())
        for block in vae.decoder_blocks:
            h = block(h)
        h = vae.decoder_output_norm(h)
        edge, face = vae.edge_embedding(h), vae.face_embedding(h)
    return edge - edge.mean(0, keepdim=True), face - face.mean(0, keepdim=True)


def triangles(adjacency: np.ndarray, cap: int | None = None) -> tuple[np.ndarray, bool]:
    """All (i<j<k) with all three edges in the upper-triangular boolean adjacency; stops past `cap`."""
    n = len(adjacency)
    found, count = [], 0
    for i in range(n):
        js = np.flatnonzero(adjacency[i])
        if len(js) < 2:
            continue
        a, b = np.nonzero(adjacency[np.ix_(js, js)])
        if len(a):
            found.append(np.stack([np.full(len(a), i, dtype=np.int64), js[a], js[b]], axis=1))
            count += len(a)
            if cap is not None and count > cap:
                return np.concatenate(found), True
    if not found:
        return np.zeros((0, 3), dtype=np.int64), False
    return np.concatenate(found), False


def score_mesh(edge_emb: Tensor, face_emb: Tensor, gt_edges: np.ndarray, gt_faces: np.ndarray,
               cap: int | None = None, pair_chunk: int = 1 << 20, face_chunk: int = 1 << 18) -> dict:
    """Strict metric for one mesh. gt_edges [E,2] and gt_faces [F,3] are sorted-unique int64."""
    n = len(edge_emb)
    device = edge_emb.device
    pairs = torch.triu_indices(n, n, 1, device=device).T
    logits = torch.cat([edge_logits(edge_emb, pairs[s:s + pair_chunk])
                        for s in range(0, len(pairs), pair_chunk)]) if len(pairs) else edge_emb.new_zeros(0)
    if not torch.isfinite(logits).all():
        raise FloatingPointError("nonfinite edge logits")
    predicted = (logits > 0).cpu().numpy()
    pairs_np = pairs.cpu().numpy()
    truth = np.isin(pairs_np[:, 0] * n + pairs_np[:, 1], gt_edges[:, 0] * n + gt_edges[:, 1])
    edge = dict(tp=int((predicted & truth).sum()), fp=int((predicted & ~truth).sum()),
                fn=int((~predicted & truth).sum()), tn=int((~predicted & ~truth).sum()))
    adjacency = np.zeros((n, n), dtype=bool)
    adjacency[pairs_np[predicted, 0], pairs_np[predicted, 1]] = True
    candidates, overflow = triangles(adjacency, cap)
    gt_keys = (gt_faces[:, 0] * n + gt_faces[:, 1]) * n + gt_faces[:, 2]
    labels = np.isin((candidates[:, 0] * n + candidates[:, 1]) * n + candidates[:, 2], gt_keys)
    face_pred = np.zeros(len(candidates), dtype=bool)
    for s in range(0, len(candidates), face_chunk):
        ids = torch.from_numpy(candidates[s:s + face_chunk]).to(device)
        scores = face_logits(face_emb, ids)
        if not torch.isfinite(scores).all():
            raise FloatingPointError("nonfinite face logits")
        face_pred[s:s + face_chunk] = (scores > 0).cpu().numpy()
    tp = int((face_pred & labels).sum())
    covered = (adjacency[gt_faces[:, 0], gt_faces[:, 1]] & adjacency[gt_faces[:, 0], gt_faces[:, 2]]
               & adjacency[gt_faces[:, 1], gt_faces[:, 2]])
    face = dict(tp=tp, fp=int((face_pred & ~labels).sum()), fn=len(gt_faces) - tp,
                tn=int((~face_pred & ~labels).sum()))
    result = dict(vertices=n, edge=edge, face=face, candidates=len(candidates), overflow=bool(overflow),
                  face_fn_missing=int((~covered).sum()), face_fn_present=int(covered.sum()) - tp)
    for task in ("edge", "face"):
        result[task]["micro_f1"] = micro_f1(result[task]["tp"], result[task]["fp"], result[task]["fn"])
    result["edge_strict"] = edge["fp"] == edge["fn"] == 0
    result["joint_strict"] = result["edge_strict"] and face["fp"] == face["fn"] == 0 and not overflow
    return result


def default_cap(gt_faces: np.ndarray) -> int:
    return max(200_000, 20 * len(gt_faces))


@torch.no_grad()
def evaluate_latents(vae, data, latents: Tensor, normalized: bool) -> list:
    """Decode and score every object. `latents` is [T,512], packed in data order."""
    rows = []
    for i, uid in enumerate(data.uids):
        sl = data.token_slice(i)
        z = latents[sl].float()
        z_norm = z if normalized else (z - data.mean) / data.std
        z_raw = z * data.std + data.mean if normalized else z
        edge_emb, face_emb = decode(vae, z_raw)
        row = score_mesh(edge_emb, face_emb, data.gt_edges[i], data.gt_faces[i], default_cap(data.gt_faces[i]))
        row["uid"] = uid
        row["latent_rmse"] = float((z_norm - data.mu_normalized[sl]).square().mean().sqrt())
        rows.append(row)
    return rows


def aggregate(rows: list) -> dict:
    out = {}
    for task in ("edge", "face"):
        sums = {k: sum(r[task][k] for r in rows) for k in ("tp", "fp", "fn", "tn")}
        sums["micro_f1"] = micro_f1(sums["tp"], sums["fp"], sums["fn"])
        out[task] = sums
    out["objects"] = len(rows)
    out["joint_strict"] = sum(r["joint_strict"] for r in rows)
    out["edge_strict"] = sum(r["edge_strict"] for r in rows)
    out["overflow_objects"] = [r["uid"] for r in rows if r["overflow"]]
    out["face_fn_missing"] = sum(r["face_fn_missing"] for r in rows)
    out["face_fn_present"] = sum(r["face_fn_present"] for r in rows)
    tokens = sum(r["vertices"] for r in rows)
    out["latent_rmse_token_weighted"] = float(np.sqrt(sum(r["latent_rmse"] ** 2 * r["vertices"] for r in rows) / tokens))
    by_size = {}
    for low, high in SIZE_BINS:
        group = [r for r in rows if low <= r["vertices"] < high]
        if not group:
            continue
        f = {k: sum(r["face"][k] for r in group) for k in ("tp", "fp", "fn")}
        by_size[size_bin(low)] = dict(objects=len(group), face_micro_f1=micro_f1(f["tp"], f["fp"], f["fn"]),
                                      joint_strict=sum(r["joint_strict"] for r in group),
                                      latent_rmse_mean=float(np.mean([r["latent_rmse"] for r in group])))
    out["by_size"] = by_size
    out["lower_bound"] = bool(out["overflow_objects"])
    return out
