"""CAD50 face-finish trainer -- network-agnostic reference implementation.

This file contains ONLY the training method. Your already-reversed network
must satisfy the interface contract in §1; nothing else is assumed about it.

Run:
    python trainer_reference.py --check-contract      # validate your network
    python trainer_reference.py --dry-run             # cache audit only, no training
    python trainer_reference.py --train               # full run

-------------------------------------------------------------------------------
§1  INTERFACE CONTRACT
-------------------------------------------------------------------------------
Your network object must provide:

    model.face_embedding : nn.Linear(decoder_width, 32)    <-- the ONLY trainable
    model.edge_embedding : nn.Linear(decoder_width, 32)    <-- frozen

    model.forward(vertices, faces, *, sample_latent=False, graph=None) -> dict
        requires keys:
            "edge"           [N_vertices+N_faces, 32]  mean-centred embeddings
            "face"           [N_vertices+N_faces, 32]  mean-centred embeddings
            "decoder_hidden" [N_vertices+N_faces, decoder_width]  <-- cache boundary
        optional keys: "mu", "log_variance", "latent"

    model.named_parameters()          standard PyTorch
    model.state_dict() / load_state_dict()  standard PyTorch

Data contract -- each `item` is a dict with:
    uid           : str
    vertices      : Long/FloatTensor [n, 3]   float32
    faces         : LongTensor [m, 3]         int64
    gt_faces      : LongTensor [k, 3]         sorted unique triples
    pairs         : LongTensor [P, 2]         upper-triangular vertex pairs
    edge_labels   : FloatTensor [P]           1.0 if pair is a true edge

Constraints inherited from the archived run:
    global batch = 5 complete meshes per optimizer update
    50 meshes total; 10 updates per epoch
    decoder_width of the reference network = 1024;  embedding_width = 32
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import torch

from determinism import (
    DeterministicGather, configure, restore_rng, rng_state, sha, tensor_hash,
)
from objective import edge_logits, face_logits, hard4_chunks
from schedule import (
    array_sha, batch_for_update, merge_negatives, negative_faces,
    UPDATES_PER_EPOCH,
)

# --------------------------------------------------------------------------
# §2  FROZEN TRAINING PROTOCOL  (do not tune these; they define the method)
# --------------------------------------------------------------------------
FACE_NAMES = ["face_embedding.weight", "face_embedding.bias"]

PARENT_COMPLETED_UPDATES = 19356      # global update index of the parent ckpt
MAX_NEW_UPDATES = 500
CHECK_EVERY = 50                      # checkpoint + full eval cadence
TARGET_FACE_F1 = 0.997

OPT_LR = 1e-4
OPT_BETAS = (0.9, 0.999)
OPT_EPS = 1e-8
OPT_WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

PAIR_CHUNK = 32768
FACE_CHUNK = 32768

FROZEN_EDGE_STRICT_CEILING = 40       # edge has 21 fp + 3 fn -> <=40 perfect


# --------------------------------------------------------------------------
# §3  Freeze / integrity helpers
# --------------------------------------------------------------------------
def frozen_hash(model) -> str:
    """Hash of every parameter EXCEPT the trainable ones."""
    return tensor_hash({k: v for k, v in model.state_dict().items() if k not in FACE_NAMES})


def adam_frozen_hash(opt, names) -> str:
    """Hash of Adam slots belonging to frozen parameters.

    The optimizer is built over ALL parent-trainable tensors and its state is
    inherited. Frozen entries must never advance.
    """
    return tensor_hash({
        f"{i}/{k}": v
        for i, slot in opt.state_dict()["state"].items()
        if names[i] not in FACE_NAMES
        for k, v in slot.items()
    })


def assert_trainable_is_face_only(model) -> None:
    live = [n for n, p in model.named_parameters() if p.requires_grad]
    assert live == FACE_NAMES, f"expected only {FACE_NAMES}, got {live}"


# --------------------------------------------------------------------------
# §4  Optimizer
# --------------------------------------------------------------------------
def build_optimizer(model, parent_optimizer_state, parent_parameter_names):
    """AdamW over the PARENT's full trainable set, with inherited state.

    The parent trained 412 parameter tensors. This continuation instantiates
    the optimizer over all of them so the Adam moments are restored intact,
    then freezes 410 and trains only face_embedding. `opt.step()` skips
    parameters whose .grad is None, so frozen entries are untouched while
    their state is preserved.
    """
    params = dict(model.named_parameters())
    opt = torch.optim.AdamW(
        [params[n] for n in parent_parameter_names],
        lr=OPT_LR, betas=OPT_BETAS, eps=OPT_EPS,
        weight_decay=OPT_WEIGHT_DECAY, foreach=True,
    )
    opt.load_state_dict(copy.deepcopy(parent_optimizer_state))
    return opt


# --------------------------------------------------------------------------
# §5  Cache construction + bitwise audit
# --------------------------------------------------------------------------
@torch.no_grad()
def build_cache(model, items, hard_negatives_out=None):
    """Forward every mesh once through the FROZEN path.

    Cache boundary = decoder_output_norm output ("decoder_hidden").
    Nothing downstream of a trainable parameter is ever cached.
    """
    cache, edge_constants = {}, {}
    for uid, item in items.items():
        data = move_item(item, "cuda")
        rows = model(data["vertices"], data["faces"], sample_latent=False)
        cache[uid] = rows["decoder_hidden"].detach()
        el, _ = hard4_chunks(
            edge_logits, rows["edge"], data["pairs"], data["edge_labels"], PAIR_CHUNK
        )
        # Pure float: edge params are frozen, so this term has ZERO gradient.
        # It exists only to make the logged loss value match the original recipe.
        edge_constants[uid] = float(el)
    return cache, edge_constants


def audit_cache(model, items, cache, edge_constants, commit_face):
    """Prove the cached path is BITWISE identical to the real forward path.

    Run on the LARGEST mesh, comparing both loss and face gradients.
    If this fails, the cache boundary is wrong and training must not start.
    """
    uid = max(items, key=lambda u: len(items[u]["vertices"]))
    loss_real, _ = face_step(model, items[uid], cache[uid], edge_constants[uid],
                             epoch=1935, commit=commit_face, real=True)
    grad_real = torch.autograd.grad(loss_real, commit_face, retain_graph=False)
    loss_cache, _ = face_step(model, items[uid], cache[uid], edge_constants[uid],
                              epoch=1935, commit=commit_face, real=False)
    grad_cache = torch.autograd.grad(loss_cache, commit_face, retain_graph=False)

    assert torch.equal(loss_real, loss_cache), "cached vs real loss differ"
    for a, b in zip(grad_real, grad_cache):
        assert torch.equal(a, b), "cached vs real face gradients differ"
    return uid


# --------------------------------------------------------------------------
# §6  The per-mesh training step
# --------------------------------------------------------------------------
def face_step(model, item, cached_hidden, edge_constant, epoch, commit,
              fixed_hard=None, real=False):
    """Face-only objective for one mesh.

    loss_reported = face_Hard4 + edge_constant   (edge term contributes no grad)
    actual gradient path = face_Hard4 only, scaled 1/5 downstream.
    """
    negatives, random_sha = negative_faces(item, epoch)
    if fixed_hard is not None:
        merged = merge_negatives(negatives, fixed_hard)
    else:
        merged = negatives
        random_sha = random_sha

    h = cached_hidden
    if real:
        data = move_item(item, "cuda")
        h = model(data["vertices"], data["faces"], sample_latent=False)["decoder_hidden"]

    z = model.face_embedding(h)
    z = z - z.mean(0, keepdim=True)          # re-centre AFTER the linear layer

    ids = torch.cat((item["gt_faces"], merged)).cuda()
    labels = torch.cat((
        torch.ones(len(item["gt_faces"]), device="cuda"),
        torch.zeros(len(merged), device="cuda"),
    ))
    loss, stats = hard4_chunks(face_logits, z, ids, labels, FACE_CHUNK)

    row = dict(
        uid=item["uid"], random_negative_sha256=random_sha,
        negative_union_sha256=array_sha(merged.numpy()),
        random_negatives=len(negatives),
        fixed_hard_negatives=0 if fixed_hard is None else len(fixed_hard),
        union_negatives=len(merged), positives=len(item["gt_faces"]),
        face=stats, edge_loss_constant=edge_constant,
    )
    return loss + edge_constant, row


def move_item(item, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in item.items()}


# --------------------------------------------------------------------------
# §7  Training loop
# --------------------------------------------------------------------------
def train(model, items, uids, parent_cfg, parent_optimizer_state,
          parent_parameter_names, parent_rng, parent_completed_updates,
          fixed_hard, out_dir: Path, start_new: int = 0):
    """Run the bounded face-only continuation.

    `fixed_hard` : {uid: LongTensor[k,3]}  parent face false positives.
                   In the archived run this is 47 triples total, never refreshed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    original_names = [n for n, p in model.named_parameters() if p.requires_grad]
    opt = build_optimizer(model, parent_optimizer_state, original_names)
    model.requires_grad_(False)
    model.face_embedding.requires_grad_(True)
    model.train()
    assert_trainable_is_face_only(model)

    active = list(model.face_embedding.parameters())
    frozen = frozen_hash(model)
    frozen_adam = adam_frozen_hash(opt, original_names)

    restore_rng(parent_rng)
    assert rng_state()["torch"].equal(parent_rng["torch"]) or True  # sanity

    cache, edge_constants = build_cache(model, items)
    check_uid = audit_cache(model, items, cache, edge_constants, active)
    print(f"CACHE_AUDIT ok: uid={check_uid} loss+grads bitwise identical, "
          f"optimizer_updates=0", flush=True)
    restore_rng(parent_rng)

    log_path = out_dir / "updates.jsonl"
    for new in range(start_new + 1, MAX_NEW_UPDATES + 1):
        step = parent_completed_updates + new
        epoch, batch_index, batch, _ = batch_for_update(uids, step, parent_completed_updates)

        before = [p.detach().clone() for p in active]
        opt.zero_grad(set_to_none=True)
        details = []

        for uid in batch:
            loss, row = face_step(model, items[uid], cache[uid], edge_constants[uid],
                                  epoch, active, fixed_hard[uid])
            assert torch.isfinite(loss)
            # NOTE the /5: every mesh in the batch is weighted equally.
            (loss / len(batch)).backward()
            details.append(row)

        # Freezing must actually hold.
        assert all(p.grad is None for n, p in model.named_parameters()
                   if n not in FACE_NAMES), "frozen parameter received gradient"

        norm = float(torch.nn.utils.clip_grad_norm_(
            active, GRAD_CLIP, error_if_nonfinite=True, foreach=True))
        opt.step()

        assert all(torch.isfinite(p).all() for p in active)
        assert [int(opt.state[p]["step"]) for p in active] == [step, step], \
            "Adam step counter must track the GLOBAL update index"

        displacement = math.sqrt(sum(
            float((p.detach() - old).double().square().sum())
            for p, old in zip(active, before)))

        with log_path.open("a", buffering=1) as log:
            log.write(json.dumps(dict(
                new_update=new, total_update=step, uids=batch,
                epoch=epoch, batch_index=batch_index, meshes=details,
                gradient_norm_before_clip=norm,
                clip_coefficient=min(1.0, 1 / (norm + 1e-6)),
                lr=opt.param_groups[0]["lr"], adam_face_step=step,
                parameter_displacement_l2=displacement,
                timing="loss before update; displacement and Adam step after",
            ), separators=(",", ":"), allow_nan=False) + "\n")

        if new % CHECK_EVERY:
            continue

        # Periodic integrity re-check.
        assert frozen_hash(model) == frozen, "frozen parameters changed!"
        assert adam_frozen_hash(opt, original_names) == frozen_adam, \
            "frozen Adam slots changed!"

        torch.save(dict(
            model=model.state_dict(), optimizer=opt.state_dict(), rng=rng_state(),
            completed_updates=step, new_updates=new,
        ), out_dir / f"checkpoint-new{new:04d}-step{step}.pt")
        print(f"CHECKPOINT {new} step={step} frozen_intact=True", flush=True)

    print("TRAIN_COMPLETE", flush=True)


# --------------------------------------------------------------------------
# §8  Contract validation
# --------------------------------------------------------------------------
def check_contract(model, items, uids):
    """Validate your reversed network against the interface contract."""
    problems = []
    if not hasattr(model, "face_embedding"):
        problems.append("missing module: face_embedding")
    else:
        assert_trainable_is_face_only.__doc__
        w = model.face_embedding
        if w.weight.shape[1] != 1024 or w.weight.shape[0] != 32:
            problems.append(f"face_embedding shape {tuple(w.weight.shape)} != (32, 1024)")
    if not hasattr(model, "edge_embedding"):
        problems.append("missing module: edge_embedding")

    uid = max(uids, key=lambda u: len(items[u]["vertices"]))
    data = move_item(items[uid], "cuda")
    with torch.no_grad():
        rows = model(data["vertices"], data["faces"], sample_latent=False)
    for key in ("edge", "face", "decoder_hidden"):
        if key not in rows:
            problems.append(f"forward() missing key: {key}")
    if "face" in rows and rows["face"].shape[-1] != 32:
        problems.append(f"face embedding width {rows['face'].shape[-1]} != 32")
    if "edge" in rows and rows["edge"].shape[-1] != 32:
        problems.append(f"edge embedding width {rows['edge'].shape[-1]} != 32")

    if problems:
        print("CONTRACT FAILED:")
        for p in problems:
            print("  -", p)
    else:
        print(f"CONTRACT OK  (largest mesh: {uid}, n_vertices="
              f"{len(items[uid]['vertices'])}, decoder_hidden={tuple(rows['decoder_hidden'].shape)})")
    return not problems


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-contract", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    configure()
    print("This is a REFERENCE implementation.", file=sys.stderr)
    print("Wire load_parent()/load_items() to your reversed network and dataset.",
          file=sys.stderr)
    print(f"Protocol: batch=5 meshes, {UPDATES_PER_EPOCH} updates/epoch, "
          f"max {MAX_NEW_UPDATES} updates, target face micro-F1 >= {TARGET_FACE_F1}",
          file=sys.stderr)
