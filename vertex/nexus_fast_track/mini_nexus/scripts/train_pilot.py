#!/usr/bin/env python3
"""Train one mini-Nexus stage with batching, checkpoints and torchrun DDP."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mini_nexus.data import (  # noqa: E402
    PilotStage2Dataset,
    collate_octree_level,
    collate_pilot_samples,
    load_ready_uids,
)
from mini_nexus.topology import TopologyAutoencoder  # noqa: E402
from mini_nexus.training import (  # noqa: E402
    TopologyAESystem,
    TopologyFlowSystem,
    VertexStageSystem,
)


DEFAULT_STAGE2 = PROJECT_ROOT.parent / "data_pilot32" / "stage2_outputs"
DEFAULT_AUDIT = PROJECT_ROOT / "outputs" / "pilot_algorithm_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pilot training; model defaults are scaled for smoke runs.")
    parser.add_argument(
        "--stage", choices=("vertex", "topology-ae", "topology-flow"), required=True
    )
    parser.add_argument("--stage2-root", type=Path, default=DEFAULT_STAGE2)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--topology-ae-checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1, help="每个 GPU 的 batch size")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--precision", choices=("fp32", "bf16"))
    parser.add_argument(
        "--force-ddp",
        action="store_true",
        help="允许 torchrun --nproc_per_node=1 也经过真实 DDP 路径",
    )
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--condition-dim", type=int, help="Vertex VecSet width; default: hidden-dim")
    parser.add_argument("--condition-heads", type=int, help="Vertex VecSet heads; default: num-heads")
    parser.add_argument("--condition-layers", type=int, default=8)
    parser.add_argument("--condition-tokens", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--spacetime-dim", type=int, default=16)
    parser.add_argument("--max-depth", type=int, default=9)
    parser.add_argument(
        "--activation-checkpointing", action="store_true",
        help="Checkpoint Vertex VecSet/DiT blocks to reduce activation memory",
    )
    args = parser.parse_args()
    if args.condition_dim is None:
        args.condition_dim = args.hidden_dim
    if args.condition_heads is None:
        args.condition_heads = args.num_heads
    if args.stage == "vertex":
        args.vertex_architecture = "point_cloud_vertex_v1"
    return args


def initialize_distributed(force_ddp: bool) -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 or force_ddp:
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL DDP requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, local_rank, world_size, device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_topology_autoencoder(args: argparse.Namespace) -> TopologyAutoencoder:
    if args.topology_ae_checkpoint is None:
        raise ValueError("topology-flow requires --topology-ae-checkpoint")
    checkpoint = load_checkpoint(args.topology_ae_checkpoint, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    autoencoder_state = {
        key.removeprefix("autoencoder."): value
        for key, value in state.items()
        if key.startswith("autoencoder.")
    }
    if not autoencoder_state:
        autoencoder_state = state
    autoencoder = TopologyAutoencoder(
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        spacetime_dim=args.spacetime_dim,
        num_heads=args.num_heads,
        num_layers=max(1, args.num_layers // 2),
    )
    autoencoder.load_state_dict(autoencoder_state, strict=True)
    return autoencoder


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.stage == "vertex":
        return VertexStageSystem(
            hidden_dim=args.hidden_dim,
            condition_dim=args.condition_dim,
            condition_tokens=args.condition_tokens,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            condition_heads=args.condition_heads,
            condition_layers=args.condition_layers,
            max_depth=args.max_depth,
            use_checkpoint=args.activation_checkpointing,
        )
    if args.stage == "topology-ae":
        return TopologyAESystem(
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            spacetime_dim=args.spacetime_dim,
            num_heads=args.num_heads,
            num_layers=max(1, args.num_layers // 2),
        )
    return TopologyFlowSystem(
        load_topology_autoencoder(args),
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        condition_tokens=args.condition_tokens,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )


def model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    if isinstance(model, DistributedDataParallel):
        return model.module.state_dict()
    return model.state_dict()


def load_checkpoint(path: Path, map_location: Any) -> dict[str, Any]:
    """Load only project-generated checkpoints; spell out the PyTorch 2.6 change."""

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    ready_uids: list[str],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "stage": args.stage,
            **({"vertex_architecture": "point_cloud_vertex_v1"} if args.stage == "vertex" else {}),
            "step": step,
            "model": model_state(model),
            "optimizer": optimizer.state_dict(),
            "args": json_safe_args(args),
            "ready_uids": ready_uids,
        },
        temporary,
    )
    temporary.replace(path)


def json_safe_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def validate_vertex_resume(args: argparse.Namespace, checkpoint: dict[str, Any]) -> None:
    if args.stage != "vertex":
        return
    if checkpoint.get("stage") != "vertex":
        raise ValueError("resume checkpoint stage mismatch")
    if checkpoint.get("vertex_architecture") != "point_cloud_vertex_v1":
        raise ValueError("Vertex resume requires point_cloud_vertex_v1; legacy prototype weights are incompatible")
    saved_args = checkpoint.get("args", {})
    for name in (
        "hidden_dim", "condition_dim", "condition_tokens", "num_layers",
        "num_heads", "condition_heads", "condition_layers", "max_depth",
    ):
        if saved_args.get(name) != getattr(args, name):
            raise ValueError(f"Vertex resume model configuration mismatch: {name}")


def main() -> int:
    args = parse_args()
    if args.precision is None:
        args.precision = "fp32" if args.stage == "topology-ae" else "bf16"
    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("steps and batch-size must be positive")
    if args.hidden_dim % args.num_heads:
        raise ValueError("hidden-dim must be divisible by num-heads")
    if args.stage == "vertex":
        if min(args.condition_dim, args.condition_heads, args.condition_layers) <= 0:
            raise ValueError("Vertex condition dimensions must be positive")
        if args.condition_dim % args.condition_heads:
            raise ValueError("condition-dim must be divisible by condition-heads")
    if args.stage == "topology-ae" and args.precision != "fp32":
        raise ValueError("topology-ae requires --precision fp32")
    if args.stage == "topology-ae":
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if torch.get_float32_matmul_precision() != "highest":
            raise RuntimeError("failed to require highest FP32 matmul precision")
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            raise RuntimeError("failed to disable TF32 for topology-ae")
    rank, local_rank, world_size, device = initialize_distributed(args.force_ddp)
    distributed = dist.is_initialized()
    seed_everything(args.seed + rank)

    output = args.output or PROJECT_ROOT / "outputs" / "pilot_training" / args.stage
    output = output.resolve()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    ready_uids = load_ready_uids(args.audit.resolve())
    dataset = PilotStage2Dataset(args.stage2_root.resolve(), ready_uids)
    sampler = (
        DistributedSampler(dataset, shuffle=True, seed=args.seed)
        if distributed
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_pilot_samples,
        drop_last=False,
    )

    checkpoint = None
    if args.resume is not None and args.stage == "vertex":
        checkpoint = load_checkpoint(args.resume, map_location="cpu")
        validate_vertex_resume(args, checkpoint)

    model = build_model(args).to(device)
    if args.stage == "topology-ae":
        model = model.to(dtype=torch.float32)
        non_fp32_parameters = [
            name
            for name, parameter in model.named_parameters()
            if parameter.is_floating_point() and parameter.dtype != torch.float32
        ]
        if non_fp32_parameters:
            raise RuntimeError(
                f"topology-ae parameters are not FP32: {non_fp32_parameters[:8]}"
            )
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    start_step = 0
    if args.resume is not None:
        if checkpoint is None:
            checkpoint = load_checkpoint(args.resume, map_location=device)
        target = model.module if isinstance(model, DistributedDataParallel) else model
        target.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])

    if rank == 0:
        config = {
            **json_safe_args(args),
            "output": str(output),
            "device": str(device),
            "world_size": world_size,
            "ready_uids": ready_uids,
            "warning": (
                None
                if len(ready_uids) >= world_size
                else "ready sample count is smaller than world size; DDP sampler repeats samples"
            ),
        }
        if args.stage == "vertex":
            config.update(
                {
                    "vertex_architecture": "point_cloud_vertex_v1",
                    "independent_reimplementation": True,
                    "claim_boundary": "training_process_not_generation_quality",
                    "paper_deviations": [
                        "point_cloud_condition_only",
                        "vertex_architecture_details_independently_reconstructed",
                        "uniform_time_linear_rectified_flow_training",
                        "pilot_dataset_and_configurable_model_dimensions",
                    ],
                }
            )
        if args.stage == "topology-ae":
            config.update(
                {
                    "parameter_precision": "fp32",
                    "network_precision": "fp32",
                    "loss_precision": "fp32",
                    "gradient_reduce_precision": "fp32",
                    "optimizer_state_precision": "fp32",
                    "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
                    "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                }
            )
        (output / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    iterator = iter(loader)
    epoch = 0
    log_path = output / "train.jsonl"
    for step in range(start_step, args.steps):
        try:
            cpu_batch = next(iterator)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            iterator = iter(loader)
            cpu_batch = next(iterator)
        batch = cpu_batch.to(device)
        if args.stage == "topology-ae" and batch.vertices.dtype != torch.float32:
            raise RuntimeError(
                f"topology-ae vertex inputs are not FP32: {batch.vertices.dtype}"
            )
        optimizer.zero_grad(set_to_none=True)
        autocast_enabled = (
            args.precision == "bf16"
            and args.stage != "topology-ae"
            and device.type == "cuda"
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            if args.stage == "vertex":
                current_depth = step % args.max_depth + 1
                level = collate_octree_level(
                    cpu_batch, current_depth, args.max_depth
                ).to(device)
                loss = model(batch.condition, level)
            else:
                current_depth = None
                loss = model(batch)
        if args.stage == "topology-ae" and loss.dtype != torch.float32:
            raise RuntimeError(f"topology-ae loss is not FP32: {loss.dtype}")
        loss.backward()
        if args.stage == "topology-ae":
            non_fp32_gradients = [
                name
                for name, parameter in model.named_parameters()
                if parameter.grad is not None and parameter.grad.dtype != torch.float32
            ]
            if non_fp32_gradients:
                raise RuntimeError(
                    f"topology-ae gradients are not FP32: {non_fp32_gradients[:8]}"
                )
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if args.stage == "topology-ae" and step == start_step:
            non_fp32_optimizer_states = []
            for state in optimizer.state.values():
                for name, value in state.items():
                    if (
                        torch.is_tensor(value)
                        and value.is_floating_point()
                        and value.dtype != torch.float32
                    ):
                        non_fp32_optimizer_states.append(f"{name}:{value.dtype}")
            if non_fp32_optimizer_states:
                raise RuntimeError(
                    "topology-ae optimizer states are not FP32: "
                    f"{non_fp32_optimizer_states[:8]}"
                )

        if rank == 0:
            record = {
                "step": step + 1,
                "stage": args.stage,
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm),
                "depth": current_depth,
                "uids": list(batch.uids),
                "world_size": world_size,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
                save_checkpoint(
                    output / f"checkpoint-{step + 1:06d}.pt",
                    model,
                    optimizer,
                    step + 1,
                    args,
                    ready_uids,
                )

    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
