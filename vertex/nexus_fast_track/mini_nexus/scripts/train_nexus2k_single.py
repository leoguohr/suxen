#!/usr/bin/env python3
"""Train Nexus2K Vertex or Topology Flow on one GPU with resume metadata."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mini_nexus.data_2k import Nexus2KManifestDataset, collate_nexus2k_samples  # noqa: E402
from mini_nexus.topology import TopologyAutoencoder  # noqa: E402
from mini_nexus.topology_checkpoint import (  # noqa: E402
    load_topology_checkpoint,
    load_topology_system_from_checkpoint,
)
from mini_nexus.training import TopologyFlowSystem, VertexStageSystem  # noqa: E402


PAPER_DEVIATIONS = (
    "point_cloud_condition_only",
    "single_gpu_stagewise_training",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("vertex", "topology-flow"), required=True
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--topology-ae-checkpoint", type=Path)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--validation-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--hidden-dim", type=int,
        help="Vertex default: 1536; topology-flow default: 128",
    )
    parser.add_argument(
        "--condition-dim", type=int,
        help="Vertex VecSet width: 2048 at hidden-dim 1536, otherwise hidden-dim",
    )
    parser.add_argument(
        "--condition-heads", type=int,
        help="Vertex VecSet heads: 16 at hidden-dim 1536, otherwise num-heads",
    )
    parser.add_argument("--condition-layers", type=int, default=8)
    parser.add_argument(
        "--condition-tokens", type=int,
        help="Vertex default: 1024; topology-flow default: 32",
    )
    parser.add_argument(
        "--num-layers", type=int,
        help="Vertex default: 36; topology-flow default: 2",
    )
    parser.add_argument(
        "--num-heads", type=int,
        help="Vertex default: 12; topology-flow default: 4",
    )
    parser.add_argument(
        "--activation-checkpointing", action="store_true",
        help="Checkpoint Vertex VecSet/DiT blocks to reduce activation memory",
    )
    parser.add_argument("--latent-dim", type=int, default=32)
    args = parser.parse_args()
    defaults = (
        {"hidden_dim": 1536, "condition_tokens": 1024, "num_layers": 36, "num_heads": 12}
        if args.stage == "vertex"
        else {"hidden_dim": 128, "condition_tokens": 32, "num_layers": 2, "num_heads": 4}
    )
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    full_vertex_defaults = args.stage == "vertex" and args.hidden_dim == 1536
    if args.condition_dim is None:
        args.condition_dim = 2048 if full_vertex_defaults else args.hidden_dim
    if args.condition_heads is None:
        args.condition_heads = 16 if full_vertex_defaults else args.num_heads
    if args.stage == "vertex":
        args.vertex_architecture = "point_cloud_vertex_v1"
    return args


def paper_deviations(args: argparse.Namespace) -> tuple[str, ...]:
    if args.stage != "vertex":
        return PAPER_DEVIATIONS + ("scaled_model_dimensions",)
    deviations = PAPER_DEVIATIONS + (
        "vertex_architecture_details_independently_reconstructed",
        "uniform_time_linear_rectified_flow_training",
    )
    dimensions = (
        args.hidden_dim, args.condition_dim, args.condition_tokens, args.num_layers,
        args.num_heads, args.condition_heads, args.condition_layers,
    )
    if dimensions != (1536, 2048, 1024, 36, 12, 16, 8):
        deviations += ("scaled_model_dimensions",)
    return deviations


def validate_vertex_resume(args: argparse.Namespace, checkpoint: dict[str, Any]) -> None:
    if args.stage != "vertex":
        return
    if checkpoint.get("vertex_architecture") != "point_cloud_vertex_v1":
        raise ValueError("Vertex resume requires point_cloud_vertex_v1; legacy prototype weights are incompatible")
    saved_args = checkpoint.get("args", {})
    for name in (
        "hidden_dim", "condition_dim", "condition_tokens", "num_layers",
        "num_heads", "condition_heads", "condition_layers",
    ):
        if saved_args.get(name) != getattr(args, name):
            raise ValueError(f"Vertex resume model configuration mismatch: {name}")


def json_safe_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint(path: Path, map_location: Any) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_autoencoder(args: argparse.Namespace) -> TopologyAutoencoder:
    if args.topology_ae_checkpoint is None:
        raise ValueError("topology-flow requires --topology-ae-checkpoint")
    checkpoint = load_topology_checkpoint(args.topology_ae_checkpoint)
    autoencoder = load_topology_system_from_checkpoint(
        checkpoint, device="cpu"
    ).autoencoder
    if autoencoder.mu.out_features != args.latent_dim:
        raise ValueError("--latent-dim must match the Topology AE checkpoint")
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
            max_depth=9,
            use_checkpoint=args.activation_checkpointing,
        )
    return TopologyFlowSystem(
        load_autoencoder(args),
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        condition_tokens=args.condition_tokens,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )


@dataclass
class DeterministicIndexStream:
    """按固定随机顺序不断给出“数据集下标”，而不是返回训练样本本身。

    例如 ``next()`` 返回整数 ``7``，含义只是“下一次取
    ``train_dataset[7]``”。遍历完一轮后会按照 ``seed + epoch`` 生成下一轮
    的确定性乱序，因此同一 seed 和同一恢复位置会得到相同的数据顺序。
    """

    length: int
    seed: int
    position: int = 0
    _epoch: int = -1
    _permutation: list[int] | None = None

    def _permutation_for_epoch(self, epoch: int) -> list[int]:
        generator = torch.Generator(device="cpu").manual_seed(self.seed + epoch)
        return torch.randperm(self.length, generator=generator).tolist()

    def next(self) -> int:
        # position 是从训练开始以来已经取过多少个样本。divmod 同时得到：
        # epoch  = 已经完整遍历数据集多少轮；
        # offset = 当前轮中应取乱序表的第几个位置。
        epoch, offset = divmod(self.position, self.length)
        if epoch != self._epoch:
            self._permutation = self._permutation_for_epoch(epoch)
            self._epoch = epoch
        assert self._permutation is not None
        # index 的类型是 int。它不是 batch，也不是 Tensor，只是 Dataset 下标。
        index = self._permutation[offset]
        self.position += 1
        return index


def stage_loss(
    model: torch.nn.Module,
    batch: object,
    stage: str,
    *,
    depth: int,
) -> torch.Tensor:
    if stage == "vertex":
        return model(batch.condition, batch.octree_levels[depth - 1])
    return model(batch)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    *,
    step: int,
    samples_seen: int,
    manifest_sha256: str,
    validation_loss: float | None,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 2 if args.stage == "vertex" else 1,
            "independent_reimplementation": True,
            "paper_deviations": paper_deviations(args),
            **({"vertex_architecture": "point_cloud_vertex_v1"} if args.stage == "vertex" else {}),
            "stage": args.stage,
            "step": step,
            "samples_seen": samples_seen,
            "manifest_sha256": manifest_sha256,
            "validation_loss": validation_loss,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": json_safe_args(args),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
            },
        },
        temporary,
    )
    temporary.replace(path)
    (path.parent / "last_checkpoint.txt").write_text(str(path) + "\n", encoding="utf-8")


@torch.no_grad()
def validate(
    args: argparse.Namespace,
    model: torch.nn.Module,
    dataset: Nexus2KManifestDataset,
    device: torch.device,
) -> float:
    model.eval()
    count = min(args.validation_samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, num=count, dtype=np.int64).tolist()
    losses = []
    for order, index in enumerate(indices):
        batch = collate_nexus2k_samples([dataset[index]]).to(device)
        depth = order % 9 + 1
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=args.precision == "bf16",
        ):
            loss = stage_loss(
                model,
                batch,
                args.stage,
                depth=depth,
            )
        losses.append(float(loss))
    model.train()
    return float(np.mean(losses))


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "steps",
        "gradient_accumulation",
        "save_every",
        "validate_every",
        "validation_samples",
        "hidden_dim",
        "condition_tokens",
        "num_layers",
        "num_heads",
        "latent_dim",
    )
    if any(getattr(args, name) <= 0 for name in positive):
        raise ValueError("steps, intervals and model dimensions must be positive")
    if args.hidden_dim % args.num_heads:
        raise ValueError("hidden-dim must be divisible by num-heads")
    if args.stage == "vertex":
        if min(args.condition_dim, args.condition_heads, args.condition_layers) <= 0:
            raise ValueError("Vertex condition dimensions must be positive")
        if args.condition_dim % args.condition_heads:
            raise ValueError("condition-dim must be divisible by condition-heads")


def main() -> int:
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("formal Nexus2K training requires CUDA")
    seed_everything(args.seed)
    device = torch.device("cuda")
    train_dataset = Nexus2KManifestDataset(args.manifest, "train")
    val_dataset = Nexus2KManifestDataset(args.manifest, "val")
    if len(train_dataset) != 1060 or len(val_dataset) != 118:
        raise RuntimeError("final manifest must contain train=1060 and val=118")

    checkpoint = None
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume, "cpu")
        if checkpoint.get("stage") != args.stage:
            raise ValueError("resume checkpoint stage mismatch")
        if checkpoint.get("manifest_sha256") != train_dataset.manifest_sha256:
            raise ValueError("resume checkpoint manifest hash mismatch")
        validate_vertex_resume(args, checkpoint)

    args.output.mkdir(parents=True, exist_ok=True)
    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    start_step = 0
    samples_seen = 0
    best_validation = math.inf
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        samples_seen = int(checkpoint["samples_seen"])
        if checkpoint.get("validation_loss") is not None:
            best_validation = float(checkpoint["validation_loss"])
        rng = checkpoint.get("rng")
        if rng is not None:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            torch.cuda.set_rng_state_all(rng["cuda"])

    config = {
        **json_safe_args(args),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "independent_reimplementation": True,
        "claim_boundary": "training_process_not_generation_quality",
        "paper_deviations": paper_deviations(args),
        **({"vertex_architecture": "point_cloud_vertex_v1"} if args.stage == "vertex" else {}),
        "manifest_sha256": train_dataset.manifest_sha256,
        "train_count": len(train_dataset),
        "val_count": len(val_dataset),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    atomic_json(args.output / "config.json", config)

    stream = DeterministicIndexStream(len(train_dataset), args.seed, samples_seen)
    log_path = args.output / "train.jsonl"
    model.train()
    for step in range(start_step, args.steps):
        # 这里的 step 是“一次参数更新”。在下面完成若干次 micro-step 的
        # forward/backward 后，循环末尾才执行一次 optimizer.step()。
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        accumulated_loss = 0.0
        uids = []
        depths = []
        try:
            for micro_step in range(args.gradient_accumulation):
                # (1) stream.next() 返回一个整数下标，例如 7。
                sample_index = stream.next()
                # (2) 用整数下标从 Dataset 加载一个 Nexus2KSample 对象。
                sample = train_dataset[sample_index]
                uids.append(sample.uid)
                depth = (samples_seen + micro_step) % 9 + 1
                depths.append(depth)
                # (3) [sample] 是长度为 1 的 Python list，所以这里构成的
                # micro-batch 只有一个 mesh。collate 负责补 batch 维、padding
                # 顶点并保存 mask；它不会凭空把一个 sample 变成多个样本。
                batch = collate_nexus2k_samples([sample]).to(device)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=args.precision == "bf16",
                ):
                    raw_loss = stage_loss(
                        model,
                        batch,
                        args.stage,
                        depth=depth,
                    )
                    # 每个 micro-step 的梯度先除以累计次数。默认累计 4 次，
                    # 所以四个单样本梯度相加后等价于四个样本 loss 的平均值。
                    # 这叫 effective batch size=4；但每次 forward 的物理
                    # micro-batch size 仍然是 1，四个 mesh 没有同时送入网络。
                    loss = raw_loss / args.gradient_accumulation
                loss.backward()
                accumulated_loss += float(raw_loss.detach())
            samples_seen += args.gradient_accumulation
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("gradient norm is not finite")
            # 到这里才使用累计后的梯度更新一次参数。因此外层一个 step
            # 默认消费 4 个样本，而不是每个 micro-step 都更新一次。
            optimizer.step()
            torch.cuda.synchronize(device)
        except Exception as error:
            failure = {
                "stage": args.stage,
                "step": step + 1,
                "samples_seen": samples_seen,
                "uids": uids,
                "error_type": type(error).__name__,
                "error": str(error),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            atomic_json(args.output / "failure.json", failure)
            raise

        validation_loss = None
        if (step + 1) % args.validate_every == 0 or step + 1 == args.steps:
            validation_loss = validate(
                args,
                model,
                val_dataset,
                device,
            )
        record = {
            "step": step + 1,
            "stage": args.stage,
            "train_loss": accumulated_loss / args.gradient_accumulation,
            "validation_loss": validation_loss,
            "gradient_norm": float(gradient_norm),
            "samples_seen": samples_seen,
            "uids": uids,
            "depths": depths if args.stage == "vertex" else None,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)

        if validation_loss is not None and validation_loss < best_validation:
            best_validation = validation_loss
            save_checkpoint(
                args.output / "checkpoint-best.pt",
                model,
                optimizer,
                args,
                step=step + 1,
                samples_seen=samples_seen,
                manifest_sha256=train_dataset.manifest_sha256,
                validation_loss=best_validation,
            )
        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            save_checkpoint(
                args.output / f"checkpoint-{step + 1:07d}.pt",
                model,
                optimizer,
                args,
                step=step + 1,
                samples_seen=samples_seen,
                manifest_sha256=train_dataset.manifest_sha256,
                validation_loss=(None if math.isinf(best_validation) else best_validation),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
