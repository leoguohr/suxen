#!/usr/bin/env python3
"""固定 20 个 mesh 的 Topology AE packed/DDP 训练主循环。

这是当前双卡训练真正执行的主体。建议按 ``main`` 从上到下阅读，数据流是::

    读取固定 manifest 的 20 个 Nexus2KSample
      -> 按二次计算量分配给各 DDP rank
      -> 每个 rank 内把相近大小对象组成 packed batches
      -> 同一次前向并行算多 mesh，但保持序列隔离
      -> 每个 mesh 单独算 edge/face/KL
      -> 20 个 mesh 等权形成一个 optimizer step
      -> DDP 同步梯度 -> clip -> Adam.step -> 写日志/权重

同一个入口既可由单进程调用，也可由 ``torchrun`` 启动多进程。packing 与 DDP
只改变执行调度，不改变这 20 个对象、每对象 loss 或总目标。这里没有通用正式
训练入口的额外功能，目的就是让 overfit 实验容易审计。
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mini_nexus.data_2k import Nexus2KManifestDataset  # noqa: E402
from mini_nexus.negative_candidates import TopologyNegativeCandidateStore  # noqa: E402
from mini_nexus.packed_topology import (  # noqa: E402
    balance_samples_across_ranks,
    collate_packed_topology,
    plan_padded_batches,
    topology_attention_cost,
)
from mini_nexus.training_2k import Nexus2KTopologyAESystem  # noqa: E402


MODEL_PROFILE = "topology_ae_final_layernorm_no_rms_v1"
TRAINING_PROFILE = "fixed_edges_first_face_negatives_v2"
# 这一个 dict 同时用于构造模型和写入 checkpoint，避免“实际训练参数”与
# “评估重建参数”分别维护后发生漂移。24 encoder layers 在模型类中解释为
# 12 个 MeanSAGEConv+Transformer 组合；20 条 overfit 明确关闭 dropout。
MODEL_ARGS = {
    "hidden_dim": 512,
    "latent_dim": 64,
    "spacetime_dim": 32,
    "num_heads": 8,
    "encoder_layers": 24,
    "decoder_layers": 16,
    "decoder_hidden_dim": 1024,
    "encoder_dropout": 0.0,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """只暴露本实验确实需要改变的运行参数。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--negative-candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=20)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="bounded benchmark mode: write config/logs but no model checkpoint",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=200,
        help="linearly increase LR from 0.1x to 1x; 0 disables warmup",
    )
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp32_flash_bf16"),
        default="fp32",
    )
    parser.add_argument("--pair-chunk-size", type=int, default=2_000_000)
    parser.add_argument("--max-packed-meshes", type=int, default=4)
    parser.add_argument("--max-padding-ratio", type=float, default=1.35)
    parser.add_argument(
        "--calibrate-logit-scales",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="calibrate frozen edge/face scales on the fixed manifest before step 1",
    )
    parser.add_argument("--logit-rms-target", type=float, default=1.0)
    parser.add_argument("--logit-calibration-epsilon-draws", type=int, default=4)
    parser.add_argument("--edge-logit-scale", type=float)
    parser.add_argument("--face-logit-scale", type=float)
    parser.add_argument(
        "--face-interval-factor",
        type=float,
        choices=(0.25, 1.0),
        default=0.25,
        help="1.0 uses the Gram determinant; 0.25 uses triangle area squared",
    )
    return parser.parse_args(argv)


def learning_rate_at_step(base_lr: float, step: int, warmup_steps: int) -> float:
    """Preserve the linear warmup used by the 1,000-step LayerNorm run."""

    if not warmup_steps:
        return base_lr
    progress = min((step - 1) / max(warmup_steps - 1, 1), 1.0)
    return base_lr * (0.1 + 0.9 * progress)


def stable_uid_seed(base_seed: int, step: int, uid: str) -> int:
    """为每个 ``(seed,step,UID)`` 生成稳定 VAE 噪声种子。

    若只调用全局 RNG，把 mesh 换到另一 rank 或另一 pack 就会改变它抽到的
    epsilon，导致“调度优化前后”不再是可比实验。这里消除这种无关差异。
    """

    uid_value = int.from_bytes(hashlib.sha256(uid.encode()).digest()[:8], "little")
    return (base_seed + step * 1_000_003 + uid_value) % (2**63 - 1)


def distributed_mean_sums(
    local_loss_sum: torch.Tensor,
    local_component_sums: dict[str, float],
    total_samples: int,
) -> tuple[float, dict[str, float]]:
    """汇总所有 rank 的逐 mesh 和，再除以全局 mesh 数。

    训练梯度与日志都以对象等权为语义。这里不能先求每个 rank 的均值再平均，
    因为两个 rank 分到的对象数或计算组数不一定相同。
    """

    dist.all_reduce(local_loss_sum, op=dist.ReduceOp.SUM)
    names = sorted(local_component_sums)
    values = torch.tensor(
        [local_component_sums[name] for name in names],
        dtype=torch.float64,
        device=local_loss_sum.device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return float(local_loss_sum / total_samples), {
        name: float(value / total_samples) for name, value in zip(names, values)
    }


def resolve_scoring_args(args: argparse.Namespace) -> None:
    """Validate scoring controls and fill explicit non-calibrated defaults."""

    positive = {
        "logit_rms_target": args.logit_rms_target,
        "face_interval_factor": args.face_interval_factor,
    }
    for name, value in positive.items():
        if value <= 0 or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and positive")
    if args.logit_calibration_epsilon_draws <= 0:
        raise ValueError("logit_calibration_epsilon_draws must be positive")
    if args.calibrate_logit_scales:
        if args.edge_logit_scale is not None or args.face_logit_scale is not None:
            raise ValueError(
                "explicit logit scales cannot be combined with automatic calibration"
            )
        return
    args.edge_logit_scale = 1.0 if args.edge_logit_scale is None else args.edge_logit_scale
    args.face_logit_scale = 1.0 if args.face_logit_scale is None else args.face_logit_scale
    for name in ("edge_logit_scale", "face_logit_scale"):
        value = float(getattr(args, name))
        if value <= 0 or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and positive")


def scoring_profile_from_args(args: argparse.Namespace) -> str:
    """Name the behavior actually selected instead of trusting one launcher."""

    scale_mode = "calibrated" if args.calibrate_logit_scales else "explicit"
    face_factor = format(float(args.face_interval_factor), "g").replace(".", "p")
    return (
        f"per_mesh_center_only_fixed_{scale_mode}_logits_"
        f"face_factor_{face_factor}_v1"
    )


def calibrate_fixed_logit_scales(
    model: Nexus2KTopologyAESystem,
    batches: list[object],
    args: argparse.Namespace,
    *,
    rank: int,
    world_size: int,
) -> dict[str, object]:
    """Calibrate frozen scales from per-mesh raw interval second moments.

    Each mesh contributes equally.  Face moments are measured before the
    optional triangle-area factor, so a ``0.25`` experiment is not silently
    cancelled by a four-times-larger calibrated face scale.
    """

    if not model.training:
        raise ValueError("logit calibration requires training mode for VAE epsilon draws")
    local = torch.zeros(3, dtype=torch.float64, device=next(model.parameters()).device)
    for seed_step in range(args.logit_calibration_epsilon_draws):
        for batch in batches:
            sample_seeds = tuple(
                stable_uid_seed(args.seed, seed_step, uid) for uid in batch.uids
            )
            local += model.calibration_interval_moment_sums(
                batch,
                negative_seed=args.seed,
                fixed_negatives=True,
                sample_seeds=sample_seeds,
            )
    if world_size > 1:
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
    expected_calibration_rows = (
        args.expected_samples * args.logit_calibration_epsilon_draws
    )
    if int(local[2].item()) != expected_calibration_rows:
        raise RuntimeError(
            f"logit calibration saw {int(local[2].item())} meshes, "
            f"expected {expected_calibration_rows}"
        )

    edge_raw_rms = math.sqrt(float(local[0] / local[2]))
    face_raw_rms = math.sqrt(float(local[1] / local[2]))
    for name, value in (
        ("edge_raw_rms", edge_raw_rms),
        ("face_raw_rms", face_raw_rms),
    ):
        if not math.isfinite(value) or value <= 1e-6:
            raise FloatingPointError(
                f"{name} is degenerate ({value}); refusing to create a huge scale"
            )

    calibrated = torch.zeros(2, dtype=torch.float64, device=local.device)
    if rank == 0:
        calibrated[0] = args.logit_rms_target / edge_raw_rms
        calibrated[1] = args.logit_rms_target / face_raw_rms
    if world_size > 1:
        dist.broadcast(calibrated, src=0)
    edge_scale, face_scale = map(float, calibrated.tolist())
    model.set_fixed_logit_scales(edge_scale, face_scale)
    args.edge_logit_scale = edge_scale
    args.face_logit_scale = face_scale
    return {
        "enabled": True,
        "seed_steps": list(range(args.logit_calibration_epsilon_draws)),
        "epsilon_draws_per_mesh": args.logit_calibration_epsilon_draws,
        "mesh_count": args.expected_samples,
        "aggregation": "sqrt(mean_per_mesh_raw_interval_second_moment)",
        "calibration_excludes_face_interval_factor": True,
        "target_raw_logit_rms": args.logit_rms_target,
        "edge_raw_rms": edge_raw_rms,
        "face_raw_rms": face_raw_rms,
        "edge_logit_scale": edge_scale,
        "face_logit_scale": face_scale,
        "effective_initial_face_logit_rms": (
            args.logit_rms_target * args.face_interval_factor
        ),
    }


def save_checkpoint(
    path: Path,
    model: Nexus2KTopologyAESystem,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    manifest_sha256: str,
    step: int,
    scoring_profile: str,
    requested_args: dict[str, object],
    calibration: dict[str, object],
) -> None:
    """原子保存模型、优化器以及足够重建本次实验的元数据。"""

    # 先写 .tmp，再用同文件系统的 replace 原子替换目标；中途被终止时不会留下
    # 一个名字正常但内容只写了一半的 checkpoint。
    temporary = path.with_suffix(path.suffix + ".tmp")
    saved_args = vars(args).copy()
    saved_args.update(MODEL_ARGS)
    torch.save(
        {
            "format_version": 4,
            "independent_reimplementation": True,
            "stage": "topology-ae",
            "experiment": "20_sample_training_set_overfit_packed_ddp",
            "model_profile": MODEL_PROFILE,
            "training_profile": TRAINING_PROFILE,
            "scoring_profile": scoring_profile,
            "step": step,
            "samples_seen": args.expected_samples * step,
            "manifest_sha256": manifest_sha256,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": saved_args,
            "requested_args": requested_args,
            "spacetime_scoring": model.scoring_contract(),
            "runtime": model.runtime_contract(),
            "logit_scale_calibration": calibration,
        },
        temporary,
    )
    temporary.replace(path)


def main(
    system_class: type[Nexus2KTopologyAESystem] = Nexus2KTopologyAESystem,
    *,
    required_precision: str | None = None,
) -> int:
    args = parse_args()
    if args.warmup_steps < 0:
        raise ValueError("warmup_steps must be nonnegative")
    requested_args = vars(args).copy()
    resolve_scoring_args(args)
    scoring_profile = scoring_profile_from_args(args)
    # 默认 System 是端到端 FP32；Flash thin entry 显式要求 selective profile。
    # 两种 profile 都不允许用一个外层 autocast 降低整个网络精度。
    active_required_precision = required_precision or "fp32"
    if args.precision != active_required_precision:
        raise ValueError(
            f"this training system requires --precision {active_required_precision}"
        )
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("failed to select highest FP32 matmul precision")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("non-Flash FP32 computation requires TF32 to be disabled")
    # torchrun 为每个子进程注入 WORLD_SIZE/RANK/LOCAL_RANK：
    # RANK 是全局进程号，LOCAL_RANK 是本机 GPU 编号。
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    visible_gpus = torch.cuda.device_count()
    if visible_gpus == 0:
        raise RuntimeError("packed Topology-AE training requires CUDA")
    if local_rank >= visible_gpus:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank}, but only {visible_gpus} GPU(s) are visible; "
            "do not validate two-GPU DDP on a one-GPU instance"
        )
    if world_size > 1:
        # 单机 NVIDIA 多卡使用 NCCL；每个进程随后只绑定自己的 local_rank。
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # 固定 Python、NumPy、PyTorch RNG。VAE epsilon 还会在每 step 按 UID 单独固定。
    random.seed(args.seed + rank)
    np.random.seed((args.seed + rank) % (2**32))
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    # Dataset 直接读冻结 manifest；这里不重新划分、不重新量化，也不改写 NPZ。
    dataset = Nexus2KManifestDataset(args.manifest, "train")
    if len(dataset) != args.expected_samples:
        raise ValueError(
            f"expected {args.expected_samples} fixed meshes, found {len(dataset)}"
        )
    # overfit 实验要求“一个目录只对应一次运行”。若日志已存在就拒绝追加，
    # 防止两次实验的 step/loss 混在同一条曲线里。
    if (args.output / "train.jsonl").exists():
        raise ValueError("refusing to append to an existing packed run")
    # 20 条数据很小，可以一次只读加载；随后固定 rank 分配和 pack 计划。
    samples = [dataset[index] for index in range(len(dataset))]
    assignments = balance_samples_across_ranks(samples, world_size)
    rank_samples = assignments[rank]
    sample_groups = plan_padded_batches(
        rank_samples,
        max_meshes=args.max_packed_meshes,
        max_padding_ratio=args.max_padding_ratio,
    )
    batches = [collate_packed_topology(group).to(device) for group in sample_groups]
    if any(batch.vertices.dtype != torch.float32 for batch in batches):
        raise TypeError("Topology AE vertex inputs must be FP32")

    # sidecar 是只读候选库。当前 overfit 使用固定 face negatives；Edge 分支始终
    # 遍历所有 V choose 2，无需也不会消费 sampled negative edges。
    store = TopologyNegativeCandidateStore(args.negative_candidate_root, "mixed_medium")
    model = system_class(
        **MODEL_ARGS,
        pair_chunk_size=args.pair_chunk_size,
        negative_candidate_store=store,
        fixed_overfit_face_negatives=True,
        edge_logit_scale=(
            1.0 if args.edge_logit_scale is None else args.edge_logit_scale
        ),
        face_logit_scale=(
            1.0 if args.face_logit_scale is None else args.face_logit_scale
        ),
        face_interval_factor=args.face_interval_factor,
    ).to(device=device, dtype=torch.float32)
    non_fp32_parameters = [
        name
        for name, parameter in model.named_parameters()
        if parameter.is_floating_point() and parameter.dtype != torch.float32
    ]
    if non_fp32_parameters:
        raise TypeError(
            "Topology AE parameters must be FP32: "
            + ", ".join(non_fp32_parameters[:5])
        )
    if world_size > 1:
        # DDP 不改变模型；它为每个 rank 保存副本，并在 backward 时 all-reduce 梯度。
        wrapped: Nexus2KTopologyAESystem | DistributedDataParallel = (
            DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False,
            )
        )
    else:
        wrapped = model
    target = wrapped.module if isinstance(wrapped, DistributedDataParallel) else wrapped
    calibration: dict[str, object] = {"enabled": False}
    if args.calibrate_logit_scales:
        calibration = calibrate_fixed_logit_scales(
            target,
            batches,
            args,
            rank=rank,
            world_size=world_size,
        )
    # 与当前论文对齐方案一致使用 Adam；weight_decay=0.0，不沿用旧 AdamW 基线。
    optimizer = torch.optim.Adam(
        wrapped.parameters(), lr=args.learning_rate, weight_decay=0.0
    )

    if rank == 0:
        # 只有 rank0 写公共输出，避免两个进程同时创建/覆盖 config 与 checkpoint。
        args.output.mkdir(parents=True, exist_ok=True)
        config = {
            "experiment": "20_sample_training_set_overfit_packed_ddp",
            "model_profile": MODEL_PROFILE,
            "training_profile": TRAINING_PROFILE,
            "scoring_profile": scoring_profile,
            "model_args": MODEL_ARGS,
            "manifest_sha256": dataset.manifest_sha256,
            "uids": list(dataset.uids),
            "world_size": world_size,
            "global_meshes_per_optimizer_step": args.expected_samples,
            "per_mesh_equal_weight": True,
            "precision_profile": args.precision,
            "runtime": model.runtime_contract(),
            "parameter_precision": "fp32",
            "gradient_reduce_precision": "fp32",
            "optimizer_state_precision": "fp32",
            "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "edge_supervision": "all_unordered_vertex_pairs_per_mesh",
            "face_negative_sampling": (
                "fixed_all_gt_edge_false_cycles_plus_1x_wedge_plus_0.5x_uniform"
            ),
            "spacetime_scoring": {
                "scale_mode": (
                    "fixed_calibrated"
                    if args.calibrate_logit_scales
                    else "fixed_explicit"
                ),
                **target.scoring_contract(),
            },
            "logit_scale_calibration": calibration,
            "rank_plans": [
                {
                    "rank": rank_index,
                    "uids": [sample.uid for sample in assigned],
                    "cost": sum(topology_attention_cost(sample) for sample in assigned),
                }
                for rank_index, assigned in enumerate(assignments)
            ],
            "requested_args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in requested_args.items()
            },
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        }
        (args.output / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    if world_size > 1:
        dist.barrier()

    # target 指向无 DDP 外壳的真实模块，读取诊断量和保存 state_dict 时使用它。
    log_path = args.output / "train.jsonl"
    wrapped.train()
    for step in range(1, args.steps + 1):
        # 一个 step 的定义：20 个固定 mesh 全部完成一次前向/反向，然后只执行
        # 一次 optimizer.step()。pack 数或 GPU 数变化不会改变这个定义。
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        optimizer.param_groups[0]["lr"] = learning_rate_at_step(
            args.learning_rate, step, args.warmup_steps
        )
        optimizer.zero_grad(set_to_none=True)
        local_loss_sum = torch.zeros((), device=device)
        local_component_sums: dict[str, float] = {}
        for batch_index, batch in enumerate(batches):
            is_last_local_batch = batch_index == len(batches) - 1
            # 非最后一个本地 pack 先积累梯度但不做跨卡通信；最后一个 pack 才
            # 触发一次 DDP all-reduce。数学等价，减少重复通信。
            sync_context = (
                contextlib.nullcontext()
                if world_size == 1 or is_last_local_batch
                else wrapped.no_sync()
            )
            # 除 selective profile 的 external FlashAttention I/O 外，网络、VAE、
            # projection、Spacetime 和 loss 均为 FP32。DDP 只控制通信时机。
            with sync_context:
                # 每个 UID 独立确定 epsilon；改变 pack/rank 仍得到相同采样。
                sample_seeds = tuple(
                    stable_uid_seed(args.seed, step, uid) for uid in batch.uids
                )
                batch_loss = wrapped(
                    batch,
                    negative_seed=args.seed,
                    fixed_negatives=True,
                    packed=True,
                    sample_seeds=sample_seeds,
                )
                if batch_loss.dtype != torch.float32:
                    raise TypeError("Topology AE loss must remain FP32")
                sample_count = len(batch.uids)
                # DDP averages rank gradients.  This scale restores the global
                # sum divided by exactly 20 meshes, even when ranks hold unequal
                # mesh counts or different numbers of packed groups.
                backward_loss = (
                    batch_loss * sample_count * world_size / args.expected_samples
                )
            backward_loss.backward()
            # 保存未缩放的逐对象统计，供 rank0 日志汇总；这不参与梯度。
            local_loss_sum += batch_loss.detach() * sample_count
            for name, value in target.last_loss_components.items():
                local_component_sums[name] = (
                    local_component_sums.get(name, 0.0) + value * sample_count
                )
        non_fp32_gradients = [
            name
            for name, parameter in wrapped.named_parameters()
            if parameter.grad is not None and parameter.grad.dtype != torch.float32
        ]
        if non_fp32_gradients:
            raise TypeError(
                "Topology AE gradients must be FP32: "
                + ", ".join(non_fp32_gradients[:5])
            )
        # 此返回值是裁剪前的全局 L2 梯度范数。大于 1 不代表错误；参数更新实际
        # 使用的是缩放到不超过 1 的梯度。NaN/Inf 才立即终止实验。
        gradient_norm = torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("gradient norm is not finite")
        # 到这里才真正更新一次参数，所以日志中的 step 与 optimizer step 一一对应。
        optimizer.step()
        if step == 1:
            non_fp32_optimizer_states = []
            for state in optimizer.state.values():
                for name, value in state.items():
                    if (
                        isinstance(value, torch.Tensor)
                        and value.is_floating_point()
                        and value.dtype != torch.float32
                    ):
                        non_fp32_optimizer_states.append(f"{name}:{value.dtype}")
            if non_fp32_optimizer_states:
                raise TypeError(
                    "Adam floating-point state must be FP32: "
                    + ", ".join(non_fp32_optimizer_states[:5])
                )
        torch.cuda.synchronize(device)

        if world_size > 1:
            loss_value, components = distributed_mean_sums(
                local_loss_sum, local_component_sums, args.expected_samples
            )
        else:
            loss_value = float(local_loss_sum / args.expected_samples)
            components = {
                name: value / args.expected_samples
                for name, value in local_component_sums.items()
            }
        elapsed = time.perf_counter() - started
        peak_allocated = torch.cuda.max_memory_allocated(device) / 2**30
        peak_reserved = torch.cuda.max_memory_reserved(device) / 2**30
        if rank == 0:
            # train.jsonl 每行是一个完整 JSON 记录，便于中途 tail、断点画曲线和审计。
            record = {
                "step": step,
                "loss": loss_value,
                "components": components,
                "gradient_norm": float(gradient_norm),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "seconds": elapsed,
                "peak_allocated_gib_rank0": peak_allocated,
                "peak_reserved_gib_rank0": peak_reserved,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if step == 1 or step % 10 == 0:
                print(json.dumps(record, ensure_ascii=False), flush=True)
            if not args.no_checkpoint and (
                step % args.save_every == 0 or step == args.steps
            ):
                save_checkpoint(
                    args.output / f"checkpoint-{step:07d}.pt",
                    target,
                    optimizer,
                    args,
                    dataset.manifest_sha256,
                    step,
                    scoring_profile,
                    requested_args,
                    calibration,
                )
    if world_size > 1:
        # 正常退出时释放 NCCL 进程组；训练异常则由 torchrun 负责终止其他 rank。
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
