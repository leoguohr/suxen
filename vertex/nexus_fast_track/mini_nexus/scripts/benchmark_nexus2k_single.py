#!/usr/bin/env python3
"""Measure one-A100 time and memory before launching long Nexus2K training."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mini_nexus.data_2k import Nexus2KManifestDataset, collate_nexus2k_samples  # noqa: E402
from mini_nexus.topology import TopologyAutoencoder  # noqa: E402
from mini_nexus.training import TopologyFlowSystem, VertexStageSystem  # noqa: E402
from mini_nexus.training_2k import Nexus2KTopologyAESystem  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resource benchmark; model defaults are scaled.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("vertex", "topology-ae", "topology-flow"),
        default=("vertex", "topology-ae", "topology-flow"),
    )
    parser.add_argument(
        "--quantiles", nargs="+", type=float, default=(0.0, 0.5, 0.9, 1.0)
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--condition-dim", type=int, help="Vertex VecSet width; default: hidden-dim")
    parser.add_argument("--condition-heads", type=int, help="Vertex VecSet heads; default: num-heads")
    parser.add_argument("--condition-layers", type=int, default=8)
    parser.add_argument("--condition-tokens", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--spacetime-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--activation-checkpointing", action="store_true",
        help="Checkpoint Vertex VecSet/DiT blocks to reduce activation memory",
    )
    args = parser.parse_args()
    if args.condition_dim is None:
        args.condition_dim = args.hidden_dim
    if args.condition_heads is None:
        args.condition_heads = args.num_heads
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scan_sizes(dataset: Nexus2KManifestDataset) -> list[dict[str, int | str]]:
    records = []
    for index, row in enumerate(dataset.rows):
        path = Path(row["mesh_quantized_training"])
        with np.load(path, allow_pickle=False) as archive:
            vertices = archive["vertices_norm"]
            faces = archive["faces"]
            records.append(
                {
                    "index": index,
                    "uid": row["uid"],
                    "vertices": int(vertices.shape[0]),
                    "faces": int(faces.shape[0]),
                }
            )
    return records


def choose_quantiles(
    records: list[dict[str, int | str]], quantiles: list[float]
) -> list[dict[str, int | str | float]]:
    if any(value < 0.0 or value > 1.0 for value in quantiles):
        raise ValueError("quantiles must be within [0,1]")
    ordered = sorted(records, key=lambda item: (int(item["vertices"]), str(item["uid"])))
    selected = []
    seen: set[int] = set()
    for quantile in quantiles:
        position = int(round(quantile * (len(ordered) - 1)))
        index = int(ordered[position]["index"])
        if index in seen:
            continue
        seen.add(index)
        selected.append({**ordered[position], "quantile": quantile})
    return selected


def build_model(args: argparse.Namespace, stage: str) -> torch.nn.Module:
    topology_layers = max(1, args.num_layers // 2)
    if stage == "vertex":
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
    if stage == "topology-ae":
        return Nexus2KTopologyAESystem(
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            spacetime_dim=args.spacetime_dim,
            num_heads=args.num_heads,
            num_layers=topology_layers,
            face_interval_factor=0.25,
        )
    autoencoder = TopologyAutoencoder(
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        spacetime_dim=args.spacetime_dim,
        num_heads=args.num_heads,
        num_layers=topology_layers,
    )
    return TopologyFlowSystem(
        autoencoder,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        condition_tokens=args.condition_tokens,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )


def one_step(
    args: argparse.Namespace,
    stage: str,
    cpu_batch: object,
    sample_seed: int,
) -> dict[str, object]:
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    seed_everything(sample_seed)
    batch = cpu_batch.to(device)
    if stage == "topology-ae" and batch.vertices.dtype != torch.float32:
        raise RuntimeError(
            f"topology-ae vertex inputs are not FP32: {batch.vertices.dtype}"
        )
    model = build_model(args, stage).to(device)
    if stage == "topology-ae":
        model = model.to(dtype=torch.float32)
        if any(
            parameter.is_floating_point() and parameter.dtype != torch.float32
            for parameter in model.parameters()
        ):
            raise RuntimeError("topology-ae parameters are not FP32")
    if stage == "topology-ae":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    autocast_enabled = args.precision == "bf16" and stage != "topology-ae"
    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled
    ):
        if stage == "vertex":
            loss = model(batch.condition, batch.octree_levels[-1])
        elif stage == "topology-ae":
            loss = model(batch, negative_seed=sample_seed)
        else:
            loss = model(batch)
    if stage == "topology-ae" and loss.dtype != torch.float32:
        raise RuntimeError(f"topology-ae loss is not FP32: {loss.dtype}")
    loss.backward()
    if stage == "topology-ae" and any(
        parameter.grad is not None and parameter.grad.dtype != torch.float32
        for parameter in model.parameters()
    ):
        raise RuntimeError("topology-ae gradients are not FP32")
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    gradients_finite = all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    optimizer.step()
    if stage == "topology-ae" and any(
        torch.is_tensor(value)
        and value.is_floating_point()
        and value.dtype != torch.float32
        for state in optimizer.state.values()
        for value in state.values()
    ):
        raise RuntimeError("topology-ae optimizer states are not FP32")
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result = {
        "status": "passed",
        "loss": float(loss.detach()),
        "gradient_norm": float(gradient_norm),
        "gradients_finite": bool(gradients_finite),
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "parameter_count": parameter_count,
        "effective_precision": (
            "fp32" if stage == "topology-ae" else args.precision
        ),
    }
    if stage == "topology-ae":
        result.update(
            {
                "scoring_contract": model.scoring_contract(),
                "logit_scale_calibration": (
                    "not_run_in_resource_benchmark; explicit unit scales"
                ),
            }
        )
    return result


def write_report(path: Path, report: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if "topology-ae" in args.stages:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if torch.get_float32_matmul_precision() != "highest":
            raise RuntimeError("failed to require highest FP32 matmul precision")
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            raise RuntimeError("failed to disable TF32 for topology-ae")
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    if args.hidden_dim % args.num_heads:
        raise ValueError("hidden-dim must be divisible by num-heads")

    args.output.mkdir(parents=True, exist_ok=True)
    dataset = Nexus2KManifestDataset(args.manifest, "train")
    size_records = scan_sizes(dataset)
    selected = choose_quantiles(size_records, list(args.quantiles))
    report: dict[str, object] = {
        "status": "running",
        "claim_boundary": "single_step_resource_benchmark_not_model_quality",
        "independent_reimplementation": True,
        "paper_deviations": [
            "configurable_model_dimensions_with_scaled_defaults",
            "point_cloud_condition_only",
            *(
                ["vertex_architecture_details_independently_reconstructed",
                 "uniform_time_linear_rectified_flow_training"]
                if "vertex" in args.stages else []
            ),
        ],
        "benchmark_limitations": [
            "topology_ae_uses_explicit_unit_logit_scales_without_fixed20_calibration"
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "manifest_sha256": dataset.manifest_sha256,
        "train_count": len(dataset),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "precision": args.precision,
        "topology_ae_effective_precision": (
            "fp32" if "topology-ae" in args.stages else None
        ),
        "model_config": {
            "hidden_dim": args.hidden_dim,
            "vertex_architecture": "point_cloud_vertex_v1",
            "condition_dim": args.condition_dim,
            "condition_heads": args.condition_heads,
            "condition_layers": args.condition_layers,
            "activation_checkpointing": args.activation_checkpointing,
            "condition_tokens": args.condition_tokens,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "latent_dim": args.latent_dim,
            "spacetime_dim": args.spacetime_dim,
        },
        "selected_samples": selected,
        "results": [],
    }
    report_path = args.output / "benchmark_report.json"
    (args.output / "size_index.json").write_text(
        json.dumps(size_records, indent=2) + "\n", encoding="utf-8"
    )
    write_report(report_path, report)

    results: list[dict[str, object]] = report["results"]  # type: ignore[assignment]
    for sample_order, selected_sample in enumerate(selected):
        sample = dataset[int(selected_sample["index"])]
        cpu_batch = collate_nexus2k_samples([sample])
        for stage_order, stage in enumerate(args.stages):
            record = {
                "uid": sample.uid,
                "quantile": selected_sample["quantile"],
                "vertices": len(sample.vertices),
                "faces": len(sample.faces),
                "depth9_parents": len(sample.octree_levels[-1].parent_codes),
                "stage": stage,
            }
            try:
                record.update(
                    one_step(
                        args,
                        stage,
                        cpu_batch,
                        args.seed + sample_order * 100 + stage_order,
                    )
                )
            except torch.OutOfMemoryError as error:
                record.update({"status": "oom", "error": str(error)})
                torch.cuda.empty_cache()
            results.append(record)
            write_report(report_path, report)
            print(json.dumps(record, ensure_ascii=False), flush=True)

    report["status"] = (
        "passed" if all(item["status"] == "passed" for item in results) else "completed_with_oom"
    )
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_report(report_path, report)
    print(f"report={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
