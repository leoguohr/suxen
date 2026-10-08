#!/usr/bin/env python3
"""按 Nexus 的 edges-first 顺序评估 Topology AE checkpoint。

这里评估的是“已知 GT 顶点与 GT encoder topology 时，AE 能否重建连接关系”，
不是从点云端到端生成 mesh。正式恢复顺序是：

    所有顶点对的一阶 interval > 0
      -> predicted edge graph
      -> 只枚举该图中的 3-cycles
      -> 二阶 interval > 0
      -> predicted faces
      -> orientation correction

因此 face 绝不会在全部 V choose 3 上独立阈值；它必须先通过 edge graph。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mini_nexus.data_2k import Nexus2KManifestDataset  # noqa: E402
from mini_nexus.packed_topology import collate_packed_topology  # noqa: E402
from mini_nexus.topology import (  # noqa: E402
    mesh_edges,
    orient_faces_consistently,
    recover_topology,
)
from mini_nexus.topology_evaluation import (  # noqa: E402
    aggregate_set_metrics,
    set_metrics,
)
from mini_nexus.topology_checkpoint import (  # noqa: E402
    load_topology_checkpoint,
    load_topology_system_from_checkpoint,
    topology_runtime_resolution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-samples", type=int, default=16)
    parser.add_argument("--allow-manifest-mismatch", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def evaluate_checkpoint(
    path: Path,
    dataset: Nexus2KManifestDataset,
    indices: list[int],
    *,
    allow_manifest_mismatch: bool = False,
) -> dict[str, object]:
    checkpoint = load_topology_checkpoint(path)
    checkpoint_manifest = checkpoint.get("manifest_sha256")
    if (
        checkpoint_manifest is not None
        and checkpoint_manifest != dataset.manifest_sha256
        and not allow_manifest_mismatch
    ):
        raise ValueError(
            "evaluation manifest does not match checkpoint; pass "
            "--allow-manifest-mismatch only for an intentional cross-manifest audit"
        )
    # eval() 关闭 dropout，并让 VAE 使用 mu 而不是重新采样 epsilon。
    model = load_topology_system_from_checkpoint(checkpoint, device="cuda")

    edge_metric_rows = []
    face_metric_rows = []
    samples = []
    for index in indices:
        # 这里把 GT faces 交给 encoder；这是 Topology AE 重建实验的定义，
        # 不能误报为 Topology Flow 或完整 condition-to-mesh 生成结果。
        sample = dataset[index]
        vertices = sample.vertices.cuda()
        truth_face_tensor = sample.faces.cuda()
        packed = collate_packed_topology([sample]).to("cuda")
        _, _, edge_embedding_rows, face_embedding_rows = model.topology_embedding_rows(
            packed,
            sample_seeds=None,
        )
        edge_embedding = edge_embedding_rows[0]
        face_embedding = face_embedding_rows[0]

        # Paper inference: all pairs -> edge graph -> 3-cycles -> face test.
        predicted_edge_tensor, predicted_face_tensor = recover_topology(
            edge_embedding,
            face_embedding,
            edge_logit_scale=model.edge_logit_scale,
            face_logit_scale=model.face_logit_scale,
            face_interval_factor=model.face_interval_factor,
        )
        # interval 只决定无向 face membership；最终 winding 由确定性后处理修正。
        oriented_faces = orient_faces_consistently(vertices, predicted_face_tensor)

        predicted_edges = {tuple(row) for row in predicted_edge_tensor.cpu().tolist()}
        truth_edges = {
            tuple(row) for row in mesh_edges(truth_face_tensor).cpu().tolist()
        }
        predicted_faces = {
            tuple(row)
            for row in torch.sort(oriented_faces, dim=1).values.cpu().tolist()
        }
        truth_faces = {
            tuple(row)
            for row in torch.sort(truth_face_tensor, dim=1).values.cpu().tolist()
        }

        edge_result = set_metrics(predicted_edges, truth_edges)
        face_result = set_metrics(predicted_faces, truth_faces)
        edge_metric_rows.append(edge_result)
        face_metric_rows.append(face_result)
        samples.append(
            {
                "uid": sample.uid,
                "vertices": len(vertices),
                "faces": len(truth_face_tensor),
                "edge": edge_result,
                "face": face_result,
            }
        )

    return {
        "checkpoint": str(path),
        "training_step": int(checkpoint["step"]),
        "checkpoint_manifest_sha256": checkpoint.get("manifest_sha256"),
        "evaluation_manifest_sha256": dataset.manifest_sha256,
        "manifest_matches_checkpoint": (
            checkpoint.get("manifest_sha256") == dataset.manifest_sha256
        ),
        "scoring_profile": checkpoint.get("scoring_profile", "legacy_unspecified"),
        "scoring_profile_status": "informational; scoring_contract validated separately",
        "scoring_contract": model.scoring_contract(),
        "checkpoint_runtime": checkpoint.get("runtime", "legacy_unspecified"),
        "evaluation_runtime": model.runtime_contract(),
        "runtime_resolution": topology_runtime_resolution(checkpoint),
        "metric_aggregation": "micro_counts_across_meshes",
        "threshold_comparator": ">",
        "edge": aggregate_set_metrics(edge_metric_rows),
        "face": aggregate_set_metrics(face_metric_rows),
        "samples": samples,
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("evaluation requires CUDA")

    dataset = Nexus2KManifestDataset(args.manifest, "val")
    count = min(args.validation_samples, len(dataset))
    # 用确定性的均匀位置选择，保证不同 checkpoint 比较的是同一批 validation UID。
    indices = np.linspace(0, len(dataset) - 1, num=count, dtype=np.int64).tolist()
    checkpoints = sorted(args.checkpoint_dir.glob("checkpoint-0*.pt"))
    if not checkpoints:
        raise ValueError("no numbered checkpoint found")

    report = {
        "claim_boundary": "Topology AE recovery, not end-to-end Nexus generation",
        "recovery": "all pairs -> edge graph -> 3-cycles -> faces -> orientation",
        "thresholds": {"edge": 0.0, "face": 0.0},
        "threshold_comparator": ">",
        "manifest": str(args.manifest),
        "evaluation_manifest_sha256": dataset.manifest_sha256,
        "validation_count": count,
        "results": [
            evaluate_checkpoint(
                path,
                dataset,
                indices,
                allow_manifest_mismatch=args.allow_manifest_mismatch,
            )
            for path in checkpoints
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
