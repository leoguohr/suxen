import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from mini_nexus.data_2k import Nexus2KManifestDataset, collate_nexus2k_samples
from mini_nexus.octree import build_octree_levels, decode_leaf_centers


FIELDS = [
    "uid",
    "split",
    "condition_point",
    "mesh_quantized_training",
    "octree",
    "topology",
    "decision",
]


def _write_sample(root: Path, uid: str, split: str, cells: torch.Tensor) -> dict[str, str]:
    stage2 = root / "stage2_outputs" / uid
    stage3 = root / "stage3_outputs" / uid
    stage4 = root / "stage4_outputs" / uid
    stage2.mkdir(parents=True)
    stage3.mkdir(parents=True)
    stage4.mkdir(parents=True)

    vertices = decode_leaf_centers(cells, 9).numpy()
    faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)
    points = np.resize(vertices, (8192, 3)).astype(np.float32)
    normals = np.zeros((8192, 3), dtype=np.float32)
    normals[:, 2] = 1.0
    condition_path = stage2 / "condition_point.npz"
    np.savez(condition_path, points=points, normals=normals, sampled_face=np.zeros(8192))

    mesh_path = stage3 / "mesh_quantized_training.npz"
    np.savez(
        mesh_path,
        vertices_norm=vertices,
        faces=faces,
        quantized_vertices=cells.numpy().astype(np.int32),
    )

    levels = build_octree_levels(cells, 9)
    octree_payload: dict[str, np.ndarray] = {
        "quantized_vertices": torch.cat([cells, cells[:1]], dim=0).numpy(),
        "unique_leaves": cells.numpy(),
        "recovered_leaves": cells.numpy(),
        "cell_centers": vertices,
    }
    for level in levels:
        octree_payload[f"depth_{level.depth}_parent_xyz"] = level.parent_codes.numpy()
        octree_payload[f"depth_{level.depth}_target"] = level.target.numpy().astype(np.uint8)
    octree_path = stage3 / "octree_d9.npz"
    np.savez(octree_path, **octree_payload)

    pairs = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]], axis=0
    )
    edge_index = np.unique(np.sort(pairs, axis=1), axis=0).T
    face_set = np.unique(np.sort(faces, axis=1), axis=0)
    face_nodes = len(vertices) + np.arange(len(faces), dtype=np.int64)
    source = faces.reshape(-1)
    destination = np.repeat(face_nodes, 3)
    incidence = np.stack(
        [np.concatenate([source, destination]), np.concatenate([destination, source])]
    )
    topology_path = stage4 / "topology.npz"
    np.savez(
        topology_path,
        edge_index=edge_index,
        face_set=face_set,
        faces_oriented=faces,
        face_centroid=vertices[faces].mean(axis=1),
        incidence_index=incidence,
        face_node_offset=np.array(len(vertices), dtype=np.int64),
    )
    return {
        "uid": uid,
        "split": split,
        "condition_point": str(condition_path),
        "mesh_quantized_training": str(mesh_path),
        "octree": str(octree_path),
        "topology": str(topology_path),
        "decision": "keep",
    }


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_final_manifest_loads_train_val_and_variable_batch(tmp_path: Path):
    train = _write_sample(
        tmp_path,
        "nexus_2k_train",
        "train",
        torch.tensor([[1, 193, 65], [1, 193, 129], [65, 193, 65], [193, 1, 65]]),
    )
    val = _write_sample(
        tmp_path,
        "nexus_2k_val",
        "val",
        torch.tensor(
            [[11, 21, 31], [41, 51, 61], [71, 81, 91], [101, 111, 121], [131, 141, 151]]
        ),
    )
    manifest = tmp_path / "training_manifest_manual_dedup.csv"
    _write_manifest(manifest, [train, val])

    train_dataset = Nexus2KManifestDataset(manifest, "train")
    val_dataset = Nexus2KManifestDataset(manifest, "val")
    train_sample = train_dataset[train_dataset.index_for_uid("nexus_2k_train")]
    val_sample = val_dataset[val_dataset.index_for_uid("nexus_2k_val")]
    batch = collate_nexus2k_samples([train_sample, val_sample])

    assert train_dataset.manifest_sha256 == val_dataset.manifest_sha256
    assert batch.uids == ("nexus_2k_train", "nexus_2k_val")
    assert batch.splits == ("train", "val")
    assert batch.condition.shape == (2, 8192, 6)
    assert batch.vertices.shape == (2, 5, 3)
    assert batch.vertex_mask.sum(dim=1).tolist() == [4, 5]
    assert len(batch.faces[0]) == len(batch.face_set[0]) == 2
    assert len(batch.octree_levels) == 9
    assert batch.octree_levels[-1].target.shape[-1] == 8


def test_loader_rejects_topology_that_does_not_match_training_mesh(tmp_path: Path):
    row = _write_sample(
        tmp_path,
        "nexus_2k_bad",
        "train",
        torch.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]]),
    )
    topology_path = Path(row["topology"])
    with np.load(topology_path, allow_pickle=False) as archive:
        payload = {key: np.array(archive[key], copy=True) for key in archive.files}
    payload["faces_oriented"][0] = np.array([0, 2, 3])
    np.savez(topology_path, **payload)
    manifest = tmp_path / "training_manifest_manual_dedup.csv"
    _write_manifest(manifest, [row])

    dataset = Nexus2KManifestDataset(manifest, "train")
    with pytest.raises(ValueError, match="faces_oriented"):
        dataset[0]


def test_loader_rejects_cross_uid_manifest_path(tmp_path: Path):
    first = _write_sample(
        tmp_path,
        "nexus_2k_first",
        "train",
        torch.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]]),
    )
    second = _write_sample(
        tmp_path,
        "nexus_2k_second",
        "val",
        torch.tensor([[11, 21, 31], [41, 51, 61], [71, 81, 91], [101, 111, 121]]),
    )
    first["topology"] = second["topology"]
    manifest = tmp_path / "training_manifest_manual_dedup.csv"
    _write_manifest(manifest, [first, second])

    dataset = Nexus2KManifestDataset(manifest, "train")
    with pytest.raises(ValueError, match="same UID"):
        dataset[0]
