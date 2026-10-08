"""只读加载最终人工几何去重后的 Nexus2K manifest。

初学者先把本文件理解成“数据守门员”，而不是神经网络：

* manifest 的一行就是一个 UID；四条路径必须都属于这个 UID；
* ``condition_point`` 保存 8192 个点及法向；
* ``mesh_quantized_training`` 保存 Vertex/Topology 两阶段共同使用的顶点与面；
* ``octree`` 保存 D=9 八叉树标签；
* ``topology`` 保存由同一组 faces 精确推导的 edge/face/incidence 标签。

当前 Topology AE overfit20 的前向传播只消费 ``vertices/faces/edge_index/face_set``。
condition 与 octree 仍被加载和校验，是为了确保同一 batch 没有 UID 错配；它们会
在后续 Vertex Diffusion 和条件生成训练中使用。loader 从不写回或修复 NPZ。
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .data import VertexLevelBatch
from .octree import OctreeLevel, expand_occupied_children, metadata_for_parents


REQUIRED_PATH_FIELDS = (
    "condition_point",
    "mesh_quantized_training",
    "octree",
    "topology",
)


@dataclass(frozen=True)
class Nexus2KSample:
    """一个对象的未 padding 数据；每个变长张量仍保持自己的真实长度。"""

    uid: str
    split: str
    condition: Tensor
    vertices: Tensor
    faces: Tensor
    quantized_vertices: Tensor
    octree_levels: tuple[OctreeLevel, ...]
    edge_index: Tensor
    face_set: Tensor
    faces_oriented: Tensor
    face_centroid: Tensor
    incidence_index: Tensor
    face_node_offset: int
    paths: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Nexus2KBatch:
    """多个对象的 batch。

    只有 ``vertices`` 需要 padding 成 ``[B,V_max,3]``，并由 ``vertex_mask``
    标出真实顶点。faces/edges 等 topology 张量长度差异很大，所以保留 tuple，
    后续训练系统会逐对象计算 loss。
    """

    uids: tuple[str, ...]
    splits: tuple[str, ...]
    condition: Tensor
    vertices: Tensor
    vertex_mask: Tensor
    quantized_vertices: tuple[Tensor, ...]
    faces: tuple[Tensor, ...]
    octree_levels: tuple[VertexLevelBatch, ...]
    edge_index: tuple[Tensor, ...]
    face_set: tuple[Tensor, ...]
    faces_oriented: tuple[Tensor, ...]
    face_centroid: tuple[Tensor, ...]
    incidence_index: tuple[Tensor, ...]
    face_node_offset: tuple[int, ...]
    paths: tuple[tuple[tuple[str, str], ...], ...]

    def to(self, device: torch.device | str) -> "Nexus2KBatch":
        return Nexus2KBatch(
            uids=self.uids,
            splits=self.splits,
            condition=self.condition.to(device),
            vertices=self.vertices.to(device),
            vertex_mask=self.vertex_mask.to(device),
            quantized_vertices=tuple(value.to(device) for value in self.quantized_vertices),
            faces=tuple(value.to(device) for value in self.faces),
            octree_levels=tuple(level.to(device) for level in self.octree_levels),
            edge_index=tuple(value.to(device) for value in self.edge_index),
            face_set=tuple(value.to(device) for value in self.face_set),
            faces_oriented=tuple(value.to(device) for value in self.faces_oriented),
            face_centroid=tuple(value.to(device) for value in self.face_centroid),
            incidence_index=tuple(value.to(device) for value in self.incidence_index),
            face_node_offset=self.face_node_offset,
            paths=self.paths,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _owned_array(archive: np.lib.npyio.NpzFile, key: str, dtype: np.dtype) -> np.ndarray:
    if key not in archive.files:
        raise ValueError(f"NPZ is missing required field: {key}")
    return np.array(archive[key], dtype=dtype, copy=True)


def _expected_edges(faces: np.ndarray) -> np.ndarray:
    pairs = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]], axis=0
    )
    return np.unique(np.sort(pairs, axis=1), axis=0).T


def _expected_incidence(faces: np.ndarray, vertex_count: int) -> np.ndarray:
    face_nodes = vertex_count + np.arange(len(faces), dtype=np.int64)
    vertices = faces.reshape(-1)
    nodes = np.repeat(face_nodes, 3)
    return np.stack(
        [np.concatenate([vertices, nodes]), np.concatenate([nodes, vertices])],
        axis=0,
    )


def _validate_manifest_path(uid: str, field: str, raw_path: str) -> Path:
    path = Path(raw_path)
    expected = {
        "condition_point": ("stage2_outputs", "condition_point.npz"),
        "mesh_quantized_training": ("stage3_outputs", "mesh_quantized_training.npz"),
        "octree": ("stage3_outputs", "octree_d9.npz"),
        "topology": ("stage4_outputs", "topology.npz"),
    }[field]
    stage, filename = expected
    if path.name != filename or path.parent.name != uid or stage not in path.parts:
        raise ValueError(f"{uid}: manifest field {field} does not identify the same UID/stage")
    if not path.is_file():
        raise ValueError(f"{uid}: manifest file does not exist: {path}")
    return path


def _load_octree(
    uid: str, path: Path, quantized_vertices: np.ndarray, vertices: np.ndarray
) -> tuple[OctreeLevel, ...]:
    with np.load(path, allow_pickle=False) as archive:
        unique_leaves = _owned_array(archive, "unique_leaves", np.int64)
        recovered_leaves = _owned_array(archive, "recovered_leaves", np.int64)
        cell_centers = _owned_array(archive, "cell_centers", np.float32)
        raw_quantized = _owned_array(archive, "quantized_vertices", np.int64)
        if not np.array_equal(unique_leaves, quantized_vertices):
            raise ValueError(f"{uid}: octree unique_leaves disagree with Stage-3 vertices")
        if not np.array_equal(recovered_leaves, quantized_vertices):
            raise ValueError(f"{uid}: decoded octree leaves disagree with Stage-3 vertices")
        if not np.array_equal(cell_centers, vertices):
            raise ValueError(f"{uid}: decoded octree centers disagree with vertices_norm")
        raw_set = {tuple(value) for value in raw_quantized.tolist()}
        if any(tuple(value) not in raw_set for value in quantized_vertices.tolist()):
            raise ValueError(f"{uid}: Stage-3 cell is absent from the pre-clean octree record")

        current_parents = np.zeros((1, 3), dtype=np.int64)
        levels: list[OctreeLevel] = []
        for depth in range(1, 10):
            parents = _owned_array(archive, f"depth_{depth}_parent_xyz", np.int64)
            target = _owned_array(archive, f"depth_{depth}_target", np.float32)
            if parents.ndim != 2 or parents.shape[1] != 3:
                raise ValueError(f"{uid}: depth-{depth} parents must have shape [N,3]")
            if target.shape != (len(parents), 8):
                raise ValueError(f"{uid}: depth-{depth} target must have shape [N,8]")
            if not np.array_equal(parents, current_parents):
                raise ValueError(f"{uid}: depth-{depth} parents do not follow prior occupancy")
            if not np.isin(target, (0.0, 1.0)).all() or (target.sum(axis=1) == 0).any():
                raise ValueError(f"{uid}: depth-{depth} occupancy is not nonempty multi-hot")
            parent_tensor = torch.from_numpy(parents)
            target_tensor = torch.from_numpy(target)
            levels.append(
                OctreeLevel(
                    depth=depth,
                    parent_codes=parent_tensor,
                    target=target_tensor,
                    metadata=metadata_for_parents(parent_tensor, depth, 9),
                )
            )
            current_parents = (
                np.unique(
                    expand_occupied_children(parent_tensor, target_tensor.bool())
                    .cpu()
                    .numpy(),
                    axis=0,
                )
            )
        if not np.array_equal(current_parents, quantized_vertices):
            raise ValueError(f"{uid}: occupancy traversal does not recover Stage-3 leaves")
    return tuple(levels)


def load_nexus2k_sample(row: dict[str, str]) -> Nexus2KSample:
    """加载并交叉审计 manifest 的一行，成功后才构造训练样本。"""

    # 第一关：行级身份检查。decision=keep 才属于冻结训练集。
    uid = row.get("uid", "")
    split = row.get("split", "")
    if not uid:
        raise ValueError("manifest row is missing UID")
    if split not in {"train", "val"}:
        raise ValueError(f"{uid}: split must be train or val")
    if row.get("decision") != "keep":
        raise ValueError(f"{uid}: final training manifest row must have decision=keep")
    paths = {
        field: _validate_manifest_path(uid, field, row.get(field, ""))
        for field in REQUIRED_PATH_FIELDS
    }

    # 第二关：读取条件点云。拼接后每个点是 [x,y,z,nx,ny,nz] 六维。
    with np.load(paths["condition_point"], allow_pickle=False) as archive:
        points = _owned_array(archive, "points", np.float32)
        normals = _owned_array(archive, "normals", np.float32)
    if points.shape != (8192, 3) or normals.shape != (8192, 3):
        raise ValueError(f"{uid}: points and normals must both have shape [8192,3]")
    if not np.isfinite(points).all() or not np.isfinite(normals).all():
        raise ValueError(f"{uid}: condition contains NaN or Inf")

    # 第三关：读取 Stage-3 共同 mesh 真值。这里绝不重新量化或重新焊接顶点。
    with np.load(paths["mesh_quantized_training"], allow_pickle=False) as archive:
        vertices = _owned_array(archive, "vertices_norm", np.float32)
        faces = _owned_array(archive, "faces", np.int64)
        quantized_vertices = _owned_array(archive, "quantized_vertices", np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"{uid}: vertices_norm must have shape [V,3]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"{uid}: faces must have shape [F,3]")
    if quantized_vertices.shape != vertices.shape:
        raise ValueError(f"{uid}: quantized_vertices must have shape [V,3]")
    if not np.isfinite(vertices).all():
        raise ValueError(f"{uid}: vertices contain NaN or Inf")
    if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError(f"{uid}: face index is outside [0,V)")
    if len(np.unique(quantized_vertices, axis=0)) != len(quantized_vertices):
        raise ValueError(f"{uid}: Stage-3 mesh still has a depth-9 collision")

    # 第四关：完整走一遍八叉树 occupancy，确认叶子精确恢复同一组顶点。
    octree_levels = _load_octree(
        uid, paths["octree"], quantized_vertices, vertices
    )

    # 第五关：读取 Stage-4 topology，并从 faces 重新推导后逐项比对。
    with np.load(paths["topology"], allow_pickle=False) as archive:
        edge_index = _owned_array(archive, "edge_index", np.int64)
        face_set = _owned_array(archive, "face_set", np.int64)
        faces_oriented = _owned_array(archive, "faces_oriented", np.int64)
        face_centroid = _owned_array(archive, "face_centroid", np.float32)
        incidence_index = _owned_array(archive, "incidence_index", np.int64)
        if "face_node_offset" not in archive.files:
            raise ValueError(f"{uid}: topology is missing face_node_offset")
        face_node_offset = int(archive["face_node_offset"])
    if not np.array_equal(faces_oriented, faces):
        raise ValueError(f"{uid}: faces_oriented disagree with Stage-3 faces")
    expected_face_set = np.unique(np.sort(faces, axis=1), axis=0)
    if not np.array_equal(face_set, expected_face_set):
        raise ValueError(f"{uid}: face_set cannot be derived exactly from faces")
    if not np.array_equal(edge_index, _expected_edges(faces)):
        raise ValueError(f"{uid}: edge_index cannot be derived exactly from faces")
    if face_node_offset != len(vertices):
        raise ValueError(f"{uid}: face_node_offset must equal V")
    if not np.array_equal(incidence_index, _expected_incidence(faces, len(vertices))):
        raise ValueError(f"{uid}: incidence_index disagrees with faces")
    expected_centroid = vertices[faces].mean(axis=1)
    if not np.allclose(face_centroid, expected_centroid, atol=1e-7, rtol=0.0):
        raise ValueError(f"{uid}: face_centroid disagrees with Stage-3 geometry")

    return Nexus2KSample(
        uid=uid,
        split=split,
        condition=torch.from_numpy(np.concatenate([points, normals], axis=1)),
        vertices=torch.from_numpy(vertices),
        faces=torch.from_numpy(faces),
        quantized_vertices=torch.from_numpy(quantized_vertices),
        octree_levels=octree_levels,
        edge_index=torch.from_numpy(edge_index),
        face_set=torch.from_numpy(face_set),
        faces_oriented=torch.from_numpy(faces_oriented),
        face_centroid=torch.from_numpy(face_centroid),
        incidence_index=torch.from_numpy(incidence_index),
        face_node_offset=face_node_offset,
        paths=tuple((field, str(paths[field])) for field in REQUIRED_PATH_FIELDS),
    )


class Nexus2KManifestDataset(Dataset[Nexus2KSample]):
    """Lazy, read-only dataset over one frozen train or val manifest split."""

    def __init__(self, manifest_path: Path, split: str):
        if split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_file():
            raise ValueError(f"manifest does not exist: {self.manifest_path}")
        self.manifest_sha256 = _sha256(self.manifest_path)
        with self.manifest_path.open(newline="", encoding="utf-8") as handle:
            all_rows = list(csv.DictReader(handle))
        if not all_rows:
            raise ValueError("manifest is empty")
        uids = [row.get("uid", "") for row in all_rows]
        if any(not uid for uid in uids) or len(uids) != len(set(uids)):
            raise ValueError("manifest UIDs must be present and globally unique")
        invalid_splits = sorted({row.get("split", "") for row in all_rows} - {"train", "val"})
        if invalid_splits:
            raise ValueError(f"manifest contains unsupported splits: {invalid_splits}")
        if any(row.get("decision") != "keep" for row in all_rows):
            raise ValueError("final manifest may only contain decision=keep rows")
        self.rows = tuple(row for row in all_rows if row["split"] == split)
        if not self.rows:
            raise ValueError(f"manifest has no rows for split={split}")
        self.uids = tuple(row["uid"] for row in self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Nexus2KSample:
        return load_nexus2k_sample(dict(self.rows[index]))

    def index_for_uid(self, uid: str) -> int:
        try:
            return self.uids.index(uid)
        except ValueError as error:
            raise KeyError(f"UID {uid} is not in split") from error


def collate_nexus2k_samples(samples: list[Nexus2KSample]) -> Nexus2KBatch:
    """把传入列表组成 batch，不改变任何对象内部的 vertex identity。

    ``len(samples)`` 才是这里的 batch 对象数。传 ``[sample]`` 得到 B=1；传
    ``[sample0, ..., sample19]`` 得到 B=20。collate 只负责整理/padding 数据，
    不决定后续网络是否真正并行处理这 B 个变长 mesh。
    """

    if not samples:
        raise ValueError("cannot collate an empty Nexus2K batch")
    # 顶点坐标用零 padding；mask 使后续代码能恢复每个对象真实的 V。
    maximum_vertices = max(len(sample.vertices) for sample in samples)
    vertices = torch.zeros((len(samples), maximum_vertices, 3), dtype=torch.float32)
    vertex_mask = torch.zeros((len(samples), maximum_vertices), dtype=torch.bool)
    for batch_index, sample in enumerate(samples):
        count = len(sample.vertices)
        vertices[batch_index, :count] = sample.vertices
        vertex_mask[batch_index, :count] = True

    # 八叉树的每个 depth 也分别 padding；当前 Topology AE 不消费它，
    # 但同一个 Nexus2KBatch 可直接交给 Vertex Stage 使用。
    batched_levels: list[VertexLevelBatch] = []
    for depth_index in range(9):
        levels = [sample.octree_levels[depth_index] for sample in samples]
        maximum_parents = max(len(level.parent_codes) for level in levels)
        target = torch.zeros((len(samples), maximum_parents, 8), dtype=torch.float32)
        metadata = torch.zeros((len(samples), maximum_parents, 4), dtype=torch.float32)
        positions = torch.zeros((len(samples), maximum_parents, 3), dtype=torch.float32)
        depths = torch.full(
            (len(samples), maximum_parents), depth_index + 1, dtype=torch.long
        )
        mask = torch.zeros((len(samples), maximum_parents), dtype=torch.bool)
        for batch_index, level in enumerate(levels):
            count = len(level.parent_codes)
            target[batch_index, :count] = level.target
            metadata[batch_index, :count] = level.metadata
            positions[batch_index, :count] = level.parent_codes.to(torch.float32)
            mask[batch_index, :count] = True
        batched_levels.append(VertexLevelBatch(target, metadata, positions, depths, mask))

    return Nexus2KBatch(
        uids=tuple(sample.uid for sample in samples),
        splits=tuple(sample.split for sample in samples),
        condition=torch.stack([sample.condition for sample in samples]),
        vertices=vertices,
        vertex_mask=vertex_mask,
        quantized_vertices=tuple(sample.quantized_vertices for sample in samples),
        faces=tuple(sample.faces for sample in samples),
        octree_levels=tuple(batched_levels),
        edge_index=tuple(sample.edge_index for sample in samples),
        face_set=tuple(sample.face_set for sample in samples),
        faces_oriented=tuple(sample.faces_oriented for sample in samples),
        face_centroid=tuple(sample.face_centroid for sample in samples),
        incidence_index=tuple(sample.incidence_index for sample in samples),
        face_node_offset=tuple(sample.face_node_offset for sample in samples),
        paths=tuple(sample.paths for sample in samples),
    )
