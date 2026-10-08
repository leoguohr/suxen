"""从有版本、已审计的 topology 负样本 sidecar 中只读采样。

``candidate_manifest.csv`` 负责 UID 对齐，``presets.json`` 决定各种困难负样本
的比例，每个 UID 的 NPZ 保存实际候选。当前 medium overfit20 中：

* edge loss 监督全部 ``V choose 2`` 顶点对，所以这里返回的负 edge 仅用于日志/API
  兼容，真正 edge loss 不消费它；
* face 的全部正例来自 topology 真值，负 face 从这里的 ``mixed_medium`` 候选池
  按 step 可复现地重采样。候选文件和比例保持冻结，只有每一步选中的子集变化。

本文件只选择已有候选，不改变 mesh、faces 或任何 NPZ。
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


EDGE_FIELDS = {
    "knn": "edge_negative_knn",
    "two_hop": "edge_negative_two_hop",
    "uniform": "edge_negative_uniform",
}
FACE_FIELDS = {
    "cycle": "face_negative_cycle",
    "wedge": "face_negative_wedge",
    "uniform": "face_negative_uniform",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _largest_remainder(total: int, weights: dict[str, float]) -> dict[str, int]:
    """按比例分配整数配额，并把舍入余数给小数部分最大的来源。"""

    if total < 0 or not weights:
        raise ValueError("total must be non-negative and weights must be nonempty")
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("sampling weights must sum to a positive value")
    raw = {name: total * value / weight_sum for name, value in weights.items()}
    quotas = {name: int(np.floor(value)) for name, value in raw.items()}
    remainder = total - sum(quotas.values())
    order = sorted(weights, key=lambda name: (-(raw[name] - quotas[name]), name))
    for name in order[:remainder]:
        quotas[name] += 1
    return quotas


def _stable_seed(seed: int, uid: str, kind: str, source: str) -> int:
    """把实验 seed、UID、类别和来源编码成跨进程稳定的随机种子。"""

    payload = f"{seed}:{uid}:{kind}:{source}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


@dataclass(frozen=True)
class SampledTopologyNegatives:
    """一次采样结果以及各困难来源实际贡献的数量。"""

    edges: Tensor
    faces: Tensor
    edge_source_counts: dict[str, int]
    face_source_counts: dict[str, int]


class TopologyNegativeCandidateStore:
    """Lazy reader and deterministic sampler for one audited sidecar version."""

    def __init__(
        self,
        root: Path,
        preset_name: str,
        *,
        expected_uids: dict[str, str] | None = None,
    ):
        # 初始化只读元数据；真正的大 NPZ 等某个 UID 被访问时才懒加载。
        self.root = Path(root)
        manifest_path = self.root / "candidate_manifest.csv"
        presets_path = self.root / "presets.json"
        protocol_path = self.root / "evaluation_protocol.json"
        for path in (manifest_path, presets_path, protocol_path):
            if not path.is_file():
                raise ValueError(f"negative-candidate file does not exist: {path}")

        with presets_path.open(encoding="utf-8") as handle:
            presets_document = json.load(handle)
        presets = presets_document.get("presets", {})
        if preset_name not in presets:
            raise ValueError(f"unknown negative-candidate preset: {preset_name}")
        self.preset_name = preset_name
        self.preset = dict(presets[preset_name])
        with protocol_path.open(encoding="utf-8") as handle:
            self.evaluation_protocol = json.load(handle)

        with manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError("candidate manifest is empty")
        self.rows: dict[str, dict[str, str]] = {}
        for row in rows:
            uid = row.get("uid", "")
            split = row.get("split", "")
            if not uid or uid in self.rows:
                raise ValueError("candidate manifest UIDs must be present and unique")
            if split not in {"train", "val"}:
                raise ValueError(f"{uid}: candidate split must be train or val")
            self.rows[uid] = row
        if expected_uids is not None:
            if set(self.rows) != set(expected_uids):
                raise ValueError("candidate UIDs disagree with the final training manifest")
            mismatched = [
                uid for uid, split in expected_uids.items()
                if self.rows[uid]["split"] != split
            ]
            if mismatched:
                raise ValueError(f"candidate splits disagree for {mismatched[:3]}")

        self.manifest_sha256 = _sha256(manifest_path)
        self.presets_sha256 = _sha256(presets_path)
        self.protocol_sha256 = _sha256(protocol_path)
        self._cache: dict[str, dict[str, np.ndarray]] = {}

    def metadata(self) -> dict[str, object]:
        return {
            "negative_candidate_root": str(self.root),
            "negative_candidate_count": len(self.rows),
            "negative_candidate_manifest_sha256": self.manifest_sha256,
            "negative_presets_sha256": self.presets_sha256,
            "negative_protocol_sha256": self.protocol_sha256,
            "negative_preset": self.preset_name,
        }

    def protocol_uids(self, group: str) -> tuple[str, ...]:
        section = self.evaluation_protocol.get(group)
        if not isinstance(section, dict) or not isinstance(section.get("uids"), list):
            raise ValueError(f"evaluation protocol has no UID group: {group}")
        uids = tuple(str(uid) for uid in section["uids"])
        missing = [uid for uid in uids if uid not in self.rows]
        if missing:
            raise ValueError(f"protocol group {group} has unknown UIDs: {missing[:3]}")
        return uids

    def _resolve_candidate_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            if path.is_file():
                return path
            raise ValueError(f"candidate file does not exist: {path}")
        for base in (self.root, *self.root.parents):
            candidate = base / path
            if candidate.is_file():
                return candidate
        raise ValueError(f"cannot resolve candidate path relative to {self.root}: {path}")

    def _load(self, uid: str) -> dict[str, np.ndarray]:
        """加载一个 UID 的候选池，并缓存内存副本以供后续 step 复用。"""

        if uid in self._cache:
            return self._cache[uid]
        try:
            row = self.rows[uid]
        except KeyError as error:
            raise KeyError(f"UID {uid} is absent from candidate manifest") from error
        raw_path = (
            row.get("candidate_path")
            or row.get("candidate_npz")
            or row.get("path")
            or ""
        )
        if not raw_path:
            raise ValueError(f"{uid}: candidate manifest row has no NPZ path")
        path = self._resolve_candidate_path(raw_path)
        arrays: dict[str, np.ndarray] = {}
        with np.load(path, allow_pickle=False) as archive:
            for field in (*EDGE_FIELDS.values(), *FACE_FIELDS.values()):
                if field not in archive.files:
                    raise ValueError(f"{uid}: candidate NPZ is missing {field}")
                width = 2 if field.startswith("edge_") else 3
                value = np.array(archive[field], dtype=np.int64, copy=True)
                if value.ndim != 2 or value.shape[1] != width:
                    raise ValueError(f"{uid}: {field} must have shape [N,{width}]")
                if len(value) and not np.array_equal(value, np.sort(value, axis=1)):
                    raise ValueError(f"{uid}: {field} rows must be canonical tuples")
                arrays[field] = value
        self._cache[uid] = arrays
        return arrays

    @staticmethod
    def _sample_kind(
        arrays: dict[str, np.ndarray],
        field_map: dict[str, str],
        mix: dict[str, float],
        target_count: int,
        *,
        uid: str,
        kind: str,
        seed: int,
        fixed: bool,
    ) -> tuple[np.ndarray, dict[str, int]]:
        # 先算每种来源应拿多少，再无放回地选取并跨来源去重。
        quotas = _largest_remainder(target_count, mix)
        selected: list[tuple[int, ...]] = []
        selected_set: set[tuple[int, ...]] = set()
        source_counts = {source: 0 for source in mix}
        ordered_pools: dict[str, np.ndarray] = {}
        for source in mix:
            pool = arrays[field_map[source]]
            if fixed or len(pool) < 2:
                ordered_pools[source] = pool
            else:
                rng = np.random.default_rng(_stable_seed(seed, uid, kind, source))
                ordered_pools[source] = pool[rng.permutation(len(pool))]

        def take(source: str, limit: int) -> None:
            if limit <= 0:
                return
            added = 0
            for row in ordered_pools[source]:
                item = tuple(int(value) for value in row)
                if item in selected_set:
                    continue
                selected.append(item)
                selected_set.add(item)
                source_counts[source] = source_counts.get(source, 0) + 1
                added += 1
                if added == limit or len(selected) == target_count:
                    break

        for source, quota in quotas.items():
            take(source, quota)
        if len(selected) < target_count:
            hard_sources = [source for source in mix if source != "uniform"]
            fallback_sources = hard_sources + (["uniform"] if "uniform" in mix else [])
            for source in fallback_sources:
                take(source, target_count - len(selected))
                if len(selected) == target_count:
                    break

        width = 2 if kind == "edge" else 3
        result = np.asarray(selected, dtype=np.int64).reshape(-1, width)
        return result, source_counts

    def sample(
        self,
        uid: str,
        *,
        positive_edge_count: int,
        positive_face_count: int,
        seed: int,
        fixed: bool,
        include_edges: bool = True,
    ) -> SampledTopologyNegatives:
        """按照 preset 生成该 UID 的 edge/face 负样本张量。"""

        arrays = self._load(uid)
        face_target = int(round(positive_face_count * self.preset["face_negative_ratio"]))
        if include_edges:
            edge_target = int(
                round(positive_edge_count * self.preset["edge_negative_ratio"])
            )
            edges, edge_counts = self._sample_kind(
                arrays,
                EDGE_FIELDS,
                dict(self.preset["edge_mix"]),
                edge_target,
                uid=uid,
                kind="edge",
                seed=seed,
                fixed=fixed,
            )
        else:
            # Paper-aligned edge loss监督全部 V choose 2 pair，不消费负 edge。
            # 跳过这一步只删除无用 CPU 采样，不改变任何训练标签或 loss。
            edges = np.empty((0, 2), dtype=np.int64)
            edge_counts = {}
        faces, face_counts = self._sample_kind(
            arrays,
            FACE_FIELDS,
            dict(self.preset["face_mix"]),
            face_target,
            uid=uid,
            kind="face",
            seed=seed,
            fixed=fixed,
        )
        return SampledTopologyNegatives(
            edges=torch.from_numpy(edges),
            faces=torch.from_numpy(faces),
            edge_source_counts=edge_counts,
            face_source_counts=face_counts,
        )

    def sample_fixed_overfit_faces(
        self,
        uid: str,
        *,
        positive_edges: Tensor,
        positive_faces: Tensor,
        vertex_count: int,
        seed: int,
        wedge_ratio: float = 1.0,
        uniform_ratio: float = 0.5,
    ) -> SampledTopologyNegatives:
        """Build the fixed face-negative set used by the small-set overfit test.

        Every three-cycle in the ground-truth edge graph that is not a ground-truth
        face is included.  Fixed wedge and uniform subsets are then read from the
        audited sidecar.  This deliberately matches the edges-first recovery path
        more closely than drawing a fresh mixed subset at every optimizer step.
        """

        if vertex_count < 0 or wedge_ratio < 0 or uniform_ratio < 0:
            raise ValueError("vertex_count and overfit face ratios must be non-negative")

        if positive_edges.ndim != 2:
            raise ValueError("positive_edges must have shape [2,E] or [E,2]")
        if positive_edges.shape[0] == 2:
            edge_rows = positive_edges.T
        elif positive_edges.shape[1] == 2:
            edge_rows = positive_edges
        else:
            raise ValueError("positive_edges must have shape [2,E] or [E,2]")
        if positive_faces.ndim != 2 or positive_faces.shape[1] != 3:
            raise ValueError("positive_faces must have shape [F,3]")
        face_rows = positive_faces
        edge_set = {
            tuple(sorted((int(row[0]), int(row[1]))))
            for row in edge_rows.detach().cpu().tolist()
        }
        face_set = {
            tuple(sorted((int(row[0]), int(row[1]), int(row[2]))))
            for row in face_rows.detach().cpu().tolist()
        }
        if any(
            left < 0 or right >= vertex_count or left == right
            for left, right in edge_set
        ):
            raise ValueError(f"{uid}: positive edge is outside [0,V) or degenerate")
        if any(
            face[0] < 0 or face[2] >= vertex_count or len(set(face)) != 3
            for face in face_set
        ):
            raise ValueError(f"{uid}: positive face is outside [0,V) or degenerate")

        adjacency = [set() for _ in range(vertex_count)]
        for left, right in edge_set:
            adjacency[left].add(right)
            adjacency[right].add(left)

        false_cycles: list[tuple[int, int, int]] = []
        for left, neighbors in enumerate(adjacency):
            for middle in (value for value in neighbors if value > left):
                for right in adjacency[left].intersection(adjacency[middle]):
                    if right > middle:
                        triplet = (left, middle, right)
                        if triplet not in face_set:
                            false_cycles.append(triplet)
        false_cycles.sort()

        arrays = self._load(uid)
        selected = list(false_cycles)
        selected_set = set(false_cycles)
        source_counts = {
            "cycle": len(false_cycles),
            "wedge": 0,
            "uniform": 0,
        }

        def add_fixed_subset(source: str, target: int) -> None:
            if target <= 0:
                return
            pool = arrays[FACE_FIELDS[source]]
            rng = np.random.default_rng(
                _stable_seed(seed, uid, "face_overfit", source)
            )
            order = rng.permutation(len(pool)) if len(pool) > 1 else np.arange(len(pool))
            for index in order:
                item = tuple(int(value) for value in pool[index])
                if item in selected_set or item in face_set:
                    continue
                selected.append(item)
                selected_set.add(item)
                source_counts[source] += 1
                if source_counts[source] == target:
                    break

        positive_face_count = len(face_set)
        add_fixed_subset("wedge", int(np.ceil(positive_face_count * wedge_ratio)))
        add_fixed_subset("uniform", int(np.ceil(positive_face_count * uniform_ratio)))

        faces = np.asarray(selected, dtype=np.int64).reshape(-1, 3)
        return SampledTopologyNegatives(
            edges=torch.empty((0, 2), dtype=torch.int64),
            faces=torch.from_numpy(faces),
            edge_source_counts={},
            face_source_counts=source_counts,
        )
