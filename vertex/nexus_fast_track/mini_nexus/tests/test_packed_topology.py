from types import SimpleNamespace

import torch

from mini_nexus.packed_topology import (
    balance_samples_across_ranks,
    plan_padded_batches,
    topology_attention_cost,
)


def sample(uid: str, vertex_count: int, face_count: int) -> SimpleNamespace:
    return SimpleNamespace(
        uid=uid,
        vertices=torch.zeros((vertex_count, 3)),
        faces=torch.zeros((face_count, 3), dtype=torch.long),
    )


def test_padded_batch_plan_keeps_every_uid_once_and_bounds_batch_size():
    samples = [
        sample("small-a", 8, 12),
        sample("small-b", 9, 14),
        sample("medium", 100, 200),
        sample("large", 1000, 2000),
    ]
    batches = plan_padded_batches(
        samples, max_meshes=2, max_padding_ratio=1.25
    )
    assert sorted(value.uid for batch in batches for value in batch) == sorted(
        value.uid for value in samples
    )
    assert all(len(batch) <= 2 for batch in batches)
    assert [value.uid for value in batches[-1]] == ["large"]


def test_rank_balancer_assigns_every_sample_once():
    samples = [sample(f"uid-{index}", index + 2, 2 * index + 1) for index in range(8)]
    assignments = balance_samples_across_ranks(samples, world_size=2)
    assert sorted(value.uid for rank in assignments for value in rank) == sorted(
        value.uid for value in samples
    )
    loads = [sum(topology_attention_cost(value) for value in rank) for rank in assignments]
    assert max(loads) / min(loads) < 1.5
