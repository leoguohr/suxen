import torch

from mini_nexus.data import PilotSample, collate_octree_level, collate_pilot_samples
from mini_nexus.topology import TopologyAutoencoder
from mini_nexus.training import TopologyAESystem, TopologyFlowSystem, VertexStageSystem


def make_sample(uid: str, offset: float = 0.0) -> PilotSample:
    vertices = torch.tensor(
        [
            [-0.75 + offset, -0.75, -0.75],
            [0.75, -0.75 + offset, -0.75],
            [-0.75, 0.75, -0.75 + offset],
            [-0.75, -0.75, 0.75],
        ]
    )
    faces = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    points = vertices.repeat(2, 1)
    condition = torch.cat([points, torch.nn.functional.normalize(points, dim=-1)], dim=-1)
    return PilotSample(uid, vertices, faces, condition, {})


def test_all_training_systems_support_a_two_object_batch():
    torch.manual_seed(2)
    batch = collate_pilot_samples([make_sample("a"), make_sample("b", 0.05)])

    vertex = VertexStageSystem(
        hidden_dim=24, condition_tokens=2, num_layers=1, num_heads=3, max_depth=3,
        condition_dim=24, condition_heads=3,
    )
    level = collate_octree_level(batch, current_depth=2, max_depth=3)
    vertex_loss = vertex(batch.condition, level)
    vertex_loss.backward()
    assert torch.isfinite(vertex_loss)

    topology_ae_system = TopologyAESystem(
        hidden_dim=24, latent_dim=8, spacetime_dim=8, num_heads=3, num_layers=1
    )
    topology_ae_loss = topology_ae_system(batch)
    topology_ae_loss.backward()
    assert torch.isfinite(topology_ae_loss)

    frozen_ae = TopologyAutoencoder(
        hidden_dim=24, latent_dim=8, spacetime_dim=8, num_heads=3, num_layers=1
    )
    topology_flow = TopologyFlowSystem(
        frozen_ae,
        hidden_dim=24,
        latent_dim=8,
        condition_tokens=2,
        num_layers=1,
        num_heads=3,
    )
    topology_flow_loss = topology_flow(batch)
    topology_flow_loss.backward()
    assert torch.isfinite(topology_flow_loss)
