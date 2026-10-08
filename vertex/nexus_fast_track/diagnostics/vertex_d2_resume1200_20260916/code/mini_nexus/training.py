"""Stage-specific trainable systems shared by single-GPU and DDP entrypoints."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .data import PilotBatch, VertexLevelBatch
from .flow import flow_matching_batch, flow_matching_loss
from .models import ConditionalFlowTransformer, VecSetConditionEncoder
from .topology import TopologyAutoencoder, topology_autoencoder_loss
from .vertex import VertexConditionEncoder, VertexDiT


class VertexStageSystem(nn.Module):
    """Joint point-cloud VecSet + octree velocity model, independent Nexus baseline.

    Dimensions default to the agreed full architecture. Pass explicit smaller
    dimensions for smoke tests; no weights from the old prototype are compatible.
    """

    architecture = "point_cloud_vertex_v1"

    def __init__(
        self,
        hidden_dim: int = 1536,
        condition_tokens: int = 1024,
        num_layers: int = 36,
        num_heads: int = 12,
        max_depth: int = 9,
        *,
        condition_dim: int = 2048,
        condition_heads: int = 16,
        condition_layers: int = 8,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.condition_encoder = VertexConditionEncoder(
            hidden_dim=condition_dim,
            num_tokens=condition_tokens,
            num_heads=condition_heads,
            num_layers=condition_layers,
            use_checkpoint=use_checkpoint,
        )
        self.flow = VertexDiT(
            hidden_dim=hidden_dim,
            condition_dim=condition_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_depth=max_depth,
            use_checkpoint=use_checkpoint,
        )

    def forward(
        self,
        condition: Tensor,
        level: VertexLevelBatch,
        *,
        noise: Tensor | None = None,
        time: Tensor | None = None,
        condition_mask: Tensor | None = None,
    ) -> Tensor:
        if level.target.ndim != 3 or level.target.shape[-1] != 8:
            raise ValueError("vertex target must have shape [B,N,8]")
        if level.mask.shape != level.target.shape[:2] or level.mask.dtype != torch.bool:
            raise ValueError("vertex mask must be boolean [B,N]")
        counts = level.mask.sum(dim=1)
        if torch.any(counts == 0):
            raise ValueError("each training object must contain a valid parent")
        valid = level.mask.unsqueeze(-1)
        clean = level.target.masked_fill(~valid, 0)
        if noise is not None:
            if noise.shape != clean.shape:
                raise ValueError("noise must match vertex target shape")
            noise = noise.masked_fill(~valid, 0)
        noisy, target_velocity, time, _ = flow_matching_batch(
            clean, noise=noise, time=time
        )
        prediction = self.flow(
            noisy,
            time,
            level.positions.to(torch.long),
            level.depths,
            self.condition_encoder(condition, condition_mask),
            level.mask,
        )
        # Average within each object first: a mesh with more parents must not
        # silently receive a larger training weight. Reduce in fp32 under AMP.
        error = prediction.float() - target_velocity.float().masked_fill(~valid, 0)
        squared = error.masked_fill(~valid, 0).square()
        per_object = squared.sum(dim=(1, 2)) / (counts * 8)
        return per_object.mean()


class TopologyAESystem(nn.Module):
    """One DDP forward may contain several variable-size mesh graphs."""

    def __init__(
        self,
        hidden_dim: int = 96,
        latent_dim: int = 16,
        spacetime_dim: int = 16,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.autoencoder = TopologyAutoencoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            spacetime_dim=spacetime_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )

    def forward(self, batch: PilotBatch) -> Tensor:
        losses = []
        for vertices_padded, mask, faces in zip(
            batch.vertices, batch.vertex_mask, batch.faces
        ):
            vertices = vertices_padded[mask]
            _, mu, log_variance, edge_embedding, face_embedding = self.autoencoder(
                vertices, faces
            )
            loss, _ = topology_autoencoder_loss(
                vertices,
                faces,
                mu,
                log_variance,
                edge_embedding,
                face_embedding,
            )
            losses.append(loss)
        return torch.stack(losses).mean()


class TopologyFlowSystem(nn.Module):
    def __init__(
        self,
        topology_autoencoder: TopologyAutoencoder,
        hidden_dim: int = 96,
        latent_dim: int = 16,
        condition_tokens: int = 16,
        num_layers: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        self.topology_autoencoder = topology_autoencoder.eval()
        for parameter in self.topology_autoencoder.parameters():
            parameter.requires_grad_(False)
        self.condition_encoder = VecSetConditionEncoder(
            hidden_dim=hidden_dim,
            num_tokens=condition_tokens,
            num_heads=num_heads,
        )
        self.flow = ConditionalFlowTransformer(
            data_dim=latent_dim,
            metadata_dim=3,
            hidden_dim=hidden_dim,
            condition_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            use_depth_embedding=False,
        )

    def train(self, mode: bool = True) -> "TopologyFlowSystem":
        super().train(mode)
        self.topology_autoencoder.eval()
        return self

    def forward(self, batch: PilotBatch) -> Tensor:
        target = torch.zeros(
            (*batch.vertices.shape[:2], self.flow.data_projection.in_features),
            device=batch.vertices.device,
            dtype=batch.vertices.dtype,
        )
        with torch.no_grad():
            for batch_index, (vertices_padded, mask, faces) in enumerate(
                zip(batch.vertices, batch.vertex_mask, batch.faces)
            ):
                vertices = vertices_padded[mask]
                mu, _ = self.topology_autoencoder.encode(vertices, faces)
                target[batch_index, : len(vertices)] = mu
        noisy, target_velocity, time, _ = flow_matching_batch(target)
        prediction = self.flow(
            noisy,
            time,
            batch.vertices,
            self.condition_encoder(batch.condition),
            batch.vertex_mask,
            positions=batch.vertices,
        )
        return flow_matching_loss(prediction, target_velocity, batch.vertex_mask)
