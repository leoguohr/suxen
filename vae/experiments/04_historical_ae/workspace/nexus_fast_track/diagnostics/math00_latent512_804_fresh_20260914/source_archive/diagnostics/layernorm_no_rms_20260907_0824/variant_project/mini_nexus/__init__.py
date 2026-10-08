"""Small, readable, independent implementation of the public Nexus method."""

from .flow import euler_integrate, flow_matching_batch, flow_matching_loss
from .octree import (
    OctreeLevel,
    build_octree_levels,
    decode_leaf_centers,
    quantize_vertex_cells,
    quantize_vertices,
)
from .topology import (
    TopologyAutoencoder,
    edge_interval_logits,
    enumerate_edge_triangles,
    face_interval_logits,
    first_order_interval,
    normalize_embedding,
    orient_faces_consistently,
    recover_topology,
    second_order_interval,
    vae_kl_loss,
)

__all__ = [
    "OctreeLevel",
    "TopologyAutoencoder",
    "build_octree_levels",
    "decode_leaf_centers",
    "edge_interval_logits",
    "euler_integrate",
    "enumerate_edge_triangles",
    "face_interval_logits",
    "first_order_interval",
    "normalize_embedding",
    "flow_matching_batch",
    "flow_matching_loss",
    "quantize_vertex_cells",
    "quantize_vertices",
    "orient_faces_consistently",
    "recover_topology",
    "second_order_interval",
    "vae_kl_loss",
]
