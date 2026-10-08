import pytest
import torch

from mini_nexus.topology import (
    TopologyAutoencoder,
    build_vertex_face_incidence,
    edge_interval_logits,
    face_interval_logits,
    first_order_interval,
    orient_faces_consistently,
    recover_topology,
    sample_face_triplets,
    second_order_interval,
    topology_autoencoder_loss,
)


def test_spacetime_interval_signs():
    origin = torch.tensor([0.0, 0.0, 0.0, 0.0])
    spatial = torch.tensor([1.0, 0.0, 0.0, 0.0])
    temporal = torch.tensor([0.0, 0.0, 2.0, 0.0])
    assert first_order_interval(origin, spatial) > 0
    assert first_order_interval(origin, temporal) < 0

    a = torch.tensor([0.0, 0.0, 0.0, 0.0])
    b = torch.tensor([1.0, 0.0, 0.0, 0.0])
    c = torch.tensor([0.0, 1.0, 0.0, 0.0])
    assert second_order_interval(a, b, c) > 0


def test_face_triangle_area_factor_changes_magnitude_not_sign():
    a = torch.tensor([0.0, 0.0, 0.0, 0.0])
    b = torch.tensor([1.0, 0.0, 0.0, 0.0])
    c = torch.tensor([0.0, 1.0, 0.0, 0.0])
    parallelogram = second_order_interval(a, b, c, area_factor=1.0)
    triangle = second_order_interval(a, b, c, area_factor=0.25)
    assert torch.equal(parallelogram, torch.tensor(1.0))
    assert torch.equal(triangle, torch.tensor(0.25))
    assert torch.sign(parallelogram) == torch.sign(triangle)


def test_interval_logit_helpers_apply_positive_scales_and_reject_invalid_factor():
    origin = torch.tensor([0.0, 0.0, 0.0, 0.0])
    first = torch.tensor([1.0, 0.0, 0.0, 0.0])
    second = torch.tensor([0.0, 1.0, 0.0, 0.0])

    assert torch.equal(
        edge_interval_logits(origin, first, logit_scale=2.0),
        torch.tensor(2.0),
    )
    assert torch.equal(
        face_interval_logits(
            origin,
            first,
            second,
            logit_scale=4.0,
            area_factor=0.25,
        ),
        torch.tensor(1.0),
    )
    for invalid in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="area_factor"):
            second_order_interval(origin, first, second, area_factor=invalid)


def test_edges_first_face_recovery():
    edge_embedding = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    )
    face_embedding = edge_embedding.clone()
    edges, faces = recover_topology(edge_embedding, face_embedding)
    assert edges.shape == (3, 2)
    assert torch.equal(faces, torch.tensor([[0, 1, 2]]))

    scaled_edges, scaled_faces = recover_topology(
        edge_embedding,
        face_embedding,
        edge_logit_scale=0.01,
        face_logit_scale=100.0,
        face_interval_factor=0.25,
    )
    assert torch.equal(scaled_edges, edges)
    assert torch.equal(scaled_faces, faces)


def test_nonzero_recovery_thresholds_use_scaled_logit_units():
    embedding = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    )

    unscaled_edges, unscaled_faces = recover_topology(
        embedding,
        embedding,
        edge_threshold=1.5,
        face_threshold=0.75,
    )
    assert unscaled_edges.shape == (1, 2)
    assert unscaled_faces.shape == (0, 3)

    scaled_edges, scaled_faces = recover_topology(
        embedding,
        embedding,
        edge_threshold=1.5,
        face_threshold=0.75,
        edge_logit_scale=2.0,
        face_interval_factor=0.25,
    )
    assert scaled_edges.shape == (3, 2)
    assert scaled_faces.shape == (0, 3)

    _, restored_faces = recover_topology(
        embedding,
        embedding,
        edge_threshold=1.5,
        face_threshold=0.75,
        edge_logit_scale=2.0,
        face_logit_scale=4.0,
        face_interval_factor=0.25,
    )
    assert torch.equal(restored_faces, torch.tensor([[0, 1, 2]]))


def test_orientation_makes_shared_edge_directions_opposite():
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    # Both faces currently traverse the shared edge 2 -> 0.
    faces = torch.tensor([[0, 1, 2], [2, 0, 3]])
    oriented = orient_faces_consistently(vertices, faces)

    directed_edges = []
    for face in oriented.tolist():
        directed_edges.extend(zip(face, face[1:] + face[:1]))
    assert directed_edges.count((0, 2)) + directed_edges.count((2, 0)) == 2
    assert directed_edges.count((0, 2)) == directed_edges.count((2, 0))


def test_orientation_makes_closed_component_positive_volume():
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    inward_faces = torch.tensor([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
    oriented = orient_faces_consistently(vertices, inward_faces)
    triangles = vertices[oriented]
    signed_volume = torch.sum(
        triangles[:, 0] * torch.cross(triangles[:, 1], triangles[:, 2], dim=1)
    )
    assert signed_volume > 0


def test_topology_autoencoder_forward_and_loss():
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    model = TopologyAutoencoder(
        hidden_dim=32,
        latent_dim=8,
        spacetime_dim=8,
        num_heads=4,
        encoder_layers=2,
        decoder_layers=1,
        decoder_hidden_dim=32,
    )
    _, mu, log_variance, edge_embedding, face_embedding = model(vertices, faces)
    loss, parts = topology_autoencoder_loss(
        vertices, faces, mu, log_variance, edge_embedding, face_embedding
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert set(parts) == {"edge", "face", "kl"}


def test_topology_autoencoder_can_disable_encoder_dropout():
    model = TopologyAutoencoder(
        hidden_dim=32,
        latent_dim=8,
        spacetime_dim=8,
        num_heads=4,
        encoder_layers=2,
        decoder_layers=1,
        decoder_hidden_dim=32,
        encoder_dropout=0.0,
    )
    block = model.encoder_blocks[0].transformer
    assert block.dropout.p == 0.0
    assert block.dropout1.p == 0.0
    assert block.dropout2.p == 0.0


def test_topology_decoder_centers_without_rescaling_each_embedding():
    torch.manual_seed(7)
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    model = TopologyAutoencoder(
        hidden_dim=32,
        latent_dim=8,
        spacetime_dim=8,
        num_heads=4,
        encoder_layers=2,
        decoder_layers=1,
        decoder_hidden_dim=32,
    ).eval()
    raw = {}
    handles = [
        getattr(model, name).register_forward_hook(
            lambda _module, _inputs, output, name=name: raw.update({name: output})
        )
        for name in ("edge_embedding", "face_embedding")
    ]
    try:
        outputs = model(vertices, faces)[3:]
    finally:
        for handle in handles:
            handle.remove()

    for name, embedding in zip(("edge_embedding", "face_embedding"), outputs):
        assert embedding.dtype == torch.float32
        torch.testing.assert_close(
            embedding, raw[name].float() - raw[name].float().mean(dim=0, keepdim=True)
        )
        torch.testing.assert_close(embedding.mean(dim=0), torch.zeros(8), atol=1e-6, rtol=0)
        assert not torch.isclose(embedding.square().mean(), torch.tensor(1.0))


@pytest.mark.parametrize("packed", [False, True])
def test_output_layernorms_run_before_heads_and_receive_gradients(packed):
    torch.manual_seed(71)
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
    incidence = torch.stack(build_vertex_face_incidence(faces, len(vertices)))
    model = TopologyAutoencoder(
        hidden_dim=32, latent_dim=8, spacetime_dim=8, num_heads=4,
        encoder_layers=2, decoder_layers=1, decoder_hidden_dim=32,
        encoder_dropout=0.0,
    )
    observed = {}
    handles = []
    for name in ("encoder_output_norm", "decoder_output_norm", "mu", "log_variance",
                 "edge_embedding", "face_embedding"):
        handles.append(getattr(model, name).register_forward_hook(
            lambda _module, inputs, output, name=name:
            observed.update({name: (inputs[0], output)})
        ))
    try:
        if packed:
            outputs = model.forward_packed(
                vertices.unsqueeze(0), torch.ones((1, len(vertices)), dtype=torch.bool),
                (faces,), (incidence,),
            )
            outputs = tuple(value[0] for value in outputs)
        else:
            outputs = model(vertices, faces, incidence)
        loss, _ = topology_autoencoder_loss(vertices, faces, *outputs[1:])
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()

    for norm_name, head_names in (
        ("encoder_output_norm", ("mu", "log_variance")),
        ("decoder_output_norm", ("edge_embedding", "face_embedding")),
    ):
        norm = getattr(model, norm_name)
        assert isinstance(norm, torch.nn.LayerNorm)
        normalized = observed[norm_name][1]
        assert normalized.dtype == torch.float32
        for head_name in head_names:
            torch.testing.assert_close(observed[head_name][0], normalized)
        for parameter in norm.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        # The decoder beta cancels under a linear head followed by centering.
        assert torch.count_nonzero(norm.weight.grad) > 0


def test_vertex_face_incidence_is_bidirectional():
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
    source, target = build_vertex_face_incidence(faces, vertex_count=4)
    directed_edges = set(zip(source.tolist(), target.tolist()))
    assert len(directed_edges) == 12
    for face_index, face in enumerate(faces.tolist()):
        face_node = 4 + face_index
        for vertex in face:
            assert (vertex, face_node) in directed_edges
            assert (face_node, vertex) in directed_edges


def test_autoencoder_reuses_precomputed_incidence_without_changing_output():
    torch.manual_seed(17)
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    faces = torch.tensor(
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=torch.long
    )
    source, target = build_vertex_face_incidence(faces, len(vertices))
    incidence = torch.stack((source, target))
    model = TopologyAutoencoder(
        hidden_dim=32,
        latent_dim=8,
        encoder_layers=2,
        decoder_layers=2,
        decoder_hidden_dim=32,
        encoder_dropout=0.0,
    ).eval()

    torch.manual_seed(29)
    direct = model(vertices, faces)
    torch.manual_seed(29)
    precomputed = model(vertices, faces, incidence)

    for direct_tensor, precomputed_tensor in zip(direct, precomputed):
        assert torch.equal(direct_tensor, precomputed_tensor)


def test_packed_autoencoder_matches_independent_meshes_in_eval_mode():
    """Padding masks must not introduce cross-mesh graph or attention paths."""

    first_vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    first_faces = torch.tensor([[0, 1, 2]])
    second_vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    second_faces = torch.tensor(
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    )
    incidences = []
    for vertices, faces in (
        (first_vertices, first_faces),
        (second_vertices, second_faces),
    ):
        source, target = build_vertex_face_incidence(faces, len(vertices))
        incidences.append(torch.stack((source, target)))

    model = TopologyAutoencoder(
        hidden_dim=24,
        latent_dim=8,
        spacetime_dim=8,
        num_heads=3,
        encoder_layers=2,
        decoder_layers=2,
        decoder_hidden_dim=24,
        encoder_dropout=0.0,
    ).eval()
    with torch.no_grad():
        for norm in (model.encoder_output_norm, model.decoder_output_norm):
            norm.weight.copy_(torch.linspace(0.7, 1.3, 24))
            norm.bias.copy_(torch.linspace(-0.2, 0.2, 24))
    independent = [
        model(vertices, faces, incidence)
        for vertices, faces, incidence in zip(
            (first_vertices, second_vertices),
            (first_faces, second_faces),
            incidences,
        )
    ]
    padded_vertices = torch.zeros((2, 4, 3))
    padded_vertices[0, :3] = first_vertices
    padded_vertices[1] = second_vertices
    vertex_mask = torch.tensor(
        [[True, True, True, False], [True, True, True, True]]
    )
    packed = model.forward_packed(
        padded_vertices,
        vertex_mask,
        (first_faces, second_faces),
        tuple(incidences),
    )

    for sample_index, vertex_count in enumerate((3, 4)):
        for output_index in range(1, 5):
            assert torch.allclose(
                packed[output_index][sample_index, :vertex_count],
                independent[sample_index][output_index],
                atol=2e-6,
                rtol=2e-6,
            )
        for output_index in (3, 4):
            valid = packed[output_index][sample_index, :vertex_count]
            assert torch.allclose(valid.mean(dim=0), torch.zeros(8), atol=1e-6)
    assert torch.equal(packed[3][0, 3], torch.zeros(8))
    assert torch.equal(packed[4][0, 3], torch.zeros(8))

    changed_vertices = padded_vertices.clone()
    changed_vertices[1] = changed_vertices[1] * -7 + 3
    changed_vertices[0, 3] = 1e6
    changed = model.forward_packed(
        changed_vertices, vertex_mask, (first_faces, second_faces), tuple(incidences)
    )
    for before, after in zip(packed, changed):
        torch.testing.assert_close(after[0], before[0], atol=2e-6, rtol=2e-6)


def test_topology_encoder_uses_interleaved_graph_transformer_blocks():
    model = TopologyAutoencoder(
        hidden_dim=24,
        latent_dim=8,
        spacetime_dim=8,
        num_heads=3,
        encoder_layers=4,
        decoder_layers=2,
        decoder_hidden_dim=24,
    )
    assert len(model.encoder_blocks) == 2
    assert len(model.decoder_blocks) == 2
    # Nexus only says "standard Transformer"; the 4x FFN ratio is the
    # independent standard-Transformer interpretation recorded in config.json.
    assert model.encoder_blocks[0].transformer.linear1.in_features == 24
    assert model.encoder_blocks[0].transformer.linear1.out_features == 4 * 24


def test_decoder_uses_latent_only_and_full_hidden_width_attention():
    """The paper-aligned decoder does not re-inject GT vertex coordinates.

    Full-width MHA is an explicit independent choice: Q/K/V project from and to
    the complete decoder hidden width.  Nexus does not publish these dimensions.
    """

    model = TopologyAutoencoder(
        hidden_dim=24,
        latent_dim=8,
        spacetime_dim=8,
        num_heads=4,
        encoder_layers=2,
        decoder_layers=1,
        decoder_hidden_dim=32,
    )
    assert model.latent_input.in_features == 8
    assert model.latent_input.out_features == 32
    attention = model.decoder_blocks[0].attention
    assert attention.embed_dim == 32
    assert attention.in_proj_weight.shape == (3 * 32, 32)
    assert not hasattr(model.decoder_blocks[0], "ffn")


def test_face_sampler_keeps_all_positives_and_adds_negatives():
    faces = torch.tensor([[0, 1, 2], [0, 1, 3]])
    triplets, labels = sample_face_triplets(
        faces, 6, negative_ratio=2.0, generator=torch.Generator().manual_seed(3)
    )
    positives = {tuple(row.tolist()) for row in triplets[labels.bool()]}
    assert positives == {(0, 1, 2), (0, 1, 3)}
    assert int((~labels.bool()).sum()) == 4
