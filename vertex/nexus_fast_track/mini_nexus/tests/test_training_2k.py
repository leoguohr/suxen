import copy
from types import SimpleNamespace

import pytest
import torch

import mini_nexus.training_2k as training_2k
from mini_nexus.topology import (
    all_vertex_pairs,
    build_vertex_face_incidence,
    first_order_interval,
    mesh_edges,
    second_order_interval,
)
from mini_nexus.negative_candidates import SampledTopologyNegatives
from mini_nexus.topology_checkpoint import load_topology_system_from_checkpoint
from mini_nexus.training_2k import (
    Nexus2KTopologyAESystem,
    paper_balanced_binary_loss,
    paper_edge_loss_all_pairs,
    raw_interval_second_moments,
)
from scripts.train_topology_ae_overfit_packed import (
    calibrate_fixed_logit_scales,
    learning_rate_at_step,
    parse_args,
    resolve_scoring_args,
    save_checkpoint,
    scoring_profile_from_args,
)


class RecordingCandidateStore:
    def __init__(self):
        self.calls: list[tuple[str, int]] = []

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
        self.calls.append((uid, seed))
        return SampledTopologyNegatives(
            edges=torch.tensor([[3, 4]]),
            faces=torch.tensor([[0, 1, 4]]),
            edge_source_counts={"uniform": 1},
            face_source_counts={"uniform": 1},
        )

    def sample_fixed_overfit_faces(
        self,
        uid: str,
        *,
        positive_edges: torch.Tensor,
        positive_faces: torch.Tensor,
        vertex_count: int,
        seed: int,
        wedge_ratio: float = 1.0,
        uniform_ratio: float = 0.5,
    ) -> SampledTopologyNegatives:
        self.calls.append((uid, seed))
        return SampledTopologyNegatives(
            edges=torch.empty((0, 2), dtype=torch.long),
            faces=torch.tensor([[0, 1, 4]]),
            edge_source_counts={},
            face_source_counts={"cycle": 0, "wedge": 1, "uniform": 0},
        )


def tetrahedron() -> tuple[torch.Tensor, torch.Tensor]:
    vertices = torch.tensor(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
        ]
    )
    faces = torch.tensor(
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    )
    return vertices, faces


def packed_tetrahedron_batch(uid: str = "one") -> SimpleNamespace:
    vertices, faces = tetrahedron()
    positive_edges = mesh_edges(faces).T
    positive_faces = torch.unique(torch.sort(faces, dim=-1).values, dim=0)
    source, target = build_vertex_face_incidence(faces, len(vertices))
    return SimpleNamespace(
        uids=(uid,),
        vertices=vertices.unsqueeze(0),
        vertex_mask=torch.ones((1, len(vertices)), dtype=torch.bool),
        faces=(faces,),
        edge_index=(positive_edges,),
        face_set=(positive_faces,),
        incidence_index=(torch.stack((source, target)),),
    )


def test_nexus2k_topology_ae_sampled_loss_backward():
    torch.manual_seed(3)
    vertices, faces = tetrahedron()
    batch = SimpleNamespace(
        vertices=vertices.unsqueeze(0),
        vertex_mask=torch.ones((1, len(vertices)), dtype=torch.bool),
        faces=(faces,),
        edge_index=(mesh_edges(faces).T,),
        face_set=(torch.unique(torch.sort(faces, dim=-1).values, dim=0),),
    )
    model = Nexus2KTopologyAESystem(
        hidden_dim=24,
        latent_dim=8,
        spacetime_dim=8,
        num_heads=3,
        encoder_layers=2,
        decoder_layers=1,
        decoder_hidden_dim=24,
    )
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    loss = model(batch, negative_seed=11)
    assert loss.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert all(
        parameter.grad is None or parameter.grad.dtype == torch.float32
        for parameter in model.parameters()
    )


def test_paper_balanced_binary_loss_averages_tp_tn_fp_fn_groups():
    logits = torch.tensor([2.0, -2.0, 2.0, -2.0], requires_grad=True)
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    loss, counts = paper_balanced_binary_loss(logits, labels)
    expected = torch.stack(
        [
            torch.nn.functional.binary_cross_entropy_with_logits(
                logits[index : index + 1], labels[index : index + 1]
            )
            for index in range(4)
        ]
    ).mean()
    assert torch.allclose(loss, expected)
    assert counts == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_paper_edge_loss_supervises_every_vertex_pair():
    edge_embedding = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        requires_grad=True,
    )
    positive_edges = torch.tensor([[0, 1], [0, 3]]).T
    loss, counts = paper_edge_loss_all_pairs(
        edge_embedding, positive_edges, pair_chunk_size=2
    )
    assert counts["pairs"] == 6
    assert counts["positive"] == 2
    assert counts["negative"] == 4
    loss.backward()
    assert torch.isfinite(edge_embedding.grad).all()


def test_edge_logit_scale_is_applied_after_interval_without_changing_groups():
    embedding = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        requires_grad=True,
    )
    positive_edges = torch.tensor([[0, 1], [0, 3]]).T
    scale = 0.125
    actual, counts = paper_edge_loss_all_pairs(
        embedding, positive_edges, pair_chunk_size=2, logit_scale=scale
    )

    pairs = all_vertex_pairs(len(embedding), embedding.device)
    positive = {tuple(row) for row in positive_edges.T.tolist()}
    labels = torch.tensor(
        [tuple(row) in positive for row in pairs.tolist()], dtype=torch.float32
    )
    raw = first_order_interval(embedding[pairs[:, 0]], embedding[pairs[:, 1]])
    expected, expected_counts = paper_balanced_binary_loss(scale * raw, labels)
    assert torch.allclose(actual, expected)
    assert {name: counts[name] for name in ("tp", "tn", "fp", "fn")} == expected_counts


def test_raw_interval_moments_include_distinct_positive_and_negative_faces():
    edge_embedding = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
        ]
    )
    face_embedding = edge_embedding.clone()
    positive_faces = torch.tensor([[0, 1, 2]])
    negative_faces = torch.tensor([[0, 1, 3]])
    edge_m2, face_m2 = raw_interval_second_moments(
        edge_embedding,
        face_embedding,
        positive_faces,
        negative_faces,
        pair_chunk_size=2,
    )
    pairs = all_vertex_pairs(len(edge_embedding), edge_embedding.device)
    expected_edge = first_order_interval(
        edge_embedding[pairs[:, 0]], edge_embedding[pairs[:, 1]]
    ).double().square().mean()
    face_rows = torch.cat((positive_faces, negative_faces))
    expected_face = second_order_interval(
        face_embedding[face_rows[:, 0]],
        face_embedding[face_rows[:, 1]],
        face_embedding[face_rows[:, 2]],
    ).double().square().mean()
    assert torch.equal(edge_m2, expected_edge)
    assert torch.equal(face_m2, expected_face)


def test_system_rejects_non_positive_fixed_scales():
    with pytest.raises(ValueError, match="edge_logit_scale"):
        Nexus2KTopologyAESystem(edge_logit_scale=0.0)
    with pytest.raises(ValueError, match="face_interval_factor"):
        Nexus2KTopologyAESystem(face_interval_factor=0.0)


def test_scoring_contract_round_trips_in_strict_state_dict():
    kwargs = {
        "hidden_dim": 24,
        "latent_dim": 8,
        "spacetime_dim": 8,
        "num_heads": 3,
        "encoder_layers": 2,
        "decoder_layers": 1,
        "decoder_hidden_dim": 24,
        "encoder_dropout": 0.0,
    }
    source = Nexus2KTopologyAESystem(
        **kwargs,
        edge_logit_scale=0.25,
        face_logit_scale=0.125,
        face_interval_factor=0.25,
    )
    state = source.state_dict()
    assert state["_extra_state"]["edge_logit_scale"] == 0.25

    restored = Nexus2KTopologyAESystem(**kwargs)
    restored.load_state_dict(state, strict=True)
    assert restored.scoring_contract() == source.scoring_contract()

    legacy_state = state.copy()
    del legacy_state["_extra_state"]
    legacy_target = Nexus2KTopologyAESystem(**kwargs)
    with pytest.raises(RuntimeError, match="_extra_state"):
        legacy_target.load_state_dict(legacy_state, strict=True)


def test_checkpoint_records_resolved_scoring_contract(tmp_path):
    model = Nexus2KTopologyAESystem(
        hidden_dim=24,
        latent_dim=8,
        spacetime_dim=8,
        num_heads=3,
        encoder_layers=2,
        decoder_layers=1,
        decoder_hidden_dim=24,
        encoder_dropout=0.0,
        edge_logit_scale=0.5,
        face_logit_scale=0.125,
        face_interval_factor=0.25,
    )
    optimizer = torch.optim.Adam(model.parameters())
    args = SimpleNamespace(
        expected_samples=20,
        edge_logit_scale=0.5,
        face_logit_scale=0.125,
        face_interval_factor=0.25,
    )
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        args,
        "manifest-hash",
        100,
        "scoring-profile",
        {"edge_logit_scale": None},
        {"enabled": True},
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    assert checkpoint["format_version"] == 4
    assert checkpoint["scoring_profile"] == "scoring-profile"
    assert checkpoint["spacetime_scoring"] == model.scoring_contract()
    assert checkpoint["runtime"] == model.runtime_contract()
    assert checkpoint["model"]["_extra_state"]["face_interval_factor"] == 0.25
    assert checkpoint["requested_args"] == {"edge_logit_scale": None}


def test_from_saved_args_reconstructs_one_scoring_contract():
    saved = {
        "hidden_dim": 24,
        "latent_dim": 8,
        "spacetime_dim": 8,
        "num_heads": 3,
        "encoder_layers": 2,
        "decoder_layers": 1,
        "decoder_hidden_dim": 24,
        "encoder_dropout": 0.0,
        "edge_logit_scale": 0.5,
        "face_logit_scale": 0.25,
        "face_interval_factor": 0.25,
    }
    model = Nexus2KTopologyAESystem.from_saved_args(saved)
    assert model.scoring_contract() == {
        "embedding_normalization": "per_mesh_center_only",
        "edge_logit_scale": 0.5,
        "face_logit_scale": 0.25,
        "face_interval_factor": 0.25,
        "zero_threshold_sign_preserved": True,
    }


def test_checkpoint_loader_restores_recorded_backend_and_rejects_contract_drift():
    from mini_nexus.flash_varlen_topology import FlashVarlenNexus2KTopologyAESystem

    saved = {
        "hidden_dim": 24,
        "latent_dim": 8,
        "spacetime_dim": 8,
        "num_heads": 3,
        "encoder_layers": 2,
        "decoder_layers": 1,
        "decoder_hidden_dim": 24,
        "encoder_dropout": 0.0,
        "precision": "fp32_flash_bf16",
        "edge_logit_scale": 0.5,
        "face_logit_scale": 0.125,
        "face_interval_factor": 0.25,
    }
    source = FlashVarlenNexus2KTopologyAESystem.from_saved_args(saved)
    checkpoint = {
        "format_version": 4,
        "args": saved,
        "model": source.state_dict(),
        "spacetime_scoring": source.scoring_contract(),
        "runtime": source.runtime_contract(),
    }
    restored = load_topology_system_from_checkpoint(checkpoint, device="cpu")
    assert isinstance(restored, FlashVarlenNexus2KTopologyAESystem)
    assert restored.scoring_contract() == source.scoring_contract()
    assert restored.runtime_contract() == source.runtime_contract()

    diagnostic = copy.deepcopy(checkpoint)
    for metadata in (diagnostic["args"], diagnostic["model"]["_extra_state"],
                     diagnostic["spacetime_scoring"]):
        metadata["normalize_spacetime_embeddings"] = False
        metadata["embedding_normalization_eps"] = 1e-6
    restored_diagnostic = load_topology_system_from_checkpoint(diagnostic, device="cpu")
    assert restored_diagnostic.scoring_contract() == source.scoring_contract()
    for name, parameter in source.named_parameters():
        torch.testing.assert_close(restored_diagnostic.state_dict()[name], parameter)

    missing_layernorm = copy.deepcopy(checkpoint)
    del missing_layernorm["model"]["autoencoder.encoder_output_norm.weight"]
    with pytest.raises(ValueError, match="final LayerNorm"):
        load_topology_system_from_checkpoint(missing_layernorm, device="cpu")

    for location in ("args", "model", "spacetime_scoring"):
        legacy_rms = copy.deepcopy(checkpoint)
        metadata = legacy_rms[location]
        if location == "model":
            metadata = metadata["_extra_state"]
        metadata["normalize_spacetime_embeddings"] = True
        metadata["embedding_normalization"] = "per_mesh_center_global_unit_rms"
        with pytest.raises(ValueError):
            load_topology_system_from_checkpoint(legacy_rms, device="cpu")

    drifted = copy.deepcopy(checkpoint)
    drifted["spacetime_scoring"]["edge_logit_scale"] = 9.0
    with pytest.raises(ValueError, match="top-level scoring metadata"):
        load_topology_system_from_checkpoint(drifted, device="cpu")

    contradictory_runtime = copy.deepcopy(checkpoint)
    contradictory_runtime["args"]["precision"] = "fp32"
    with pytest.raises(ValueError, match="precision and runtime backend"):
        load_topology_system_from_checkpoint(contradictory_runtime, device="cpu")

    ambiguous_legacy = copy.deepcopy(checkpoint)
    del ambiguous_legacy["runtime"]
    ambiguous_legacy["args"]["precision"] = "bf16"
    with pytest.raises(ValueError, match="runtime metadata"):
        load_topology_system_from_checkpoint(ambiguous_legacy, device="cpu")

    unsupported_format = copy.deepcopy(checkpoint)
    unsupported_format["format_version"] = 999
    with pytest.raises(ValueError, match="unsupported Topology AE checkpoint format"):
        load_topology_system_from_checkpoint(unsupported_format, device="cpu")


def test_calibration_uses_explicit_moment_path_without_training_loss(monkeypatch):
    model = Nexus2KTopologyAESystem(
        hidden_dim=24,
        latent_dim=8,
        spacetime_dim=8,
        num_heads=3,
        encoder_layers=2,
        decoder_layers=1,
        decoder_hidden_dim=24,
        encoder_dropout=0.0,
        pair_chunk_size=3,
        negative_candidate_store=RecordingCandidateStore(),
        fixed_overfit_face_negatives=True,
        face_interval_factor=0.25,
    )

    def fail_if_training_loss_runs(*_args, **_kwargs):
        raise AssertionError("calibration must not compute the training loss")

    monkeypatch.setattr(
        training_2k, "topology_autoencoder_loss_with_face_negatives", fail_if_training_loss_runs
    )
    args = SimpleNamespace(
        logit_calibration_epsilon_draws=2,
        expected_samples=1,
        seed=101,
        logit_rms_target=1.0,
        face_interval_factor=0.25,
        edge_logit_scale=None,
        face_logit_scale=None,
    )
    metadata = calibrate_fixed_logit_scales(
        model, [packed_tetrahedron_batch()], args, rank=0, world_size=1
    )

    assert metadata["edge_raw_rms"] * model.edge_logit_scale == pytest.approx(1.0)
    assert metadata["face_raw_rms"] * model.face_logit_scale == pytest.approx(1.0)
    assert metadata["effective_initial_face_logit_rms"] == 0.25

    model.eval()
    with pytest.raises(ValueError, match="requires training mode"):
        calibrate_fixed_logit_scales(
            model, [packed_tetrahedron_batch()], args, rank=0, world_size=1
        )


def test_scoring_profile_describes_resolved_behavior():
    args = SimpleNamespace(
        logit_rms_target=1.0,
        face_interval_factor=1.0,
        logit_calibration_epsilon_draws=1,
        calibrate_logit_scales=False,
        edge_logit_scale=None,
        face_logit_scale=None,
    )
    resolve_scoring_args(args)
    assert args.edge_logit_scale == args.face_logit_scale == 1.0
    assert scoring_profile_from_args(args) == (
        "per_mesh_center_only_fixed_explicit_logits_face_factor_1_v1"
    )

    args.calibrate_logit_scales = True
    args.edge_logit_scale = None
    args.face_logit_scale = None
    resolve_scoring_args(args)
    assert scoring_profile_from_args(args) == (
        "per_mesh_center_only_fixed_calibrated_logits_face_factor_1_v1"
    )


def test_packed_trainer_defaults_to_centered_calibrated_face_area_scoring():
    args = parse_args(
        [
            "--manifest",
            "manifest.csv",
            "--negative-candidate-root",
            "negatives",
            "--output",
            "output",
        ]
    )

    assert not hasattr(args, "normalize_spacetime_embeddings")
    assert not hasattr(args, "embedding_normalization_eps")
    assert args.calibrate_logit_scales is True
    assert args.face_interval_factor == 0.25
    assert args.warmup_steps == 200
    assert scoring_profile_from_args(args).endswith("face_factor_0p25_v1")


@pytest.mark.parametrize("step, expected", [(1, 1e-5), (200, 1e-4), (1000, 1e-4)])
def test_warmup_uses_absolute_optimizer_step(step, expected):
    assert learning_rate_at_step(1e-4, step, warmup_steps=200) == pytest.approx(expected)
    assert learning_rate_at_step(1e-4, step, warmup_steps=0) == 1e-4


@pytest.mark.parametrize("option", ["--normalize-spacetime-embeddings", "--embedding-normalization-eps"])
def test_packed_trainer_rejects_removed_rms_options(option):
    with pytest.raises(SystemExit):
        parse_args([
            "--manifest", "manifest.csv", "--negative-candidate-root", "negatives",
            "--output", "output", option,
        ])


def test_packed_system_passes_fixed_scoring_to_reconstruction_loss(monkeypatch):
    observed = {}

    def probe_loss(*values, **kwargs):
        observed.update(
            edge_logit_scale=kwargs["edge_logit_scale"],
            face_logit_scale=kwargs["face_logit_scale"],
            face_interval_factor=kwargs["face_interval_factor"],
        )
        loss = values[5].square().mean()
        return loss, {"probe": loss}

    monkeypatch.setattr(
        training_2k, "topology_autoencoder_loss_with_face_negatives", probe_loss
    )
    model = Nexus2KTopologyAESystem(
        hidden_dim=24,
        latent_dim=8,
        spacetime_dim=8,
        num_heads=3,
        encoder_layers=2,
        decoder_layers=1,
        decoder_hidden_dim=24,
        encoder_dropout=0.0,
        negative_candidate_store=RecordingCandidateStore(),
        fixed_overfit_face_negatives=True,
        edge_logit_scale=0.5,
        face_logit_scale=0.125,
        face_interval_factor=0.25,
    ).eval()
    loss = model(
        packed_tetrahedron_batch(),
        negative_seed=7,
        fixed_negatives=True,
        packed=True,
        sample_seeds=(11,),
    )

    assert torch.isfinite(loss)
    assert observed == {
        "edge_logit_scale": 0.5,
        "face_logit_scale": 0.125,
        "face_interval_factor": 0.25,
    }


def test_vectorized_pair_chunks_match_torch_combinations():
    from mini_nexus.training_2k import _all_pair_chunks

    for vertex_count in (2, 3, 4, 17, 65):
        expected = torch.combinations(torch.arange(vertex_count), r=2)
        for chunk_size in (1, 2, 7, 128):
            actual = torch.cat(
                list(_all_pair_chunks(vertex_count, torch.device("cpu"), chunk_size))
            )
            assert torch.equal(actual, expected)


def test_large_pair_fallback_matches_torch_combinations():
    from mini_nexus.training_2k import _all_pair_chunks

    vertex_count = 65
    expected = torch.combinations(torch.arange(vertex_count), r=2)
    actual = torch.cat(
        list(
            _all_pair_chunks(
                vertex_count,
                torch.device("cpu"),
                chunk_size=37,
                materialize_limit=0,
            )
        )
    )
    assert torch.equal(actual, expected)


def test_edge_loss_and_gradient_do_not_depend_on_pair_chunk_size():
    torch.manual_seed(13)
    embedding_small = torch.randn(19, 8, requires_grad=True)
    embedding_large = embedding_small.detach().clone().requires_grad_(True)
    positive_edges = torch.tensor([[0, 1], [0, 3], [2, 8], [7, 18]]).T

    loss_small, counts_small = paper_edge_loss_all_pairs(
        embedding_small, positive_edges, pair_chunk_size=5
    )
    loss_large, counts_large = paper_edge_loss_all_pairs(
        embedding_large, positive_edges, pair_chunk_size=10_000
    )
    loss_small.backward()
    loss_large.backward()

    assert counts_small == counts_large
    assert torch.allclose(loss_small, loss_large, atol=1e-6)
    assert torch.allclose(embedding_small.grad, embedding_large.grad, atol=1e-6)


def test_micro_batch_seed_offset_preserves_global_sample_identity():
    vertices, faces = tetrahedron()
    positive_edges = mesh_edges(faces).T
    positive_faces = torch.unique(torch.sort(faces, dim=-1).values, dim=0)
    batch = SimpleNamespace(
        uids=("first", "second"),
        vertices=torch.stack([vertices, vertices]),
        vertex_mask=torch.ones((2, len(vertices)), dtype=torch.bool),
        faces=(faces, faces),
        edge_index=(positive_edges, positive_edges),
        face_set=(positive_faces, positive_faces),
    )
    store = RecordingCandidateStore()
    model = Nexus2KTopologyAESystem(
        hidden_dim=24,
        latent_dim=8,
        spacetime_dim=8,
        num_heads=3,
        encoder_layers=2,
        decoder_layers=1,
        decoder_hidden_dim=24,
        encoder_dropout=0.0,
        negative_candidate_store=store,
    ).eval()

    model(batch, negative_seed=100, sample_seed_offset=7)

    assert store.calls == [("first", 107), ("second", 108)]


def test_fixed_overfit_face_negatives_are_built_once_per_uid():
    vertices, faces = tetrahedron()
    positive_edges = mesh_edges(faces).T
    positive_faces = torch.unique(torch.sort(faces, dim=-1).values, dim=0)
    batch = SimpleNamespace(
        uids=("one",),
        vertices=vertices.unsqueeze(0),
        vertex_mask=torch.ones((1, len(vertices)), dtype=torch.bool),
        faces=(faces,),
        edge_index=(positive_edges,),
        face_set=(positive_faces,),
    )
    store = RecordingCandidateStore()
    model = Nexus2KTopologyAESystem(
        hidden_dim=24,
        latent_dim=8,
        spacetime_dim=8,
        num_heads=3,
        encoder_layers=2,
        decoder_layers=1,
        decoder_hidden_dim=24,
        encoder_dropout=0.0,
        negative_candidate_store=store,
        fixed_overfit_face_negatives=True,
    ).eval()

    model(batch, negative_seed=17)
    model(batch, negative_seed=999)

    assert store.calls == [("one", 17)]


def test_two_micro_batches_match_one_effective_batch_gradient():
    vertices, faces = tetrahedron()
    positive_edges = mesh_edges(faces).T
    positive_faces = torch.unique(torch.sort(faces, dim=-1).values, dim=0)

    def make_batch(uids: tuple[str, ...]) -> SimpleNamespace:
        count = len(uids)
        return SimpleNamespace(
            uids=uids,
            vertices=torch.stack([vertices] * count),
            vertex_mask=torch.ones((count, len(vertices)), dtype=torch.bool),
            faces=tuple(faces for _ in uids),
            edge_index=tuple(positive_edges for _ in uids),
            face_set=tuple(positive_faces for _ in uids),
        )

    torch.manual_seed(5)
    full_model = Nexus2KTopologyAESystem(
        hidden_dim=24,
        latent_dim=8,
        spacetime_dim=8,
        num_heads=3,
        encoder_layers=2,
        decoder_layers=1,
        decoder_hidden_dim=24,
        encoder_dropout=0.0,
        negative_candidate_store=RecordingCandidateStore(),
    ).eval()
    split_model = copy.deepcopy(full_model)

    full_loss = full_model(
        make_batch(("first", "second")), negative_seed=100, sample_seed_offset=0
    )
    full_loss.backward()

    split_loss = torch.zeros(())
    for offset, uid in enumerate(("first", "second")):
        part = split_model(
            make_batch((uid,)), negative_seed=100, sample_seed_offset=offset
        )
        (0.5 * part).backward()
        split_loss = split_loss + 0.5 * part.detach()

    assert torch.allclose(full_loss.detach(), split_loss, atol=1e-6)
    for full_parameter, split_parameter in zip(
        full_model.parameters(), split_model.parameters()
    ):
        assert torch.allclose(full_parameter.grad, split_parameter.grad, atol=1e-6)
