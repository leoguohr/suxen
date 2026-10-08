from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

import mini_nexus.flash_varlen_topology as flash_backend
import mini_nexus.training_2k as training_backend
from mini_nexus.flash_varlen_topology import (
    FlashVarlenNexus2KTopologyAESystem,
    _cu_seqlens,
    _flash_self_attention,
    flash_attn_varlen_qkvpacked_func,
)
from mini_nexus.negative_candidates import SampledTopologyNegatives
from mini_nexus.topology import build_vertex_face_incidence, mesh_edges
from mini_nexus.training_2k import Nexus2KTopologyAESystem


requires_flash_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or flash_attn_varlen_qkvpacked_func is None,
    reason="requires CUDA and flash-attn",
)


class FixedCandidateStore:
    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    def sample_fixed_overfit_faces(
        self,
        uid,
        *,
        positive_edges,
        positive_faces,
        vertex_count,
        seed,
        **_kwargs,
    ):
        self.calls.append((uid, seed))
        return SampledTopologyNegatives(
            edges=torch.empty((0, 2), dtype=torch.long),
            faces=torch.tensor([[0, 1, vertex_count - 1]], dtype=torch.long),
            edge_source_counts={},
            face_source_counts={"uniform": 1},
        )


def two_mesh_batch():
    vertex_rows = (
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
                [2.0, 2.0, 2.0],
            ]
        ),
    )
    face_rows = (
        torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long),
        torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 4]], dtype=torch.long),
    )
    vertices = torch.zeros((2, 5, 3), dtype=torch.float32)
    vertex_mask = torch.zeros((2, 5), dtype=torch.bool)
    incidence_rows = []
    edge_rows = []
    positive_face_rows = []
    for index, (vertex_row, face_row) in enumerate(zip(vertex_rows, face_rows)):
        vertices[index, : len(vertex_row)] = vertex_row
        vertex_mask[index, : len(vertex_row)] = True
        source, target = build_vertex_face_incidence(face_row, len(vertex_row))
        incidence_rows.append(torch.stack((source, target)))
        edge_rows.append(mesh_edges(face_row).T)
        positive_face_rows.append(
            torch.unique(torch.sort(face_row, dim=-1).values, dim=0)
        )
    return SimpleNamespace(
        uids=("four", "five"),
        vertices=vertices,
        vertex_mask=vertex_mask,
        faces=face_rows,
        incidence_index=tuple(incidence_rows),
        edge_index=tuple(edge_rows),
        face_set=tuple(positive_face_rows),
    )


def small_flash_system(**kwargs):
    defaults = {
        "hidden_dim": 32,
        "latent_dim": 8,
        "spacetime_dim": 16,
        "num_heads": 4,
        "num_layers": 1,
        "encoder_dropout": 0.0,
    }
    defaults.update(kwargs)
    return FlashVarlenNexus2KTopologyAESystem(**defaults)


def fake_flash(qkv, _cu, _maximum, **_kwargs):
    return qkv[:, 0]


def math_flash(qkv, cu, _maximum, *, dropout_p, softmax_scale, causal):
    """CPU math substitute, not a test of the installed CUDA Flash kernel."""
    assert qkv.dtype == torch.bfloat16
    assert dropout_p == 0.0 and softmax_scale is None and not causal
    rows = []
    for start, end in zip(cu.tolist()[:-1], cu.tolist()[1:]):
        q, k, v = qkv[start:end].float().unbind(dim=1)
        q, k, v = (value.transpose(0, 1) for value in (q, k, v))
        scores = (q @ k.transpose(-2, -1)) / q.shape[-1] ** 0.5
        rows.append((scores.softmax(dim=-1) @ v).transpose(0, 1))
    return torch.cat(rows).to(dtype=torch.bfloat16)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_flash_cuda)])
def test_full_flash_path_matches_serial_layernorm_and_isolates_meshes(monkeypatch, device):
    if device == "cpu":
        monkeypatch.setattr(flash_backend, "flash_attn_varlen_qkvpacked_func", math_flash)
    torch.manual_seed(71)
    model = small_flash_system().autoencoder.to(device).eval()
    with torch.no_grad():
        for norm in (model.encoder_output_norm, model.decoder_output_norm):
            norm.weight.copy_(torch.linspace(0.7, 1.3, 32, device=device))
            norm.bias.copy_(torch.linspace(-0.2, 0.2, 32, device=device))
    batch = two_mesh_batch()
    vertices = batch.vertices.to(device)
    mask = batch.vertex_mask.to(device)
    faces = tuple(value.to(device) for value in batch.faces)
    incidence = tuple(value.to(device) for value in batch.incidence_index)

    def forward(values):
        return flash_backend._flash_varlen_autoencoder_forward(
            model, values, mask, faces, incidence, sample_seeds=(17, 23)
        )

    expected = [model(vertices[index, valid], face, inc)[1:]
                for index, (valid, face, inc) in enumerate(zip(mask, faces, incidence))]
    actual = forward(vertices)
    for index, outputs in enumerate(expected):
        for output, rows in zip(outputs, actual):
            # Retain BF16 QKV/output rounding; bitwise equality is not expected.
            torch.testing.assert_close(rows[index], output, atol=2e-2, rtol=2e-2)

    changed_vertices = vertices.clone()
    changed_vertices[1] = changed_vertices[1] * -7 + 3
    changed_vertices[~mask] = 1e6
    changed = forward(changed_vertices)
    for before_rows, after_rows in zip(actual, changed):
        torch.testing.assert_close(after_rows[0], before_rows[0], atol=1e-6, rtol=1e-6)

    sum(row.square().mean() for rows in actual for row in rows).backward()
    for norm in (model.encoder_output_norm, model.decoder_output_norm):
        for parameter in norm.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(norm.weight.grad) > 0


def test_flash_system_adds_no_trainable_parameters():
    kwargs = {
        "hidden_dim": 32,
        "latent_dim": 8,
        "spacetime_dim": 16,
        "num_heads": 4,
        "num_layers": 1,
        "encoder_dropout": 0.0,
    }
    baseline = Nexus2KTopologyAESystem(**kwargs)
    experiment = FlashVarlenNexus2KTopologyAESystem(**kwargs)
    baseline_schema = {
        name: tuple(parameter.shape) for name, parameter in baseline.named_parameters()
    }
    experiment_schema = {
        name: tuple(parameter.shape)
        for name, parameter in experiment.named_parameters()
    }
    assert experiment_schema == baseline_schema


def test_flash_attention_uses_an_explicit_bf16_kernel_island():
    """QKV/output projections stay FP32 around the required BF16 kernel I/O."""

    observed: dict[str, torch.dtype] = {}
    original_flash = flash_backend.flash_attn_varlen_qkvpacked_func

    def fake_flash(qkv, _cu, _maximum, **_kwargs):
        observed["kernel_input"] = qkv.dtype
        output = qkv[:, 0]
        observed["kernel_output"] = output.dtype
        return output

    flash_backend.flash_attn_varlen_qkvpacked_func = fake_flash
    try:
        attention = nn.MultiheadAttention(16, 4, batch_first=True, dropout=0.0).to(
            dtype=torch.float32
        )
        tokens = torch.randn((5, 16), dtype=torch.float32, requires_grad=True)
        output = _flash_self_attention(
            attention,
            tokens,
            _cu_seqlens([3, 2], tokens.device),
            3,
        )
        output.square().mean().backward()
    finally:
        flash_backend.flash_attn_varlen_qkvpacked_func = original_flash

    assert observed == {
        "kernel_input": torch.bfloat16,
        "kernel_output": torch.bfloat16,
    }
    assert output.dtype == torch.float32
    assert tokens.grad is not None and tokens.grad.dtype == torch.float32
    assert all(parameter.dtype == torch.float32 for parameter in attention.parameters())
    assert all(
        parameter.grad is None or parameter.grad.dtype == torch.float32
        for parameter in attention.parameters()
    )


def test_flash_system_keeps_every_non_flash_module_in_fp32_on_cpu():
    """A fake kernel makes the complete selective-precision boundary CPU-testable."""

    observed_flash_dtypes = []
    observed_loss_dtypes: dict[str, torch.dtype] = {}
    observed_embeddings: dict[str, torch.Tensor] = {}
    observed_module_dtypes = []
    original_flash = flash_backend.flash_attn_varlen_qkvpacked_func

    def fake_flash(qkv, _cu, _maximum, **_kwargs):
        observed_flash_dtypes.append(qkv.dtype)
        return qkv[:, 0]

    flash_backend.flash_attn_varlen_qkvpacked_func = fake_flash
    system = FlashVarlenNexus2KTopologyAESystem(
        hidden_dim=32,
        latent_dim=8,
        spacetime_dim=16,
        num_heads=4,
        num_layers=1,
        encoder_dropout=0.0,
    ).to(dtype=torch.float32)
    system.train()
    optimizer = torch.optim.Adam(system.parameters(), lr=1e-4)
    vertices = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
    source, target = build_vertex_face_incidence(faces, 3)
    batch = SimpleNamespace(
        uids=("triangle",),
        vertices=vertices,
        vertex_mask=torch.ones((1, 3), dtype=torch.bool),
        faces=(faces,),
        incidence_index=(torch.stack((source, target)),),
        edge_index=(torch.tensor([[0, 0, 1], [1, 2, 2]]),),
        face_set=(faces,),
    )

    def probe_loss(self, **kwargs):
        for name in ("mu", "log_variance", "edge_embedding", "face_embedding"):
            observed_loss_dtypes[name] = kwargs[name].dtype
        for name in ("edge_embedding", "face_embedding"):
            observed_embeddings[name] = kwargs[name].detach()
        loss = sum(kwargs[name].square().mean() for name in observed_loss_dtypes)
        return loss, {"probe": loss.detach()}

    def probe_module(_module, inputs, output):
        tensors = [value for value in inputs if torch.is_tensor(value)]
        if torch.is_tensor(output):
            tensors.append(output)
        for value in tensors:
            if value.is_floating_point():
                observed_module_dtypes.append(value.dtype)

    system._loss_with_face_negatives = MethodType(probe_loss, system)
    handles = [
        module.register_forward_hook(probe_module)
        for module in system.modules()
        if isinstance(module, (nn.Linear, nn.LayerNorm))
    ]
    try:
        loss = system(
            batch,
            negative_seed=1,
            packed=True,
            sample_seeds=(2,),
        )
        loss.backward()
        optimizer.step()
    finally:
        for handle in handles:
            handle.remove()
        flash_backend.flash_attn_varlen_qkvpacked_func = original_flash

    expected_flash_calls = len(system.autoencoder.encoder_blocks) + len(
        system.autoencoder.decoder_blocks
    )
    assert observed_flash_dtypes == [torch.bfloat16] * expected_flash_calls
    assert set(observed_module_dtypes) == {torch.float32}
    assert set(observed_loss_dtypes.values()) == {torch.float32}
    for embedding in observed_embeddings.values():
        torch.testing.assert_close(
            embedding.mean(dim=0), torch.zeros(16), atol=1e-6, rtol=0.0
        )
    for norm in (system.autoencoder.encoder_output_norm, system.autoencoder.decoder_output_norm):
        assert norm.weight.grad is not None
        assert torch.isfinite(norm.weight.grad).all()
        assert torch.count_nonzero(norm.weight.grad) > 0
    assert loss.dtype == torch.float32
    assert all(parameter.dtype == torch.float32 for parameter in system.parameters())
    assert all(
        parameter.grad is None or parameter.grad.dtype == torch.float32
        for parameter in system.parameters()
    )
    assert all(
        not torch.is_tensor(value)
        or not value.is_floating_point()
        or value.dtype == torch.float32
        for state in optimizer.state.values()
        for value in state.values()
    )


def test_flash_centers_each_mesh_without_rescaling(monkeypatch):
    """Centering must preserve each head's scale and keep objects isolated."""

    monkeypatch.setattr(flash_backend, "flash_attn_varlen_qkvpacked_func", fake_flash)
    system = small_flash_system().eval()
    batch = two_mesh_batch()
    feature_pattern = torch.linspace(0.5, 1.5, 16)

    def controlled_rows(first_scale, second_scale):
        first = first_scale * torch.arange(4).unsqueeze(1) * feature_pattern
        second = second_scale * torch.arange(5).unsqueeze(1) * feature_pattern
        return torch.cat((first, second), dim=0)

    def replace_output(values):
        def hook(_module, _inputs, output):
            return values.to(output)

        return hook

    raw_rows = {
        "edge_embedding": controlled_rows(1.0, 100.0),
        "face_embedding": controlled_rows(50.0, 0.5),
    }
    handles = (
        system.autoencoder.edge_embedding.register_forward_hook(
            replace_output(raw_rows["edge_embedding"])
        ),
        system.autoencoder.face_embedding.register_forward_hook(
            replace_output(raw_rows["face_embedding"])
        ),
    )
    observed = {"edge_embedding": [], "face_embedding": []}

    def probe_loss(self, **kwargs):
        for name in observed:
            observed[name].append(kwargs[name].detach().clone())
        loss = kwargs["mu"].square().mean()
        return loss, {"probe": loss.detach()}

    system._loss_with_face_negatives = MethodType(probe_loss, system)
    try:
        loss = system(
            batch,
            negative_seed=7,
            packed=True,
            sample_seeds=(11, 12),
        )
    finally:
        for handle in handles:
            handle.remove()

    assert torch.isfinite(loss)
    for name, rows in observed.items():
        assert [len(row) for row in rows] == [4, 5]
        for row, raw in zip(rows, raw_rows[name].split((4, 5))):
            torch.testing.assert_close(
                row.mean(dim=0), torch.zeros(16), atol=2e-5, rtol=0.0
            )
            torch.testing.assert_close(
                row, raw - raw.mean(dim=0, keepdim=True), atol=2e-5, rtol=0.0
            )
            assert not torch.isclose(row.square().mean(), torch.tensor(1.0))


def test_flash_calibration_uses_moments_without_training_loss(monkeypatch):
    monkeypatch.setattr(flash_backend, "flash_attn_varlen_qkvpacked_func", fake_flash)
    candidate_store = FixedCandidateStore()
    system = small_flash_system(
        negative_candidate_store=candidate_store,
        fixed_overfit_face_negatives=True,
    ).train()

    def fail_if_training_loss_runs(self, **_kwargs):
        raise AssertionError("calibration must not compute the training loss")

    system._loss_with_face_negatives = MethodType(fail_if_training_loss_runs, system)
    moments = system.calibration_interval_moment_sums(
        two_mesh_batch(),
        negative_seed=13,
        fixed_negatives=True,
        sample_seeds=(17, 19),
    )

    assert moments.dtype == torch.float64
    assert moments.shape == (3,)
    assert torch.isfinite(moments).all()
    assert moments[2].item() == 2
    assert candidate_store.calls == [("four", 13), ("five", 13)]


def test_flash_loss_passes_fixed_scoring_controls_to_reconstruction(monkeypatch):
    monkeypatch.setattr(flash_backend, "flash_attn_varlen_qkvpacked_func", fake_flash)
    observed = []

    def probe_loss(*values, **kwargs):
        observed.append(
            {
                "edge_logit_scale": kwargs["edge_logit_scale"],
                "face_logit_scale": kwargs["face_logit_scale"],
                "face_interval_factor": kwargs["face_interval_factor"],
            }
        )
        loss = values[5].square().mean()
        return loss, {"probe": loss.detach()}

    monkeypatch.setattr(
        training_backend, "topology_autoencoder_loss_with_face_negatives", probe_loss
    )
    system = small_flash_system(
        negative_candidate_store=FixedCandidateStore(),
        fixed_overfit_face_negatives=True,
        edge_logit_scale=0.5,
        face_logit_scale=0.125,
        face_interval_factor=0.25,
    ).eval()
    loss = system(
        two_mesh_batch(),
        negative_seed=23,
        fixed_negatives=True,
        packed=True,
        sample_seeds=(29, 31),
    )

    assert torch.isfinite(loss)
    assert observed == [
        {
            "edge_logit_scale": 0.5,
            "face_logit_scale": 0.125,
            "face_interval_factor": 0.25,
        },
        {
            "edge_logit_scale": 0.5,
            "face_logit_scale": 0.125,
            "face_interval_factor": 0.25,
        },
    ]


@requires_flash_cuda
def test_flash_varlen_attention_matches_independent_mha_sequences_and_backpropagates():
    """The packed kernel must neither mix meshes nor break finite gradients."""

    torch.manual_seed(7)
    device = torch.device("cuda")
    attention = nn.MultiheadAttention(32, 4, batch_first=True, dropout=0.0).to(device)
    attention.eval()
    lengths = [5, 3]
    tokens = torch.randn((sum(lengths), 32), device=device, requires_grad=True)

    expected_rows = []
    offset = 0
    for length in lengths:
        row = tokens[offset : offset + length].unsqueeze(0)
        expected, _ = attention(row, row, row, need_weights=False)
        expected_rows.append(expected.squeeze(0))
        offset += length
    expected = torch.cat(expected_rows)
    actual = _flash_self_attention(
        attention,
        tokens,
        _cu_seqlens(lengths, device),
        max(lengths),
    )

    # The algorithms are mathematically the same, but BF16 kernels use different
    # reduction orders.  This tolerance checks for implementation mistakes rather
    # than requiring bitwise equality between two CUDA attention kernels.
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    assert actual.dtype == torch.float32
    actual.square().mean().backward()
    assert tokens.grad is not None
    assert tokens.grad.dtype == torch.float32
    assert torch.isfinite(tokens.grad).all()
    assert all(
        parameter.grad is None or parameter.grad.dtype == torch.float32
        for parameter in attention.parameters()
    )


@requires_flash_cuda
def test_flash_system_keeps_reconstruction_inputs_and_loss_in_fp32():
    """The BF16 Flash island must return to FP32 before reconstruction."""

    device = torch.device("cuda")
    system = FlashVarlenNexus2KTopologyAESystem(
        hidden_dim=32,
        latent_dim=8,
        spacetime_dim=16,
        num_heads=4,
        num_layers=1,
        encoder_dropout=0.0,
    ).to(device)
    system.train()
    vertices = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        device=device,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)
    source, target = build_vertex_face_incidence(faces, 3)
    batch = SimpleNamespace(
        uids=("triangle",),
        vertices=vertices,
        vertex_mask=torch.ones((1, 3), dtype=torch.bool, device=device),
        faces=(faces,),
        incidence_index=(torch.stack((source, target)),),
        edge_index=(torch.tensor([[0, 0, 1], [1, 2, 2]], device=device),),
        face_set=(faces,),
    )
    observed: dict[str, torch.dtype] = {}

    def probe_loss(self, **kwargs):
        for name in ("mu", "log_variance", "edge_embedding", "face_embedding"):
            observed[name] = kwargs[name].dtype
        loss = sum(kwargs[name].square().mean() for name in observed)
        return loss, {"probe": loss.detach()}

    system._loss_with_face_negatives = MethodType(probe_loss, system)
    loss = system(
        batch,
        negative_seed=1,
        packed=True,
        sample_seeds=(2,),
    )

    assert set(observed.values()) == {torch.float32}
    assert loss.dtype == torch.float32
    loss.backward()
    assert all(
        parameter.grad is None
        or (
            parameter.grad.dtype == torch.float32
            and torch.isfinite(parameter.grad).all()
        )
        for parameter in system.parameters()
    )
