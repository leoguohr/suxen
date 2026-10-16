"""CPU gate. Run before any GPU job:  CUDA_VISIBLE_DEVICES= python -m pytest -q tests/test_cpu.py
(or `python tests/test_cpu.py`). Uses tiny models and synthetic meshes only; no server data needed."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import metrics  # noqa: E402
from data import Segments, TopologyData  # noqa: E402
from model import TINY, TopologyFlowSystem, transfer_vertex_weights  # noqa: E402
from sampling import generate  # noqa: E402
from train import EMA, lr_at, make_batch, object_groups, train_update  # noqa: E402
from vendor import vertex_s0  # noqa: E402

torch.set_num_threads(2)


def perturbed_system(seed=0):
    torch.manual_seed(seed)
    system = TopologyFlowSystem(TINY).float()
    with torch.no_grad():  # zero-initialized layers would make the comparisons trivial
        for p in system.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return system


def fan_mesh(n, rng):
    """Triangle fan around vertex 0 plus a strip: valid faces/edges for n >= 4."""
    faces = [(0, i, i + 1) for i in range(1, n - 1)]
    faces = np.array(faces, dtype=np.int64)
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [0, 2]], faces[:, [1, 2]]]), axis=1), axis=0)
    vertices = rng.uniform(-1, 1, size=(n, 3)).astype(np.float32)
    return vertices, faces, edges


def synthetic_data(lengths, latent=TINY.latent_dim, points=40, seed=0, device="cpu"):
    rng = np.random.default_rng(seed)
    records = []
    for i, n in enumerate(lengths):
        vertices, faces, edges = fan_mesh(n, rng)
        pts = np.concatenate([rng.uniform(-1, 1, (points, 3)), rng.normal(size=(points, 3))], 1).astype(np.float32)
        mask = np.ones(points, dtype=bool)
        mask[-3:] = i % 2 == 0  # some objects with padded/invalid points
        records.append(dict(uid=f"obj{i}", vertices=vertices, faces=faces, edges=edges, points=pts,
                            point_mask=mask, mu=rng.normal(size=(n, latent)).astype(np.float32),
                            logvar=np.full((n, latent), -6.0, dtype=np.float32)))
    data = TopologyData(records, torch.zeros(latent), torch.ones(latent), device)
    data.compute_fps(TINY.condition_tokens)
    return data


def test_vendor_files_are_byte_identical():
    sources = json.loads((ROOT / "vendor" / "SOURCES.json").read_text())["files"]
    for name, info in sources.items():
        assert hashlib.sha256((ROOT / "vendor" / name).read_bytes()).hexdigest() == info["sha256"], name


def test_packed_forward_matches_s0_block_forward():
    system = perturbed_system()
    lengths, context = [5, 9, 2], 6
    seg = Segments.build(lengths, "cpu")
    g = torch.Generator().manual_seed(1)
    tokens = sum(lengths)
    noisy = torch.randn(tokens, TINY.latent_dim, generator=g)
    vertices = torch.rand(tokens, 3, generator=g) * 2 - 1
    time = torch.rand(len(lengths), generator=g)
    condition = torch.randn(len(lengths), context, TINY.condition_dim, generator=g)
    packed = system.flow.forward_packed(noisy, time, vertices, seg, condition, attention="sdpa")
    width = max(lengths)
    pn = torch.zeros(len(lengths), width, TINY.latent_dim)
    pv = torch.zeros(len(lengths), width, 3)
    mask = torch.zeros(len(lengths), width, dtype=torch.bool)
    for i, (s, e) in enumerate(seg.bounds):
        pn[i, :e - s], pv[i, :e - s], mask[i, :e - s] = noisy[s:e], vertices[s:e], True
    padded = system.flow.forward_padded(pn, time, pv, condition, mask)
    for i, (s, e) in enumerate(seg.bounds):
        assert torch.allclose(packed[s:e], padded[i, :e - s], atol=1e-5, rtol=1e-5), i
    # Gradients agree too, with activation checkpointing on the packed path.
    system.zero_grad()
    out = system.flow.forward_packed(noisy, time, vertices, seg, condition, attention="sdpa", use_checkpoint=True)
    out.square().sum().backward()
    packed_grads = {n: p.grad.clone() for n, p in system.named_parameters() if p.grad is not None}
    system.zero_grad()
    out = system.flow.forward_padded(pn, time, pv, condition, mask)
    (out.square() * mask.unsqueeze(-1)).sum().backward()
    for name, p in system.named_parameters():
        if p.grad is not None:
            assert torch.allclose(packed_grads[name], p.grad, atol=1e-4, rtol=1e-4), name


def test_condition_encoder_with_cached_fps_matches_forward():
    system = perturbed_system()
    data = synthetic_data([6, 7, 5])
    direct = system.condition_encoder(data.points, data.point_valid)
    cached = system.encode_condition(data.points, data.point_valid, data.fps_indices)
    assert torch.allclose(direct, cached, atol=1e-6)
    checkpointed = system.encode_condition(data.points, data.point_valid, data.fps_indices, use_checkpoint=True)
    assert torch.allclose(direct, checkpointed, atol=1e-6)


def test_triangles_match_brute_force():
    rng = np.random.default_rng(3)
    for n, p in [(1, .5), (2, .5), (8, .3), (12, .5), (15, .9)]:
        upper = np.triu(rng.random((n, n)) < p, 1)
        tri, overflow = metrics.triangles(upper)
        brute = {(i, j, k) for i, j, k in itertools.combinations(range(n), 3)
                 if upper[i, j] and upper[i, k] and upper[j, k]}
        assert not overflow and {tuple(t) for t in tri.tolist()} == brute and len(tri) == len(brute)
    upper = np.triu(np.ones((10, 10), dtype=bool), 1)
    assert metrics.triangles(upper, cap=5)[1]


def test_indicator_formulas():
    emb = torch.tensor([[1., 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]])
    assert torch.allclose(metrics.edge_logits(emb, torch.tensor([[0, 1]])), torch.tensor([metrics.EDGE_SCALE]))
    emb2 = torch.tensor([[0., 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]])
    expected = metrics.FACE_SCALE * metrics.FACE_FACTOR * 1.0
    assert torch.allclose(metrics.face_logits(emb2, torch.tensor([[0, 1, 2]])), torch.tensor([expected]))
    emb3 = torch.tensor([[0., 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])  # pure time area -> negative
    assert float(metrics.face_logits(emb3, torch.tensor([[0, 1, 2]]))) < 0


def test_score_mesh_counts_by_hand(monkeypatch=None):
    gt_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    gt_edges = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [2, 3]], dtype=np.int64)
    predicted_edges = {(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)}       # misses (2,3), adds (1,3)
    positive_faces = {(0, 1, 2), (0, 1, 3)}
    old_edge, old_face = metrics.edge_logits, metrics.face_logits
    metrics.edge_logits = lambda emb, pairs: torch.tensor(
        [1.0 if tuple(p) in predicted_edges else -1.0 for p in pairs.tolist()])
    metrics.face_logits = lambda emb, ids: torch.tensor(
        [1.0 if tuple(t) in positive_faces else -1.0 for t in ids.tolist()])
    try:
        row = metrics.score_mesh(torch.zeros(5, 4), torch.zeros(5, 4), gt_edges, gt_faces)
    finally:
        metrics.edge_logits, metrics.face_logits = old_edge, old_face
    assert row["edge"] == dict(tp=4, fp=1, fn=1, tn=4, micro_f1=8 / 10)
    assert row["candidates"] == 2 and row["face_fn_missing"] == 1 and row["face_fn_present"] == 0
    assert {k: row["face"][k] for k in ("tp", "fp", "fn", "tn")} == dict(tp=1, fp=1, fn=1, tn=0)
    assert not row["joint_strict"]


def test_lr_schedule():
    args = SimpleNamespace(lr=1e-4, min_lr=0.0, warmup=500, max_updates=10000, decay_fraction=0.2)
    assert abs(lr_at(args, 1) - 2e-7) < 1e-15 and lr_at(args, 500) == 1e-4 and lr_at(args, 8000) == 1e-4
    assert abs(lr_at(args, 9000) - 5e-5) < 1e-12 and lr_at(args, 10000) == 0.0


def test_batch_is_stratified_and_deterministic():
    data = synthetic_data([4, 5, 6, 4, 7, 5, 4])
    t, noisy, target = make_batch(data, seed=7, update=3)
    strata = sorted(int(x) for x in (t.double() * data.count + 1e-5).floor())
    assert strata == list(range(data.count))
    t2, noisy2, target2 = make_batch(data, seed=7, update=3)
    assert torch.equal(t, t2) and torch.equal(noisy, noisy2) and torch.equal(target, target2)
    assert not torch.equal(make_batch(data, seed=7, update=4)[1], noisy)


def args_for_test(**kw):
    base = dict(seed=5, target="posterior", attention="sdpa", checkpoint_activations=1, clip=1e9, lr=0.1,
                min_lr=0.0, warmup=1, max_updates=100, decay_fraction=0.0)
    base.update(kw)
    return SimpleNamespace(**base)


def test_chunked_update_equals_single_group():
    data = synthetic_data([5, 9, 4, 6])
    results = []
    for chunk in (0, 10):
        system = perturbed_system(1)
        optimizer = torch.optim.SGD(system.parameters(), lr=0.1)
        train_update(system, optimizer, data, object_groups(data, chunk), args_for_test(), 1)
        results.append(torch.cat([p.detach().flatten() for p in system.parameters()]))
    assert len(object_groups(data, 10)) > 1
    assert torch.allclose(results[0], results[1], atol=1e-6)


def test_resume_is_exact():
    data = synthetic_data([5, 9, 4])
    args = args_for_test(lr=1e-3, clip=1.0, warmup=2)
    groups = object_groups(data, 0)

    def fresh():
        system = perturbed_system(2)
        return system, torch.optim.AdamW(system.parameters(), lr=1e-3, weight_decay=0.01), EMA(system, 0.9)

    system, opt, ema = fresh()
    for u in range(1, 5):
        train_update(system, opt, data, groups, args, u)
        ema.update()
    straight = (copy.deepcopy(system.state_dict()), ema.state(system))
    system, opt, ema = fresh()
    for u in range(1, 3):
        train_update(system, opt, data, groups, args, u)
        ema.update()
    saved = (copy.deepcopy(system.state_dict()), copy.deepcopy(opt.state_dict()), ema.state(system))
    system, opt, ema = fresh()
    system.load_state_dict(saved[0])
    opt.load_state_dict(saved[1])
    ema.load(system, saved[2])
    for u in range(3, 5):
        train_update(system, opt, data, groups, args, u)
        ema.update()
    for name, tensor in straight[0].items():
        assert torch.equal(tensor, system.state_dict()[name]), name
    for name, tensor in straight[1].items():
        assert torch.equal(tensor, ema.state(system)[name]), name


def test_vertex_warm_start_mapping():
    class S0Like(nn.Module):
        def __init__(self):
            super().__init__()
            self.condition_encoder = vertex_s0.VertexConditionEncoder(
                input_dim=6, hidden_dim=TINY.condition_dim, num_tokens=TINY.condition_tokens,
                num_heads=TINY.condition_heads, num_layers=TINY.condition_layers)
            self.flow = vertex_s0.VertexDiT(hidden_dim=TINY.hidden_dim, condition_dim=TINY.condition_dim,
                                            num_layers=TINY.num_layers, num_heads=TINY.num_heads)
    torch.manual_seed(11)
    s0 = S0Like()
    with torch.no_grad():
        for p in s0.parameters():
            p.add_(torch.randn_like(p))
    target = perturbed_system(3)
    before = {k: v.clone() for k, v in target.state_dict().items()}
    report = transfer_vertex_weights(target, s0.state_dict())
    after = target.state_dict()
    s0_state = s0.state_dict()
    for name in after:
        if name.startswith(("flow.blocks.", "condition_encoder.", "flow.time_embedding.", "flow.position_embedding.")):
            assert torch.equal(after[name], s0_state[name]), name
        else:
            assert torch.equal(after[name], before[name]), name
    assert "flow.depth_embedding.weight" in report["s0_unused"] and "flow.data_embedding.weight" in report["s0_unused"]


def test_generation_is_independent_of_grouping():
    system = perturbed_system(4).eval()
    data = synthetic_data([5, 9, 4, 6])
    one = generate(system, data, seed=3, steps=4, dtype="fp32", attention="sdpa", max_tokens=10**6)
    split = generate(system, data, seed=3, steps=4, dtype="fp32", attention="sdpa", max_tokens=10)
    assert torch.allclose(one, split, atol=1e-5)
    dpm = generate(system, data, seed=3, steps=4, sampler="dpm2m", dtype="fp32", attention="sdpa")
    assert dpm.shape == one.shape and torch.isfinite(dpm).all()


def test_decode_and_score_with_a_tiny_vae():
    from vendor.ownae_v2_native_models import Config, NativeTopologyAE
    torch.manual_seed(5)
    cfg = Config(model_variant="B_v2_teacher_blocks", encoder_width=16, latent_width=512, decoder_width=16,
                 encoder_composite_blocks=1, decoder_blocks=1, heads=2)
    vae = NativeTopologyAE(cfg).float().eval().requires_grad_(False)
    data = synthetic_data([6, 8], latent=512)
    rows = metrics.evaluate_latents(vae, data, data.mu, normalized=False)
    summary = metrics.aggregate(rows)
    assert summary["objects"] == 2 and summary["face"]["tp"] + summary["face"]["fn"] == sum(len(f) for f in data.gt_faces)
    assert all(abs(r["latent_rmse"]) < 1e-6 for r in rows)  # decoding mu itself: zero latent error
    # The decoder half must equal the native model's own forward on the same latent.
    vertices = torch.from_numpy(data.records[0]["vertices"])
    faces = torch.from_numpy(data.records[0]["faces"])
    with torch.no_grad():
        native = vae(vertices, faces)
        edge, face = metrics.decode(vae, native["mu"])
    assert torch.allclose(edge, native["edge"], atol=1e-6) and torch.allclose(face, native["face"], atol=1e-6)


def test_cascade_matching_mapping_and_scoring():
    import cascade
    gt = np.array([[10, 10, 10], [11, 10, 10], [10, 11, 10], [11, 11, 10], [12, 11, 10]], dtype=np.int64)
    gt_faces = np.array([[0, 1, 2], [0, 2, 3], [2, 3, 4]], dtype=np.int64)
    gt_edges = np.unique(np.sort(np.concatenate([gt_faces[:, [0, 1]], gt_faces[:, [0, 2]], gt_faces[:, [1, 2]]]),
                                 axis=1), axis=0)
    shifted = gt[4] + np.array([1, 0, 0])
    pred = np.stack([gt[3], gt[0], shifted, gt[1], gt[2], np.array([40, 40, 40])])
    strict = cascade.match_vertices(pred, gt, one_cell=False)
    loose = cascade.match_vertices(pred, gt, one_cell=True)
    assert strict.tolist() == [3, 0, -1, 1, 2, -1] and loose.tolist() == [3, 0, 4, 1, 2, -1]
    edges, faces, lost_e, lost_f = cascade.map_gt_to_pred(strict, len(gt), gt_edges, gt_faces)
    assert {tuple(f) for f in faces.tolist()} == {(1, 3, 4), (0, 1, 4)} and lost_f == 1 and lost_e == 2
    perfect_edges, perfect_faces, _, _ = cascade.map_gt_to_pred(loose, len(gt), gt_edges, gt_faces)
    assert len(perfect_faces) == 3 and len(perfect_edges) == len(gt_edges)
    # A decoder that predicts exactly the one-cell-mapped mesh on the 6 predicted vertices.
    predicted_edges = {tuple(e) for e in perfect_edges.tolist()}
    predicted_faces = {tuple(f) for f in perfect_faces.tolist()}
    old_edge, old_face = metrics.edge_logits, metrics.face_logits
    metrics.edge_logits = lambda emb, pairs: torch.tensor([1.0 if tuple(p) in predicted_edges else -1.0 for p in pairs.tolist()])
    metrics.face_logits = lambda emb, ids: torch.tensor([1.0 if tuple(t) in predicted_faces else -1.0 for t in ids.tolist()])
    try:
        emb = torch.zeros(len(pred), 4)
        loose_row = cascade.score_tree(emb, emb, loose, len(gt), gt_edges, gt_faces)
        strict_row = cascade.score_tree(emb, emb, strict, len(gt), gt_edges, gt_faces)
    finally:
        metrics.edge_logits, metrics.face_logits = old_edge, old_face
    assert loose_row["joint_strict"] and loose_row["face"]["tp"] == 3 and loose_row["edge"]["tp"] == len(gt_edges)
    # Strict matching: the shifted vertex is unmatched, so its 2 edges / 1 face are both false and missed.
    assert (strict_row["edge"]["tp"], strict_row["edge"]["fp"], strict_row["edge"]["fn"]) == (len(gt_edges) - 2, 2, 2)
    assert (strict_row["face"]["tp"], strict_row["face"]["fp"], strict_row["face"]["fn"]) == (2, 1, 1)
    assert cascade.xyz_to_cells(cascade.cells_to_xyz(gt)).tolist() == gt.tolist()
    failed = cascade.failed_tree(gt_edges, gt_faces)
    summary = cascade.aggregate_trees([dict(loose_row, vertex_set_exact=False, vertex_seed=1),
                                       dict(failed, vertex_set_exact=False, vertex_seed=2)])
    assert summary["all"]["face"]["fn"] == 3 and summary["all"]["trees"] == 2 and summary["exact_vertex_trees"]["trees"] == 0


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}", flush=True)
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(error).__name__}: {error}", flush=True)
    print("CPU_GATE_OK" if failures == 0 else f"CPU_GATE_FAILED ({failures})")
    sys.exit(1 if failures else 0)
