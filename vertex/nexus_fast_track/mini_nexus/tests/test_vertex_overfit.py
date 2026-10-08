import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from mini_nexus.octree import build_octree_levels
from mini_nexus.vertex_evaluation import (
    cell_metrics, chamfer_mean_distance, denoise_level, generate_cells, trace_denoising,
)
from mini_nexus.training import VertexStageSystem
from scripts.train_vertex_overfit import pair_at_step
from test_data_2k import _write_manifest, _write_sample


class OracleFlow:
    max_depth = 2

    def __init__(self, cells):
        self.levels = build_octree_levels(cells, self.max_depth)

    def __call__(self, noisy, time, codes, depths, context):
        level = self.levels[int(depths[0]) - 1]
        matches = (codes[0, :, None] == level.parent_codes[None]).all(-1).float()
        target = (matches @ level.target).unsqueeze(0)
        return (target - noisy) / (1 - time[:, None, None])


class OracleModel:
    def __init__(self, cells):
        self.flow = OracleFlow(cells)


def test_pair_schedule_covers_every_uid_and_depth_without_aliasing():
    first = [pair_at_step(step, 20, 9, 17) for step in range(180)]
    second = [pair_at_step(step, 20, 9, 17) for step in range(180, 360)]
    expected = {(uid, depth) for uid in range(20) for depth in range(1, 10)}
    assert set(first) == set(second) == expected
    assert first != second


def test_oracle_denoising_and_octree_generation_recover_known_cells():
    cells = torch.tensor([[0, 0, 0], [1, 2, 3], [3, 3, 3]])
    model, context = OracleModel(cells), torch.zeros(1, 4, 8)
    for level in model.flow.levels:
        metrics = denoise_level(model, level, context, time=0.1, seed=17)
        assert metrics["f1"] == metrics["iou"] == 1
    recovered, report = generate_cells(model, context, steps=4, seed=17)
    assert report["status"] == "complete"
    assert cell_metrics(recovered, cells)["iou"] == 1


def test_generation_never_repairs_empty_or_exploding_predictions():
    class ConstantModel:
        def __init__(self, target):
            self.target = target
            self.flow = self
            self.max_depth = 2

        def __call__(self, noisy, time, codes, depths, context):
            return (self.target - noisy) / (1 - time[:, None, None])

    context = torch.zeros(1, 4, 8)
    empty, report = generate_cells(ConstantModel(0), context, steps=2, seed=1)
    assert empty.shape == (0, 3) and report["status"] == "empty"
    aborted, report = generate_cells(ConstantModel(1), context, steps=2, seed=1, max_parents=4)
    assert aborted is None and report["status"] == "capacity_abort"


def test_chamfer_is_in_normalized_coordinate_units():
    a, b = torch.tensor([[0, 0, 0]]), torch.tensor([[1, 0, 0]])
    assert chamfer_mean_distance(a, b, depth=2) == pytest.approx(0.5)
    assert chamfer_mean_distance(a[:0], b, depth=2) is None


def test_activation_trace_preserves_predictions_and_removes_hooks():
    model = VertexStageSystem(hidden_dim=24, condition_dim=32, condition_tokens=4,
                              num_layers=2, num_heads=3, condition_heads=4, max_depth=2).eval()
    context = torch.randn(1, 4, 32)
    level = build_octree_levels(torch.tensor([[0, 0, 0], [3, 3, 3]]), 2)[1]
    before = denoise_level(model, level, context, time=0.1, seed=1)
    trace = trace_denoising(model, level, context, time=0.1, seed=1)
    after = denoise_level(model, level, context, time=0.1, seed=1)
    assert before == after == trace["denoising"]
    assert trace["activations"]["depth_embedding"]["rms"] > 0
    assert set(trace["activations"]["first_modulation"]) == {
        "shift_sa", "scale_sa", "gate_sa", "shift_ff", "scale_ff", "gate_ff"}
    assert all(not module._forward_hooks for module in model.modules())


def test_overfit_cli_saves_resumes_and_rejects_changed_configuration(tmp_path):
    cells = torch.tensor([[0, 0, 0], [0, 0, 511], [0, 511, 0], [511, 0, 0]])
    rows = [_write_sample(tmp_path, f"sample_{index}", "train", cells) for index in range(2)]
    manifest, output = tmp_path / "manifest.csv", tmp_path / "run"
    _write_manifest(manifest, rows)
    script = Path(__file__).resolve().parents[1] / "scripts/train_vertex_overfit.py"
    base = [sys.executable, str(script), "--manifest", str(manifest), "--output", str(output),
            "--expected-samples", "2", "--device", "cpu", "--precision", "fp32",
            "--hidden-dim", "24", "--num-heads", "3", "--num-layers", "2",
            "--condition-dim", "32", "--condition-heads", "4", "--condition-tokens", "4",
            "--max-depth", "2", "--generation-steps", "2", "--eval-every", "1"]
    first = subprocess.run(base + ["--steps", "1"], capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    checkpoint = output / "checkpoint-last.pt"
    resumed = subprocess.run(base + ["--steps", "2", "--resume", str(checkpoint)], capture_output=True, text=True)
    assert resumed.returncode == 0, resumed.stderr
    assert json.loads((output / "status.json").read_text())["state"] == "complete"
    records = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert [row["step"] for row in records] == [1, 2]
    report = json.loads((output / "evaluation-000002.json").read_text())
    assert len(report["generation"]) == 2
    assert report["swapped_summary_by_time"]
    mismatch = subprocess.run(base + ["--steps", "3", "--resume", str(checkpoint), "--seed", "9"], capture_output=True, text=True)
    assert mismatch.returncode != 0 and "resume configuration mismatch: seed" in mismatch.stderr
