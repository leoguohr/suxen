import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from scripts.check_vertex_sampler import GTOracle, check_cells, exact_cells
from test_data_2k import _write_manifest, _write_sample


def test_depth9_oracle_checks_asymmetric_coordinates_and_boundaries():
    cells = torch.tensor([[0, 0, 0], [511, 511, 511], [1, 2, 4], [7, 33, 129]])
    report = check_cells(cells, name='asymmetric', seeds=(17, 29), steps_list=(1, 20))
    assert report['passed'] and report['oracle_only_not_model_score']
    assert report['level_sampler_checks'] == 36
    assert report['continuous_max_abs_error'] < 1e-5
    assert all(r['exact_coordinate_set'] for r in report['generation'])


def test_oracle_rejects_singular_endpoint():
    oracle = GTOracle(torch.tensor([[1, 2, 4]]))
    with pytest.raises(ValueError, match='t < 1'):
        oracle(torch.zeros(1, 1, 8), torch.ones(1), torch.zeros(1, 1, 3, dtype=torch.long),
               torch.ones(1, dtype=torch.long), None)


def test_coordinate_gate_rejects_same_count_wrong_positions():
    cells = torch.tensor([[i, i * 2, i * 4] for i in range(8)])
    assert not exact_cells(cells + 1, cells)
    assert exact_cells(cells.flip(0), cells)


def test_oracle_catches_reversed_time_in_actual_generation_sampler(monkeypatch):
    import mini_nexus.vertex_evaluation as evaluation
    from mini_nexus.flow import euler_integrate

    def reversed_time(fn, initial, *, steps):
        return euler_integrate(lambda x, t: -fn(x, t), initial, steps=steps)

    monkeypatch.setattr(evaluation, 'euler_integrate', reversed_time)
    with pytest.raises(AssertionError):
        check_cells(torch.tensor([[1, 2, 4], [7, 33, 129]]), name='negative_control',
                    seeds=(17,), steps_list=(4,))


def test_oracle_catches_wrong_xyz_child_numbering(monkeypatch):
    import mini_nexus.vertex_evaluation as evaluation
    original = evaluation.expand_occupied_children
    monkeypatch.setattr(evaluation, 'expand_occupied_children',
                        lambda parents, occupancy: original(parents, occupancy)[:, [2, 1, 0]])
    with pytest.raises(AssertionError):
        check_cells(torch.tensor([[1, 2, 4], [7, 33, 129]]), name='wrong_bits',
                    seeds=(17,), steps_list=(4,))


def test_stage_a_cli_freezes_problem_and_failed_gate_blocks_b(tmp_path):
    cells = torch.tensor([[0, 0, 0], [0, 0, 511], [0, 511, 0], [511, 0, 0]])
    rows = [_write_sample(tmp_path, 'fixture', 'train', cells)]
    manifest, output = tmp_path / 'manifest.csv', tmp_path / 'run'
    _write_manifest(manifest, rows)
    script = Path(__file__).resolve().parents[1] / 'scripts/train_vertex_staged.py'
    result = subprocess.run([sys.executable, str(script), '--manifest', str(manifest),
        '--output', str(output), '--uids', 'fixture', '--last-stage', 'C', '--smoke-model',
        '--device', 'cpu', '--precision', 'fp32', '--stage-steps', '3', '3', '3', '3',
        '--eval-every', '3'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    records = [json.loads(s) for s in (output / 'A/train.jsonl').read_text().splitlines()]
    assert len(records) == 3
    assert len({r['noise_sha256'] for r in records}) == 1
    assert {r['time'] for r in records} == {.5}
    assert {r['depth'] for r in records} == {9}
    assert {r['weight_decay'] for r in records} == {0.}
    assert json.loads((output / 'oracle_before_training.json').read_text())['passed']
    assert json.loads((output / 'status.json').read_text())['state'] == 'gate_failed'
    assert not (output / 'B').exists()
