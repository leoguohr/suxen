"""Read-only verification for the added 3x LR branch."""
from pathlib import Path
import json
import math
import torch


ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent / "math00_four_mesh_20260913" / "checkpoint-update1000.pt"


def equal(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def read(path):
    return json.loads(path.read_text())


parent = torch.load(PARENT, map_location="cpu", weights_only=False, mmap=True)
start = torch.load(ROOT / "mid3x/checkpoint-update0000.pt", map_location="cpu", weights_only=False, mmap=True)
end = torch.load(ROOT / "mid3x/checkpoint-update0400.pt", map_location="cpu", weights_only=False, mmap=True)
assert equal(parent["model"], start["model"])
assert equal(parent["optimizer"]["state"], start["optimizer"]["state"])
assert equal(parent["training_rng_state"], start["training_rng_state"])
assert start["training_noise_draws"] == 5200

expected_lrs = {
    "encoder_mu": 3e-8,
    "decoder_body": 3e-7,
    "edge_head": 3e-7,
    "face_head": 3e-7,
    "logvar": 1e-4,
}
actual_lrs = {g["name"]: g["lr"] for g in start["optimizer"]["param_groups"]}
assert actual_lrs.keys() == expected_lrs.keys()
assert all(math.isclose(actual_lrs[name], value, rel_tol=1e-12) for name, value in expected_lrs.items())
assert end["completed_updates"] == 1400
assert end["additional_updates"] == 400
assert end["training_noise_draws"] == 6800

steps = {}
for group in end["optimizer"]["param_groups"]:
    steps[group["name"]] = sorted(
        set(float(end["optimizer"]["state"][i]["step"]) for i in group["params"])
    )
    assert steps[group["name"]] == ([2000.0] if group["name"] == "logvar" else [2200.0])

rows = {
    name: [json.loads(line) for line in (ROOT / name / "updates.jsonl").read_text().splitlines()]
    for name in ["control", "high10x", "mid3x"]
}
assert all(len(branch) == 400 for branch in rows.values())
for control, high, mid in zip(rows["control"], rows["high10x"], rows["mid3x"]):
    assert control["epsilon_sha256"] == high["epsilon_sha256"] == mid["epsilon_sha256"]
    assert control["rng_before_sha256"] == high["rng_before_sha256"] == mid["rng_before_sha256"]
    assert control["rng_after_sha256"] == high["rng_after_sha256"] == mid["rng_after_sha256"]
assert len({digest for row in rows["mid3x"] for digest in row["epsilon_sha256"]}) == 1600

manifest = read(ROOT / "mid3x/manifest.json")
complete = read(ROOT / "mid3x/complete.json")
assert manifest["noise_checks"] == [0, 100, 200, 400]
assert complete["training_epsilon_hashes"] == 1600
assert complete["all_groups_updated_every_step"]
assert complete["nonzero_sampling_every_step"]
assert all((ROOT / f"mid3x/independent_summary_step{step:04d}.json").is_file() for step in [0, 100, 200, 400])

result = {
    "parent_weights_exact": True,
    "parent_Adam_moments_exact": True,
    "parent_training_RNG_exact": True,
    "effective_lrs": expected_lrs,
    "paired_training_epsilon_exact_all_three_branches": True,
    "unique_training_epsilon_hashes": 1600,
    "noise_checks_present": [0, 100, 200, 400],
    "all_five_groups_updated_each_step": True,
    "nonzero_sampling_each_step": True,
    "final_Adam_steps": steps,
}
(ROOT / "mid3x_verification.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
