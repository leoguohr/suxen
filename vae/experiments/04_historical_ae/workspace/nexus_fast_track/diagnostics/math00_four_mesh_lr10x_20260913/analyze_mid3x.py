"""Build the three-branch structural comparison without rerunning training."""
from pathlib import Path
import json
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
BRANCHES = ["control", "mid3x", "high10x"]
LABELS = {"control": "Control", "mid3x": "MidLR 3x", "high10x": "HighLR 10x"}
COLORS = {"control": "#3977b8", "mid3x": "#15976d", "high10x": "#b1458f"}
CHECKS = [0, 100, 200, 400]
FULL_CHECKS = [0, 1, 10, 20, 50, 100, 200, 300, 400]


def read(path):
    return json.loads(path.read_text())


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def noise_path(branch, step, summary=False):
    suffix = f"independent_{'summary_' if summary else ''}step{step:04d}.{('json' if summary else 'jsonl')}"
    if branch != "mid3x" and step == 100:
        return ROOT / "replay100" / branch / suffix
    return ROOT / branch / suffix


def structural_record(rec):
    faces = rec["face"]["tp"] + rec["face"]["fn"]
    missing = rec["gt_faces_missing_from_edge_candidates"]
    return {
        "uid": rec["uid"],
        "edge": {key: rec["edge"][key] for key in ["tp", "fp", "fn", "f1"]},
        "face": {key: rec["face"][key] for key in ["tp", "fp", "fn", "f1"]},
        "gt_faces_missing_from_edge_candidates": missing,
        "gt_face_candidate_coverage": (faces - missing) / faces,
    }


summary = {
    "question": "Can 3x retain acceleration while reducing 10x oscillation and old-sample degradation?",
    "checks": CHECKS,
    "verification": read(ROOT / "mid3x_verification.json"),
    "branches": {},
}

for branch in BRANCHES:
    branch_root = ROOT / branch
    updates = read_rows(branch_root / "updates.jsonl")
    reconstruction = [row["reconstruction_before_update"] for row in updates]
    differences = [right - left for left, right in zip(reconstruction, reconstruction[1:])]
    clips = [row["clip_coefficient"] for row in updates]
    gradients = [row["global_norm_preclip"] for row in updates]
    mu = {step: read(branch_root / f"mu_step{step:04d}.json") for step in CHECKS}
    noise_rows = {step: read_rows(noise_path(branch, step)) for step in CHECKS}
    noise_summaries = {step: read(noise_path(branch, step, summary=True)) for step in CHECKS}

    noise_structure = {}
    for step, rows in noise_rows.items():
        per_mesh = []
        for mesh_index in [2, 3]:
            face_total = rows[0]["rec"][mesh_index]["face"]["tp"] + rows[0]["rec"][mesh_index]["face"]["fn"]
            per_mesh.append({
                "uid": rows[0]["rec"][mesh_index]["uid"],
                "edge": {
                    key: statistics.fmean(row["rec"][mesh_index]["edge"][key] for row in rows)
                    for key in ["tp", "fp", "fn"]
                },
                "face": {
                    key: statistics.fmean(row["rec"][mesh_index]["face"][key] for row in rows)
                    for key in ["tp", "fp", "fn"]
                },
                "gt_face_candidate_coverage": statistics.fmean(
                    (face_total - row["rec"][mesh_index]["gt_faces_missing_from_edge_candidates"]) / face_total
                    for row in rows
                ),
            })
        noise_structure[str(step)] = per_mesh

    strict = {}
    for mode in ["mu", "sample"]:
        strict[mode] = {
            str(step): read(branch_root / f"{mode}_step{step:04d}.json")["old_meshes_perfect"]
            for step in FULL_CHECKS
        }

    summary["branches"][branch] = {
        "lr": read(branch_root / "manifest.json")["lr"],
        "mu_structure": {
            str(step): [structural_record(mu[step]["rec"][index]) for index in [2, 3]]
            for step in CHECKS
        },
        "noise_structure_mean_50": noise_structure,
        "noise_old_strict_perfect": {
            str(step): noise_summaries[step]["old_perfect"] for step in CHECKS
        },
        "noise_mean_soft4": {
            str(step): noise_summaries[step]["mean_soft4"] for step in CHECKS
        },
        "final_unseen_old_strict_perfect": read(branch_root / "final_unseen_summary_step0400.json")["old_perfect"],
        "old_strict_at_full_mu_and_fixed_checks": strict,
        "dynamics": {
            "reconstruction_first50_mean": statistics.fmean(reconstruction[:50]),
            "reconstruction_last50_mean": statistics.fmean(reconstruction[-50:]),
            "step_difference_std": statistics.pstdev(differences),
            "maximum_single_step_increase": max(differences),
            "steps_with_increase": sum(value > 0 for value in differences),
            "steps_with_increase_over_1e-4": sum(value > 1e-4 for value in differences),
            "gradient_norm_median": statistics.median(gradients),
            "gradient_norm_maximum": max(gradients),
            "clipped_steps": sum(value < 1 for value in clips),
            "clip_coefficient_median": statistics.median(clips),
            "median_relative_parameter_update": {
                group: statistics.median(row["actual_updates"][group]["relative_l2"] for row in updates)
                for group in updates[0]["actual_updates"]
            },
        },
    }

control_updates = summary["branches"]["control"]["dynamics"]["median_relative_parameter_update"]
mid_updates = summary["branches"]["mid3x"]["dynamics"]["median_relative_parameter_update"]
summary["mid3x_actual_update_ratio_vs_control"] = {
    group: mid_updates[group] / control_updates[group] for group in control_updates
}
summary["checkpoint_sha256"] = {
    "parent": "9874a273209137017a63521d7ef1ba1e3dabdeef5cc3f43c1cb59d044d92d35d",
    "control_step400": "83c70b93188c985525ec1a4343cedf92271ce06381aa5189e0ff1191081dd0a9",
    "mid3x_step400": "640a981738f96992ba5dd8a6f8a5b0d0306a32560af8edabc195f010d23ba8b1",
    "high10x_step400": "c814e2c878e48446b494fede8fde99b3d7d1edefee86c17c15711ea8c4ed7ced",
}
(ROOT / "mid3x_summary.json").write_text(json.dumps(summary, indent=2))

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for branch in BRANCHES:
    data = summary["branches"][branch]
    for mesh_index, linestyle in enumerate(["-", "--"]):
        label = f"{LABELS[branch]} {data['mu_structure']['0'][mesh_index]['uid'][-6:]}"
        axes[0, 0].plot(CHECKS, [data["mu_structure"][str(step)][mesh_index]["edge"]["tp"] for step in CHECKS], marker="o", linestyle=linestyle, color=COLORS[branch], label=label)
        axes[0, 1].plot(CHECKS, [data["mu_structure"][str(step)][mesh_index]["face"]["tp"] for step in CHECKS], marker="o", linestyle=linestyle, color=COLORS[branch], label=label)
        axes[0, 2].plot(CHECKS, [100 * data["mu_structure"][str(step)][mesh_index]["gt_face_candidate_coverage"] for step in CHECKS], marker="o", linestyle=linestyle, color=COLORS[branch], label=label)
    axes[1, 0].plot(CHECKS, [data["noise_old_strict_perfect"][str(step)] for step in CHECKS], marker="o", color=COLORS[branch], label=LABELS[branch])
    axes[1, 1].plot(CHECKS, [data["noise_mean_soft4"][str(step)] for step in CHECKS], marker="o", color=COLORS[branch], label=LABELS[branch])
    updates = read_rows(ROOT / branch / "updates.jsonl")
    axes[1, 2].plot([row["update"] for row in updates], [row["reconstruction_before_update"] for row in updates], color=COLORS[branch], linewidth=1, label=LABELS[branch])

for axis, title in zip(axes.flat, [
    "New meshes: Mu Edge TP", "New meshes: Mu Face TP", "GT-face candidate coverage",
    "Old pair strict / 50 noise groups", "Same-monitor mean Soft4", "Paired-epsilon training reconstruction",
]):
    axis.set_title(title)
    axis.set_xlabel("Additional updates")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7)
axes[0, 2].set_ylabel("percent")
axes[1, 0].set_ylim(48.5, 50.5)
fig.tight_layout()
fig.savefig(ROOT / "mid3x_comparison.png", dpi=170)
print(json.dumps(summary, indent=2))
