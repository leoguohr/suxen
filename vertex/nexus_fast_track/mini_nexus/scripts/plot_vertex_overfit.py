#!/usr/bin/env python3
"""Plot downloaded Vertex overfit evidence without treating aborted trees as leaves."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    train = [json.loads(line) for line in (root / "train.jsonl").read_text().splitlines()]
    evaluations = [json.loads(line) for line in (root / "evaluations.jsonl").read_text().splitlines()]
    final_step = evaluations[-1]["step"]
    final = json.loads((root / f"evaluation-{final_step:06d}.json").read_text())
    initial = json.loads((root / "evaluation-000000.json").read_text())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    steps, losses = np.array([r["step"] for r in train]), np.array([r["loss"] for r in train])
    ax = axes[0, 0]
    ax.plot(steps, losses, color="#98a5b8", alpha=.35, linewidth=.7)
    window = min(50, len(losses))
    ax.plot(steps[window-1:], np.convolve(losses, np.ones(window)/window, mode="valid"), color="#1c5677")
    ax.set(title="Loss stays finite but does not establish recovery", xlabel="Optimizer step", ylabel="Velocity MSE (training)")
    ax = axes[0, 1]
    for t, color in (("0.1", "#bd4a42"), ("0.5", "#277b86")):
        ax.plot([e["step"] for e in evaluations], [e["summary_by_time"][t]["f1"] for e in evaluations], marker="o", color=color, label=f"Denoising F1, t={t}")
    ax.scatter([final_step], [final["swapped_summary_by_time"]["0.1"]["f1"]], marker="x", s=120, color="black", label="Swapped condition, t=0.1", zorder=5)
    ax.set(title="Point-cloud swap has almost no effect", xlabel="Optimizer step", ylabel="Macro F1", ylim=(0, 1))
    ax.legend(fontsize=8)
    ax = axes[0, 2]
    for report, name, color in ((initial, "Initial", "#98a5b8"), (final, "Step 1000", "#bd4a42")):
        values = [np.mean([r["f1"] for r in report["per_object_depth_time"] if r["time"] == .1 and r["depth"] == d]) for d in range(1, 10)]
        ax.plot(range(1, 10), values, marker="o", label=name, color=color)
    ax.set(title="Deep-level recovery remains poor", xlabel="Target child depth", ylabel="Denoising F1, t=0.1", ylim=(0, 1))
    ax.legend()
    ax = axes[1, 0]
    for row in final["generation"]:
        ax.plot([r["depth"] for r in row["levels"]], [r["children"] for r in row["levels"]], alpha=.5, color="#bd4a42")
    ax.axhline(final["max_generated_parents"], color="black", linestyle="--", label="Capacity limit")
    ax.set(title="20 / 20 trees abort before depth 9", xlabel="Depth", ylabel="Predicted occupied children", yscale="log")
    ax.legend()
    ax = axes[1, 1]
    grads = [r for r in train if r.get("vecset_gradient_norm_after_clip", 0) > 0]
    for key, label in (("vecset_gradient_norm_after_clip", "VecSet"), ("dit_gradient_norm_after_clip", "DiT")):
        ax.plot([r["step"] for r in grads], [r[key] for r in grads], marker="o", label=label)
    ax.set(title="Condition encoder gradients are much smaller", xlabel="Optimizer step", ylabel="L2 gradient norm after clipping", yscale="log")
    ax.legend()
    ax = axes[1, 2]
    diagnostic = root / "post_diagnostics.json"
    if diagnostic.exists():
        trace = json.loads(diagnostic.read_text())["embedding_trace"]
        ax.plot([r["block"]+1 for r in trace["trace"]], [r["hidden_rms"] for r in trace["trace"]], color="#bd4a42")
        ax.axhline(trace["depth_embedding_rms"], linestyle="--", label="Depth embedding RMS")
        ax.set(title="Residual feature magnitudes grow strongly", xlabel="DiT block", ylabel="Hidden RMS (one diagnostic case)", yscale="log")
        ax.legend()
    fig.suptitle("Vertex overfit20: 1,000 steps completed, overfit check FAILED", fontsize=17)
    fig.savefig(root / "training_diagnostics.png", dpi=160)
    plt.close(fig)

    selected = [final["generation"][i] for i in (0, 10, 12)]
    fig = plt.figure(figsize=(12, 12))
    for index, row in enumerate(selected):
        data = np.load(root / f"generation-{final_step:06d}" / f"{row['uid']}.npz")
        target = -1 + 2 * (data["target"].astype(float) + .5) / 512
        predicted = -1 + 2 * (data["predicted"].astype(float) + .5) / 512
        for column, (points, title) in enumerate(((target, f"Target: {len(target)} vertices"), (predicted, "Model: no completed depth-9 output"))):
            ax = fig.add_subplot(3, 2, index*2+column+1, projection="3d")
            if column == 0 or row["status"] in ("complete", "empty"):
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, color="#277b86" if column == 0 else "#bd4a42", alpha=.8)
            else:
                ax.text2D(.5, .48, f"Capacity abort at depth {row['depth']}\n{row.get('child_count', row.get('parent_count')):,} occupied children\n\nPartial tree is NOT a final vertex set", transform=ax.transAxes, ha="center", color="#a13931", fontsize=11)
            ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1), title=f"{row['uid']}\n{title}")
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=20, azim=45)
    fig.suptitle("Target / prediction audit — representative small, medium and large objects", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(root / "target_prediction_audit.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
