#!/usr/bin/env python3
"""Compare single-GPU and DDP checkpoints after the same optimizer step."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mini_nexus.topology_checkpoint import load_topology_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("single", type=Path)
    parser.add_argument("ddp", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    single = load_topology_checkpoint(args.single)
    ddp = load_topology_checkpoint(args.ddp)
    if single["step"] != ddp["step"]:
        raise ValueError("checkpoints must come from the same optimizer step")
    if single["manifest_sha256"] != ddp["manifest_sha256"]:
        raise ValueError("checkpoints use different manifests")
    if single["model"].keys() != ddp["model"].keys():
        raise ValueError("checkpoints contain different model parameters")

    maximum_absolute = 0.0
    maximum_relative = 0.0
    largest_parameter = ""
    total_elements = 0
    absolute_sum = 0.0
    squared_sum = 0.0
    thresholds = (1e-7, 1e-6, 1e-5, 1e-4)
    counts_above = {threshold: 0 for threshold in thresholds}
    for name, single_value in single["model"].items():
        ddp_value = ddp["model"][name]
        if not torch.is_tensor(single_value) or not torch.is_tensor(ddp_value):
            if single_value != ddp_value:
                raise ValueError(f"checkpoint metadata differs at model state {name}")
            continue
        differences = (single_value - ddp_value).abs().float()
        absolute = differences.max().item()
        scale = torch.maximum(single_value.abs(), ddp_value.abs()).max().item()
        relative = absolute / max(scale, 1e-12)
        total_elements += differences.numel()
        absolute_sum += differences.sum().item()
        squared_sum += (differences * differences).sum().item()
        for threshold in thresholds:
            counts_above[threshold] += int((differences > threshold).sum())
        if absolute > maximum_absolute:
            maximum_absolute = absolute
            largest_parameter = name
        maximum_relative = max(maximum_relative, relative)

    print(
        json.dumps(
            {
                "step": single["step"],
                "manifest_sha256": single["manifest_sha256"],
                "maximum_parameter_absolute_difference": maximum_absolute,
                "maximum_parameter_relative_difference": maximum_relative,
                "mean_parameter_absolute_difference": absolute_sum / total_elements,
                "rms_parameter_difference": (squared_sum / total_elements) ** 0.5,
                "parameter_elements": total_elements,
                "elements_above_absolute_threshold": {
                    str(threshold): count
                    for threshold, count in counts_above.items()
                },
                "largest_difference_parameter": largest_parameter,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
