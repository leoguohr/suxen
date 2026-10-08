"""Validated checkpoint loading for Topology-AE evaluation tools."""

from __future__ import annotations

from pathlib import Path

import torch

from .training_2k import Nexus2KTopologyAESystem


def _project_scoring_contract(
    metadata: dict[str, object], expected: dict[str, object]
) -> dict[str, object]:
    """Compare required scoring fields without inventing missing defaults."""

    if metadata.get("normalize_spacetime_embeddings", False):
        raise ValueError("RMS checkpoints require their immutable historical snapshot")
    if not expected.keys() <= metadata.keys():
        raise ValueError("checkpoint scoring metadata is incomplete")
    return {key: metadata[key] for key in expected}


def _resolve_system_name(
    saved_args: dict[str, object], recorded_runtime: dict[str, object] | None
) -> tuple[str, str]:
    """Validate the explicitly recorded backend; never guess historical models."""

    if not isinstance(recorded_runtime, dict):
        raise ValueError("checkpoint runtime metadata is required")
    precision = saved_args.get("precision")
    system_name = recorded_runtime.get("training_system")
    expected_precision = {
        "Nexus2KTopologyAESystem": "fp32",
        "FlashVarlenNexus2KTopologyAESystem": "fp32_flash_bf16",
    }.get(system_name)
    if expected_precision is None:
        raise ValueError(f"unsupported Topology AE training system: {system_name}")
    if precision is not None and precision != expected_precision:
        raise ValueError("checkpoint precision and runtime backend disagree")
    return system_name, "recorded_runtime"


def topology_runtime_resolution(checkpoint: dict[str, object]) -> str:
    """Confirm that evaluation uses an explicitly recorded backend."""

    saved_args = checkpoint.get("args")
    recorded_runtime = checkpoint.get("runtime")
    if not isinstance(saved_args, dict):
        raise ValueError("checkpoint must contain an args dictionary")
    if recorded_runtime is not None and not isinstance(recorded_runtime, dict):
        raise ValueError("checkpoint runtime metadata must be a dictionary")
    _, source = _resolve_system_name(saved_args, recorded_runtime)
    return source


def load_topology_checkpoint(path: Path) -> dict[str, object]:
    """Load a trusted local Topology-AE checkpoint."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Topology AE checkpoint must contain a dictionary")
    return checkpoint


def load_topology_system_from_checkpoint(
    checkpoint: dict[str, object],
    *,
    device: torch.device | str,
) -> Nexus2KTopologyAESystem:
    """Strictly rebuild the recorded backend and validate duplicated metadata."""

    if checkpoint.get("stage", "topology-ae") != "topology-ae":
        raise ValueError("checkpoint is not a Topology AE checkpoint")
    format_version = checkpoint.get("format_version")
    if format_version != 4:
        raise ValueError(f"unsupported Topology AE checkpoint format: {format_version}")

    saved_args = checkpoint.get("args")
    state_dict = checkpoint.get("model")
    if not isinstance(saved_args, dict) or not isinstance(state_dict, dict):
        raise ValueError("checkpoint must contain args and model dictionaries")
    output_norm_keys = (
        f"autoencoder.{side}_output_norm.{parameter}"
        for side in ("encoder", "decoder")
        for parameter in ("weight", "bias")
    )
    if any(key not in state_dict for key in output_norm_keys):
        raise ValueError(
            "checkpoint lacks final LayerNorm parameters; "
            "use its immutable historical snapshot"
        )

    recorded_runtime = checkpoint.get("runtime")
    if recorded_runtime is not None and not isinstance(recorded_runtime, dict):
        raise ValueError("checkpoint runtime metadata must be a dictionary")
    system_name, _ = _resolve_system_name(saved_args, recorded_runtime)
    if system_name == "Nexus2KTopologyAESystem":
        system_class = Nexus2KTopologyAESystem
    elif system_name == "FlashVarlenNexus2KTopologyAESystem":
        from .flash_varlen_topology import FlashVarlenNexus2KTopologyAESystem

        system_class = FlashVarlenNexus2KTopologyAESystem
    else:
        raise ValueError(f"unsupported Topology AE training system: {system_name}")

    model = system_class.from_saved_args(saved_args)
    args_contract = model.scoring_contract()
    extra_state = state_dict.get("_extra_state")
    if not isinstance(extra_state, dict):
        raise ValueError("checkpoint Topology AE extra state must be a dictionary")
    state_contract = _project_scoring_contract(extra_state, args_contract)
    if state_contract != args_contract:
        raise ValueError("checkpoint args and model scoring state disagree")
    recorded_scoring = checkpoint.get("spacetime_scoring")
    if not isinstance(recorded_scoring, dict):
        raise ValueError("checkpoint spacetime scoring metadata must be a dictionary")
    expected = _project_scoring_contract(recorded_scoring, args_contract)
    if expected != args_contract:
        raise ValueError("checkpoint args and top-level scoring metadata disagree")

    model.load_state_dict(state_dict, strict=True)
    if model.scoring_contract() != args_contract:
        raise ValueError("loaded model scoring state disagrees with checkpoint args")
    for key, value in model.runtime_contract().items():
        if recorded_runtime.get(key) != value:
            raise ValueError(f"checkpoint runtime metadata disagrees at {key}")
    return model.to(device=device, dtype=torch.float32).eval()
