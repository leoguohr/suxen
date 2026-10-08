"""Validated checkpoint loading for Topology-AE evaluation tools."""

from __future__ import annotations

from pathlib import Path

import torch

from .training_2k import Nexus2KTopologyAESystem


_SCORING_ARGUMENT_KEYS = (
    "normalize_spacetime_embeddings",
    "embedding_normalization_eps",
    "edge_logit_scale",
    "face_logit_scale",
    "face_interval_factor",
)


def _project_scoring_contract(
    metadata: dict[str, object], fallback: dict[str, object]
) -> dict[str, object]:
    """Read one scoring contract while filling fields absent in legacy metadata."""

    return {key: metadata.get(key, value) for key, value in fallback.items()}


def _resolve_system_name(
    saved_args: dict[str, object], recorded_runtime: dict[str, object] | None
) -> tuple[str, str]:
    """Resolve a non-ambiguous backend and explain where that fact came from."""

    precision = saved_args.get("precision")
    if recorded_runtime is not None:
        system_name = recorded_runtime.get("training_system")
        if not isinstance(system_name, str):
            raise ValueError("checkpoint runtime is missing training_system")
        expected_precision = {
            "Nexus2KTopologyAESystem": "fp32",
            "FlashVarlenNexus2KTopologyAESystem": "fp32_flash_bf16",
        }.get(system_name)
        if expected_precision is None:
            raise ValueError(f"unsupported Topology AE training system: {system_name}")
        if precision is not None and precision != expected_precision:
            raise ValueError("checkpoint precision and runtime backend disagree")
        return system_name, "recorded_runtime"

    if precision == "fp32_flash_bf16":
        return "FlashVarlenNexus2KTopologyAESystem", "inferred_from_precision"
    if precision == "bf16":
        raise ValueError(
            "legacy whole-network BF16 checkpoint has no unambiguous faithful "
            "runtime metadata; evaluate it with its immutable historical snapshot"
        )
    if precision not in (None, "fp32"):
        raise ValueError(f"unsupported Topology AE precision: {precision}")
    return "Nexus2KTopologyAESystem", "legacy_default_standard_backend"


def topology_runtime_resolution(checkpoint: dict[str, object]) -> str:
    """Describe whether an evaluation backend was recorded or inferred."""

    saved_args = checkpoint.get("args")
    recorded_runtime = checkpoint.get("runtime")
    if not isinstance(saved_args, dict):
        raise ValueError("checkpoint must contain an args dictionary")
    if recorded_runtime is not None and not isinstance(recorded_runtime, dict):
        raise ValueError("checkpoint runtime metadata must be a dictionary")
    _, source = _resolve_system_name(saved_args, recorded_runtime)
    return source


def load_topology_checkpoint(path: Path) -> dict[str, object]:
    """Load a trusted local Topology-AE checkpoint across PyTorch versions."""

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
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
    if format_version is not None and not isinstance(format_version, int):
        raise ValueError("checkpoint format_version must be an integer")
    if format_version is not None and format_version not in {1, 2, 3, 4}:
        raise ValueError(f"unsupported Topology AE checkpoint format: {format_version}")

    saved_args = checkpoint.get("args")
    state_dict = checkpoint.get("model")
    if not isinstance(saved_args, dict) or not isinstance(state_dict, dict):
        raise ValueError("checkpoint must contain args and model dictionaries")

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
    if extra_state is not None:
        if not isinstance(extra_state, dict):
            raise ValueError("checkpoint Topology AE extra state must be a dictionary")
        state_contract = _project_scoring_contract(extra_state, args_contract)
        scoring_args_present = any(key in saved_args for key in _SCORING_ARGUMENT_KEYS)
        if scoring_args_present and state_contract != args_contract:
            raise ValueError("checkpoint args and model scoring state disagree")
    recorded_scoring = checkpoint.get("spacetime_scoring")
    if recorded_scoring is not None:
        if not isinstance(recorded_scoring, dict):
            raise ValueError("checkpoint spacetime scoring metadata must be a dictionary")
        expected = _project_scoring_contract(recorded_scoring, args_contract)
        if expected != args_contract:
            raise ValueError("checkpoint args and top-level scoring metadata disagree")

    model.load_state_dict(state_dict, strict=True)
    if model.scoring_contract() != args_contract:
        raise ValueError("loaded model scoring state disagrees with checkpoint args")
    if isinstance(recorded_runtime, dict):
        actual_runtime = model.runtime_contract()
        for key, value in actual_runtime.items():
            if recorded_runtime.get(key) != value:
                raise ValueError(f"checkpoint runtime metadata disagrees at {key}")
    return model.to(device=device, dtype=torch.float32).eval()
