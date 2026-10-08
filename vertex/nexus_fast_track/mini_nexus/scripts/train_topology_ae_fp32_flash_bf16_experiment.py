#!/usr/bin/env python3
"""Topology AE selective-precision entry: FP32 except BF16 FlashAttention."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mini_nexus.flash_varlen_topology import (  # noqa: E402
    FlashVarlenNexus2KTopologyAESystem,
)
import train_topology_ae_overfit_packed as packed_trainer  # noqa: E402


def main() -> int:
    return packed_trainer.main(
        system_class=FlashVarlenNexus2KTopologyAESystem,
        required_precision="fp32_flash_bf16",
    )


if __name__ == "__main__":
    raise SystemExit(main())
