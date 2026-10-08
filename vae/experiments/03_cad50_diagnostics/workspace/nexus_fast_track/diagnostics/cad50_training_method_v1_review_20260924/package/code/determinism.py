"""Determinism / reproducibility policy for the CAD50 face-finish run.

Extracted from the archived `run_support.py`. This is not decoration: the
archived run asserts BITWISE equality between cached and real forward paths,
and between cold and warm evaluations. None of that is achievable without
every switch below being set exactly this way.

Call `configure()` first, before importing/constructing anything else that
touches CUDA.
"""
from __future__ import annotations

import os

# Must be set BEFORE torch initialises CUDA.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import random  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = None  # set by the caller if artifact writing is needed


def configure(seed: int = 0) -> None:
    """Pin every source of numerical nondeterminism."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    # No TF32 anywhere: it changes reduction order and breaks bitwise equality.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    # Attention must go through the MATH backend.
    # flash / mem-efficient SDPA have different accumulation orders.
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    torch.set_float32_matmul_precision("highest")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    """Full RNG snapshot -- the trainer stores this in every checkpoint."""
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng(value: dict) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    if value["cuda"]:
        torch.cuda.set_rng_state_all(value["cuda"])


def tensor_hash(state: dict) -> str:
    """Order-sensitive hash of a named-tensor dict. Used to prove freezing."""
    import hashlib

    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def sha(path) -> str:
    import hashlib
    from pathlib import Path

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Deterministic neighbourhood aggregation
# --------------------------------------------------------------------------
class DeterministicGather(torch.autograd.Function):
    """Gather whose backward uses a deterministic index_add_.

    torch's default scatter-add backward for gather is nondeterministic on
    CUDA (atomics). This replaces it.
    """

    @staticmethod
    def forward(ctx, nodes, source):
        ctx.save_for_backward(source)
        ctx.shape = nodes.shape
        return nodes[source]

    @staticmethod
    def backward(ctx, gradient):
        (source,) = ctx.saved_tensors
        torch.use_deterministic_algorithms(True)
        result = gradient.new_zeros(ctx.shape)
        result.index_add_(0, source, gradient)
        return result, None
