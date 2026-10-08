"""Flow matching shared by the vertex and topology stages."""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor


def flow_matching_batch(
    data: Tensor,
    *,
    noise: Tensor | None = None,
    time: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Construct one linear conditional-flow training batch.

    The path is x_t = (1-t) x_0 + t x_1, where x_0 is Gaussian noise and
    x_1 is the clean training target. Its exact velocity is x_1 - x_0.
    """

    if data.ndim < 2:
        raise ValueError("data must have a batch dimension and a feature dimension")
    if noise is None:
        noise = torch.randn_like(data)
    if noise.shape != data.shape:
        raise ValueError("noise and data must have the same shape")

    batch_size = data.shape[0]
    if time is None:
        time = torch.rand(batch_size, device=data.device, dtype=data.dtype)
    if time.shape != (batch_size,):
        raise ValueError(f"time must have shape [{batch_size}]")

    broadcast_shape = (batch_size,) + (1,) * (data.ndim - 1)
    t = time.reshape(broadcast_shape)
    noisy_data = (1.0 - t) * noise + t * data
    target_velocity = data - noise
    return noisy_data, target_velocity, time, noise


def flow_matching_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Mean squared velocity error, optionally ignoring padded tokens."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    squared_error = (prediction - target).square()
    if mask is None:
        return squared_error.mean()
    if mask.shape != prediction.shape[:-1]:
        raise ValueError("mask must match all prediction dimensions except the last")
    weights = mask.to(squared_error.dtype).unsqueeze(-1)
    denominator = weights.sum().clamp_min(1.0) * prediction.shape[-1]
    return (squared_error * weights).sum() / denominator


@torch.no_grad()
def euler_integrate(
    velocity_fn: Callable[[Tensor, Tensor], Tensor],
    initial: Tensor,
    *,
    steps: int = 20,
) -> Tensor:
    """Integrate dx/dt = velocity_fn(x,t) from t=0 to t=1 with Euler."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    x = initial
    dt = 1.0 / steps
    for step in range(steps):
        time = torch.full(
            (x.shape[0],), step / steps, device=x.device, dtype=x.dtype
        )
        velocity = velocity_fn(x, time)
        if velocity.shape != x.shape:
            raise ValueError("velocity_fn must return the same shape as x")
        x = x + dt * velocity
    return x

