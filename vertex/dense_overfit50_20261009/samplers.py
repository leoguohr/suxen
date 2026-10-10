"""ODE samplers for the rectified flow used here: x_t = t * x1 + (1 - t) * x0, t = 0 noise, t = 1 data,
velocity v = x1 - x0. Plain arithmetic only, so the same code runs on torch tensors and numpy arrays.

The Nexus paper (Sec. 4.1) samples each octree level with 20 DPM-Solver steps and does not name the
variant. `dpm_solver_pp_2m` is DPM-Solver++(2M) (Lu et al. 2022) in data-prediction form, our choice:
  alpha_t = t, sigma_t = 1 - t, lambda_t = log(alpha_t / sigma_t), data prediction x_hat = x + (1 - t) v.
Its first-order step is exactly the Euler step of this flow, so the two samplers differ only by the
second-order correction. The grid is uniform in t like the Euler sampler. The first two steps are
first order (lambda_0 = -inf), and so is the last one (lower-order final, standard for few steps).
"""
from __future__ import annotations

import math


def euler(velocity, x, steps: int):
    """Reference Euler, same grid as mini_nexus.flow.euler_integrate (velocity(x, s) with s = i/steps)."""
    for i in range(steps):
        x = x + (1.0 / steps) * velocity(x, i / steps)
    return x


def dpm_solver_pp_2m(velocity, x, steps: int):
    if steps < 1:
        raise ValueError("steps must be positive")
    lam = lambda u: math.log(u / (1.0 - u))  # noqa: E731
    prev_data, prev_s = None, None
    for i in range(steps):
        s, t = i / steps, (i + 1) / steps
        data = x + (1.0 - s) * velocity(x, s)
        if i == steps - 1:
            return data
        if prev_data is None or prev_s == 0.0:
            d = data
        else:
            r = (lam(s) - lam(prev_s)) / (lam(t) - lam(s))
            d = (1.0 + 0.5 / r) * data - (0.5 / r) * prev_data
        x = ((1.0 - t) / (1.0 - s)) * x + ((t - s) / (1.0 - s)) * d
        prev_data, prev_s = data, s
    return x


SAMPLERS = ("euler", "dpm2m")
