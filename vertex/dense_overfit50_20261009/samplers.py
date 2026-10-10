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
    """The 2M update x_t = (sigma_t/sigma_s) x + (t - s)/(1 - s) * D, D = data + (data - prev_data)/(2r),
    rewritten as  x_t = [Euler step x + dt*v] + dt/(1 - s) * (D - data).
    First-order steps (the first two and the last) skip the correction, so they run exactly the same
    floating-point operations as mini_nexus euler_integrate and match it bit for bit."""
    if steps < 1:
        raise ValueError("steps must be positive")
    lam = lambda u: math.log(u / (1.0 - u))  # noqa: E731
    dt = 1.0 / steps
    prev_data, prev_s = None, None
    for i in range(steps):
        s, t = i / steps, (i + 1) / steps
        v = velocity(x, s)
        data = x + (1.0 - s) * v
        second_order = prev_data is not None and prev_s != 0.0 and i != steps - 1
        x_next = x + dt * v
        if second_order:
            r = (lam(s) - lam(prev_s)) / (lam(t) - lam(s))
            x_next = x_next + (dt / (1.0 - s)) * (0.5 / r) * (data - prev_data)
        x, prev_data, prev_s = x_next, data, s
    return x


SAMPLERS = ("euler", "dpm2m")
