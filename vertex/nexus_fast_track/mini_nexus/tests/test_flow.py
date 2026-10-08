import torch

from mini_nexus.flow import euler_integrate, flow_matching_batch, flow_matching_loss


def test_linear_flow_batch_has_exact_path_and_velocity():
    data = torch.tensor([[[1.0, 3.0]]])
    noise = torch.tensor([[[-1.0, 1.0]]])
    time = torch.tensor([0.25])
    noisy, target_velocity, returned_time, returned_noise = flow_matching_batch(
        data, noise=noise, time=time
    )
    assert torch.allclose(noisy, torch.tensor([[[-0.5, 1.5]]]))
    assert torch.allclose(target_velocity, torch.tensor([[[2.0, 2.0]]]))
    assert torch.equal(returned_time, time)
    assert torch.equal(returned_noise, noise)
    assert flow_matching_loss(target_velocity, target_velocity).item() == 0.0


def test_euler_integrates_constant_velocity():
    initial = torch.zeros((2, 3, 4))

    def constant_velocity(x, time):
        del time
        return torch.ones_like(x) * 2.0

    result = euler_integrate(constant_velocity, initial, steps=8)
    assert torch.allclose(result, torch.full_like(initial, 2.0))

