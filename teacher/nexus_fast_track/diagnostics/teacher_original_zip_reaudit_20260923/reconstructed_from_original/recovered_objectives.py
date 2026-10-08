"""Loss components supported by original records; NOT the missing trainer.

The caller must choose unknown coefficients/reductions/training protocol.
Original candidate-negative sampler, posterior clamp/dropout, count coefficient,
frozen-output cache construction and RNG ordering remain unrecovered.
"""
import torch
from torch.nn import functional as F


def grouped_bce(scores, labels):
    labels = labels.bool().reshape(-1)
    scores = scores.float().reshape(-1)
    predicted = scores.detach() > 0
    entries = F.binary_cross_entropy_with_logits(scores, labels.float(), reduction='none')
    groups = [labels & predicted, ~labels & ~predicted, ~labels & predicted, labels & ~predicted]
    return sum(entries[g].sum() / g.sum().clamp_min(1) for g in groups) / 4


def standard_normal_kl_elements(mean, log_variance):
    # Returns per-element KL; the original aggregate reduction is not proven.
    return (mean.square() + log_variance.exp() - log_variance - 1) * .5


def masked_square_mean(error, valid):
    return (error.square() * valid[..., None]).sum() / (valid.sum().clamp_min(1) * error.shape[-1])


def topology_velocity_mse(network, target_latent, vertices, noise, time, valid):
    # Target latent must use the matching AE and its saved normalization.
    fraction = time[:, None, None]
    intermediate = noise * (1 - fraction) + target_latent * fraction
    prediction = network(intermediate, time, vertices, valid)
    return masked_square_mean(prediction - (target_latent - noise), valid)


def point_velocity_count_loss(network, vertices, features, noise, time, valid, counts, *, count_weight):
    fraction = time[:, None, None]
    intermediate = noise * (1 - fraction) + vertices * fraction
    clean = network(intermediate, time, features, valid)
    velocity = (clean - intermediate) / (1 - fraction)
    flow = masked_square_mean(velocity - (vertices - noise), valid)
    count = F.cross_entropy(network.count_logits(features), counts)
    return {'total':flow + count_weight * count, 'velocity_mse':flow, 'count_ce':count}


def cached_prior_mse(network, features, frozen_clean_output, target_vertices, valid):
    # Plausible candidate objective; original cache generation is unavailable.
    prediction = network.prior(features)[:, :target_vertices.shape[1]]
    prediction = prediction + network.residual_scale * frozen_clean_output.detach()
    return masked_square_mean(prediction - target_vertices, valid)
