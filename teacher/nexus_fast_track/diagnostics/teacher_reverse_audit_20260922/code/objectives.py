"""Reconstructed training objective building blocks, NOT original training loops.

Source supports four-group BCE, KL, velocity-MSE flow matching and point count CE.
These functions do not recover the original sampler, dropout, optimizer resets,
step ordering or exact loss coefficients. Missing coefficients are explicit args.
"""
from __future__ import annotations
import torch
from torch import Tensor
from torch.nn import functional as F


def hard4_bce(logits: Tensor, labels: Tensor) -> Tensor:
    """Whole-task reduction. Empty groups contribute zero, outer divisor=4.

    If chunking is needed, aggregate group SUMS and COUNTS across all chunks
    before reducing; do not average chunk losses.
    """
    s=logits.float().reshape(-1);y=labels.float().reshape(-1)
    if s.shape!=y.shape or not bool(((y==0)|(y==1)).all()):
        raise ValueError('Expected matching logits and binary labels')
    gt=y.bool();pr=s.detach()>0
    groups=(gt&pr,~gt&~pr,~gt&pr,gt&~pr)
    losses=F.binary_cross_entropy_with_logits(s,y,reduction='none')
    return sum((losses[g].sum()/g.sum().clamp_min(1) for g in groups))/4


def kl_standard_normal(mu: Tensor, logvar: Tensor) -> Tensor:
    """Per-element mean. Original KL reduction/clamping not proven by weights."""
    return .5*(mu.square()+logvar.exp()-1-logvar).mean()


def topology_velocity_loss(model, z1: Tensor, vertices: Tensor, t: Tensor,
                           noise: Tensor, mask: Tensor | None = None) -> Tensor:
    """z1 must use that AE's own stored latent mean/std normalization."""
    if z1.shape!=noise.shape:raise ValueError('Noise shape mismatch')
    tau=t[:,None,None]
    zt=(1-tau)*noise+tau*z1
    e=(model(zt,t,vertices,mask)-(z1-noise)).square()
    if mask is None:return e.mean()
    return (e*mask[...,None]).sum()/(mask.sum().clamp_min(1)*z1.shape[-1])


def point_velocity_and_count_loss(model, target: Tensor, features: Tensor,
    t: Tensor, noise: Tensor, mask: Tensor, counts: Tensor,
    *, count_coefficient: float) -> dict[str,Tensor]:
    """X1-parameterized velocity MSE + CE, as labelled in the supplied config.

    count_coefficient is required because its exact original value is unavailable.
    The final text-prior training was a separate cached-clean objective; this is
    not a claim that the final candidate was trained with this combined loss.
    """
    if not bool(((t>=0)&(t<1)).all()):raise ValueError('Require 0<=t<1')
    tau=t[:,None,None];xt=(1-tau)*noise+tau*target
    x1=model(xt,t,features,mask)
    predicted_velocity=(x1-xt)/(1-tau)
    e=(predicted_velocity-(target-noise)).square()
    velocity=(e*mask[...,None]).sum()/(mask.sum().clamp_min(1)*3)
    count=F.cross_entropy(model.count_logits(features),counts)
    return dict(loss=velocity+count_coefficient*count,velocity_mse=velocity,count_ce=count)


def cached_prior_loss(model, text_features: Tensor, cached_frozen_output: Tensor,
                      target: Tensor, mask: Tensor) -> Tensor:
    """Plausible final cached-clean objective supported by candidate history.

    Only coordinate_prior and residual_scale should be trainable for this phase.
    How historical cached_frozen_output was sampled is not fully recoverable.
    """
    clean=model.prior(text_features)[:,:target.shape[1]]+model.residual_scale*cached_frozen_output.detach()
    e=(clean-target).square()
    return (e*mask[...,None]).sum()/(mask.sum().clamp_min(1)*3)
