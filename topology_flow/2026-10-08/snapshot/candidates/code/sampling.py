"""Euler baseline for x(t)=(1-t)*noise+t*data, integrated from 0 to 1.

This is deliberately not labelled NEXUS's DPM-Solver implementation.
"""
import torch
from latent_data import transform_latent


@torch.no_grad()
def integrate_euler(velocity, initial, steps, mask, stop=lambda:False):
    if not isinstance(steps,int) or steps < 1: raise ValueError('Euler steps must be positive')
    x = initial.masked_fill(~mask.unsqueeze(-1),0)
    for index in range(steps):
        if stop(): raise InterruptedError('Sampling budget/STOP reached before a complete mesh')
        time = x.new_full((len(x),),index/steps)
        v = velocity(x,time)
        if v.shape != x.shape or not torch.isfinite(v).all(): raise FloatingPointError('Invalid flow velocity')
        x = (x+v/steps).masked_fill(~mask.unsqueeze(-1),0)
    if not torch.isfinite(x).all(): raise FloatingPointError('Nonfinite generated latent')
    return x


@torch.no_grad()
def generate_latents(model, vertices, points, normalization, seed, steps, point_mask=None, stop=lambda:False):
    """One complete mesh. No posterior cache, faces or GT Edge graph is accepted."""
    if model.training: raise ValueError('Use a fixed eval model for generation')
    device = next(model.parameters()).device
    if vertices.ndim!=2 or vertices.shape[1]!=3: raise ValueError('Expected [N,3] coordinates')
    # A private CPU generator makes evaluation independent of every training RNG.
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn((1,len(vertices),512),generator=generator,dtype=torch.float32).to(device)
    vertices,points = vertices[None].to(device),points[None].to(device)
    mask = torch.ones(vertices.shape[:2],dtype=torch.bool,device=device)
    pmask = None if point_mask is None else point_mask[None].to(device)
    # Trainable condition features may be reused only within this no-grad solve.
    context = model.encode_condition(points,pmask)
    normalized = integrate_euler(lambda x,t:model.flow(x,t,vertices,context,mask),noise,steps,mask,stop)
    return transform_latent(normalized[0],normalization,inverse=True)
