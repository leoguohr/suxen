"""Separate frozen VAE posterior export and decoder-only generation interfaces.

No GT faces, Edge labels or Encoder call are accepted by decode_latents.
The source architecture is an unchanged local snapshot, not teacher weights.
"""
import torch
from _vae_reference import Graph, Config, NativeTopologyAE
from _faces_reference import file_sha

SOURCE_CHECKPOINT_SHA256 = 'e89b8078b7abb0ca7b0c2c44f9f20382ec5b742f2aee948f209d642852714d15'


def load_frozen_vae(path, device, expected_sha256=SOURCE_CHECKPOINT_SHA256):
    """Only deserialize a hash-verified trusted local checkpoint, never teacher weights."""
    if file_sha(path) != expected_sha256:
        raise ValueError('VAE checkpoint SHA256 differs from the pinned source')
    state = torch.load(path, map_location='cpu', weights_only=False)
    cfg = Config(**state['model_config'])
    if cfg.model_variant != 'B_v2_teacher_blocks' or cfg.latent_width != 512:
        raise ValueError('Checkpoint is not the selected OwnAE-v2 latent512 model')
    model = NativeTopologyAE(cfg)
    model.load_state_dict(state['model'], strict=True)
    del state
    return model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)


def check_codec(model):
    if model.cfg.model_variant != 'B_v2_teacher_blocks' or model.cfg.latent_width != 512:
        raise ValueError('Requires the native OwnAE-v2 latent512 architecture')
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Freeze all VAE parameters and use eval mode before export/decode')


@torch.no_grad()
def encode_posterior(model, vertices, faces):
    check_codec(model)
    graph = Graph.from_faces(faces, len(vertices))
    h = torch.cat((model.vertex_input(model.position_features(vertices)),
                   model.face_input(model.position_features(vertices[faces].mean(1)))))
    for block in model.encoder_blocks:
        h = block(h, graph)
    h = model.encoder_output_norm(h[:len(vertices)])
    return dict(mu=model.mu(h), logvar=model.log_variance(h).clamp(-20, 10))


def sample_posterior(mu, logvar, generator):
    if mu.shape != logvar.shape or mu.shape[-1] != 512:
        raise ValueError('mu and effective logvar must share per-vertex latent512 shape')
    epsilon = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)
    return mu+(logvar*.5).exp()*epsilon


@torch.no_grad()
def decode_latents(model, latent):
    check_codec(model)
    if latent.ndim != 2 or latent.shape[1] != 512 or latent.dtype != torch.float32:
        raise ValueError('Decode one complete mesh: FP32 [N,512], without padding')
    h = model.latent_input(latent)
    for block in model.decoder_blocks:
        h = block(h)
    h = model.decoder_output_norm(h)
    edge, face = model.edge_embedding(h), model.face_embedding(h)
    return dict(decoder_hidden=h, edge=edge-edge.mean(0, keepdim=True), face=face-face.mean(0, keepdim=True))
