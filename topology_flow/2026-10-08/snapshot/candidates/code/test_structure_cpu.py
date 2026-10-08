"""Synthetic CPU-only structure tests. No optimizer, real checkpoint or GPU use."""
import json
from dataclasses import replace
from pathlib import Path
import torch
from topology_flow import TopologyFlowConfig, PointCloudTopologyFlow, linear_flow_target, equal_mesh_velocity_loss
from vae_codec import encode_posterior, decode_latents, sample_posterior
from _vae_reference import NativeTopologyAE, Config


def main():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    torch.use_deterministic_algorithms(True)
    cfg = TopologyFlowConfig(hidden_dim=24, num_layers=2, num_heads=3, condition_dim=32,
        condition_layers=2, condition_heads=4, condition_tokens=4, rope_scale=1., recompute=False)
    model = PointCloudTopologyFlow(cfg)
    z = torch.randn(2, 7, 512)
    xyz = torch.randn(2, 7, 3)
    points = torch.randn(2, 12, 6)
    time = torch.tensor([.3, .7])
    mask = torch.tensor([[True]*7, [True]*4+[False]*3])
    point_mask = torch.tensor([[True]*12, [True]*9+[False]*3])
    initial = model(z, time, xyz, points, mask, point_mask)
    assert initial.shape == z.shape and initial.count_nonzero() == 0
    # Nonzero test-only weights avoid vacuous invariance tests on zero outputs.
    with torch.no_grad():
        model.flow.output.weight.normal_(std=.1)
        for block in model.flow.blocks:
            block.modulation[-1].weight.normal_(std=.03)
            block.modulation[-1].bias.normal_(std=.03)
            block.cross_attention.output.weight.normal_(std=.03)
    model.eval()
    out = model(z, time, xyz, points, mask, point_mask)
    assert out.abs().max() > .01 and not out[~mask].count_nonzero()
    perm = torch.tensor([6, 1, 3, 0, 5, 4, 2])
    changed = model(z[:,perm], time, xyz[:,perm], points, mask[:,perm], point_mask)
    torch.testing.assert_close(changed, out[:,perm], atol=2e-5, rtol=2e-5)
    zp, vp, pp = z.clone(), xyz.clone(), points.clone()
    zp[~mask] = 10000; vp[~mask] = 10000; pp[~point_mask] = 10000
    torch.testing.assert_close(model(zp,time,vp,pp,mask,point_mask), out, atol=2e-5, rtol=2e-5)
    singles = torch.cat([model(z[i:i+1],time[i:i+1],xyz[i:i+1],points[i:i+1],mask[i:i+1],point_mask[i:i+1]) for i in range(2)])
    torch.testing.assert_close(singles, out, atol=2e-5, rtol=2e-5)
    assert (model(z,time,xyz,points+1,mask,point_mask)-out).abs().max() > 1e-5
    assert (model(z,time,xyz*2,points,mask,point_mask)-out).abs().max() > 1e-5
    assert (model(z,time+.1,xyz,points,mask,point_mask)-out).abs().max() > 1e-5

    clean, noise = torch.randn_like(z), torch.randn_like(z)
    mixed, velocity = linear_flow_target(clean, noise, torch.tensor([0.,1.]), mask)
    torch.testing.assert_close(mixed[0], noise[0])
    torch.testing.assert_close(mixed[1][mask[1]], clean[1][mask[1]])
    torch.testing.assert_close((noise+velocity)[mask], clean[mask])
    # Equal mesh mean, not a global mean weighted by valid vertex count.
    prediction = torch.tensor([[[1.],[1.],[1.]], [[3.],[99.],[99.]]])
    unequal_mask = torch.tensor([[True,True,True],[True,False,False]])
    assert equal_mesh_velocity_loss(prediction, torch.zeros_like(prediction), unequal_mask).item() == 5.
    model.train()
    target = torch.randn_like(z)
    loss = equal_mesh_velocity_loss(model(z,time,xyz,points,mask,point_mask),target,mask)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(p.grad.square().sum() for p in model.condition_encoder.parameters()) > 0
    recompute = PointCloudTopologyFlow(replace(cfg, recompute=True))
    recompute.load_state_dict(model.state_dict()); recompute.train()
    other_loss = equal_mesh_velocity_loss(recompute(z,time,xyz,points,mask,point_mask),target,mask)
    other_loss.backward()
    torch.testing.assert_close(loss, other_loss)
    for p, q in zip(model.parameters(), recompute.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-5)

    vae = NativeTopologyAE(Config(model_variant='B_v2_teacher_blocks', latent_width=512,
        encoder_width=24, decoder_width=24, heads=3, encoder_composite_blocks=1, decoder_blocks=2,
        activation_checkpointing=False)).eval().requires_grad_(False)
    vertices = torch.randn(5, 3)
    faces = torch.tensor([[0,1,2],[1,3,4]])
    with torch.no_grad(): full = vae(vertices, faces, sample_latent=False)
    posterior = encode_posterior(vae, vertices, faces)
    torch.testing.assert_close(posterior['mu'], full['mu'], atol=0, rtol=0)
    torch.testing.assert_close(posterior['logvar'], full['log_variance'], atol=0, rtol=0)
    decoded = decode_latents(vae, posterior['mu'])
    for key in ('edge','face','decoder_hidden'):
        torch.testing.assert_close(decoded[key], full[key], atol=0, rtol=0)
    rng = torch.Generator().manual_seed(7)
    saved = rng.get_state().clone()
    z1 = sample_posterior(posterior['mu'],posterior['logvar'],rng)
    z2 = sample_posterior(posterior['mu'],posterior['logvar'],rng)
    assert not torch.equal(z1,z2)
    rng.set_state(saved)
    torch.testing.assert_close(z1,sample_posterior(posterior['mu'],posterior['logvar'],rng),atol=0,rtol=0)
    # Decode continues to work when every Encoder entry is deliberately forbidden.
    def forbidden(*args, **kwargs): raise AssertionError('Encoder called during decode')
    vae.vertex_input.forward = forbidden
    vae.face_input.forward = forbidden
    decode_latents(vae,z1)
    assert all(p.grad is None for p in vae.parameters())
    result = dict(passed=True, device='cpu', torch_version=torch.__version__, optimizer_updates=0,
        real_checkpoint_loaded=False, production_scale_tested=False,
        tests=['latent512 shape and zero initialization', 'nonzero permutation equivariance',
            'padding isolation', 'mesh batch isolation', 'point/position/time influence',
            'linear path endpoints and velocity sign', 'equal mesh loss reduction',
            'finite flow and VecSet gradients', 'MATH recompute output and gradient equality',
            'frozen synthetic VAE encode/decode equality', 'posterior RNG continuation and restore',
            'decoder cannot read GT faces or call Encoder'])
    path = Path(__file__).resolve().parent.parent/'evidence/cpu_tests.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
