"""Small CPU protocol checks; no checkpoint, CUDA, dataset, or server required."""
import copy
import dataclasses
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch

from native_models import Config, NativeTopologyAE
from data_objective import epoch_batches, resume_cursor
from run_support import tensor_hash, sha
from evaluate_checkpoint import evaluate
from stream_faces import stream_faces
import train_vae
from train_vae import probe_gradients, append_committed_logs, commit_checkpoint
from vae_protocol import (BETA, START, kl_parts, initialize_logvar, make_optimizer,
    optimizer_group_names, group_steps, evaluation_generator, uid_noise_seed,
    snapshot_parameters, measured_deltas)


def small_config(checkpointing):
    return Config(model_variant='B_v2_teacher_blocks', encoder_width=16, latent_width=8,
        decoder_width=16, encoder_composite_blocks=1, decoder_blocks=1,
        heads=4, ffn_ratio=2, fourier_bands=2, activation_checkpointing=checkpointing)


def item(uid):
    faces = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    return dict(uid=uid, vertices=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],
        [0.,0.,1.],[1.,1.,0.],[1.,0.,1.],[0.,1.,1.]], dtype=torch.float32),
        faces=faces, gt_faces=faces,
        edges=torch.tensor([[0,1],[0,2],[1,2],[1,3],[2,3]], dtype=torch.long))


def source_model_class():
    path = Path(__file__).resolve().parent.parent/'source_snapshot/native_models.py'
    spec = importlib.util.spec_from_file_location('source_native_models_for_cpu_test', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_native_and_recompute():
    torch.manual_seed(7)
    cfg = small_config(True)
    source = source_model_class()
    old = source.NativeTopologyAE(source.Config(**cfg.to_dict()))
    model = NativeTopologyAE(cfg)
    model.load_state_dict(old.state_dict())
    mesh = item('one')
    with torch.no_grad():
        a = old(mesh['vertices'], mesh['faces'])
        b = model(mesh['vertices'], mesh['faces'])
    for key in ('mu', 'log_variance', 'latent', 'edge', 'face'):
        assert torch.equal(a[key], b[key]), key
    assert b['latent'] is b['mu']
    initialize_logvar(model)
    assert torch.count_nonzero(model.log_variance.weight) == 0
    assert torch.all(model.log_variance.bias == -6)
    assert all(torch.equal(old.state_dict()[key], model.state_dict()[key])
               for key in old.state_dict() if not key.startswith('log_variance.'))
    plain = NativeTopologyAE(dataclasses.replace(cfg, activation_checkpointing=False))
    plain.load_state_dict(model.state_dict()); plain.log_variance.requires_grad_(True)
    expected_generator = torch.Generator().manual_seed(42)
    expected_epsilon = torch.randn((7, 8), generator=expected_generator)
    before_global = torch.get_rng_state().clone()
    sampled = model(mesh['vertices'], mesh['faces'], sample_latent=True,
                    generator=torch.Generator().manual_seed(42))
    assert torch.equal(sampled['epsilon'], expected_epsilon)
    assert torch.equal(torch.get_rng_state(), before_global)
    assert torch.allclose(sampled['latent'], sampled['mu']+torch.exp(sampled['log_variance']/2)*expected_epsilon)
    reference = plain(mesh['vertices'], mesh['faces'], sample_latent=True,
                      generator=torch.Generator().manual_seed(42))
    for key in ('mu', 'log_variance', 'latent', 'edge', 'face'):
        assert torch.allclose(sampled[key], reference[key], atol=1e-6, rtol=1e-6), key
    loss = sampled['edge'].square().mean()+sampled['face'].square().mean()
    loss.backward()
    other_loss = reference['edge'].square().mean()+reference['face'].square().mean()
    other_loss.backward()
    for (name, p), (_, q) in zip(model.named_parameters(), plain.named_parameters()):
        assert p.grad is not None and q.grad is not None, name
        assert torch.allclose(p.grad, q.grad, atol=1e-5, rtol=1e-4), name
    assert model.log_variance.weight.grad.abs().max() > 0


def test_optimizer_inheritance_and_deltas():
    torch.manual_seed(4)
    model = NativeTopologyAE(small_config(False))
    old_params = [p for p in model.parameters() if p.requires_grad]
    old_names = [n for n, p in model.named_parameters() if p.requires_grad]
    old = torch.optim.AdamW(old_params, lr=1e-4, betas=(.9,.999), eps=1e-8,
                            weight_decay=.01, foreach=True)
    mesh = item('one')
    out = model(mesh['vertices'], mesh['faces'])
    (out['edge'].square().mean()+out['face'].square().mean()).backward(); old.step()
    state = copy.deepcopy(old.state_dict())
    for value in state['state'].values(): value['step'].fill_(START)
    fresh = NativeTopologyAE(small_config(False))
    opt = make_optimizer(fresh, state, expected_old_count=len(old_params))
    assert optimizer_group_names(fresh, opt)[0] == old_names
    assert optimizer_group_names(fresh, opt)[1] == ['log_variance.weight','log_variance.bias']
    assert group_steps(opt) == [[START], []]
    assert len(opt.state) == len(old_params)
    assert all(p not in opt.state for p in fresh.log_variance.parameters())
    for source_state, target_state in zip(state['state'].values(), opt.state_dict()['state'].values()):
        for key in ('step','exp_avg','exp_avg_sq'):
            assert torch.equal(source_state[key], target_state[key])
    before = snapshot_parameters(opt)
    with torch.no_grad():
        fresh.encoder_blocks[0].graph_norm.weight.add_(.125)
        fresh.log_variance.bias.add_(.25)
    group, module = measured_deltas(fresh, opt, before)
    assert np.isclose(group[0], .125*np.sqrt(fresh.encoder_blocks[0].graph_norm.weight.numel()))
    assert np.isclose(group[1], .25*np.sqrt(fresh.log_variance.bias.numel()))
    assert module['encoder_mu'] == group[0] and module['logvar'] == group[1]
    assert module['decoder'] == module['edge_head'] == module['face_head'] == 0
    # New checkpoint restoration must use group names, not model.named_parameters order.
    cp_model = copy.deepcopy(fresh.state_dict()); cp_opt = copy.deepcopy(opt.state_dict())
    resumed = NativeTopologyAE(small_config(False))
    ropt = make_optimizer(resumed, expected_old_count=len(old_params))
    resumed.load_state_dict(cp_model); ropt.load_state_dict(cp_opt)
    assert optimizer_group_names(resumed, ropt) == optimizer_group_names(fresh, opt)
    assert tensor_hash(resumed.state_dict()) == tensor_hash(fresh.state_dict())
    assert group_steps(ropt) == [[START], []]


def test_kl_eval_rng_and_order():
    mu = torch.tensor([[1.,2.],[3.,4.]])
    lv = torch.tensor([[-6.,-5.],[-4.,-3.]])
    km, ks = kl_parts(mu, lv)
    assert torch.allclose(km, torch.tensor(.5*(1+4+9+16)/4))
    assert torch.allclose(ks, .5*(lv.exp()-1-lv).sum()/4)
    assert torch.allclose(km+ks, .5*(mu.square()+lv.exp()-1-lv).mean())
    train = torch.Generator().manual_seed(20260929)
    state = train.get_state().clone(); global_state = torch.get_rng_state().clone()
    a = torch.randn((7,8), generator=evaluation_generator('cpu',861001,'uid-a'))
    b = torch.randn((7,8), generator=evaluation_generator('cpu',861001,'uid-a'))
    c = torch.randn((7,8), generator=evaluation_generator('cpu',861002,'uid-a'))
    assert torch.equal(a,b) and not torch.equal(a,c)
    assert uid_noise_seed(861001,'uid-a') != uid_noise_seed(861001,'uid-b')
    assert torch.equal(train.get_state(),state) and torch.equal(torch.get_rng_state(),global_state)
    uids = [f'u{i:03d}' for i in range(100)]
    for completed, epoch, group in ((START,1711,0),(START+250,1723,10),(START+500,1736,0)):
        cursor = resume_cursor(uids, completed, 0)
        assert (cursor['epoch'], cursor['group'], cursor['next_mesh_position']) == (epoch,group,5*group)
        assert cursor['next_uids'] == epoch_batches(uids,epoch,0)[group]
        expected = {u:completed//20 for u in uids}
        for uid in cursor['order'][:5*group]: expected[uid] += 1
        assert sum(expected.values()) == completed*5
    assert set(expected.values()) == {1736}


def test_audit_and_faces():
    torch.manual_seed(11)
    model = NativeTopologyAE(small_config(True))
    opt = make_optimizer(model, expected_old_count=len([p for p in model.parameters() if p.requires_grad]))
    noise = torch.Generator().manual_seed(20260929)
    before_noise = noise.get_state().clone()
    items = {f'u{i:03d}': item(f'u{i:03d}') for i in range(100)}
    audit = probe_gradients(model, opt, items, dict(uids=list(items)), START, noise, device='cpu')
    assert audit['scope'] == 'next_optimizer_batch_5' and len(audit['meshes']) == 5
    assert audit['logvar_reconstruction_gradient_nonzero'] and audit['logvar_kl_gradient_nonzero']
    assert audit['optimizer_updates'] == 0 and group_steps(opt) == [[], []]
    assert torch.equal(noise.get_state(),before_noise)
    rec = audit['reconstruction_gradient_norm']; kl = audit['beta_kl_gradient_norm']
    dot = audit['reconstruction_beta_kl_gradient_inner_product']
    assert np.isclose(audit['total_gradient_norm']**2, rec**2+kl**2+2*dot, rtol=1e-5)
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
        adjacency = np.zeros((5,5), dtype=bool)
        adjacency[0,1] = adjacency[0,2] = adjacency[1,2] = True
        gt = np.array([[0,1,2],[0,1,3]], dtype=np.int64)
        result = stream_faces(adjacency, gt, lambda ids: np.ones(len(ids),np.float32),
                              Path(td), dict(group='noise-861001',uid='small'))
        assert result['complete'] and result['candidates'] == 1
        assert result['tp'] == 1 and result['fn_missing'] == 1


def test_checkpoint_resume_next_sample_and_log():
    torch.manual_seed(23)
    base = NativeTopologyAE(small_config(False))
    initialize_logvar(base)
    count = len([p for p in base.parameters() if p.requires_grad])-2
    first = NativeTopologyAE(small_config(False))
    second = NativeTopologyAE(small_config(False))
    opt_a = make_optimizer(first, expected_old_count=count)
    opt_b = make_optimizer(second, expected_old_count=count)
    first.load_state_dict(base.state_dict()); second.load_state_dict(base.state_dict())
    noise_a = torch.Generator().manual_seed(20260929)
    saved_noise = noise_a.get_state().clone()
    noise_b = torch.Generator(); noise_b.set_state(saved_noise)
    mesh = item('one')
    def update(model, opt, noise):
        opt.zero_grad(set_to_none=True)
        rows = model(mesh['vertices'], mesh['faces'], sample_latent=True, generator=noise)
        km, ks = kl_parts(rows['mu'], rows['log_variance'])
        loss = rows['edge'].square().mean()+rows['face'].square().mean()+BETA*(km+ks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']], 1.)
        opt.step()
        return rows['epsilon'].detach().clone()
    eps_a = update(first,opt_a,noise_a)
    eps_b = update(second,opt_b,noise_b)
    assert torch.equal(eps_a,eps_b) and torch.equal(noise_a.get_state(),noise_b.get_state())
    assert tensor_hash(first.state_dict()) == tensor_hash(second.state_dict())
    for a,b in zip(opt_a.state_dict()['state'].values(),opt_b.state_dict()['state'].values()):
        assert all(torch.equal(a[k],b[k]) for k in a)
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
        run = Path(td)
        records = [dict(update=START+1),dict(update=START+2)]
        append_committed_logs(run,records)
        append_committed_logs(run,records)
        assert [json.loads(x)['update'] for x in (run/'updates.jsonl').read_text().splitlines()] == [START+1,START+2]


def test_deployment_config_protocol_fields():
    project = Path(__file__).resolve().parent.parent
    config = json.loads((project/'experiment_config.json').read_text())
    server_root = Path(config['source_checkpoint']).parent.parent
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td, patch.object(train_vae, 'ROOT', server_root):
        path = Path(td)/'config.json'
        path.write_text(json.dumps(config))
        assert train_vae.validated_config(path) == config
        for key, bad in (('warmup',True),('mu_checkpoints',[0,100,500]),('mesh_per_update',4)):
            changed = dict(config, **{key:bad})
            path.write_text(json.dumps(changed))
            try: train_vae.validated_config(path)
            except AssertionError: pass
            else: raise AssertionError(f'Accepted invalid {key}')


def test_full_evaluator_actual_sampled_forward_cpu():
    torch.manual_seed(31)
    model = NativeTopologyAE(small_config(False))
    initialize_logvar(model)
    uids = [f'u{i:03d}' for i in range(100)]
    items = {uid:item(uid) for uid in uids}
    train_noise = torch.Generator().manual_seed(20260929)
    train_state = train_noise.get_state().clone()
    global_state = torch.get_rng_state().clone()
    class Budget:
        def stop(self): return False
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
        run = Path(td)
        checkpoint = run/'small.pt'
        torch.save(dict(model=model.state_dict()),checkpoint)
        entry = dict(path=str(checkpoint), sha256=sha(checkpoint),
            model_state_sha256=tensor_hash(model.state_dict()),completed_updates=START)
        data = dict(uids=uids,manifest_sha256='synthetic-only')
        mu = evaluate(model,items,data,entry,run,Budget(),device='cpu')
        noisy = evaluate(model,items,data,entry,run,Budget(),sample_seed=861001,device='cpu')
        assert mu['complete'] and noisy['complete']
        assert len(mu['meshes']) == len(noisy['meshes']) == 100
        assert all(row['posterior']['epsilon_sha256'] is None for row in mu['meshes'])
        assert all(row['posterior']['epsilon_sha256'] is not None for row in noisy['meshes'])
        assert all(row['posterior']['perturbation_rms'] > 0 for row in noisy['meshes'])
        assert all(row['actual_face_candidates'] >= 0 for row in noisy['meshes'])
        assert (run/'evaluations'/f'update-{START:08d}'/'noise-861001'/'complete.json').exists()
        again = evaluate(model,items,data,entry,run,Budget(),sample_seed=861001,device='cpu')
        assert [m['posterior']['epsilon_sha256'] for m in noisy['meshes']] == [
            m['posterior']['epsilon_sha256'] for m in again['meshes']]
    assert torch.equal(train_noise.get_state(),train_state)
    assert torch.equal(torch.get_rng_state(),global_state)


def test_checkpoint_commit_fault_windows():
    model = torch.nn.Linear(3,2)
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
        run = Path(td)
        def value(step): return dict(model=model.state_dict(), completed_updates=step)
        first = commit_checkpoint(run,START+20,None,model,lambda:value(START+20),'epoch_recovery')
        assert Path(first['path']).name == 'recovery-a.pt'
        first_sha = sha(first['path'])
        original_write = train_vae.write
        def fail_index(path, data):
            if Path(path).name == 'recovery-latest.json': raise RuntimeError('crash before index commit')
            return original_write(path,data)
        try:
            with patch.object(train_vae,'write',side_effect=fail_index):
                commit_checkpoint(run,START+21,first,model,lambda:value(START+21),'STOP')
        except RuntimeError: pass
        else: raise AssertionError('Crash was not injected')
        assert sha(first['path']) == first_sha
        assert json.loads((run/'recovery-latest.json').read_text()) == first
        second = commit_checkpoint(run,START+21,first,model,lambda:value(START+21),'STOP')
        assert Path(second['path']).name == 'recovery-b.pt'
        assert sha(first['path']) == first_sha
        assert json.loads((run/'recovery-latest.json').read_text()) == second
        orphan = run/'checkpoints/vae-0100.pt'; orphan.parent.mkdir()
        orphan.write_bytes(b'uncommitted orphan')
        milestone = commit_checkpoint(run,START+100,second,model,lambda:value(START+100),'milestone')
        assert milestone['path'] == str(orphan) and sha(orphan) == milestone['sha256']
        assert (orphan.with_suffix('.json')).exists()
        reused = commit_checkpoint(run,START+100,milestone,model,
                                   lambda: (_ for _ in ()).throw(AssertionError('rewrote valid point')),'milestone')
        assert reused == milestone
        with torch.no_grad(): model.weight.add_(1)
        try: commit_checkpoint(run,START+100,milestone,model,lambda:value(START+100),'milestone')
        except AssertionError: pass
        else: raise AssertionError('Reused a milestone with a different model')


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.cuda.init = lambda *a, **k: (_ for _ in ()).throw(AssertionError('CUDA forbidden'))
    torch.cuda._lazy_init = torch.cuda.init
    tests = (test_native_and_recompute, test_optimizer_inheritance_and_deltas,
             test_kl_eval_rng_and_order, test_audit_and_faces,
             test_checkpoint_resume_next_sample_and_log,
             test_deployment_config_protocol_fields,
             test_full_evaluator_actual_sampled_forward_cpu,
             test_checkpoint_commit_fault_windows)
    for test in tests:
        test(); print('PASS', test.__name__, flush=True)


if __name__ == '__main__': main()
