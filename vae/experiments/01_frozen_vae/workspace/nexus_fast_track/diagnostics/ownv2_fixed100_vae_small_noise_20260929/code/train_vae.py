"""Bounded step34220 -> 34720 VAE continuation of the original fixed100 AE."""
import argparse
import fcntl
import json
import math
import os
import subprocess
import time
import traceback
from pathlib import Path

from run_support import (torch, configure, read, write, sha, code_hashes, move_item,
                         tensor_hash, save_torch, rng_state, restore_rng)
from native_models import NativeTopologyAE, Config, Graph
from data_objective import load_dataset, materialize_item, epoch_batches, negative_faces, objective
from runtime_fixed100 import next_state
from evaluate_checkpoint import evaluate
from vae_protocol import (BETA, START, UPDATES, MILESTONES, SOURCE_SHA, SOURCE_MODEL_SHA,
                          kl_parts, posterior_stats, make_optimizer, old_parameter_names,
                          new_parameter_names, optimizer_group_names, group_steps,
                          gradient_norm, snapshot_parameters, measured_deltas)

ROOT = Path(__file__).resolve().parent.parent
GPU_UUID = 'GPU-3534263c-6584-9f34-9273-0ef6a7852fbe'
EVAL_SEEDS = [861001, 861002, 861003, 861004, 861005]


def verify_assigned_gpu(uuid):
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name', '--format=csv,noheader'], text=True)
    found = [row for row in raw.splitlines() if row.split(',')[0].strip() == uuid]
    assert len(found) == 1, 'Assigned GPU UUID is absent or ambiguous'
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                      '--format=csv,noheader'], text=True)
    assert not any(uuid in row for row in active.splitlines()), 'Assigned GPU already has a compute process'
    return found[0]


def validated_config(path):
    c = read(path)
    for key in ('source_checkpoint', 'source_baseline', 'data_source', 'run_directory',
                'gpu_uuid', 'train_noise_seed', 'evaluation_seeds'):
        assert key in c, f'Missing {key}'
    assert c['gpu_uuid'] == GPU_UUID
    assert c['train_noise_seed'] == 20260929 and c['evaluation_seeds'] == EVAL_SEEDS
    assert Path(c['source_checkpoint']).resolve() == (ROOT/'source/ae-step34220.pt').resolve()
    assert Path(c['source_baseline']).resolve() == (ROOT/'source/ae_mu_baseline.json').resolve()
    run = Path(c['run_directory']).resolve()
    assert run.parent == ROOT.resolve() and run.name.startswith('run'), 'Run must be an independent branch directory under the project root'
    assert c.get('source_step', START) == START and c.get('new_updates', UPDATES) == UPDATES
    assert c.get('beta', BETA) == BETA
    expected = dict(reconstruction_lr=1e-4, logvar_lr=1e-4, adamw_betas=[.9, .999],
        adamw_eps=1e-8, weight_decay=.01, global_clip=1., mesh_per_update=5,
        warmup=False, mu_checkpoints=list(MILESTONES), noise_checkpoints=[0, UPDATES],
        source_ae_passed_old_threshold=False, source_selection_user_confirmed=True)
    for key, value in expected.items():
        assert c.get(key) == value, (key, c.get(key), value)
    return c


def source_code_hashes():
    observed = {p.name: sha(p) for p in (ROOT/'source_snapshot').glob('*.py')}
    expected = read(ROOT/'source_snapshot_sha256.json')
    assert all(observed[name] == value for name, value in expected.items() if name.endswith('.py'))
    return observed


def validate_source(cp, data, items):
    assert cp['completed_updates'] == START and cp['model_config']['model_variant'] == 'B_v2_teacher_blocks'
    assert cp['source_sha256'] == source_code_hashes()
    assert cp['data'] == data and cp['cursor'] == next_state(items, data['uids'], START, 0)
    assert len(cp['optimizer_parameter_names']) == 412
    assert len(cp['optimizer']['param_groups']) == 1
    assert cp['optimizer']['param_groups'][0]['lr'] == 1e-4
    assert cp['scheduler']['completed_updates'] == START and cp['scheduler']['last_lr'] == 1e-4
    assert tensor_hash({k.removeprefix('log_variance.'): v for k, v in cp['model'].items()
                        if k.startswith('log_variance.')}) == cp['frozen_logvar_hash']
    assert sum(cp['participation'].values()) == START*5
    assert all(v == 1711 for v in cp['participation'].values())
    assert tensor_hash(cp['model']) == SOURCE_MODEL_SHA


def check_baseline(result, reference):
    assert result['complete'] and len(result['meshes']) == len(reference['meshes']) == 100
    assert result['joint_perfect'] == reference['joint_perfect'] == 28
    for task, fp, fn in (('edge', 51, 64), ('face', 1508, 123)):
        assert result[task]['fp'] == reference[task]['fp'] == fp
        assert result[task]['fn'] == reference[task]['fn'] == fn
        assert result[task]['tp'] == reference[task]['tp']
        assert result[task]['tn'] == reference[task]['tn']
    for actual, expected in zip(result['meshes'], reference['meshes']):
        assert actual['uid'] == expected['uid']
        for key in ('vertices', 'edge', 'face', 'face_fn_missing', 'face_fn_present', 'actual_face_candidates'):
            assert actual[key] == expected[key], (actual['uid'], key, actual[key], expected[key])


def batch_at(data, completed):
    epoch, group = divmod(completed, 20)
    return epoch, group, epoch_batches(data['uids'], epoch, 0)[group]


def probe_gradients(model, opt, items, data, completed, noise, device='cuda'):
    """One real next batch, separate preclip reconstruction and KL gradients; no step."""
    assert completed == START
    epoch, group, batch = batch_at(data, completed)
    saved_noise = noise.get_state().clone(); saved_global = rng_state()
    parameters = [p for g in opt.param_groups for p in g['params']]
    kl_grads = [None]*len(parameters)
    meshes = []
    opt.zero_grad(set_to_none=True)
    try:
        model.train()
        for uid in batch:
            item = move_item(materialize_item(items[uid]), device)
            negatives, neg_hash = negative_faces(items[uid], epoch, 0)
            graph = Graph.from_faces(item['faces'], len(item['vertices']))
            rows = model(item['vertices'], item['faces'], sample_latent=True, graph=graph, generator=noise)
            rec, parts = objective(rows, item, negatives)
            k_mu, k_sigma = kl_parts(rows['mu'], rows['log_variance'])
            assert all(torch.isfinite(x) for x in (rec, k_mu, k_sigma))
            grads = torch.autograd.grad((k_mu+k_sigma)/5, parameters, retain_graph=True, allow_unused=True)
            for i, grad in enumerate(grads):
                if grad is not None:
                    kl_grads[i] = grad.detach().clone() if kl_grads[i] is None else kl_grads[i]+grad.detach()
            (rec/5).backward()
            meshes.append(dict(uid=uid, negative_sha256=neg_hash, reconstruction=float(rec.detach()),
                               k_mu=float(k_mu.detach()), k_sigma=float(k_sigma.detach()),
                               posterior=posterior_stats(rows), edge_loss=parts['edge_loss'], face_loss=parts['face_loss']))
            del rows, rec, grads, item, graph
        assert all(p.grad is not None for p in parameters)
        assert all(torch.isfinite(p.grad).all() for p in parameters)
        assert all(g is None or torch.isfinite(g).all() for g in kl_grads)
        def norm(xs):
            return math.sqrt(sum(float(x.double().square().sum()) for x in xs if x is not None))
        result = dict(scope='next_optimizer_batch_5', epoch=epoch, group=group, uids=batch,
                      meshes=meshes, optimizer_updates=0, beta=BETA,
                      reconstruction_gradient_norm=norm([p.grad for p in parameters]),
                      kl_gradient_norm=norm(kl_grads),
                      beta_kl_gradient_norm=BETA*norm(kl_grads),
                      reconstruction_beta_kl_gradient_inner_product=BETA*sum(
                          float((p.grad.double()*g.double()).sum()) for p, g in zip(parameters, kl_grads) if g is not None),
                      total_gradient_norm=norm([p.grad+BETA*g if g is not None else p.grad
                                                for p, g in zip(parameters, kl_grads)]),
                      reconstruction_group_gradient_norms=[gradient_norm(g['params']) for g in opt.param_groups],
                      kl_group_gradient_norms=[norm(kl_grads[:len(opt.param_groups[0]['params'])]),
                                               norm(kl_grads[len(opt.param_groups[0]['params']):])],
                      logvar_reconstruction_gradient_nonzero=all(p.grad.abs().max() > 0 for p in opt.param_groups[1]['params']),
                      logvar_kl_gradient_nonzero=all(g is not None and g.abs().max() > 0 for g in kl_grads[-2:]))
        assert result['logvar_reconstruction_gradient_nonzero'] and result['logvar_kl_gradient_nonzero']
        return result
    finally:
        opt.zero_grad(set_to_none=True)
        noise.set_state(saved_noise)
        restore_rng(saved_global)


def append_committed_logs(run, records):
    path = run/'updates.jsonl'
    last = START
    if path.exists():
        with path.open('rb') as f:
            for line in f:
                current = json.loads(line)['update']
                assert current == last+1, 'Committed update log is not consecutive'
                last = current
    with path.open('a', buffering=1) as f:
        for record in records:
            if record['update'] > last:
                assert record['update'] == last+1
                f.write(json.dumps(record, separators=(',', ':'), allow_nan=False)+'\n')
                last = record['update']


def commit_checkpoint(run, completed, latest, model, build_value, reason):
    """Commit data and metadata before publishing the recovery index."""
    relative = completed-START
    permanent = relative in MILESTONES
    if permanent:
        path = run/'checkpoints'/f'vae-{relative:04d}.pt'
    else:
        previous_slot = Path(latest['path']).name if latest is not None else ''
        path = run/('recovery-b.pt' if previous_slot == 'recovery-a.pt' else 'recovery-a.pt')
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = path.with_suffix('.json')
    model_hash = tensor_hash(model.state_dict())
    if permanent and path.exists() and metadata.exists():
        entry = read(metadata)
        assert entry['path'] == str(path) and entry['completed_updates'] == completed
        assert entry['model_state_sha256'] == model_hash and sha(path) == entry['sha256']
    else:
        entry = save_torch(path, build_value())
        entry.update(completed_updates=completed, new_updates=relative, reason=reason,
                     model_state_sha256=model_hash)
        write(metadata, entry)
    write(run/'recovery-latest.json', entry)
    return entry


def main(config_path, mode):
    config = validated_config(config_path)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == GPU_UUID
    hardware = verify_assigned_gpu(GPU_UUID)
    run = Path(config['run_directory']).resolve()
    if mode == 'start': assert not run.exists(), 'Existing run: use resume'
    else: assert (run/'recovery-latest.json').exists() and read(run/'config.json') == config
    run.mkdir(parents=True, exist_ok=True)
    lock = (run/'branch.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    model = opt = noise = None
    completed = START; update_in_progress = False; latest = None; buffer = []
    try:
        configure(0); torch.cuda.set_device(0)
        items, data = load_dataset(config['data_source'])
        baseline = read(config['source_baseline'])
        assert baseline['complete'] and baseline['identity']['checkpoint_sha256'] == SOURCE_SHA
        assert baseline['identity']['manifest_sha256'] == data['manifest_sha256']
        if mode == 'start':
            assert sha(config['source_checkpoint']) == SOURCE_SHA
            cp = torch.load(config['source_checkpoint'], map_location='cpu', weights_only=False)
            validate_source(cp, data, items)
            model = NativeTopologyAE(Config(**cp['model_config'])).cuda().float()
            assert cp['optimizer_parameter_names'] == old_parameter_names(model)
            model.load_state_dict(cp['model'], strict=True)
            opt = make_optimizer(model, cp['optimizer'])
            assert optimizer_group_names(model, opt) == [cp['optimizer_parameter_names'], new_parameter_names(model)]
            restore_rng(cp['rng']); noise = torch.Generator(device='cuda').manual_seed(config['train_noise_seed'])
            participation = cp['participation'].copy()
            del cp
            write(run/'config.json', config); write(run/'data_manifest.json', data)
        else:
            latest = read(run/'recovery-latest.json')
            assert sha(latest['path']) == latest['sha256']
            cp = torch.load(latest['path'], map_location='cpu', weights_only=False)
            assert cp['config'] == config and cp['data'] == data and cp['source_checkpoint_sha256'] == SOURCE_SHA
            assert cp['code_sha256'] == code_hashes()
            assert cp['source_snapshot_sha256'] == source_code_hashes()
            assert cp['source_baseline_sha256'] == sha(config['source_baseline'])
            completed = cp['completed_updates']; assert START <= completed <= START+UPDATES
            assert cp['cursor'] == next_state(items, data['uids'], completed, 0)
            model = NativeTopologyAE(Config(**cp['model_config'])).cuda().float()
            opt = make_optimizer(model)
            assert cp['optimizer_group_parameter_names'] == optimizer_group_names(model, opt)
            model.load_state_dict(cp['model'], strict=True); opt.load_state_dict(cp['optimizer'])
            assert group_steps(opt) == [[completed], [completed-START]] if completed > START else group_steps(opt) == [[START], []]
            restore_rng(cp['rng']); noise = torch.Generator(device='cuda'); noise.set_state(cp['train_noise_rng'])
            participation = cp['participation'].copy(); buffer = cp['unlogged_records']
            assert sum(participation.values()) == completed*5
            expected = {u: completed//20 for u in data['uids']}
            for uid in cp['cursor']['order'][:cp['cursor']['next_mesh_position']]: expected[uid] += 1
            assert participation == expected
            append_committed_logs(run, buffer); buffer = []
            del cp
        assert all(g['lr'] == 1e-4 for g in opt.param_groups)
        write(run/'runtime.json', dict(hardware=hardware, torch=torch.__version__, cuda=torch.version.cuda,
            code_sha256=code_hashes(), source_snapshot_sha256=source_code_hashes(),
            source_checkpoint_sha256=SOURCE_SHA, source_model_sha256=SOURCE_MODEL_SHA,
            optimizer_group_parameter_names=optimizer_group_names(model, opt),
            model_config=model.cfg.to_dict(), finite_update_limit=UPDATES))

        def save(reason):
            nonlocal latest, buffer
            assert not update_in_progress
            def value():
                return dict(model=model.state_dict(), optimizer=opt.state_dict(), rng=rng_state(),
                    train_noise_rng=noise.get_state(), completed_updates=completed,
                    participation=participation, cursor=next_state(items, data['uids'], completed, 0),
                    config=config, data=data, model_config=model.cfg.to_dict(),
                    optimizer_group_parameter_names=optimizer_group_names(model, opt),
                    source_checkpoint_sha256=SOURCE_SHA, source_model_sha256=SOURCE_MODEL_SHA,
                    source_snapshot_sha256=source_code_hashes(),
                    source_baseline_sha256=sha(config['source_baseline']),
                    code_sha256=code_hashes(), unlogged_records=buffer,
                    beta=BETA, no_scheduler=True)
            entry = commit_checkpoint(run, completed, latest, model, value, reason)
            append_committed_logs(run, buffer); buffer = []
            latest = entry
            return entry

        class Budget:
            def stop(self): return (run/'STOP').exists()
        budget = Budget()

        def milestone():
            relative = completed-START
            assert relative in MILESTONES
            entry = latest if latest and latest['completed_updates'] == completed else save('milestone')
            seeds = [None] + (EVAL_SEEDS if relative in (0, UPDATES) else [])
            # At step 0, the mu reproduction is a hard gate before any sampled evaluation.
            for seed in seeds:
                result = evaluate(model, items, data, entry, run, budget, sample_seed=seed)
                if not result['complete']: return False
                if relative == 0 and seed is None:
                    check_baseline(result, baseline)
                    write(run/'step0_mu_gate.json', dict(passed=True, joint_perfect=28,
                          edge=result['edge'], face=result['face'], evaluation=str(
                              run/'evaluations'/f'update-{completed:08d}'/'mu'/'complete.json')))
            if relative == 0 and not (run/'step0_gradient_audit.json').exists():
                audit = probe_gradients(model, opt, items, data, completed, noise)
                write(run/'step0_gradient_audit.json', audit)
            return True

        if mode == 'start': save('vae_initialization_no_optimizer_update')
        write(run/'status.json', dict(state='evaluating' if completed-START in MILESTONES else 'training',
            global_step=completed, new_update=completed-START,
            durable_global_step=latest['completed_updates'],
            durable_new_update=latest['completed_updates']-START,
            checkpoint=latest['path']))
        if completed-START in MILESTONES:
            if not milestone(): return
        if mode == 'evaluate': return
        model.train()
        while completed < START+UPDATES:
            if budget.stop():
                if buffer: save('requested_stop_at_update_boundary')
                write(run/'status.json', dict(state='stopped', global_step=completed,
                    new_update=completed-START, durable_global_step=completed,
                    durable_new_update=completed-START, checkpoint=latest['path']))
                return
            epoch, group, batch = batch_at(data, completed)
            opt.zero_grad(set_to_none=True); details = []
            update_in_progress = True
            batch_noise_state = noise.get_state().clone(); batch_global_state = rng_state()
            started = time.monotonic()
            for uid in batch:
                if budget.stop():
                    opt.zero_grad(set_to_none=True); noise.set_state(batch_noise_state)
                    restore_rng(batch_global_state); update_in_progress = False
                    if buffer: save('requested_stop_before_optimizer_step')
                    write(run/'status.json', dict(state='stopped', global_step=completed,
                        new_update=completed-START, durable_global_step=completed,
                        durable_new_update=completed-START, checkpoint=latest['path']))
                    return
                item = move_item(materialize_item(items[uid]), 'cuda')
                negatives, neg_hash = negative_faces(items[uid], epoch, 0)
                graph = Graph.from_faces(item['faces'], len(item['vertices']))
                rows = model(item['vertices'], item['faces'], sample_latent=True, graph=graph, generator=noise)
                rec, parts = objective(rows, item, negatives)
                k_mu, k_sigma = kl_parts(rows['mu'], rows['log_variance'])
                total = rec+BETA*(k_mu+k_sigma)
                assert all(torch.isfinite(x) for x in (rec, k_mu, k_sigma, total)), 'Nonfinite loss; no step'
                (total/5).backward()
                details.append(dict(uid=uid, negative_sha256=neg_hash, **parts,
                    k_mu=float(k_mu.detach()), k_sigma=float(k_sigma.detach()),
                    kl=float((k_mu+k_sigma).detach()), beta=BETA, total=float(total.detach()),
                    posterior=posterior_stats(rows)))
                del rows, rec, total, item, graph, negatives
            if budget.stop():
                opt.zero_grad(set_to_none=True); noise.set_state(batch_noise_state)
                restore_rng(batch_global_state); update_in_progress = False
                if buffer: save('requested_stop_before_optimizer_step')
                write(run/'status.json', dict(state='stopped', global_step=completed,
                    new_update=completed-START, durable_global_step=completed,
                    durable_new_update=completed-START, checkpoint=latest['path']))
                return
            parameters = [p for g in opt.param_groups for p in g['params']]
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
            group_grad = [gradient_norm(g['params']) for g in opt.param_groups]
            grad = float(torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True, foreach=True))
            before = snapshot_parameters(opt)
            opt.step()
            deltas, module_deltas = measured_deltas(model, opt, before); del before
            completed += 1; update_in_progress = False
            for uid in batch: participation[uid] += 1
            assert group_steps(opt) == [[completed], [completed-START]]
            assert all(bool(torch.isfinite(p).all()) for p in parameters), 'Nonfinite parameter after optimizer step'
            record = dict(update=completed, new_update=completed-START, epoch=epoch, group=group,
                uids=batch, meshes=details, reconstruction_edge_mean=sum(x['edge_loss'] for x in details)/5,
                reconstruction_face_mean=sum(x['face_loss'] for x in details)/5,
                kl_mean=sum(x['kl'] for x in details)/5,
                k_mu_mean=sum(x['k_mu'] for x in details)/5,
                k_sigma_mean=sum(x['k_sigma'] for x in details)/5, beta=BETA,
                total_mean=sum(x['total'] for x in details)/5,
                posterior_mean={k:sum(x['posterior'][k] for x in details)/5 for k in
                    ('sigma_mean', 'sigma_rms', 'sigma_p01', 'sigma_p50', 'sigma_p99',
                     'logvar_mean', 'clamp_low_fraction', 'clamp_high_fraction', 'clamp_fraction',
                     'perturbation_rms', 'perturbation_max')},
                gradient_norm_before_clip=grad, group_gradient_norms_before_clip=group_grad,
                clip_coefficient=min(1., 1./(grad+1e-6)),
                optimizer_group_lr=[g['lr'] for g in opt.param_groups],
                optimizer_group_steps=group_steps(opt), optimizer_group_delta_l2_fp32=deltas,
                module_delta_l2_fp32=module_deltas,
                full_update_seconds=time.monotonic()-started,
                timing='loss and gradients before update; Adam steps and measured parameter deltas after update')
            buffer.append(record)
            relative = completed-START
            if relative % 20 == 0 or relative in MILESTONES:
                save('milestone' if relative in MILESTONES else 'epoch_recovery')
            write(run/'status.json', dict(state='evaluating' if relative in MILESTONES else 'training',
                global_step=completed, new_update=relative,
                durable_global_step=latest['completed_updates'],
                durable_new_update=latest['completed_updates']-START,
                checkpoint=latest['path'], total_mean=record['total_mean'],
                full_update_seconds=record['full_update_seconds']))
            if relative in MILESTONES:
                if not milestone(): return
            if relative <= 3 or relative % 20 == 0:
                print('VAE_UPDATE', relative, record['total_mean'], flush=True)
        write(run/'complete.json', dict(completed_updates=completed, new_updates=UPDATES,
            final_checkpoint=latest, evaluations_complete=True,
            source_ae_joint_perfect=28, source_ae_gate_passed=False))
        write(run/'status.json', dict(state='complete', global_step=completed, new_update=UPDATES,
            durable_global_step=completed, durable_new_update=UPDATES,
            checkpoint=latest['path'], evaluations_complete=True))
    except BaseException as exc:
        scene = None
        if model is not None and opt is not None:
            try:
                scene = save_torch(run/'failure-scene-not-resumable.pt', dict(
                    model=model.state_dict(), optimizer=opt.state_dict(),
                    rng=rng_state(), train_noise_rng=None if noise is None else noise.get_state(),
                    completed_updates=completed, partial_update=update_in_progress,
                    resumable=False))
            except BaseException as save_error: scene = dict(save_error=str(save_error))
        write(run/'failure.json', dict(error=str(exc), traceback=traceback.format_exc(),
            completed_updates=completed, update_in_progress=update_in_progress,
            last_durable_checkpoint=latest, emergency_scene=scene,
            note='No automatic optimizer, LR, precision, or beta change'))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=('start', 'resume', 'evaluate'), required=True)
    args = parser.parse_args()
    main(args.config, args.mode)
