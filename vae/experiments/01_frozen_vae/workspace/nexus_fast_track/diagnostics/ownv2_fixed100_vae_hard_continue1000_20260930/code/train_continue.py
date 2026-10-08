"""Finite B35220 VAE continuation; preserve the fixed1509 hard negatives."""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
import numpy as np
from run_support import (torch, configure, read, write, sha, code_hashes, move_item,
                         tensor_hash, save_torch, rng_state, restore_rng)
from native_models import NativeTopologyAE, Config, Graph
from data_objective import load_dataset, materialize_item, epoch_batches, objective
from runtime_fixed100 import next_state
from evaluate_checkpoint import evaluate
from vae_protocol import (kl_parts, posterior_stats, optimizer_group_names, group_steps,
                          gradient_norm, snapshot_parameters, measured_deltas)
from pair_negatives import load_hard_pool, select_negatives

ROOT = Path(__file__).resolve().parent.parent
START, UPDATES, BETA = 35220, 1000, 1e-6
MILESTONES = (0, 100, 250, 500, 750, 1000)
EVAL_SEEDS = [861001, 861002, 861003, 861004, 861005]
PARENT_SHA = 'a1d650c12a4c86350f8645a0fc32e81abac550b563f979349c3946f08f12c2a9'


def digest(value):
    h = hashlib.sha256()
    def add(x):
        if torch.is_tensor(x):
            a = x.detach().cpu().contiguous().numpy()
            h.update(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
        elif isinstance(x, np.ndarray):
            h.update(str(x.dtype).encode()+str(x.shape).encode()+x.tobytes())
        elif isinstance(x, dict):
            for k,v in x.items(): add(k); add(v)
        elif isinstance(x, (list,tuple)):
            h.update(type(x).__name__.encode())
            for v in x:add(v)
        else:h.update(repr(x).encode())
    add(value)
    return h.hexdigest()


def verify_assigned_gpu(uuid):
    raw = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],text=True)
    found = [row for row in raw.splitlines() if row.split(',')[0].strip()==uuid]
    assert len(found)==1 and 'A100' in found[0]
    active = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
    assert not any(uuid in row for row in active.splitlines()), 'Assigned GPU is occupied'
    return found[0]


def validated_config(path):
    c = read(path)
    assert c['branch'] == 'B_hard'
    assert Path(c['run_directory']).resolve()==ROOT/'run'
    expected = dict(parent_sha256=PARENT_SHA,source_step=START,new_updates=UPDATES,beta=BETA,
        lr=1e-4,logvar_lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=.01,
        clip=1.,mesh_per_update=5,warmup=False,negative_seed=0,
        mu_checkpoints=list(MILESTONES),noise_checkpoints=[0,UPDATES],evaluation_seeds=EVAL_SEEDS)
    for k,v in expected.items():assert c[k]==v,(k,c[k],v)
    assert sha(c['hard_pool_path'])==c['hard_pool_sha256']
    assert sha(c['negative_schedule_path'])==c['negative_schedule_sha256']
    assert sha(c['parent_baseline'])==c['parent_baseline_sha256']
    return c


def batch_at(data, completed):
    epoch, group = divmod(completed,20)
    return epoch,group,epoch_batches(data['uids'],epoch,0)[group]


def branch_cursor(items,data,completed,config,pools):
    cursor = next_state(items,data['uids'],completed,0)
    cursor['next_uniform_negative_hashes'] = cursor.pop('next_negative_hashes')
    cursor['next_negative_hashes'] = {u:select_negatives(items[u],cursor['epoch'],config['branch'],pools)[1]['negative_sha256']
                                    for u in cursor['next_uids']}
    return cursor


def check_baseline(result, reference):
    assert result['complete'] and result['joint_perfect']==reference['joint_perfect']==31
    assert result['perfect_uids']==reference['perfect_uids']
    for task,fp,fn in [('edge',97,52),('face',1430,98)]:
        assert result[task]['fp']==reference[task]['fp']==fp
        assert result[task]['fn']==reference[task]['fn']==fn
    assert len(result['meshes'])==len(reference['meshes'])==100
    for a,b in zip(result['meshes'],reference['meshes']):
        for k in ['uid','vertices','edge','face','face_fn_missing','face_fn_present','actual_face_candidates']:
            assert a[k]==b[k],(a['uid'],k,a[k],b[k])


def restored_optimizer(model,cp):
    model.requires_grad_(True)
    lookup = dict(model.named_parameters())
    names = cp['optimizer_group_parameter_names']
    assert [len(x) for x in names]==[412,2]
    assert names[1]==['log_variance.weight','log_variance.bias']
    assert len(set(sum(names,[])))==len(lookup)==414
    opt = torch.optim.AdamW([dict(params=[lookup[n] for n in group]) for group in names],
        lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01,foreach=True)
    opt.load_state_dict(cp['optimizer'])
    assert optimizer_group_names(model,opt)==names
    for g in opt.param_groups:
        assert g['lr']==1e-4 and g['betas']==(.9,.999) and g['eps']==1e-8
        assert g['weight_decay']==.01 and g['foreach'] is True
    assert digest(opt.state_dict())==digest(cp['optimizer']), 'Adam state did not restore exactly'
    return opt


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
    config=validated_config(config_path)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==config['gpu_uuid']
    hardware=verify_assigned_gpu(config['gpu_uuid'])
    run=Path(config['run_directory']).resolve()
    if mode=='start':assert not run.exists(),'Existing branch: use resume'
    else:assert (run/'recovery-latest.json').exists() and read(run/'config.json')==config
    run.mkdir(parents=True,exist_ok=True)
    lock=(run/'branch.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    model=opt=noise=None
    completed=START;update_in_progress=False;latest=None;buffer=[]
    try:
        configure(0);torch.cuda.set_device(0)
        items,data=load_dataset(config['data_source'])
        pools,pool_manifest=load_hard_pool(config['hard_pool_path'],items,config['hard_pool_source_sha256'])
        schedule={(r['update'],r['uid']):r for r in read(config['negative_schedule_path'])}
        assert len(schedule)==5000
        baseline=read(config['parent_baseline'])
        assert baseline['identity']['checkpoint_sha256']==PARENT_SHA
        if mode=='start':
            assert sha(config['parent_checkpoint'])==PARENT_SHA
            cp=torch.load(config['parent_checkpoint'],map_location='cpu',weights_only=False)
            assert cp['completed_updates']==START and cp['beta']==BETA and cp['no_scheduler']
            assert cp['code_sha256']==config['parent_code_hashes']
            assert cp['data']==data and cp['cursor']==branch_cursor(items,data,START,config,pools)
            assert tensor_hash(cp['model'])==config['parent_model_sha256']
        else:
            latest=read(run/'recovery-latest.json');assert sha(latest['path'])==latest['sha256']
            cp=torch.load(latest['path'],map_location='cpu',weights_only=False)
            assert cp['config']==config and cp['data']==data and cp['code_sha256']==code_hashes()
            assert cp['source_checkpoint_sha256']==PARENT_SHA and cp['hard_pool_sha256']==sha(config['hard_pool_path'])
            completed=cp['completed_updates'];assert START<=completed<=START+UPDATES
            assert cp['cursor']==branch_cursor(items,data,completed,config,pools)
            buffer=cp['unlogged_records'];append_committed_logs(run,buffer);buffer=[]
        model=NativeTopologyAE(Config(**cp['model_config'])).cuda().float()
        model.load_state_dict(cp['model'],strict=True)
        assert tensor_hash(model.state_dict())==tensor_hash(cp['model'])
        opt=restored_optimizer(model,cp)
        assert group_steps(opt)==[[completed],[completed-34220]]
        restore_rng(cp['rng']);noise=torch.Generator(device='cuda');noise.set_state(cp['train_noise_rng'])
        assert digest(rng_state())==digest(cp['rng']) and torch.equal(noise.get_state(),cp['train_noise_rng'])
        participation=cp['participation'].copy()
        expected={u:completed//20 for u in data['uids']}
        cursor=next_state(items,data['uids'],completed,0)
        for uid in cursor['order'][:cursor['next_mesh_position']]:expected[uid]+=1
        assert participation==expected
        restoration=dict(model_exact=True,optimizer_exact=True,rng_exact=True,train_noise_rng_exact=True,
            restored_update=completed,optimizer_steps=group_steps(opt),model_sha256=tensor_hash(model.state_dict()),
            optimizer_sha256=digest(opt.state_dict()),rng_sha256=digest(rng_state()),noise_rng_sha256=digest(noise.get_state()),
            no_logvar_initialization=True,no_optimizer_reset=True,optimizer_group_parameter_names=optimizer_group_names(model,opt))
        write(run/f'restore-{completed:08d}.json',restoration)
        del cp
        if mode=='start':write(run/'config.json',config);write(run/'data_manifest.json',data)
        write(run/'runtime.json',dict(hardware=hardware,torch=torch.__version__,cuda=torch.version.cuda,
            code_sha256=code_hashes(),parent_sha256=PARENT_SHA,hard_pool_sha256=pool_manifest['file_sha256'],
            optimizer_group_parameter_names=optimizer_group_names(model,opt),model_config=model.cfg.to_dict(),finite_update_limit=UPDATES))
        def save(reason):
            nonlocal latest, buffer
            assert not update_in_progress
            def value():
                return dict(model=model.state_dict(), optimizer=opt.state_dict(), rng=rng_state(),
                    train_noise_rng=noise.get_state(), completed_updates=completed,
                    participation=participation, cursor=branch_cursor(items, data, completed, config, pools),
                    config=config, data=data, model_config=model.cfg.to_dict(),
                    optimizer_group_parameter_names=optimizer_group_names(model, opt),
                    source_checkpoint_sha256=config['parent_sha256'], source_model_sha256=config['parent_model_sha256'],
                    hard_pool_sha256=sha(config['hard_pool_path']),
                    source_baseline_sha256=sha(config['parent_baseline']),
                    code_sha256=code_hashes(), unlogged_records=buffer,
                    beta=BETA, no_scheduler=True)
            entry = commit_checkpoint(run, completed, latest, model, value, reason)
            append_committed_logs(run, buffer); buffer = []
            latest = entry
            return entry

        class Budget:
            def stop(self): return (run/'STOP').exists() or (ROOT/'STOP').exists()
        budget = Budget()

        def milestone():
            relative = completed-START
            assert relative in MILESTONES
            entry = latest if latest and latest['completed_updates'] == completed else save('milestone')
            seeds = [None] + (EVAL_SEEDS if relative in (0, UPDATES) else [])
            # At step 0, the mu reproduction is a hard gate before any sampled evaluation.
            for seed in seeds:
                if relative == 0 and seed is not None:
                    # Reuse only the already complete evaluation of this exact parent.
                    source = Path(config['parent_baseline']).parent.parent/f'noise-{seed}'/'complete.json'
                    result = read(source)
                    assert result['complete'] and result['native_forward'] and len(result['meshes']) == 100
                    assert result['identity']['checkpoint_sha256'] == PARENT_SHA
                    assert result['identity']['evaluation_seed'] == seed
                    folder = run/'evaluations'/f'update-{completed:08d}'/f'noise-{seed}'
                    write(folder/'complete.json', result)
                    write(folder/'reused_parent_evaluation.json', dict(source=str(source),
                          source_sha256=sha(source), checkpoint_sha256=PARENT_SHA,
                          model_state_sha256=tensor_hash(model.state_dict()),
                          evaluation_executed_in_this_run=False,
                          reason='Exact complete B35220 parent, same immutable noise conditions'))
                    continue
                noise_before = noise.get_state().clone()
                global_before = digest(rng_state())
                result = evaluate(model, items, data, entry, run, budget, sample_seed=seed)
                assert torch.equal(noise_before, noise.get_state()) and global_before == digest(rng_state())
                if not result['complete']: return False
                if relative == 0 and seed is None:
                    check_baseline(result, baseline)
                    write(run/'step0_mu_gate.json', dict(passed=True, joint_perfect=31,
                          edge=result['edge'], face=result['face'], evaluation=str(
                              run/'evaluations'/f'update-{completed:08d}'/'mu'/'complete.json')))
            return True

        if mode == 'start': save('same_parent_vae_resume_no_optimizer_update')
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
                negatives, negative_meta = select_negatives(items[uid], epoch, config['branch'], pools, 0)
                planned = schedule[(completed+1, uid)][config['branch'][0]]
                assert negative_meta == planned
                graph = Graph.from_faces(item['faces'], len(item['vertices']))
                rows = model(item['vertices'], item['faces'], sample_latent=True, graph=graph, generator=noise)
                rec, parts = objective(rows, item, negatives)
                k_mu, k_sigma = kl_parts(rows['mu'], rows['log_variance'])
                total = rec+BETA*(k_mu+k_sigma)
                assert all(torch.isfinite(x) for x in (rec, k_mu, k_sigma, total)), 'Nonfinite loss; no step'
                (total/5).backward()
                details.append(dict(uid=uid, **negative_meta, **parts,
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
            assert group_steps(opt) == [[completed], [completed-34220]]
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
            parent_joint_perfect=31, branch=config['branch'], parent_sha256=config['parent_sha256']))
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
