"""One fixed100 V2 mainline. Fresh/random or exact update-boundary resume."""
import argparse
import os
import time
import traceback
from pathlib import Path
from run_support import (torch, configure, read, write, sha, code_hashes, move_item,
                         tensor_hash, save_torch)
from native_models import NativeTopologyAE, Config, Graph
from data_objective import load_dataset, materialize_item, epoch_batches, negative_faces
from data_objective import objective
from runtime_fixed100 import (validated_config, verify_gpu, lock_mainline, claim_run,
                              Budget, build_checkpoint, restore_checkpoint)
from evaluate_checkpoint import evaluate


def main(config_path, mode):
    config = validated_config(config_path)  # Fail before any CUDA use or run creation.
    hardware = verify_gpu(config['gpu_uuid'])
    run = Path(config['run_directory']); lock = lock_mainline(config['mainline_lock'])
    if mode == 'fresh':
        assert not run.exists(), 'Existing run: use explicit resume, never implicit overwrite'
    else:
        assert (run/'recovery-latest.json').exists(), 'Full resume checkpoint is required'
        assert read(run/'config.json') == config
    claim_run(lock, run)
    run.mkdir(parents=True, exist_ok=True)
    items, data = load_dataset(config['data_source'])  # CPU identity check, no old pools.
    uids = data['uids']; largest = max(uids, key=lambda u:len(items[u]['vertices']))
    write(run/'data_manifest.json', data)
    write(run/'config.json', config)
    budget = Budget(run, config); completed = 0; participation = {u:0 for u in uids}
    model = opt = None; latest_entry = None; update_in_progress = False
    try:
        configure(config['seed']); torch.cuda.set_device(0)
        model = NativeTopologyAE(Config(model_variant='B_v2_teacher_blocks')).cuda().float().train()
        active = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(active, lr=config['lr']*(1/100), betas=tuple(config['betas']),
            eps=config['eps'], weight_decay=config['weight_decay'], foreach=True)
        frozen_hash = tensor_hash(model.log_variance.state_dict())
        if mode != 'fresh':
            entry = read(run/'recovery-latest.json'); assert sha(entry['path']) == entry['sha256']
            log_path = run/'updates.jsonl'
            if log_path.exists():
                with log_path.open('rb') as stream:
                    stream.seek(0, 2); size=stream.tell(); stream.seek(max(0,size-1024*1024))
                    last=stream.read().splitlines()[-1:]
                if last:
                    assert __import__('json').loads(last[0])['update'] == entry['completed_updates'], 'Logged updates exceed durable checkpoint; implicit replay is prohibited'
            cp = torch.load(entry['path'], map_location='cpu', weights_only=False)
            completed, participation = restore_checkpoint(cp, model, opt, items, data, config)
            frozen_hash = cp['frozen_logvar_hash']; latest_entry = entry
            assert tensor_hash(model.log_variance.state_dict()) == frozen_hash
            del cp
        else:
            assert not opt.state
        write(run/'runtime.json', dict(hardware=hardware, torch=torch.__version__, cuda=torch.version.cuda,
            pid=os.getpid(), source_sha256=code_hashes(), source_paths={k:str(Path(__file__).parent/k) for k in code_hashes()},
            model_config=model.cfg.to_dict(), trainable_parameter_names=[n for n,p in model.named_parameters() if p.requires_grad],
            trainable_parameters=sum(p.numel() for p in active), initialization='seed0 random, fresh AdamW' if mode=='fresh' else 'exact full recovery',
            GPU_preflight_updates=0))

        def save(immutable=False, reason='boundary'):
            assert not update_in_progress
            assert tensor_hash(model.log_variance.state_dict()) == frozen_hash
            previous = read(run/'recovery-latest.json') if (run/'recovery-latest.json').exists() else {}
            slot = 'recovery-b.pt' if Path(previous.get('path','')).name == 'recovery-a.pt' else 'recovery-a.pt'
            path = run/'checkpoints'/f'update-{completed:08d}.pt' if immutable else run/slot
            path.parent.mkdir(exist_ok=True)
            if immutable and path.exists():
                entry = read(path.with_suffix('.json'))
                assert sha(path) == entry['sha256'] and tensor_hash(model.state_dict()) == entry['model_state_sha256']
                return entry
            value = build_checkpoint(model, opt, completed, participation, config, data, items, frozen_hash)
            entry = save_torch(path, value)
            entry.update(completed_updates=completed, model_state_sha256=tensor_hash(model.state_dict()), reason=reason)
            write(path.with_suffix('.json'), entry)
            if not immutable: write(run/'recovery-latest.json', entry)
            return entry

        if mode == 'fresh': latest_entry = save(reason='initial_random_no_optimizer_updates')
        # Resume incomplete scheduled evaluation before changing its checkpoint state.
        pending = run/'pending_evaluation.json'
        if mode == 'evaluate' or (pending.exists() and config.get('evaluate_during_training', True)):
            entry = read(pending) if pending.exists() else save(True, 'evaluation_only')
            assert entry['completed_updates'] == completed
            result = evaluate(model, items, data, entry, run, budget)
            if result['complete'] and pending.exists(): pending.unlink()
            if mode == 'evaluate' or not result['complete']: return

        recent_seconds = []; last_eval_seconds = 0.; stop_reason = 'manual_stop'
        with (run/'updates.jsonl').open('a', buffering=1) as log:
            while True:
                if (run/'FINISH_TRAINING').exists():
                    stop_reason = 'requested_training_finish'; break
                limit = config.get('max_new_updates')
                if limit is not None and completed >= limit:
                    stop_reason = 'update_limit'; break
                reserve = max(300., last_eval_seconds) if config.get('gpu_hours') else 0.
                allowance = max(recent_seconds[-20:], default=30.)*2
                if budget.stop(reserve+allowance if reserve else 0):
                    stop_reason = 'manual_stop' if (run/'STOP').exists() else 'time_reserve'; break
                step = completed+1; epoch, batch_index = divmod(completed, 20)
                batch = epoch_batches(uids, epoch, config['seed'])[batch_index]
                lr = config['lr']*min(step/100, 1.)
                for group in opt.param_groups: group['lr'] = lr
                opt.zero_grad(set_to_none=True); details = []
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
                update_in_progress = True
                micro_seconds = []
                for uid in batch:
                    if budget.stop():
                        # No step has run. Discard partial accumulation; recovery remains at prior boundary.
                        opt.zero_grad(set_to_none=True); update_in_progress = False
                        break
                    tic = time.monotonic(); cpu_item = materialize_item(items[uid])
                    neg, neg_sha = negative_faces(items[uid], epoch, config['seed'])
                    item = move_item(cpu_item, 'cuda'); graph = Graph.from_faces(item['faces'], len(item['vertices']))
                    rows = model(item['vertices'], item['faces'], sample_latent=False, graph=graph)
                    assert rows['latent'] is rows['mu']
                    loss, parts = objective(rows, item, neg)
                    assert torch.isfinite(loss), 'Nonfinite reconstruction loss'
                    (loss/5).backward()
                    details.append(dict(uid=uid, negative_sha256=neg_sha, **parts))
                    torch.cuda.synchronize(); micro_seconds.append(time.monotonic()-tic)
                    del rows, loss, item, graph, cpu_item, neg
                if len(details) != 5:
                    # Restore the correct last-update LR before serializing a no-update stop.
                    for group in opt.param_groups: group['lr'] = config['lr']*min(max(completed,1)/100,1)
                    stop_reason = 'interrupted_before_optimizer_step'; break
                assert all(p.grad is not None for p in active), 'Trainable parameter disconnected'
                tic = time.monotonic()
                gn = float(torch.nn.utils.clip_grad_norm_(active, 1., error_if_nonfinite=True, foreach=True))
                opt.step(); torch.cuda.synchronize(); step_seconds = time.monotonic()-tic
                completed = step; update_in_progress = False
                for uid in batch: participation[uid] += 1
                assert all(p.grad is None for p in model.log_variance.parameters())
                assert all(int(opt.state[p]['step']) == completed for p in active)
                finite = all(bool(torch.isfinite(p).all()) for p in active)
                assert finite, 'Nonfinite parameter after AdamW'
                elapsed = time.monotonic()-started; recent_seconds.append(elapsed)
                record = dict(update=completed, epoch=epoch, group=batch_index, uids=batch, meshes=details,
                    loss_before=sum(x['edge_loss']+x['face_loss'] for x in details)/5,
                    lr=lr, gradient_norm_before_clip=gn, clip_coefficient=min(1.,1/(gn+1e-6)),
                    adam_step=completed, parameters_finite=finite, microbatch_seconds=micro_seconds,
                    clip_and_step_seconds=step_seconds, full_update_seconds=elapsed,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    meshes_participated=sum(participation.values()),
                    timing='loss is before optimizer update; count/Adam state are after completed update')
                log.write(__import__('json').dumps(record, separators=(',',':'), allow_nan=False)+'\n')
                if completed <= 3 or (largest in batch and not (run/'largest_group_performance.json').exists()):
                    write(run/f'performance-update-{completed:08d}.json', record)
                    if largest in batch: write(run/'largest_group_performance.json', record)
                write(run/'status.json', dict(state='training', completed_updates=completed,
                    loss_before=record['loss_before'], last_update_seconds=elapsed, charged_gpu_seconds=budget.update()))
                if completed <= 3 or completed % 20 == 0:
                    print('UPDATE', completed, record['loss_before'], elapsed, flush=True)
                if completed % 20 == 0: latest_entry = save(reason='complete_epoch_recovery')
                if completed % config.get('checkpoint_every', config['evaluation_every']) == 0:
                    save(True, 'scheduled_checkpoint')
                if config.get('evaluate_during_training', True) and completed % config['evaluation_every'] == 0:
                    latest_entry = save(reason='scheduled_evaluation_boundary')
                    entry = save(True, 'scheduled_evaluation'); write(pending, entry)
                    tic = time.monotonic(); result = evaluate(model, items, data, entry, run, budget)
                    last_eval_seconds = time.monotonic()-tic
                    if not result['complete']:
                        stop_reason = 'evaluation_incomplete'; break
                    pending.unlink()
                    print('FULL100', completed, result['face']['micro_f1'], result['joint_perfect'], flush=True)
                    if result['face']['micro_f1'] >= .997:
                        write(run/'milestone_face0997.json', dict(checkpoint=entry, face_micro_f1=result['face']['micro_f1'],
                            joint_perfect=result['joint_perfect'], note='Engineering milestone; strict100 is separate'))
        latest_entry = save(reason=stop_reason)
        entry = save(True, stop_reason)
        if config.get('evaluate_after_training', True) and not budget.stop() and stop_reason not in ('evaluation_incomplete','manual_stop'):
            write(pending, entry)
            result = evaluate(model, items, data, entry, run, budget)
            if result['complete']: pending.unlink()
        write(run/'complete.json', dict(training_stopped=True, reason=stop_reason, completed_updates=completed,
            final_checkpoint=entry, full_evaluation_pending=pending.exists(), charged_gpu_seconds=budget.update()))
    except BaseException as e:
        emergency_entry = None
        if model is not None and opt is not None:
            try:
                # Preserve the failure scene; partial accumulations/Adam mutations
                # are explicitly NOT accepted by the normal resume entrypoint.
                emergency_entry = save_torch(run/'failure-scene-not-resumable.pt', dict(
                    model=model.state_dict(), optimizer=opt.state_dict(),
                    completed_updates=completed, partial_update=update_in_progress,
                    resumable=False))
            except BaseException as save_error:
                emergency_entry = dict(save_error=str(save_error))
        write(run/'failure.json', dict(error=str(e), traceback=traceback.format_exc(), completed_updates=completed,
            update_in_progress=update_in_progress, recover_from=latest_entry,
            emergency_scene=emergency_entry,
            note='No automatic optimizer reset, precision change or retry'))
        raise
    finally:
        budget.update(finish=True)
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['fresh','resume','evaluate'], required=True)
    args = parser.parse_args(); main(args.config, args.mode)
