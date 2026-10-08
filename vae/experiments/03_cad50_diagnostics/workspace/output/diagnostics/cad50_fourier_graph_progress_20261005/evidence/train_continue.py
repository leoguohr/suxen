"""Full-state CAD50 continuation; explicitly selected GPU, bounded stage, no fresh Adam."""
import argparse
from dataclasses import replace
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
import traceback
import uuid
from protocol import (ADAM, BASE, LIMIT, MILESTONES, PARENTS, ROOT, START, acquire, check_cursor,
    check_log, info, output_folder, parent_paths, read, require, schedule, sha, verify_code, write)


def main(args):
    require(re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', args.gpu_uuid), 'Explicit full GPU UUID required')
    require(bool(args.resume) == bool(args.resume_sha256), '--resume and --resume-sha256 are required together')
    schedule(args.budget_updates)
    require(args.stop_at in MILESTONES[1:], 'Stage must end at an authorized milestone')
    folder = output_folder(args.outdir)
    folder.mkdir(parents=True, exist_ok=True)
    if args.lock_fd is None:
        execution_lock = acquire(folder/'execution.lock')
    else:
        execution_lock = os.fdopen(args.lock_fd, 'a', closefd=False)
        require(os.fstat(args.lock_fd).st_ino == (folder/'execution.lock').stat().st_ino
                and os.fstat(args.lock_fd).st_dev == (folder/'execution.lock').stat().st_dev, 'Wrong inherited execution lock')
        fcntl.flock(execution_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not (folder/'complete.json').exists(), 'Branch already complete; controller must skip it')
    require((folder/'config.json').exists() == bool(args.resume), 'Existing run requires explicit resume; new run requires parent start')
    code = verify_code()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_uuid  # Before importing torch or initializing CUDA.
    from support import configure, rng_state, restore_rng, tensor_hash, save_torch, torch
    from native_models import Config, Graph, NativeTopologyAE
    from data_objective import load_dataset, epoch_batches
    from evaluation import evaluate
    from resume_audit import (audit_restore, backward_batch, compare_evaluations, equal_state,
        fourier_gate, scientific_cfg, validate_adam, validate_parent, xyz_gate)

    attempt = f'{time.time_ns()}-{uuid.uuid4().hex[:8]}'
    audit_dir = folder/'audits'/attempt
    started = time.monotonic()
    completed = START
    model = opt = None
    boundary_safe = False
    stop_requested = []
    stage_finished = False
    last_checkpoint = None
    config = execution = None
    participation = None
    frozen_hash = None
    phase = 'load_checkpoint'

    def request_stop(signum, frame):
        stop_requested.append(signum)  # Finish current accumulation/Adam/log boundary first.

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    try:
        parent_path, expected_path = parent_paths(args.arm, args.parent_base)
        parent = info(parent_path, PARENTS[args.arm]['sha256'])
        if args.resume:
            loaded = info(args.resume, args.resume_sha256)
            require(folder in Path(loaded['path']).parents, 'Resume checkpoint must belong to this independent branch directory')
            cp = torch.load(loaded['path'], map_location='cpu', mmap=True, weights_only=False)
            require(cp.get('resumable') is True, 'Checkpoint not marked as a complete update boundary')
            config = read(folder/'config.json')
            require(cp['config'] == config and config['arm'] == args.arm, 'Resume continuation config differs')
            require(config['max_updates'] == LIMIT and config['checkpoints'] == list(MILESTONES), 'Resume budget/protocol differs')
            require(config['source_code'] == code and config['parent']['sha256'] == parent['sha256'], 'Resume provenance differs')
            cfg = Config(**cp['model_config'])
            require(scientific_cfg(cp['model_config']) == config['scientific_model_config'], 'Resume model recipe differs')
            require(cfg.activation_checkpointing is False or args.arm.startswith('Fourier'), 'Continuation checkpoint execution mode differs')
            initial_gate = read(folder/'startup-baseline-verification.json')
            require(initial_gate['passed'] and initial_gate['parent_sha256'] == parent['sha256']
                    and initial_gate['source_code'] == code, 'Verified parent baseline missing on resume')
        else:
            loaded = parent
            cp = torch.load(parent_path, map_location='cpu', mmap=True, weights_only=False)
            cfg = validate_parent(cp, args.arm)
            config = dict(schema='cad50_full_state_continue_v1', arm=args.arm, parent=parent,
                parent_config_sha256=__import__('hashlib').sha256(json.dumps(cp['config'], sort_keys=True).encode()).hexdigest(),
                scientific_model_config=scientific_cfg(cfg.to_dict()), source_code=code,
                start_updates=START, max_updates=LIMIT, additional_updates=8000,
                checkpoints=list(MILESTONES), optimizer='restored AdamW, all 412 states',
                optimizer_parameter_names=cp['optimizer_parameter_names'], lr=1e-4, warmup_updates=0,
                betas=[.9, .999], eps=1e-8, weight_decay=.01, clip=1.,
                seed=0, negative_seed=0, sampling=False, KL=0, logvar_frozen=True, dropout=0,
                microbatch=1, meshes_per_update=5, negative_ratio=1.5,
                objective=cp['config']['objective'], ordering=cp['config']['ordering'], data=cp['config']['data'],
                runtime_contract=dict(torch=cp['config']['torch'], cuda=cp['config']['cuda']),
                numerical='original deterministic Graph; FP32 SDPA MATH; TF32/autocast/fastpath off',
                best_rule='highest joint_strict; then actual Face micro_f1; then Edge micro_f1; earlier tie wins',
                termination='exact budget; no early elimination; execution failure stops controller')
            write(folder/'config.json', config)
        completed = cp['completed_updates']
        require(completed <= args.stop_at, 'Resume checkpoint is ahead of requested stage; no rollback')
        require(cp['optimizer_parameter_names'] == config['optimizer_parameter_names'], 'Resume flat optimizer map differs')
        validate_adam(cp['optimizer'], cp['optimizer_parameter_names'], cp['model'], completed)
        items, manifest = load_dataset(Path(args.source)/'data', Path(args.source)/'pools')
        require(manifest == config['data'], 'Source data/pool/UID/vertex order differs from parent')
        uids = manifest['uids']
        check_cursor(cp, uids, epoch_batches, LIMIT)
        check_log(folder/'updates.jsonl', completed)
        participation = cp['participation'].copy()
        if completed == START:
            next_uids = epoch_batches(uids, 200)[0]
            require(next_uids == [f'teacher_cad50_{i}' for i in ('31', '28', '38', '37', '00')], 'Authorized next UID order differs')
        require(set(cp['rng']) == {'python', 'numpy', 'torch', 'cuda'} and len(cp['rng']['cuda']) == 1, 'Incomplete checkpoint RNG')

        phase = 'restore_on_authorized_device'
        configure(0)
        require(torch.cuda.device_count() == 1, 'UUID must expose exactly one GPU')
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(.44, 0)
        device_row = subprocess.check_output(['nvidia-smi', '--id', args.gpu_uuid,
            '--query-gpu=uuid,name,memory.total', '--format=csv,noheader,nounits'], text=True).strip().split(',')
        require(len(device_row) == 3 and device_row[0].strip() == args.gpu_uuid
                and 'A100' in device_row[1] and int(device_row[2]) >= 80000, 'Expected explicitly approved A100 80GB')
        require(str(torch.__version__) == config['runtime_contract']['torch']
                and torch.version.cuda == config['runtime_contract']['cuda'], 'Numerical software environment differs')
        model = NativeTopologyAE(cfg).cuda().float().train()
        model.load_state_dict(cp['model'], strict=True)
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        params = [p for p in model.parameters() if p.requires_grad]
        require(names == cp['optimizer_parameter_names'], 'Model-to-Adam flat name mapping differs')
        opt = torch.optim.AdamW(params, **ADAM)
        opt.load_state_dict(cp['optimizer'])
        gpu_items = {u: {k: v.cuda() if torch.is_tensor(v) else v for k, v in x.items()} for u, x in items.items()}
        graphs = {u: Graph.from_faces(x['faces'], len(x['vertices'])) for u, x in gpu_items.items()}
        restore_rng(cp['rng'])
        restored = audit_restore(model, opt, cp, uids, epoch_batches, items)
        restored.update(checkpoint=loaded, source_code=code)
        write(audit_dir/'restoration.json', restored)
        frozen_hash = tensor_hash(model.log_variance.state_dict())
        execution = dict(gpu_uuid=args.gpu_uuid, gpu=torch.cuda.get_device_name(0),
            torch=str(torch.__version__), cuda=torch.version.cuda, allocator_fraction=.44,
            activation_checkpointing=False, no_trainable_hidden_cache=True, source_code=code,
            resume_checkpoint=loaded, resume_update=completed, stop_at=args.stop_at)
        write(audit_dir/'runtime.json', execution)  # Device identity is metadata, never config equality.
        boundary_safe = True

        def save(reason, path=None):
            require(boundary_safe, 'Refuse resumable save inside accumulation/Adam/log commit')
            require(tensor_hash(model.log_variance.state_dict()) == frozen_hash, 'Frozen logvar changed')
            value = dict(model=model.state_dict(), model_config=model.cfg.to_dict(), execution=execution,
                optimizer=opt.state_dict(), optimizer_parameter_names=names, rng=rng_state(), config=config,
                completed_updates=completed, participation=participation.copy(), next_epoch=completed//10,
                next_batch=completed%10, negative_seed=0, reason=reason, resumable=True)
            check_cursor(value, uids, epoch_batches, LIMIT)
            require(all(opt.state[p]['step'].item() == completed for p in params), 'Not all Adam steps match boundary')
            return save_torch(path or folder/f'checkpoint-{completed:05d}.pt', value)

        def evaluation(checkpoint, destination):
            state_rng = rng_state()
            write(folder/'status.json', dict(state='evaluating', completed_updates=completed, stop_at=args.stop_at, budget=LIMIT))
            result = evaluate(model, items, destination, checkpoint)
            equal_state(rng_state(), state_rng, 'Evaluation RNG unchanged')
            require(result['complete'] and len(result['meshes']) == 50, 'Incomplete evaluation')
            # Additional fields derived from unchanged full prediction evaluation, never a surrogate loss.
            summary = dict(completed_updates=completed, checkpoint=checkpoint)
            for label, rows in [('CAD50', result['meshes']), ('large16', [r for r in result['meshes'] if 66 <= r['vertices'] <= 274])]:
                gt_count = sum(r['face']['tp']+r['face']['fn'] for r in rows)
                missing = sum(r['face_fn_missing'] for r in rows)
                summary[label] = dict(result if label == 'CAD50' else result['large16'])
                for key in ('checkpoint', 'meshes', 'large16', 'complete'):
                    summary[label].pop(key, None)
                summary[label].update(denominator=len(rows), actual_face_candidates=sum(r['actual_face_candidates'] for r in rows),
                    gt_faces=gt_count, gt_face_candidates_covered=gt_count-missing,
                    candidate_coverage=(gt_count-missing)/gt_count if gt_count else 1.)
            require(len([r for r in result['meshes'] if 66 <= r['vertices'] <= 274]) == 16, 'Fixed large16 changed')
            write(destination/'derived_summary.json', summary)
            print('EVALUATION', args.arm, completed, result['joint_strict'], result['counts'], flush=True)
            return result

        phase = 'zero_update_baseline'
        # A fresh attempt directory forces real inference rather than evaluate() cache reuse.
        if completed == START:
            baseline_path = audit_dir/'parent-baseline'
            actual = evaluation(parent, baseline_path)
            expected = read(expected_path)
            require(expected['checkpoint']['sha256'] == parent['sha256'], 'Historical baseline bound to a different parent')
            compare_evaluations(actual, expected, uids)
            write(folder/'startup-baseline-verification.json', dict(passed=True, parent_sha256=parent['sha256'],
                source_code=code, all_50_UID_metrics_equal=True, optimizer_updates=0,
                evaluation_path=str(baseline_path/'evaluation.json'), historical_evaluation=str(expected_path),
                restored_model_hash=restored['model_hash']))
        else:
            evaluation(loaded, audit_dir/'resume-baseline')

        phase = 'execution_equivalence'
        if completed < LIMIT:
            epoch, cursor = divmod(completed, 10)
            batch = epoch_batches(uids, epoch)[cursor]
            if args.arm.startswith('Fourier'):
                fourier_gate(model, opt, items, gpu_items, graphs, batch, epoch, audit_dir/'FOURIER_EQUIVALENCE.json')
            else:
                write(audit_dir/'XYZ_EQUIVALENCE.json', xyz_gate(args.arm))
        # Verify startup evaluation/gate made zero model/Adam/RNG changes.
        equal_state(model.state_dict(), cp['model'], 'post_gate_model')
        equal_state(opt.state_dict(), cp['optimizer'], 'post_gate_optimizer')
        equal_state(rng_state(), cp['rng'], 'post_gate_rng')
        model.cfg = replace(model.cfg, activation_checkpointing=False)
        del cp
        if completed == START and not args.resume:
            last_checkpoint = save('verified_continuation_start')
        else:
            last_checkpoint = loaded

        def preserve_alias(checkpoint, alias):
            dest = folder/(alias+'.pt')
            tmp = folder/(alias+f'.{attempt}.tmp')
            try:
                os.link(checkpoint['path'], tmp)
            except OSError:
                shutil.copyfile(checkpoint['path'], tmp)
            tmp.replace(dest)
            entry = dict(checkpoint, path=str(dest))
            write(dest.with_suffix('.json'), entry)
            return entry

        def keep_best(checkpoint, result, result_path):
            score = [result['joint_strict'], result['counts']['face']['micro_f1'], result['counts']['edge']['micro_f1']]
            old = read(folder/'best.json') if (folder/'best.json').exists() else None
            if old is None or score > old['score']:
                best_cp = preserve_alias(checkpoint, 'best')
                write(folder/'best.json', dict(score=score, completed_updates=completed, checkpoint=best_cp,
                    evaluation_path=str(result_path), evaluation_checkpoint=result['checkpoint'],
                    note='At step2000 the new continuation state has identical model bits to the evaluated parent'))

        if completed == START:
            keep_best(last_checkpoint, actual, baseline_path/'evaluation.json')
        if stop_requested:
            raise InterruptedError('Stop requested at restored boundary')
        if args.audit_only:
            write(folder/'audit-complete.json', dict(passed=True, optimizer_updates=0, completed_updates=completed,
                checkpoint=last_checkpoint, audit_directory=str(audit_dir), source_code=code))
            print('AUDIT_COMPLETE', args.arm, completed, 'zero optimizer updates', flush=True)
            return

        def evaluation_destination(checkpoint):
            destination = folder/f'evaluations/step-{completed:05d}'
            for name in ('evaluation.json', 'progress.json'):
                metadata = destination/name
                if metadata.exists() and read(metadata)['checkpoint']['sha256'] != checkpoint['sha256']:
                    return folder/f'evaluations/step-{completed:05d}-{attempt}'
            return destination

        phase = 'training'
        with (folder/'updates.jsonl').open('a' if args.resume else 'x', buffering=1) as log:
            for step in range(completed+1, args.stop_at+1):
                if stop_requested:
                    raise InterruptedError('Stop requested at update boundary')
                epoch, cursor = divmod(completed, 10)
                batch = epoch_batches(uids, epoch)[cursor]
                require(all(group['lr'] == 1e-4 for group in opt.param_groups), 'Constant LR changed')
                boundary_safe = False
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                tic = time.monotonic()
                parts, loss_bits = backward_batch(model, items, gpu_items, graphs, batch, epoch)
                norm = float(torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True, foreach=True))
                measure = step <= restored['Adam_all_steps']+3 or step % 100 == 0
                before = [p.detach().clone() for p in params] if measure else None
                opt.step()
                torch.cuda.synchronize()
                if before is not None:
                    displacement = torch.sqrt(sum((p.detach()-q).square().sum() for p, q in zip(params, before))).item()
                    parameter_norm = torch.sqrt(sum(q.square().sum() for q in before)).item()
                    require(0 < displacement < float('inf') and 0 < parameter_norm < float('inf'), 'Invalid FP32 parameter update')
                    relative_displacement = displacement/parameter_norm
                    del before
                else:
                    displacement = parameter_norm = relative_displacement = None
                completed = step
                for uid in batch:
                    participation[uid] += 1
                require(all(opt.state[p]['step'].item() == completed for p in params), 'Adam steps disagree after update')
                record = dict(update_before=step-1, update_after=step, epoch=epoch, batch_index=cursor, uids=batch,
                    meshes=parts, mesh_loss_fp32_bits_before=loss_bits,
                    loss_before=sum(p['edge_loss']+p['face_loss'] for p in parts)/5,
                    gradient_norm_before_clip=norm, clip_coefficient=min(1., 1/(norm+1e-6)), lr=1e-4,
                    adam_step_after=step, participation_after=sum(participation.values()),
                    actual_fp32_displacement_norm=displacement, parameter_norm_before=parameter_norm,
                    relative_fp32_displacement_norm=relative_displacement,
                    activation_checkpointing=False, seconds=time.monotonic()-tic,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    timing='loss/Edge/Face counts before update; Adam step and participation after update')
                log.write(json.dumps(record, separators=(',', ':'), allow_nan=False)+'\n')
                log.flush()
                os.fsync(log.fileno())
                boundary_safe = True
                write(folder/'status.json', dict(state='training', completed_updates=completed, budget=LIMIT,
                    stop_at=args.stop_at, loss_before=record['loss_before'], seconds_per_update=record['seconds']))
                if step <= loaded.get('completed_updates', restored['Adam_all_steps'])+3 or step % 100 == 0:
                    print('UPDATE', args.arm, step, record['loss_before'], record['seconds'], flush=True)
                if completed in schedule(LIMIT):
                    last_checkpoint = save('scheduled_full_state')
                    phase = 'scheduled_evaluation'
                    destination = evaluation_destination(last_checkpoint)
                    result = evaluation(last_checkpoint, destination)
                    keep_best(last_checkpoint, result, destination/'evaluation.json')
                    write(folder/f'stage-{completed:05d}.json', dict(state='stage_complete', completed_updates=completed,
                        checkpoint=last_checkpoint, evaluation_path=str(destination/'evaluation.json'), budget=LIMIT))
                    phase = 'training'
            # If interrupted after saving but before marking a stage, explicit resume can finish its evaluation.
            if completed == args.stop_at:
                destination = evaluation_destination(last_checkpoint)
                existing = destination/'evaluation.json'
                if existing.exists() and read(existing)['checkpoint']['sha256'] == last_checkpoint['sha256']:
                    result = read(existing)
                    require(result['complete'] and len(result['meshes']) == 50, 'Stage evaluation incomplete')
                else:
                    if existing.exists():
                        destination = folder/f'evaluations/step-{completed:05d}-{attempt}'
                    result = evaluation(last_checkpoint, destination)
                keep_best(last_checkpoint, result, destination/'evaluation.json')
                write(folder/f'stage-{completed:05d}.json', dict(state='stage_complete', completed_updates=completed,
                    checkpoint=last_checkpoint, evaluation_path=str(destination/'evaluation.json'), budget=LIMIT))
                stage_finished = True
        if completed == LIMIT:
            final_cp = preserve_alias(last_checkpoint, 'final')
            write(folder/'complete.json', dict(state='complete', completed_updates=LIMIT,
                next_epoch=1000, next_batch=0, per_UID_participation=1000, mesh_participations=50000,
                additional_updates=8000, stop_reason='fixed_10000_update_budget', final_checkpoint=final_cp,
                final_evaluation_path=str(destination/'evaluation.json'), counts=result['counts'], joint_strict=result['joint_strict'],
                best=read(folder/'best.json'), logvar_unchanged=True, elapsed_seconds=time.monotonic()-started))
        write(folder/'status.json', dict(state='complete' if completed == LIMIT else 'stage_complete', completed_updates=completed, budget=LIMIT))
        print('COMPLETE' if completed == LIMIT else 'STAGE_COMPLETE', args.arm, completed, flush=True)
    except BaseException as error:
        write(audit_dir/'failure.json', dict(completed_updates=completed, phase=phase, error=str(error),
            boundary_safe=boundary_safe, stop_signals=stop_requested, traceback=traceback.format_exc()))
        # All failures remain visible. Only a fully restored/committed update boundary is resumable.
        state = None
        if model is not None and opt is not None:
            try:
                if boundary_safe and frozen_hash is not None:
                    state = save('explicit_interruption_or_failure_boundary', folder/f'boundary-{completed:05d}-{attempt}.pt')
                else:
                    state = save_torch(audit_dir/'failure-state.pt', dict(model=model.state_dict(), optimizer=opt.state_dict(),
                        optimizer_parameter_names=names, rng=rng_state(), config=config, completed_updates=completed,
                        participation=participation, resumable=False, note='Possibly inside accumulation/Adam; never resume this file'))
            except BaseException as save_error:
                write(audit_dir/'failure-save-error.json', dict(error=str(save_error)))
        write(folder/'failure.json', dict(attempt=attempt, completed_updates=completed, phase=phase,
            error=str(error), state=state, resumable=bool(boundary_safe and state), stage_finished=stage_finished))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True, choices=list(PARENTS))
    parser.add_argument('--source', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--parent-base', default=BASE)
    parser.add_argument('--budget-updates', type=int, default=LIMIT, choices=[LIMIT])
    parser.add_argument('--stop-at', type=int, default=LIMIT, choices=MILESTONES[1:])
    parser.add_argument('--resume')
    parser.add_argument('--resume-sha256')
    parser.add_argument('--audit-only', action='store_true', help='Restore, baseline and execution gate; zero optimizer updates')
    parser.add_argument('--lock-fd', type=int, help=argparse.SUPPRESS)
    main(parser.parse_args())
