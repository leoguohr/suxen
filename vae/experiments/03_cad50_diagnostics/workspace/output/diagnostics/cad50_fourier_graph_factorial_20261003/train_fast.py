"""Two missing factorial cells; reuse prior random tensors, 2000 updates each."""
import argparse
from dataclasses import replace
import fcntl
import os
from pathlib import Path
import time
import traceback
from support import *
from native_models import Config, NativeTopologyAE, Graph
from data_objective import load_dataset, epoch_batches, negative_faces, objective
from evaluation import evaluate


def main(arm, source, resume=None):
    root = Path(__file__).resolve().parent
    folder = root/'runs'/arm
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder/'execution.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if resume is None:
        assert not (folder/'config.json').exists(), 'Use explicit --resume; no duplicate start'
    configure(0)
    assert torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.44, 0)
    started = time.monotonic()
    completed = 0
    model = opt = None
    try:
        assert read(root/'CPU_TESTS.json')['Fourier_controls_forward_and_gradients_bitwise_equal']
        assert read(root/'CONTROL_GATE.json')['passed']
        initial_info = read(root/'INITIALIZATION.json')
        initial_path = Path(initial_info['checkpoint']['path'])
        assert sha(initial_path) == initial_info['checkpoint']['sha256']
        initial = torch.load(initial_path, map_location='cpu', mmap=True, weights_only=False)
        cfg = Config(model_variant='B_v2_teacher_blocks', **MODES[arm])
        model = NativeTopologyAE(cfg).cuda().float().train()
        model.load_state_dict(initial['model'], strict=True)
        assert tensor_hash(model.state_dict()) == initial_info['tensor_hash']
        params = [p for p in model.parameters() if p.requires_grad]
        names = [n for n,p in model.named_parameters() if p.requires_grad]
        assert sum(p.numel() for p in params) == 246575680
        opt = torch.optim.AdamW(params, lr=1e-4, betas=(.9,.999), eps=1e-8, weight_decay=.01, foreach=True)
        assert not opt.state
        items, manifest = load_dataset(Path(source)/'data', Path(source)/'pools')
        uids = manifest['uids']
        gpu_items = {u: {k: v.cuda() if torch.is_tensor(v) else v for k,v in x.items()} for u,x in items.items()}
        graphs = {u: Graph.from_faces(x['faces'], len(x['vertices'])) for u,x in gpu_items.items()}
        participation = {u: 0 for u in uids}
        frozen_hash = tensor_hash(model.log_variance.state_dict())
        config = dict(arm=arm, changed_operations=MODES[arm], model_config=cfg.to_dict(), seed=0,
            initial=initial_info['checkpoint'], initial_tensor_hash=initial_info['tensor_hash'], data=manifest,
            trainable_parameters=sum(p.numel() for p in params), optimizer_parameter_names=names,
            optimizer='fresh AdamW', lr=1e-4, betas=[.9,.999], eps=1e-8, weight_decay=.01,
            warmup_updates=100, warmup_formula='lr=1e-4*min(effective_update/100,1)', clip=1,
            sampling=False, KL=0, logvar_frozen=True, dropout=0, microbatch=1, meshes_per_update=5,
            max_updates=2000, checkpoints=list(CHECKS), negative_ratio=1.5,
            objective='mean5(Edge Hard4 + Face Hard4); whole-mesh groups; empty0; internal /4',
            ordering='unchanged seed0 epoch/UID SHA256 -> PCG64', source_code=code_hashes(),
            numerical='deterministic Graph; FP32 SDPA MATH; TF32/autocast/fastpath off; block recompute',
            neighbor_projection_bias=False, torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0), gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'])
        configure(0)
        # All arms start at the same RNG point after construction; no stochastic layers.
        if resume:
            saved_config = read(folder/'config.json')
            assert {k:v for k,v in config.items() if k != 'source_code'} == {k:v for k,v in saved_config.items() if k != 'source_code'}
            config = saved_config
            resume_path = Path(resume).resolve()
            cp = torch.load(resume_path, map_location='cpu', mmap=True, weights_only=False)
            assert cp['config'] == read(folder/'config.json') == config
            assert cp['optimizer_parameter_names'] == names
            model.load_state_dict(cp['model'], strict=True)
            opt.load_state_dict(cp['optimizer'])
            restore_rng(cp['rng'])
            completed = cp['completed_updates']
            participation = cp['participation']
            assert cp['next_epoch'] == completed//10 and cp['next_batch'] == completed%10
            assert sum(participation.values()) == 5*completed
            def optimizer_hash(state):
                return tensor_hash({f'{i}.{k}': v for i, values in sorted(state['state'].items())
                                    for k,v in sorted(values.items())})
            assert tensor_hash(model.state_dict()) == tensor_hash(cp['model'])
            assert optimizer_hash(opt.state_dict()) == optimizer_hash(cp['optimizer'])
            restored_rng = rng_state()
            assert torch.equal(restored_rng['torch'], cp['rng']['torch'])
            assert all(torch.equal(a,b) for a,b in zip(restored_rng['cuda'], cp['rng']['cuda']))
            next_uids = epoch_batches(uids, completed//10)[completed%10]
            write(folder/'performance-resume-verification.json', dict(completed_updates=completed,
                model_tensors_bitwise_equal=True, Adam_state_tensors_bitwise_equal=True,
                torch_and_CUDA_rng_equal=True, next_uids=next_uids,
                next_negative_hashes={u:negative_faces(items[u], completed//10)[1] for u in next_uids}))
            log_path = folder/'updates.jsonl'
            if log_path.exists():
                last = read_last_update(log_path)
                assert last == completed, 'Uncheckpointed logged updates exist: refuse silent replay'
            write(folder/f'resume-{completed:05d}.json', dict(checkpoint=str(resume_path), sha256=sha(resume_path)))
        else:
            write(folder/'config.json', config)
        # Exact environment and data contract required when reusing historical controls.
        reference_config = read(root/'reference/config.json')
        for key in ('torch', 'cuda', 'gpu', 'data'):
            assert config[key] == reference_config[key], f'Control mismatch: {key}'
        del initial
        resume_start = completed
        model.cfg = replace(model.cfg, activation_checkpointing=False)
        execution = dict(activation_checkpointing=False, scientific_config_unchanged=True,
            allocator_fraction=0.44, entry='train_fast.py', code=code_hashes(), resumed_from=resume,
            resume_update=resume_start, performance_gate=read(root/'performance/EQUIVALENCE.json'))
        assert execution['performance_gate']['all_gradients_bitwise_equal']
        write(folder/'execution-retain-activations.json', execution)

        def save(reason):
            assert tensor_hash(model.log_variance.state_dict()) == frozen_hash
            return save_torch(folder/f'checkpoint-{completed:05d}.pt', dict(model=model.state_dict(),
                model_config=model.cfg.to_dict(), execution=execution, optimizer=opt.state_dict(), optimizer_parameter_names=names,
                rng=rng_state(), config=config, completed_updates=completed, participation=participation,
                next_epoch=completed//10, next_batch=completed%10, negative_seed=0, reason=reason))

        def check(checkpoint):
            write(folder/'status.json', dict(state='evaluating', completed_updates=completed, budget=2000))
            result = evaluate(model, items, folder/f'evaluations/step-{completed:05d}', checkpoint)
            print('EVALUATION', arm, completed, result['joint_strict'], result['counts'], flush=True)
            return result

        if completed in CHECKS:
            checkpoint = (dict(path=str(initial_path), sha256=initial_info['checkpoint']['sha256'],
                               bytes=initial_info['checkpoint']['bytes']) if completed == 0 else read(Path(resume).with_suffix('.json')))
            check(checkpoint)
        log_mode = 'a' if resume else 'x'
        last_checkpoint = None
        with (folder/'updates.jsonl').open(log_mode, buffering=1) as log:
            for step in range(completed+1, 2001):
                epoch, batch_index = divmod(step-1, 10)
                batch = epoch_batches(uids, epoch)[batch_index]
                lr = 1e-4*min(step/100, 1)
                for group in opt.param_groups:
                    group['lr'] = lr
                opt.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                tic = time.monotonic()
                parts = []
                for uid in batch:
                    item = gpu_items[uid]
                    negatives, negative_sha = negative_faces(items[uid], epoch)
                    outputs = model(item['vertices'], item['faces'], sample_latent=False, graph=graphs[uid])
                    assert outputs['latent'] is outputs['mu']
                    loss, detail = objective(outputs, item, negatives)
                    assert bool(torch.isfinite(loss)), 'Nonfinite loss'
                    (loss/5).backward()
                    parts.append(dict(uid=uid, negative_sha256=negative_sha, **detail))
                    del outputs, loss
                assert all(p.grad is None for p in model.log_variance.parameters())
                if step == 1:
                    assert all(p.grad is not None for p in params), 'Disconnected trainable parameter'
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True, foreach=True))
                measure = step <= resume_start+3 or step % 100 == 0
                before = [p.detach().clone() for p in params] if measure else None
                opt.step()
                torch.cuda.synchronize()
                seconds = time.monotonic()-tic
                if before is not None:
                    displacement = torch.sqrt(sum((p.detach()-q).square().sum() for p,q in zip(params,before))).item()
                    assert 0 < displacement < float('inf')
                    del before
                else:
                    displacement = None
                completed = step
                for uid in batch:
                    participation[uid] += 1
                assert int(opt.state[params[0]]['step']) == completed
                record = dict(activation_checkpointing=False, update=step, epoch=epoch, batch_index=batch_index, uids=batch, meshes=parts,
                    loss_before=sum(p['edge_loss']+p['face_loss'] for p in parts)/5,
                    gradient_norm_before_clip=gradient_norm, clip_coefficient=min(1.,1/(gradient_norm+1e-6)),
                    lr=lr, adam_step=step, actual_fp32_displacement_norm=displacement,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    seconds=seconds, participations=sum(participation.values()),
                    timing='loss and Edge counts before update; Adam step and participations after update')
                log.write(json.dumps(record, separators=(',',':'), allow_nan=False)+'\n')
                write(folder/'status.json', dict(state='training', completed_updates=completed, budget=2000,
                    loss_before=record['loss_before'], seconds_per_update=seconds, elapsed_seconds=time.monotonic()-started))
                if step <= resume_start+3 or step % 100 == 0:
                    print('UPDATE', arm, step, record['loss_before'], seconds, flush=True)
                if completed in CHECKS:
                    assert set(participation.values()) == {completed//10}
                    last_checkpoint = save('scheduled')
                    check(last_checkpoint)
        assert completed == 2000 and set(participation.values()) == {200}
        assert tensor_hash(model.log_variance.state_dict()) == frozen_hash
        final = read(folder/'evaluations/step-02000/evaluation.json')
        write(folder/'complete.json', dict(state='complete', completed_updates=completed,
            mesh_participations=sum(participation.values()), epochs=200, stop_reason='fixed_2000_update_budget',
            final_checkpoint=last_checkpoint, final_evaluation=final['counts'], joint_strict=final['joint_strict'],
            logvar_unchanged=True, elapsed_seconds=time.monotonic()-started))
        print('COMPLETE', arm, completed, flush=True)
    except BaseException as error:
        write(folder/'failure.json', dict(completed_updates=completed, error=str(error), traceback=traceback.format_exc()))
        if model is not None and opt is not None:
            save_torch(folder/'failure-state.pt', dict(model=model.state_dict(), optimizer=opt.state_dict(),
                rng=rng_state(), completed_updates=completed, resumable=False,
                note='Failure may be inside accumulation; use a scheduled boundary checkpoint'))
        raise


def read_last_update(path):
    with Path(path).open() as stream:
        last = None
        for line in stream:
            last = json.loads(line)['update']
    return last or 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True, choices=list(MODES))
    parser.add_argument('--source', required=True)
    parser.add_argument('--resume')
    args = parser.parse_args()
    main(args.arm, args.source, args.resume)
