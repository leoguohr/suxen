"""Explicit standalone Topology training; never touches live VAE/Vertex directories."""
import argparse
import fcntl
import json
import shutil
import time
from pathlib import Path
import torch
from _faces_reference import atomic_json, file_sha
from _vertex_reference import farthest_point_sample
from latent_data import LatentCache
from topology_flow import configure_math_backend, PointCloudTopologyFlow, TopologyFlowConfig
from flow_training import (build_training_state, train_update, checkpoint_state, restore_training,
    atomic_checkpoint, training_recipe, STAGE_KEYS)
from gpu_budget import GPUBudget


def read_stage_budget(path):
    stage = json.loads(Path(path).read_text())
    for key in ('max_updates', 'max_seconds', 'save_every', 'checkpoint_reserve_seconds'):
        if type(stage.get(key)) is not int or stage[key] <= 0:
            raise ValueError(key+' must be a positive integer in the separate stage budget')
    if stage['checkpoint_reserve_seconds'] >= stage['max_seconds']:
        raise ValueError('Stage must leave time beyond its checkpoint reserve')
    return stage


def prune_checkpoints(root, entries):
    """Only this run's indexed files; retain newest one and all protected files."""
    present = [entry for entry in entries if entry.get('retained', True)]
    recent = {entry['path'] for entry in sorted(present,key=lambda e:e['completed_updates'])[-1:]}
    for entry in present:
        path = Path(entry['path'])
        if (entry.get('protected') or path.with_suffix('.protect.json').exists()
                or str(path) in recent):
            continue
        if path.parent.resolve() != (root/'checkpoints').resolve():
            raise ValueError('Refuse retention outside this run checkpoint directory')
        if path.exists():
            if path.stat().st_size != entry['bytes']:
                raise ValueError('Refuse retention of a modified checkpoint')
            path.unlink()
        entry['retained'] = False
    atomic_json(root/'checkpoint-manifest.json',dict(keep_recent=1,checkpoints=entries,
        protection='500/1000, final/best metadata, or CHECKPOINT.protect.json are never removed'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,help='Immutable model/optimizer/sampling recipe')
    parser.add_argument('--stage-budget',required=True,help='Separately approved cumulative update target and stage time')
    parser.add_argument('--budget-file',help='Shared cumulative export/train/evaluation GPU ledger')
    parser.add_argument('--cache',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',required=True)
    parser.add_argument('--resume',help='Trusted full training checkpoint, not inference-only weights')
    parser.add_argument('--resume-sha256')
    args = parser.parse_args()
    raw_config = json.loads(Path(args.config).read_text())
    if any(raw_config.get(key) is not None for key in STAGE_KEYS):
        raise ValueError('Move max_updates/max_seconds/save_every out of the immutable recipe into --stage-budget')
    config = training_recipe(raw_config)
    stage = read_stage_budget(args.stage_budget)
    if config['batch_meshes'] != 5: raise ValueError('The fixed50 run requires five full meshes per update')
    if config['model']['rope_scale'] is None: raise ValueError('Choose and record RoPE coordinate scale before launch')
    if bool(args.resume) != bool(args.resume_sha256): raise ValueError('Resume requires both checkpoint and SHA256')
    root = Path(args.output).resolve()
    if root.exists() and not args.resume: raise FileExistsError('Fresh run requires a new output directory')
    root.mkdir(parents=True,exist_ok=bool(args.resume))
    with (root/'RUNNING.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        cache = LatentCache(args.cache,expected_meshes=50)
        if len(cache.uids) != 50 or len(set(cache.uids)) != 50:
            raise ValueError('Exactly 50 unique mesh UIDs are required')
        # No CUDA allocation is needed to establish persistent-storage requirements.
        with torch.device('meta'):
            skeleton = PointCloudTopologyFlow(TopologyFlowConfig(**config['model']))
        parameters = sum(p.numel() for p in skeleton.parameters())
        del skeleton
        full_checkpoint_bytes = parameters*12+(64 << 20)
        margin = 1 << 30
        free_bytes = shutil.disk_usage(root).free
        initial_checkpoint_bytes = parameters*4+(64 << 20)
        # This is a free-space check, not a reservation: retain flow0 while step1 commits.
        required = full_checkpoint_bytes+margin+(0 if args.resume else initial_checkpoint_bytes)
        if free_bytes < required:
            raise OSError(f'Insufficient persistent storage: free={free_bytes}, required={required}')
        device = torch.device(args.device)
        configure_math_backend()
        code = {p.name:file_sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
        # This also locks and charges setup/loading; no CUDA work precedes it.
        with GPUBudget(args.budget_file,phase='train',device=args.device) as budget:
            start = time.monotonic()
            reserve = float(stage['checkpoint_reserve_seconds'])
            if budget.stop(reserve):
                atomic_json(root/'status.json',dict(complete=False,reason='insufficient_checkpoint_reserve',
                    gpu_budget=budget.snapshot(),optimizer_started=False))
                return
            completed, prior_seconds = 0, 0.
            try:
                model, optimizer, rngs = build_training_state(config,device)
                if args.resume:
                    if file_sha(args.resume) != args.resume_sha256: raise ValueError('Resume checkpoint hash mismatch')
                    state = torch.load(args.resume,map_location='cpu',weights_only=False)
                    restore_training(state,model,optimizer,rngs,cache,config,code)
                    completed, prior_seconds = state['completed_updates'],state['elapsed_seconds']
                    del state
            except Exception as error:
                atomic_json(root/'failure.json',dict(error=repr(error),phase='model_or_resume_setup',
                    resume=args.resume,resume_sha256=args.resume_sha256,gpu_budget=budget.snapshot(),
                    note='No optimizer step; no model or precision fallback'))
                raise
            budget.require_training_cursor(completed)
            if stage['max_updates'] <= completed:
                raise ValueError('A continuation stage must authorize a larger cumulative update target')
            if budget.cuda and stage['max_updates'] > budget.state['max_optimizer_updates']:
                raise ValueError('Stage update target exceeds the shared authorized allowance')
            fps = {}
            for uid in cache.uids:
                item = cache.get(uid)
                fps[uid] = farthest_point_sample(item['points'][None,:,:3],config['model']['condition_tokens'],item['point_mask'][None])
            def elapsed(): return prior_seconds+time.monotonic()-start
            def stop():
                return ((root/'STOP').exists() or budget.updates_exhausted()
                    or time.monotonic()-start >= stage['max_seconds']-reserve or budget.stop(reserve))
            saved = completed if args.resume else -1
            latest = dict(path=str(Path(args.resume).resolve()),sha256=args.resume_sha256,
                completed_updates=completed) if args.resume else None
            entries_path = root/'checkpoint-manifest.json'
            entries = json.loads(entries_path.read_text())['checkpoints'] if entries_path.exists() else []
            attempt_id = f'from{completed:07d}-{time.time_ns()}'
            atomic_json(root/f'config-{attempt_id}.json',dict(config=config,stage_budget=stage,
                stage_budget_path=str(Path(args.stage_budget).resolve()),stage_budget_sha256=file_sha(args.stage_budget),
                cache=str(cache.root.resolve()),cache_sha256=cache.sha256,
                imported_code=str(Path(__file__).parent.resolve()),code_sha256=code,device=str(device),
                torch_version=torch.__version__,gpu_name=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                resume=args.resume,resume_sha256=args.resume_sha256,gpu_budget=budget.snapshot(),
                storage_preflight=dict(free_bytes=free_bytes,required_bytes=required,
                    estimated_full_checkpoint_bytes=full_checkpoint_bytes,filesystem=str(root),
                    full_model_adam_tensor_bytes=parameters*12,keep_recent=1,
                    planned_peak_checkpoint_count=3,planned_peak_checkpoint_bytes=3*full_checkpoint_bytes,
                    peak_note='latest plus protected500 plus incoming1000; extra best/final protections add their sizes',
                    quota='unknown; actual populated AdamW checkpoint at update1 is the write/fsync/SHA probe; no cleanup on quota error')))
            def save(final=False):
                nonlocal saved, latest, reserve
                if saved != completed:
                    if shutil.disk_usage(root).free < full_checkpoint_bytes+margin:
                        raise OSError('Insufficient space for a complete atomic model/AdamW/RNG checkpoint')
                    tick = time.monotonic()
                    path = root/'checkpoints'/f'flow-{completed:07d}.pt'
                    state = checkpoint_state(model,optimizer,rngs,cache,config,completed,elapsed(),code)
                    state.update(stage_budget=stage,gpu_budget=budget.snapshot())
                    digest = atomic_checkpoint(path,state)
                    protected = ['scheduled_evaluation'] if completed in (500,1000) else []
                    if final: protected.append('final')
                    latest = dict(path=str(path),sha256=digest,bytes=path.stat().st_size,
                        completed_updates=completed,protected=protected,retained=True)
                    entries.append(latest)
                    atomic_json(root/'latest.json',latest)
                    saved = completed
                    save_seconds = time.monotonic()-tick
                    # Never shrink the agreed shutdown reserve; observed saves can enlarge it.
                    reserve = max(reserve,2*save_seconds)
                    atomic_json(root/'last-checkpoint-timing.json',dict(completed_updates=completed,
                        seconds=save_seconds,shutdown_reserve_seconds=reserve))
                elif final:
                    for entry in entries:
                        if entry['path'] == latest['path']:
                            if 'final' not in entry.setdefault('protected',[]): entry['protected'].append('final')
                            latest = entry
                            atomic_json(root/'latest.json',latest)
                prune_checkpoints(root,entries)
            progress = {}
            try:
                if saved < 0: save()
                # First three effective updates are profiled in place and count toward every cap.
                with (root/f'updates-{attempt_id}.jsonl').open('x') as log:
                    while completed < stage['max_updates'] and not stop():
                        tick = time.monotonic()
                        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
                        progress = dict(started_after_completed_updates=completed)
                        record = train_update(model,optimizer,rngs,cache,config,completed,stop,fps,
                            profile=completed < 3,progress=progress)
                        if record is None: break
                        completed += 1
                        budget.record_update(completed)
                        if device.type=='cuda': torch.cuda.synchronize(device)
                        record.update(seconds=time.monotonic()-tick,elapsed_seconds=elapsed(),
                            gpu_budget=budget.snapshot(),
                            peak_allocated=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
                            peak_reserved=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0)
                        log.write(json.dumps(record,allow_nan=False)+'\n'); log.flush()
                        # Update1 is the real complete AdamW write/fsync/SHA storage probe.
                        if completed in (1,3,500,1000) or completed % stage['save_every']==0: save()
                save(final=True)
                atomic_json(root/'status.json',dict(complete=completed==stage['max_updates'],completed_updates=completed,
                    elapsed_seconds=elapsed(),gpu_budget=budget.snapshot(),latest=latest,
                    reason='stage_update_budget' if completed==stage['max_updates'] else 'time_update_budget_or_STOP',
                    evaluation='not_run; use evaluate_flow.py on an immutable checkpoint',
                    direct_mesh_participation=dict(complete_epochs=completed//10,
                        additional_five_mesh_groups=completed%10)))
            except Exception as error:
                atomic_json(root/'failure.json',dict(error=repr(error),completed_updates=completed,
                    last_durable_checkpoint=latest,update_progress=progress,elapsed_seconds=elapsed(),
                    gpu_budget=budget.snapshot(),
                    peak_allocated=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
                    peak_reserved=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0,
                    note='Failed update not saved; no optimizer reset, precision/model/data fallback, or silent replay'))
                raise


if __name__=='__main__': main()
