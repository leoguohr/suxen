"""One complete equal-mesh optimizer update and exact boundary resume state."""
import hashlib
import json
import os
from pathlib import Path
import time as timing
import torch
from topology_flow import PointCloudTopologyFlow, TopologyFlowConfig, linear_flow_target, equal_mesh_velocity_loss
from latent_data import transform_latent
from vae_codec import sample_posterior
from _faces_reference import file_sha


STAGE_KEYS = frozenset(('max_updates', 'max_seconds', 'save_every'))


def training_recipe(config):
    """Stage allowances may grow; the model, optimizer and sampling recipe may not."""
    return {key:value for key,value in config.items() if key not in STAGE_KEYS}


def atomic_checkpoint(path, state):
    path = Path(path)
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('xb') as stream:
        torch.save(state, stream); stream.flush(); os.fsync(stream.fileno())
    digest = file_sha(tmp)
    os.replace(tmp, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)
    if file_sha(path) != digest: raise IOError('Checkpoint SHA256 changed after atomic commit')
    return digest


def batch_cursor(uids, completed, batch_meshes, seed):
    if not uids or len(set(uids)) != len(uids): raise ValueError('Mesh UIDs must be nonempty and unique')
    if type(completed) is not int or completed < 0: raise ValueError('Invalid completed update count')
    if type(batch_meshes) is not int or batch_meshes < 1: raise ValueError('Invalid accumulation size')
    if len(uids) % batch_meshes: raise ValueError('Accumulation must divide the fixed mesh count')
    epoch, group = divmod(completed, len(uids)//batch_meshes)
    digest = hashlib.sha256(f'topology-flow/{seed}/{epoch}'.encode()).digest()
    rng = torch.Generator().manual_seed(int.from_bytes(digest[:8], 'little'))
    order = [uids[i] for i in torch.randperm(len(uids), generator=rng).tolist()]
    start = batch_meshes*group
    return dict(epoch=epoch, group=group, next_uids=order[start:start+batch_meshes], order=order)


def build_training_state(config, device):
    torch.manual_seed(config['seed'])
    model = PointCloudTopologyFlow(TopologyFlowConfig(**config['model'])).to(device=device,dtype=torch.float32).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], betas=(.9,.999), eps=1e-8, weight_decay=config['weight_decay'])
    # Three independent CPU sequences: VAE posterior, base flow noise and time.
    rngs = {name:torch.Generator().manual_seed(config['seed']+offset)
            for name,offset in (('posterior',1),('flow_noise',2),('time',3))}
    return model, optimizer, rngs


def rng_state(rngs, device):
    return dict(streams={k:r.get_state().clone() for k,r in rngs.items()},
                torch_cpu=torch.get_rng_state(),
                torch_device=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)


def restore_rng(state, rngs, device):
    for k,r in rngs.items(): r.set_state(state['streams'][k].cpu())
    torch.set_rng_state(state['torch_cpu'].cpu())
    if state['torch_device'] is not None:
        if device.type != 'cuda': raise ValueError('Exact resume requires the original device type')
        torch.cuda.set_rng_state(state['torch_device'].cpu(), device)


def checkpoint_state(model, optimizer, rngs, cache, config, completed, elapsed_seconds, code_sha256):
    device = next(model.parameters()).device
    recipe = training_recipe(config)
    return dict(model=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng_state(rngs,device),
        completed_updates=completed, elapsed_seconds=elapsed_seconds, config=recipe,
        recipe_sha256=hashlib.sha256(json.dumps(recipe,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        cursor=batch_cursor(cache.uids, completed, config['batch_meshes'], config['seed']),
        cache_sha256=cache.sha256, source_vae_sha256=cache.manifest['source_checkpoint_sha256'],
        normalization=cache.stats, uids=cache.uids, effective_code_sha256=code_sha256,
        optimizer_parameter_names=[name for name,_ in model.named_parameters()],
        device_type=device.type, model_mode='train', time_direction='noise0_data1', target='fresh_posterior')


def restore_training(state, model, optimizer, rngs, cache, config, code_sha256):
    recipe = training_recipe(config)
    if training_recipe(state['config']) != recipe or state['cache_sha256'] != cache.sha256 or state['normalization'] != cache.stats:
        raise ValueError('Resume configuration or cache differs')
    if state['uids'] != cache.uids or state['source_vae_sha256'] != cache.manifest['source_checkpoint_sha256']:
        raise ValueError('Resume ordered UIDs or frozen VAE differs')
    if state['effective_code_sha256'] != code_sha256 or state['device_type'] != next(model.parameters()).device.type:
        raise ValueError('Exact resume requires the recorded code and device type')
    if state['optimizer_parameter_names'] != [n for n,_ in model.named_parameters()]:
        raise ValueError('Optimizer parameter mapping mismatch')
    if state['cursor'] != batch_cursor(cache.uids,state['completed_updates'],config['batch_meshes'],config['seed']):
        raise ValueError('Resume data cursor mismatch')
    steps = {int(s['step']) for s in state['optimizer']['state'].values() if 'step' in s}
    if steps != ({state['completed_updates']} if state['completed_updates'] else set()):
        raise ValueError('AdamW steps differ from the completed update count')
    model.load_state_dict(state['model'],strict=True)
    optimizer.load_state_dict(state['optimizer'])
    restore_rng(state['rng'],rngs,next(model.parameters()).device)


def train_update(model, optimizer, rngs, cache, config, completed, stop=lambda:False, fps=None, profile=False, progress=None):
    """All forwards refer to the pre-update weights; STOP discards a partial group."""
    device = next(model.parameters()).device
    before = rng_state(rngs, device)
    uids = batch_cursor(cache.uids,completed,config['batch_meshes'],config['seed'])['next_uids']
    optimizer.zero_grad(set_to_none=True)
    records = []
    def clock():
        if profile and device.type == 'cuda': torch.cuda.synchronize(device)
        return timing.monotonic()
    started = clock() if profile else None
    for uid in uids:
        if stop():
            optimizer.zero_grad(set_to_none=True)
            restore_rng(before,rngs,device)
            return None
        if progress is not None: progress.update(uid=uid,phase='forward',accumulated_meshes=len(records))
        forward_start = clock() if profile else None
        item = cache.get(uid)
        clean = transform_latent(sample_posterior(item['mu'],item['logvar'],rngs['posterior']),cache.stats).unsqueeze(0)
        noise = torch.randn(clean.shape,generator=rngs['flow_noise'],dtype=torch.float32)
        time = torch.rand(1,generator=rngs['time'],dtype=torch.float32)
        mask = torch.ones(clean.shape[:2],dtype=torch.bool)
        mixed, target = linear_flow_target(clean,noise,time,mask)
        predicted = model(mixed.to(device),time.to(device),item['vertices'][None].to(device),
            item['points'][None].to(device),mask.to(device),item['point_mask'][None].to(device),
            None if fps is None else fps[uid].to(device))
        loss = equal_mesh_velocity_loss(predicted,target.to(device),mask.to(device))
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite velocity loss: '+uid)
        backward_start = clock() if profile else None
        if progress is not None: progress['phase'] = 'backward'
        (loss/len(uids)).backward()
        records.append(dict(uid=uid,velocity_mse=float(loss.detach()),time=float(time),time_bin=min(int(float(time)*10),9)))
        if profile: records[-1].update(forward_seconds=backward_start-forward_start,backward_seconds=clock()-backward_start)
    if stop():
        optimizer.zero_grad(set_to_none=True); restore_rng(before,rngs,device)
        return None
    clip_start = clock() if profile else None
    if progress is not None: progress.update(phase='clip',accumulated_meshes=len(records))
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip'],error_if_nonfinite=True)
    # Effective optimizer update, never microbatch count; no warmup restart on resume.
    lr = config['lr']*min((completed+1)/max(config['warmup_updates'],1),1.)
    for group in optimizer.param_groups: group['lr'] = lr
    adam_start = clock() if profile else None
    if progress is not None: progress['phase'] = 'adam_step'
    optimizer.step()
    adam_end = clock() if profile else None
    if not torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all():
        raise FloatingPointError('Nonfinite parameters after AdamW; resume only from the previous durable checkpoint')
    steps = sorted({int(s['step']) for s in optimizer.state.values() if 'step' in s})
    result = dict(completed_updates=completed+1,forward_at_completed_updates=completed,
        velocity_mse=sum(r['velocity_mse'] for r in records)/len(records),meshes=records,
        lr=lr,gradient_norm_preclip=float(norm),clip_coefficient=min(1.,config['clip']/(float(norm)+1e-6)),
        adam_steps=steps)
    if profile:
        result['profile'] = dict(accumulation_seconds=clip_start-started,
            forward_seconds=sum(r['forward_seconds'] for r in records),
            backward_seconds=sum(r['backward_seconds'] for r in records),
            clip_seconds=adam_start-clip_start,adam_step_seconds=adam_end-adam_start,
            complete_update_seconds=clock()-started,accumulated_meshes=len(records))
    if progress is not None: progress['phase'] = 'complete'
    return result
