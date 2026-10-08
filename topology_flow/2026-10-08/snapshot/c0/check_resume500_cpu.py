"""Read-only mmap checkpoint500 metadata preflight; no CUDA allocation or training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4 << 20),b''): h.update(block)
    return h.hexdigest()


def check(condition,message):
    if not condition: raise ValueError(message)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();started=time.monotonic()
    check(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU preflight requires CUDA_VISIBLE_DEVICES empty')
    import torch
    sys.path.insert(0,str(root/'code'))
    from flow_training import batch_cursor, training_recipe
    from latent_data import LatentCache, normalization_sha256
    from topology_flow import PointCloudTopologyFlow, TopologyFlowConfig
    cache=LatentCache(root/'cache')
    manifest=json.loads((root/'run/checkpoint-manifest.json').read_text())
    entries=[x for x in manifest['checkpoints'] if x['completed_updates']==500 and x.get('retained',True)]
    check(len(entries)==1,'Require one retained500 checkpoint')
    entry=entries[0];path=Path(entry['path']);before=path.stat()
    check(before.st_size==entry['bytes'],'Checkpoint size differs from manifest')
    state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    recipe=training_recipe(json.loads((root/'configs/user50_recipe.json').read_text()))
    recipe_sha=hashlib.sha256(json.dumps(recipe,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    check(state['completed_updates']==500,'Checkpoint is not update500')
    check(state['config']==recipe and state['recipe_sha256']==recipe_sha,'Recipe mismatch')
    check(state['cache_sha256']==cache.sha256,'Cache manifest hash mismatch')
    check(state['source_vae_sha256']==cache.manifest['source_checkpoint_sha256'],'Frozen VAE mismatch')
    check(state['normalization']==cache.stats,'Normalization changed')
    check(normalization_sha256(state['normalization'])==cache.manifest['normalization_sha256'],'Normalization hash mismatch')
    check(state['uids']==cache.uids and len(set(state['uids']))==50,'Ordered50 UID mismatch')
    code={p.name:digest(p) for p in sorted((root/'code').glob('*.py'))}
    check(state['effective_code_sha256']==code,'Frozen runtime code hash mismatch')
    check(state['device_type']=='cuda' and state['model_mode']=='train','Unexpected saved device/mode')
    check(state['time_direction']=='noise0_data1' and state['target']=='fresh_posterior','Flow recipe changed')
    cursor=batch_cursor(cache.uids,500,recipe['batch_meshes'],recipe['seed'])
    check(state['cursor']==cursor and cursor['epoch']==50 and cursor['group']==0,'Data cursor mismatch')
    with torch.device('meta'):
        meta=PointCloudTopologyFlow(TopologyFlowConfig(**recipe['model']))
    params=dict(meta.named_parameters());template=meta.state_dict()
    check(state['optimizer_parameter_names']==list(params),'Optimizer name/order mapping mismatch')
    check(set(template)==set(state['model']),'Model state keys mismatch')
    for name,tensor in state['model'].items():
        check(tensor.device.type=='cpu' and tensor.shape==template[name].shape and tensor.dtype==template[name].dtype,
              'Model tensor metadata mismatch: '+name)
    groups=state['optimizer']['param_groups'];moments=state['optimizer']['state']
    check(len(groups)==1,'Unexpected AdamW parameter groups')
    ids=groups[0]['params']
    check(len(ids)==len(params) and len(set(ids))==len(ids) and set(ids)==set(moments),'AdamW parameter/state coverage mismatch')
    for key,expected in {'lr':1e-4,'betas':(.9,.999),'eps':1e-8,'weight_decay':.01}.items():
        check(groups[0][key]==expected,'AdamW hyperparameter mismatch: '+key)
    for name,index in zip(params,ids):
        adam=moments[index]
        check(int(adam['step'].item())==500,'AdamW step differs from500: '+name)
        for key in ('exp_avg','exp_avg_sq'):
            check(adam[key].device.type=='cpu' and adam[key].shape==params[name].shape and adam[key].dtype==torch.float32,
                  'AdamW moment metadata mismatch: '+name+'/'+key)
    rng=state['rng'];check(set(rng['streams'])=={'posterior','flow_noise','time'},'Missing independent RNG streams')
    rng_evidence={}
    all_rng={**rng['streams'],'torch_cpu':rng['torch_cpu'],'torch_device':rng['torch_device']}
    for name,value in all_rng.items():
        check(isinstance(value,torch.Tensor) and value.device.type=='cpu' and value.dtype==torch.uint8 and value.ndim==1 and value.numel()>0,
              'Invalid saved RNG bytes: '+name)
        if name!='torch_device': torch.Generator(device='cpu').set_state(value)
        rng_evidence[name]=dict(bytes=value.numel(),sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest(),
            validated='CPU generator set_state accepted' if name!='torch_device' else 'bytes/shape/hash only; no CUDA initialization')
    after=path.stat()
    check((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),'Checkpoint changed during preflight')
    result=dict(passed=True,device='cpu',cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        torch_version=torch.__version__,mmap=True,real_gpu_used=False,runtime_modified=False,
        checkpoint=dict(path=str(path),bytes=before.st_size,mtime_ns=before.st_mtime_ns,sha256=entry['sha256'],
            sha_verification='Parent independently reverified full SHA immediately before this preflight; not reread here',
            stat_check='size matches manifest; mtime/size/inode unchanged during preflight; manifest has no historical mtime'),
        completed_updates=500,recipe_sha256=recipe_sha,cache_sha256=cache.sha256,
        normalization_sha256=cache.manifest['normalization_sha256'],source_vae_sha256=state['source_vae_sha256'],
        ordered_uids=cache.uids,cursor=cursor,parameter_tensors=len(params),parameters=sum(p.numel() for p in params.values()),
        model_state_tensors=len(template),adam_parameter_states=len(moments),adam_steps=[500],rng=rng_evidence,
        checked_runtime_files=len(code),seconds=time.monotonic()-started,
        limitation='Metadata/parameter mapping/RNG storage validation, not GPU resume execution or full tensor finite-value scan')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
