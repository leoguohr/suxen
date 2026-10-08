"""Read-only qualification of the existing Soft4 control; no CUDA initialization."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch as T
T.set_num_threads(1)
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent.parent
S=ROOT.parent/'teacher_cad50_soft4_stopgrad_pair_20260922/A_soft4_full'
P=ROOT.parent/'teacher_cad50_lr03_pair_20260921/B_lr03'
D=ROOT.parent/'teacher_cad50_fresh512_20260921'
SOURCE=P/'checkpoint-new0500-step2500.pt'
SOURCE_SHA='4c67709dcd3bacb6a09181b034512469b1e7f38463f1aa7b41affc8bcf435a66'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def same(a,b):
    if T.is_tensor(a):return T.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b
assert sha(SOURCE)==SOURCE_SHA
cfg=json.loads((S/'config.json').read_text())
assert cfg['source_sha256']==SOURCE_SHA and cfg['mode']=='full' and not cfg['stop_weight_grad']
assert cfg['lrs']==dict(encoder_mu=3e-6,decoder=3e-5,edge_head=3e-5,face_head=3e-5)
assert cfg['betas']==[.9,.999] and cfg['eps']==1e-8 and cfg['weight_decay']==0 and cfg['clip']==1
assert cfg['new_updates']==100 and cfg['checkpoints']==[0,25,50,75,100]
assert cfg['microbatch']==1 and cfg['meshes_per_update']==50 and not cfg['sampling'] and cfg['KL']==0 and cfg['logvar_frozen']
assert cfg['totals']==dict(meshes=50,vertices=2872,edges=8364,faces=5576,pairs=235741,face_pool=13946)
assert cfg['environment']['torch']==T.__version__ and cfg['environment']['cuda']==T.version.cuda
assert cfg['environment']['cudnn']==T.backends.cudnn.version()
assert cfg['environment']['python'].split()[0]==sys.version.split()[0]
assert sha(D/'data/manifest.json')==cfg['data_manifest_sha256']
assert sha(D/'pool_manifest.json')==cfg['pool_manifest_sha256']
data=json.loads((D/'data/manifest.json').read_text());pool=json.loads((D/'pool_manifest.json').read_text())
for row in data['meshes']:assert sha(D/'data'/row['path'])==row['sha256']
for row in pool['records']:assert sha(D/row['path'])==row['sha256']
for name,digest in cfg['code_sha256'].items():assert sha(S/name)==digest,name
dependencies=[]
for f in (S/'source_archive').rglob('*.py'):
    actual=BASE/f.relative_to(S/'source_archive')
    assert actual.is_file() and sha(actual)==sha(f),str(actual)
    dependencies.append(dict(path=str(actual),sha256=sha(f)))
logs=[json.loads(x) for x in (S/'updates.jsonl').read_text().splitlines()]
assert [x['new_update'] for x in logs]==list(range(1,101))
for step,row in enumerate(logs,1):
    assert row['completed_updates']==2500+step and row['state_before']==2499+step
    assert [m['uid'] for m in row['meshes']]==data['uids']==cfg['uids']
    assert row['lrs']==cfg['lrs'] and set(row['adam_steps'].values())=={2500+step}
    assert row['participation_per_mesh']==2500+step
    assert row['clip_coefficient']==min(1.,1/(row['total_grad_norm']+1e-6))
done=json.loads((S/'complete.json').read_text())
assert done['new_updates']==100 and done['completed_updates']==2600 and done['stopped_at_budget']
assert not (S/'failure.json').exists()
parent=T.load(SOURCE,map_location='cpu',mmap=True,weights_only=False)
start=S/'checkpoint-new0000-step2500.pt';cp=T.load(start,map_location='cpu',mmap=True,weights_only=False)
for key in ['model','optimizer','rng','participation','completed_updates']:assert same(cp[key],parent[key]),key
assert cfg['trainable_groups']==parent['config']['trainable_groups']
assert len(parent['optimizer']['state'])==sum(len(g['params']) for g in parent['optimizer']['param_groups'])
assert all(int(x['step'])==2500 for x in parent['optimizer']['state'].values())
checks=[]
for step in [0,25,50,75,100]:
    e=json.loads((S/f'eval-new{step:04d}.json').read_text())
    checkpoint=Path(e['checkpoint']);digest=sha(checkpoint);assert digest==e['checkpoint_sha256']
    assert len(e['meshes'])==50 and [x['uid'] for x in e['meshes']]==data['uids']
    for row in e['meshes']:
        assert row['face']['complete'] and sha(S/row['prediction_path'])==row['prediction_sha256']
    if step==0:
        pe=json.loads((P/'eval-new0500.json').read_text())
        assert e['counts']==pe['counts'] and e['perfect_uids']==pe['perfect_uids']
        assert e['counts']['edge']['fp']==5159 and e['counts']['edge']['fn']==1656
        assert e['counts']['face']['fp']==5359 and e['counts']['face']['fn']==3255
        assert e['counts']['face']['micro_f1']==0.350181050090525 and e['joint_perfect']==32
    checks.append(dict(step=step,path=str(checkpoint),sha256=digest,bytes=checkpoint.stat().st_size))
assert not T.cuda.is_initialized()
result=dict(passed=True,reuse_control=True,source=str(SOURCE),source_sha256=SOURCE_SHA,
    control=str(S),control_start_sha256=sha(start),model_adam_rng_progress_groups_exact=True,
    data_pool_verified=True,code_hashes_match_original_run=True,dependencies=dependencies,
    all100_updates_logged=True,checkpoint_identity_checks=checks,prediction_files_hashed=250,
    full_prediction_recount='scheduled for post-run independent audit; original evaluation records already present',
    environment_matches=True,optimizer_updates=0,gpu_used=False,
    control_config_sha256=sha(S/'config.json'),control_updates_sha256=sha(S/'updates.jsonl'))
(ROOT/'repro_outputs/CONTROL_REUSE_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
(ROOT/'S_soft4_full_control/REUSED_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
print('CONTROL_REUSE_PASS',len(dependencies),'dependencies; model/Adam/RNG exact; 100 updates; 5 checkpoints',flush=True)
