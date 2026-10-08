"""Generate two bounded continuations from the hash-verified parent runner."""
from pathlib import Path
import hashlib
import difflib

ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'math00_latent512_804_fresh_20260914'
source=(PARENT/'train.py').read_text()
assert hashlib.sha256(source.encode()).hexdigest()=='7af5153a8c68b93373d4db0dfee45cdf18cb7319455f8bde28987309c19fd5e0'
(ROOT/'parent_train.py').write_text(source)

for branch,mult in [('A_hold',1.0),('B_lr03',0.3)]:
    s=source
    def replace(old,new):
        global s
        assert s.count(old)==1,old
        s=s.replace(old,new)
    replace('"""Fresh latent512, one complete804 mesh, mu reconstruction; bounded3000 updates."""',
            '"""Paired continuation from latent512 step3000, unchanged mu objective; 500 updates."""')
    replace('CHECKS = [0, 100, 200, 500, 1000, 1500, 2000, 2500, 2750, 3000]',f'''CHECKS = [0, 100, 200, 300, 400, 500]
PARENT = ROOT.parent.parent/'math00_latent512_804_fresh_20260914'
PARENT_CHECKPOINT = PARENT/'checkpoint-update3000.pt'
PARENT_SHA = '06dc42726c816b69f06db353d4de505e9d8bceda864294c8bb75c7907ba0bdd6'
LR_MULTIPLIER = {mult}
BRANCH = '{branch}' ''')
    replace("args = json.loads((ROOT/'construction_args.json').read_text())",'''assert c.m.probe.digest(PARENT_CHECKPOINT) == PARENT_SHA
cp = T.load(PARENT_CHECKPOINT, map_location='cpu', mmap=True)
assert cp['completed_updates'] == 3000
for path,sha in cp['diagnostic']['source_sha256'].items():
    assert c.m.probe.digest(path) == sha, path
args = dict(cp['args'])''')
    replace('args.update(seed=SEED,steps=3000,expected_samples=1,max_packed_meshes=1,output=str(ROOT),',
            'args.update(seed=SEED,steps=500,expected_samples=1,max_packed_meshes=1,output=str(ROOT),')
    replace('# Constructor only: no checkpoint weights, fitted head, teacher embedding or optimizer is loaded.',
            '# Construct the same architecture, then strictly restore every parameter before any forward.')
    replace('a = model.autoencoder', '''model.load_state_dict(cp['model'], strict=True)
assert all(T.equal(p.detach().cpu(),cp['model'][n]) for n,p in model.named_parameters())
a = model.autoencoder''')
    replace('initialization_rng = T.get_rng_state().clone()',"initialization_rng = cp['initialization_rng'].clone()")
    replace("pool_path = REF/(UID+'_pool.npz')","pool_path = PARENT/(UID+'_pool.npz')")
    replace("runtime = (OLD/'sampling_forward.py').read_text()","runtime = (PARENT/'sampling_forward.py').read_text()")
    replace('assert len(optimizer.state) == 0', '''assert {g:list(ps) for g,ps in groups.items()} == cp['diagnostic']['groups']
optimizer.load_state_dict(cp['optimizer'])
restored = optimizer.state_dict()
assert restored['param_groups'] == cp['optimizer']['param_groups']
assert set(restored['state']) == set(cp['optimizer']['state'])
for key,state in restored['state'].items():
    for field,value in state.items():
        saved = cp['optimizer']['state'][key][field]
        assert T.equal(value.detach().cpu(),saved.cpu()) if isinstance(value,T.Tensor) else value==saved
    assert int(state['step']) == 3000
PARENT_LR={pg['name']:pg['lr'] for pg in optimizer.param_groups}
for pg in optimizer.param_groups:
    pg['lr'] = PARENT_LR[pg['name']]*LR_MULTIPLIER
random.setstate(cp['python_rng']); np.random.set_state(cp['numpy_rng'])
T.set_rng_state(cp['torch_rng']); T.cuda.set_rng_state_all(cp['cuda_rng'])''')
    replace('completed_updates=step,seed=SEED,python_rng=',
            'completed_updates=3000+step,additional_updates=step,seed=SEED,python_rng=')
    replace("manifest=dict(experiment='fresh512_804_mu',seed=SEED,initialization='normal constructor; no weights loaded',",'''manifest=dict(experiment='latent512_804_lr03_pair',branch=BRANCH,lr_multiplier=LR_MULTIPLIER,
              parent_checkpoint=str(PARENT_CHECKPOINT),parent_sha256=PARENT_SHA,parent_completed_updates=3000,
              lr={pg['name']:pg['lr'] for pg in optimizer.param_groups},seed=SEED,initialization='exact parent weights and four Adam groups restored', ''')
    replace("optimizer='fresh Adam'","optimizer='all four parent Adam states restored; only target LR modified'")
    replace("warmup=dict(first_update=[1e-6,1e-5],update100=[1e-5,1e-4],formula='(update-1)/99 then constant'),", "warmup=None,")
    replace('budget=3000,checks=CHECKS','budget=500,checks=CHECKS')
    replace("dict(state='preflight',completed_updates=0)","dict(state='preflight',completed_updates=3000,additional_updates=0)")
    replace("dict(state='training',completed_updates=completed,last_update_seconds=", "dict(state='training',completed_updates=3000+completed,additional_updates=completed,last_update_seconds=")
    replace("dict(state='failed',completed_updates=completed,error=", "dict(state='failed',completed_updates=3000+completed,additional_updates=completed,error=")
    replace('rec=dict(update=step,loss=', 'rec=dict(update=step,cumulative_update=3000+step,loss=')
    replace("write(ROOT/f'eval-update{step:04d}.json',rec);evaluations.append(rec)",'''if step == 0:
        parent_eval=json.loads((PARENT/'eval-update3000.json').read_text())
        for key in ['loss','parts','edge','face','actual_face_candidates','gt_face_candidates',
                    'missing_gt_face_candidates','min_margin','scales']:
            assert rec[key] == parent_eval[key], ('parent baseline mismatch',key)
        assert T.equal(T.get_rng_state(),cp['torch_rng'])
        assert all(T.equal(x,y) for x,y in zip(T.cuda.get_rng_state_all(),cp['cuda_rng']))
        write(ROOT/'parent_baseline_verified.json',dict(weights_exact=True,adam_moments_steps_exact=True,
              baseline_loss_counts_margins_scales_exact=True,rng_restored_and_unchanged_by_evaluation=True,
              parent_sha256=PARENT_SHA,lr={pg['name']:pg['lr'] for pg in optimizer.param_groups}))
    write(ROOT/f'eval-update{step:04d}.json',rec);evaluations.append(rec)''')
    replace('repeated_forward_bitwise=True,adam_states=0,',"repeated_forward_bitwise=True,adam_states=len(optimizer.state),parent_adam_step=3000,")
    replace('old_weights_loaded=False','parent_weights_loaded=True')
    replace('for step in range(1,3001):','for step in range(1,501):')
    replace("factor=1+9*min((step-1)/99,1)\n            for pg in optimizer.param_groups:pg['lr']=(1e-6 if pg['name']=='encoder_mu' else 1e-5)*factor", "for pg in optimizer.param_groups:\n                assert pg['lr']==PARENT_LR[pg['name']]*LR_MULTIPLIER")
    replace('record=dict(update=step,edge_soft4=', 'record=dict(update=step,cumulative_update=3000+step,edge_soft4=')
    replace("result=dict(state='completed',completed_updates=completed,logvar_unchanged=True,", "result=dict(state='completed',completed_updates=3000+completed,additional_updates=completed,logvar_unchanged=True,")
    replace("late_maintained_success=all(next(r for r in evaluations if r['update']==s)['perfect'] for s in [2500,2750,3000]),", "last_three_checks_perfect=all(next(r for r in evaluations if r['update']==s)['perfect'] for s in [300,400,500]),")
    compile(s,str(ROOT/branch/'train.py'),'exec')
    out=ROOT/branch;out.mkdir(exist_ok=True)
    (out/'train.py').write_text(s)
    (out/'changes_from_parent.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True),s.splitlines(True),fromfile='parent/train.py',tofile=branch+'/train.py')))
print('Prepared A_hold and B_lr03; no training executed by builder')
