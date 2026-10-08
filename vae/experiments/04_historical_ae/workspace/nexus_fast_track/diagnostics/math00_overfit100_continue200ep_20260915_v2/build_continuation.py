"""Derive a bounded continuation from the completed, archived fresh runner."""
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'math00_overfit100_fresh_20260914'
src=(PARENT/'train.py').read_text()
src=src.replace('Bounded fresh100 mu training: 4 whole meshes/update, 200 epochs, no KL.',
                'Bounded continuation: resume update5000, stop at10000, identical reconstruction training.')
src=src.replace('CHECK_EPOCHS=[0,10,25,50,100,150,200]', '''PARENT=BASE/'diagnostics/math00_overfit100_fresh_20260914'
SOURCE=PARENT/'run/checkpoint-update05000.pt'
SOURCE_SHA='6a652045b865b776ca9bfbddd68bf1fc454c1fe29ec5aa83dec6cc3629186457'
CHECK_EPOCHS=[200,250,300,350,400]

def same(a,b):
    if T.is_tensor(a):return T.is_tensor(b) and a.dtype==b.dtype and T.equal(a.detach().cpu(),b.detach().cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

def rng_snapshot(shuffle):
    import copy
    return dict(python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=T.get_rng_state(),
                cuda_rng=T.cuda.get_rng_state_all(),shuffle_rng=copy.deepcopy(shuffle.bit_generator.state))

def restore_rng(state,shuffle):
    random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);T.set_rng_state(state['torch_rng'])
    T.cuda.set_rng_state_all(state['cuda_rng']);shuffle.bit_generator.state=state['shuffle_rng']
''')
src=src.replace('def main(preflight):','def main():')
src=src.replace('import argparse,time,json,random,fcntl,shutil,sys,traceback','import time,json,random,fcntl,traceback')
src=src.replace('ROOT,BASE,SEED,T,np,c,b,setup','ROOT,BASE,T,np,c,b,setup')
start=src.index('    model,opt,groups')
end=src.index('    def objective(uid):',start)
src=src[:start]+'''    out=ROOT/'run';out.mkdir(exist_ok=True)
    assert not (out/'updates.jsonl').exists(),'Refuse overwrite or duplicate continuation'
    assert sha(SOURCE)==SOURCE_SHA,'Parent checkpoint differs from recorded epoch200 checkpoint'
    cp=T.load(SOURCE,map_location='cpu',mmap=True,weights_only=False)
    assert cp['completed_updates']==5000 and cp['epoch']==200
    old_manifest=cp['manifest']
    for p,digest in old_manifest['source_sha256'].items():assert sha(p)==digest,p
    model,opt,groups,uids,pools,forward,capture,args=setup()
    assert uids==old_manifest['uids'] and cp['participation'].keys()==set(uids)
    assert set(cp['participation'].values())=={200}
    assert {n:list(g) for n,g in groups.items()}==old_manifest['groups']
    assert [g['name'] for g in opt.param_groups]==[g['name'] for g in cp['optimizer']['param_groups']]
    model.load_state_dict(cp['model'],strict=True)
    opt.load_state_dict(cp['optimizer'])
    assert same(model.state_dict(),cp['model']) and same(opt.state_dict(),cp['optimizer'])
    for pg in opt.param_groups:
        assert abs(pg['lr']-(1e-5 if pg['name']=='encoder_mu' else 1e-4))<1e-18
        assert pg['betas']==(.9,.999) and pg['eps']==1e-8 and pg['weight_decay']==0
    assert all(p not in opt.state for p in model.autoencoder.log_variance.parameters())
    active=[p for g in groups.values() for p in g.values()];scales=model.scoring_contract()
    assert scales==old_manifest['scales']
    freeze_hash=tensor_hash(model.autoencoder.log_variance.named_parameters())
    shuffle=np.random.default_rng();restore_rng(cp,shuffle)
    assert same(rng_snapshot(shuffle),{k:cp[k] for k in rng_snapshot(shuffle)})
    participation=dict(cp['participation']);step=5000;epoch=200;first_perfect=False
    import copy
    manifest=copy.deepcopy(old_manifest)
    manifest.update(random_initialization=False,initialization='resume own fresh100 epoch200 checkpoint',
        optimizer='restored four-group Adam, all moments and counters preserved',
        parent_checkpoint=str(SOURCE),parent_sha256=SOURCE_SHA,start_epoch=200,start_update=5000,
        epochs=400,updates=10000,additional_epochs=200,additional_updates=5000,participations=40000,
        checks_epochs=CHECK_EPOCHS,warmup='none in continuation; retain terminal parent LR',
        entry_sha256={n:sha(ROOT/n) for n in ['train_continue.py','runtime.py','evaluate.py','construction_args.json']})
    manifest['args']=dict(cp['args'],output=str(ROOT),steps=10000)
    args=manifest['args']
    write(out/'manifest.json',manifest)
    write(out/'resume_verification.json',dict(parent_checkpoint=str(SOURCE),sha256=SOURCE_SHA,
        model_bitwise_equal=True,adam_all_states_bitwise_equal=True,rng_all_states_equal=True,
        uid_order_equal=True,participations_equal=True,parameter_group_names_equal=True,
        lrs={pg['name']:pg['lr'] for pg in opt.param_groups},
        adam_state_count=len(opt.state),logvar_frozen=True,optimizer_updates_at_verification=0,
        parent_source_hashes_verified=True,logvar_parameter_hash=freeze_hash))
    del cp
''' +src[end:]
start=src.index('    if preflight:')
end=src.index('    def save():',start)
src=src[:start]+src[end:]
src=src.replace("        before=tensor_hash(model.named_parameters());result=[];t=time.monotonic()", "        rng=rng_snapshot(shuffle)\n        before=tensor_hash(model.named_parameters());result=[];t=time.monotonic()")
src=src.replace("        write(out/f'eval-summary-epoch{epoch:03d}.json',summary);return summary", "        restore_rng(rng,shuffle);assert same(rng_snapshot(shuffle),rng)\n        write(out/f'eval-summary-epoch{epoch:03d}.json',summary);return summary")
src=src.replace('        path=save();evaluate(path)', '''        path=SOURCE;evaluate(path)
        old_rows=[json.loads(s) for s in (PARENT/'run/eval-epoch200.jsonl').read_text().splitlines()]
        new_rows=[json.loads(s) for s in (out/'eval-epoch200.jsonl').read_text().splitlines()]
        mismatches=[]
        for old,new in zip(old_rows,new_rows):
            for k in ['uid','parts','edge','face_training_pool','gt_face_candidates','missing_gt_face_candidates','margins','joint_perfect']:
                if old[k]!=new[k]:mismatches.append(dict(uid=old['uid'],field=k,old=old[k],new=new[k]))
            if old['face']!=new['face']:mismatches.append(dict(uid=old['uid'],field='face',old=old['face'],new=new['face']))
        write(out/'baseline_comparison.json',dict(meshes=len(new_rows),exact_match=not mismatches,mismatches=mismatches))
        assert len(new_rows)==len(old_rows)==100 and not mismatches,'Baseline differs; do not begin updates'
        write(out/'training_ready.json',dict(baseline_verified=True,start_update=5000,end_update=10000))
        print('RESUME_BASELINE_VERIFIED',flush=True)''')
src=src.replace('for epoch in range(1,201):','for epoch in range(201,401):')
src=src.replace("                    factor=1+9*min((step-1)/99,1)\n                    for pg in opt.param_groups:pg['lr']=(1e-6 if pg['name']=='encoder_mu' else 1e-5)*factor\n",'')
src=src.replace('row=dict(update=step,epoch=epoch,','row=dict(update=step,additional_update=step-5000,epoch=epoch,')
src=src.replace('if step<=3 or step%25==0:','if step<=5003 or step%25==0:')
src=src.replace('assert step==5000 and set(participation.values())=={200}','assert step==10000 and set(participation.values())=={400}')
src=src.replace("result=dict(state='completed',epochs=200,updates=5000,mesh_participations=20000,", "result=dict(state='completed',epochs=400,updates=10000,additional_updates=5000,mesh_participations=40000,")
src=src.replace("out/'eval-summary-epoch200.json').read_text()),stopped_at_budget=True", "out/'eval-summary-epoch400.json').read_text()),stopped_at_budget=True")
src=src.replace("print('COMPLETE_5000',flush=True)","print('COMPLETE_10000_ADDITIONAL5000',flush=True)")
src=src[:src.index("if __name__=='__main__':")]+"if __name__=='__main__':\n    main()\n"
(ROOT/'train_continue.py').write_text(src)
for f in ['runtime.py','evaluate.py','construction_args.json']:
    shutil.copy2(PARENT/f,ROOT/f)
compile(src,str(ROOT/'train_continue.py'),'exec')
print(ROOT/'train_continue.py')
