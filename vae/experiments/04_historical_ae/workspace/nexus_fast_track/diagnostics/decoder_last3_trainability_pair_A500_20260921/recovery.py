"""Explicit recovery from verified update204; never replay completed updates."""
import copy,fcntl,hashlib,json,os,sys,traceback
from pathlib import Path
import pipeline as P
R=P.ROOT;O=R/'repro_outputs/recovery_204'
DIGEST='4e4ddfca20111a2def7930acd8128730090dc7e41acc50c05ad72f26118ddaf6'

def verify():
    import torch as T
    import importlib.metadata as m
    T.set_num_threads(1)
    def sha(p):
        h=hashlib.sha256()
        with Path(p).open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        return h.hexdigest()
    assert sha(O/'latest.pt')==sha(R/'Control_last2/run/latest.pt')==DIGEST
    cp=T.load(O/'latest.pt',map_location='cpu',mmap=True,weights_only=False)
    assert cp['branch']=='Control_last2' and cp['new_updates']==204
    assert cp['config']==P.CFG and cp['source_full_sha256']==P.CFG['source_full_sha256']
    assert cp['source_checkpoint_sha256']==P.CFG['source_checkpoint_sha256']
    assert len(cp['per_uid_new_training_participations'])==100 and set(cp['per_uid_new_training_participations'].values())=={204}
    for group in cp['optimizer']['param_groups']:
        assert {int(cp['optimizer']['state'][i]['step']) for i in group['params']}=={1204 if group['name']=='decoder14' else 2204}
    for name in ['updates.jsonl','step_records.jsonl']:
        assert sha(R/'Control_last2/run'/name)==sha(O/name)
    for name,digest in json.loads((R/'repro_outputs/SOURCE_VERSION.json').read_text())['files'].items():assert sha(R/name)==digest,name
    for name,digest in json.loads((R/'repro_outputs/loaded_source_manifest.json').read_text()).items():assert sha(name)==digest,name
    packages={d.metadata['Name']:d.version for d in m.distributions() if d.metadata.get('Name')}
    assert '\n'.join(sorted(k+'=='+v for k,v in packages.items()))+'\n'==(R/'repro_outputs/pip_freeze.txt').read_text()
    assert not (R/'Treatment_last3').exists()
    old_process=Path('/proc/1132/cmdline')
    if old_process.exists():assert str(R/'pipeline.py') not in old_process.read_text()
    return cp,sha

def gate(cp):
    ctx=P.setup_all();meta,data=P.load_data(ctx);T=P.T;np=P.np
    old=T.load(R/'Control_last2/run/checkpoint-new0200.pt',map_location='cpu',mmap=True,weights_only=False)
    tail,opt,named,mnames=P.restore_branch(ctx['model'],ctx['full'],ctx['cp'],False)
    tail.load_state_dict(old['tail']);opt.load_state_dict(old['optimizer']);P.restore_rng(old)
    expected=json.loads((R/'Control_last2/run/actual-new0200.json').read_text());rows=[];start=P.rng()
    pred=O/'cross_container_predictions0200';pred.mkdir()
    for i,d in enumerate(data):
        real=ctx['forward'](d['uid']);h,ev,fv=ctx['core'].score(tail,d,ctx['sc'],ctx['scales'])
        assert T.equal(h.detach(),ctx['capture']['hidden']) and T.equal(ev[1],real[2][0]) and T.equal(fv[1],real[3][0])
        p=pred/(d['uid']+'.npz');actual=ctx['evaluate'].evaluate_mesh(real,d['pool'],ctx['scales'],prediction_path=p)
        with np.load(p) as a,np.load(R/expected['meshes'][i]['prediction_path']) as b:
            assert set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files),d['uid']
        e=expected['meshes'][i];assert ev[3].item()==e['edge_soft4'] and fv[3].item()==e['face_soft4']
        rows.append({'uid':d['uid'],'checkpoint200_prediction_arrays_bitwise':True})
        ctx['capture'].clear();del real,h,ev,fv
        if (i+1)%25==0:print('RECOVERY_CROSS_CONTAINER200',i+1,'/100',flush=True)
    P.same(start,P.rng())
    tail.load_state_dict(cp['tail']);opt.load_state_dict(cp['optimizer']);P.restore_rng(cp)
    P.same(tail.state_dict(),cp['tail']);P.same(opt.state_dict(),cp['optimizer']);start=P.rng()
    params=tuple(p for _,p in named)
    for i,d in enumerate(data):
        real=ctx['forward'](d['uid']);le,lf,el,fl=P.original_metrics(ctx,d,real)
        g=T.autograd.grad((le+lf)/100,params);h,ev,fv=ctx['core'].score(tail,d,ctx['sc'],ctx['scales'])
        g2=T.autograd.grad((ev[3]+fv[3])/100,params)
        assert T.equal(h.detach(),ctx['capture']['hidden']) and T.equal(ev[1],real[2][0]) and T.equal(fv[1],real[3][0])
        assert T.equal(le,ev[3]) and T.equal(lf,fv[3]) and T.equal(el,ev[2]) and T.equal(fl,fv[2])
        assert all(T.equal(a,b) for a,b in zip(g,g2)),d['uid']
        rows[i]['checkpoint204_full_cached_forward_gradient_bitwise']=True
        ctx['capture'].clear();del real,le,lf,el,fl,g,g2,h,ev,fv
        if (i+1)%25==0:print('RECOVERY_FORWARD_GRAD204',i+1,'/100',flush=True)
    P.same(start,P.rng());P.same(tail.state_dict(),cp['tail']);P.same(opt.state_dict(),cp['optimizer'])
    assert P.fingerprint(ctx['model'],mnames)==json.loads((R/'Control_last2/run/restore_verification.json').read_text())['frozen_state']
    P.write(O/'gate.json',dict(status='passed',optimizer_updates=0,resume_checkpoint_sha256=DIGEST,all100_checkpoint200_predictions_bitwise=True,all100_checkpoint204_full_cached_forward_gradient_bitwise=True,restored_adam_and_all_four_rng_exact=True,rows=rows))
    print('RECOVERY GATE PASSED optimizer_updates=0',flush=True)

if __name__=='__main__':
    stage=sys.argv[1];assert stage in ['gate','train']
    lock=(R/'execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        cp,sha=verify()
        if stage=='gate':
            assert not (O/'gate.json').exists();gate(cp)
        else:
            result=json.loads((O/'gate.json').read_text());assert result['status']=='passed' and result['optimizer_updates']==0 and result['resume_checkpoint_sha256']==DIGEST
            P.RECOVERY_CP=cp;P.RECOVERY_SHA=DIGEST;P.sha=sha
            path=R/'recovery_train.py';exec(compile(path.read_text(),str(path),'exec'),P.__dict__)
            P.train()
    except BaseException:
        (O/(stage+'_failure.txt')).write_text(traceback.format_exc())
        if P.LIVE:
            P.disk_save(R/P.LIVE['branch']/'recovery-failure-state.pt',dict(tail=P.cpu_state(P.LIVE['tail']),optimizer=P.LIVE['opt'].state_dict(),**P.rng(),branch=P.LIVE['branch'],new_updates=P.LIVE['step']))
        raise
