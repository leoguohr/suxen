"""One shared linear Edge head, three fixed feature matrices, no upstream model."""
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION','python')
import numpy as np
import torch as T

ROOT=Path(__file__).resolve().parent

def write(p,x):
    p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def norm(x):return float(x.detach().double().norm())

def setup():
    cfg=json.loads((ROOT/'config.json').read_text())
    T.set_num_threads(4);T.manual_seed(cfg['seed']);T.cuda.set_device(0)
    T.use_deterministic_algorithms(True);T.backends.cuda.matmul.allow_tf32=False;T.backends.cudnn.allow_tf32=False
    T.set_float32_matmul_precision('highest')
    initial=np.load(ROOT/'snapshots/head_initial.npz')
    head=T.nn.Linear(1024,32,bias=True).cuda().float()
    head.load_state_dict({k:T.from_numpy(initial[k].copy()).cuda() for k in ['weight','bias']})
    data=[];hashes={};first=None
    for uid,n in zip(cfg['uids'],cfg['vertices']):
        p=ROOT/'snapshots'/uid
        code=p/'effective_code/effective_loss_and_scoring.py'
        if first is None:
            first=sha(code);spec=importlib.util.spec_from_file_location('effective_scoring',code)
            scoring=importlib.util.module_from_spec(spec);spec.loader.exec_module(scoring)
        assert sha(code)==first
        d=np.load(p/'edge_all_pairs.npz');h=T.from_numpy(np.load(p/'hidden.npy').copy()).cuda()
        assert h.shape==(n,1024) and not h.requires_grad
        pairs=T.from_numpy(d['pairs'].copy()).cuda();labels=T.from_numpy(d['labels'].copy()).cuda()
        assert T.equal(pairs,T.triu_indices(n,n,1,device='cuda').T) and bool(d['in_training'].all())
        contract=json.loads((p/'effective_code/runtime_contract.json').read_text())
        assert len(pairs)<=contract['edge_chunk'] and not contract['membership_detach']
        assert contract['scales']['embedding_normalization']=='per_mesh_center_only'
        data.append(dict(uid=uid,hidden=h,pairs=pairs,labels=labels,contract=contract,
                         summary=json.loads((p/'summary.json').read_text())))
        for name in ['hidden.npy','head_gradient.npz','edge_all_pairs.npz','representations_and_gradients.npz',
                     'summary.json','effective_code/runtime_contract.json','effective_code/effective_loss_and_scoring.py']:
            hashes[str((p/name).relative_to(ROOT))]=sha(p/name)
    hashes['snapshots/head_initial.npz']=sha(ROOT/'snapshots/head_initial.npz')
    return cfg,head,data,scoring,hashes

def forward(head,d,scoring):
    raw=head(d['hidden']);center=raw-raw.mean(0,keepdim=True)
    pair=d['pairs'];logits=scoring.first_order_interval(center[pair[:,0]],center[pair[:,1]])*d['contract']['scales']['edge_logit_scale']
    ns,ms=scoring.soft4_sums(logits,d['labels']);means=ns/(ms+1e-8)
    return raw,center,logits,means.mean(),means

def metrics(d,values):
    raw,center,logits,loss,means=values;y=d['labels'];pred=logits>0
    tp=int((pred&y).sum());fp=int((pred&~y).sum());fn=int((~pred&y).sum());tn=int((~pred&~y).sum())
    return dict(uid=d['uid'],tp=tp,fp=fp,fn=fn,tn=tn,f1=2*tp/(2*tp+fp+fn),perfect=fp==fn==0,
                edge_soft4=float(loss.detach()),soft_group_means=means.detach().tolist(),
                min_margin_gt=float(logits[y].min().detach()),min_margin_non_gt=float((-logits[~y]).min().detach()),
                raw_embedding_norm=norm(raw),centered_embedding_norm=norm(center))

@T.no_grad()
def evaluate(head,data,scoring,step,coef):
    rows=[metrics(d,forward(head,d,scoring)) for d in data]
    return dict(step=step,meshes=rows,joint_perfect=all(r['perfect'] for r in rows),
                total_errors=sum(r['fp']+r['fn'] for r in rows),
                objective=coef*sum(r['edge_soft4'] for r in rows)/len(rows))

def main():
    out=ROOT/'run';assert not out.exists(),'Do not overwrite or rerun prior updates'
    out.mkdir();cfg,head,data,scoring,hashes=setup()
    coef=cfg['outer_coefficient'];params=list(head.parameters())
    baseline=evaluate(head,data,scoring,0,coef)
    checks=[];initial_weight=head.weight.detach().clone()
    # Baseline must reproduce original-network values, not an E* regression target.
    for d,row in zip(data,baseline['meshes']):
        p=ROOT/'snapshots'/d['uid'];a=np.load(p/'representations_and_gradients.npz');e=np.load(p/'edge_all_pairs.npz')
        values=forward(head,d,scoring);raw,center,logits,loss,parts=values
        gradients=T.autograd.grad(loss*coef,params)
        again=forward(head,d,scoring);gradient2=T.autograd.grad(again[3]*coef,params)
        reference=np.load(p/'head_gradient.npz')
        check=dict(uid=d['uid'],raw_exact=np.array_equal(raw.detach().cpu().numpy(),a['edge_head_raw']),
            centered_exact=np.array_equal(center.detach().cpu().numpy(),a['edge_embedding_scoring']),
            logits_exact=np.array_equal(logits.detach().cpu().numpy(),e['logits']),
            loss_exact=float(loss)==d['summary']['parts']['edge'],
            repeated_forward_exact=T.equal(logits,again[2]) and T.equal(loss,again[3]),
            repeated_backward_exact=all(T.equal(x,y) for x,y in zip(gradients,gradient2)),
            exported_head_gradient_exact=all(np.array_equal(t.cpu().numpy(),reference[k]) for t,k in zip(gradients,['weight','bias'])),
            counts_exact=all(row[k]==d['summary']['actual_reconstruction']['edge'][k] for k in ['tp','fp','fn','tn']))
        checks.append(check);write(out/'baseline_verification.json',dict(meshes=checks,optimizer_updates=0))
        assert all(v for k,v in check.items() if k!='uid'),check
        del values,raw,center,logits,loss,parts,gradients,again,gradient2
    manifest=dict(config=cfg,source_sha256=hashes,script_sha256=sha(Path(__file__)),
        trainable=['edge_head.weight','edge_head.bias'],head_instances=1,trainable_parameters=sum(p.numel() for p in params),
        fixed_hidden=True,all_three_meshes_per_update=True,gradient_accumulation='Three equal mesh contributions, one clip and one Adam step',
        scoring_dimensions=[16,16],regression_target_used=False,upstream_network_loaded=False,
        torch=T.__version__,cuda=T.version.cuda,gpu=T.cuda.get_device_name(0))
    write(out/'manifest.json',manifest)
    opt=T.optim.Adam(params,lr=cfg['lr'],betas=tuple(cfg['betas']),eps=cfg['eps'],weight_decay=cfg['weight_decay'])
    assert not opt.state

    def save(step,record,name,full=True):
        T.save(dict(step=step,head={k:v.detach().cpu() for k,v in head.state_dict().items()},
                    optimizer=opt.state_dict(),metrics=record,manifest=manifest),out/(name+'.pt'))
        write(out/(name+'.json'),record)
        if full:
            with T.no_grad():
                for d in data:
                    raw,center,logits,loss,parts=forward(head,d,scoring)
                    np.savez_compressed(out/(name+'-'+d['uid']+'.npz'),edge_head_raw=raw.cpu().numpy(),
                                        edge_embedding_scoring=center.cpu().numpy(),logits=logits.cpu().numpy())

    save(0,baseline,'checkpoint-step0000');history=[baseline]
    first=None;streak=longest=count=0;best=baseline['total_errors'];best_step=0;start=time.monotonic()
    with (out/'updates.jsonl').open('x',buffering=1) as log:
        log.write(json.dumps(baseline,allow_nan=False)+'\n')
        for step in range(1,cfg['updates']+1):
            opt.zero_grad(set_to_none=True);before=[p.detach().clone() for p in params];pre_e=[];pre_loss=[]
            for d in data:
                raw,center,logits,loss,parts=forward(head,d,scoring)
                assert bool(T.isfinite(loss))
                pre_e.append(center.detach().clone());pre_loss.append(float(loss.detach()))
                (loss*(coef/len(data))).backward()
            gn={k:norm(p.grad) for k,p in head.named_parameters()}
            total=float(T.nn.utils.clip_grad_norm_(params,cfg['clip'],error_if_nonfinite=True))
            opt.step()
            record=evaluate(head,data,scoring,step,coef)
            updates={k:dict(l2=norm(p.detach()-old),relative_l2=norm(p.detach()-old)/max(norm(old),1e-30),
                           changed_elements=int((p.detach()!=old).sum()),norm=norm(p))
                     for (k,p),old in zip(head.named_parameters(),before)}
            with T.no_grad():
                for d,row,old in zip(data,record['meshes'],pre_e):
                    e=head(d['hidden']);e=e-e.mean(0,keepdim=True)
                    row['embedding_relative_update']=norm(e-old)/max(norm(old),1e-30)
            record.update(grad_norms_preclip=gn,global_grad_norm_preclip=total,
                clip_coefficient=min(1.,cfg['clip']/(total+1e-6)),parameter_updates=updates,
                weight_norm_ratio_from_initial=norm(head.weight)/norm(initial_weight),
                objective_before_update=coef*sum(pre_loss)/len(data),elapsed_seconds=time.monotonic()-start)
            history.append(record);log.write(json.dumps(record,allow_nan=False)+'\n')
            if record['joint_perfect']:
                count+=1;streak+=1;longest=max(longest,streak)
                if first is None:first=step;save(step,record,'first-joint-perfect')
            else:streak=0
            if record['total_errors']<best:
                best=record['total_errors'];best_step=step;save(step,record,'best',full=False)
            if step in cfg['checks']:save(step,record,f'checkpoint-step{step:04d}')
            if step==1 or step%50==0:
                print('UPDATE',step,json.dumps(dict(objective=record['objective'],joint=record['joint_perfect'],
                    counts=[(r['uid'],r['fp'],r['fn']) for r in record['meshes']]),allow_nan=False),flush=True)
    assert all(sha(ROOT/k)==h for k,h in hashes.items())
    complete=dict(updates=cfg['updates'],baseline=baseline,final=history[-1],first_joint_perfect_step=first,
        joint_perfect_updates=count,longest_joint_perfect=longest,last500_joint_perfect=sum(r['joint_perfect'] for r in history[-500:]),
        best_step=best_step,minimum_total_errors=best,inputs_unchanged=True,network_updates=0,
        weight_updates_nonzero=all(r['parameter_updates']['weight']['changed_elements']>0 for r in history[1:]),
        wall_seconds=time.monotonic()-start)
    write(out/'complete.json',complete);print('COMPLETE',json.dumps(complete),flush=True)

if __name__=='__main__':main()
