"""One original-shaped shared linear head on 100 frozen Decoder feature matrices."""
import hashlib,importlib.util,json,os,time
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
import numpy as np
import torch as T
ROOT=Path(__file__).resolve().parent
def write(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def norm(x):return float(x.detach().double().norm())
def load():
    cfg=json.loads((ROOT/'config.json').read_text());meta=json.loads((ROOT/'cache/manifest.json').read_text())
    T.set_num_threads(4);T.manual_seed(cfg['seed']);T.cuda.set_device(0)
    T.use_deterministic_algorithms(True);T.backends.cuda.matmul.allow_tf32=False;T.backends.cudnn.allow_tf32=False
    T.set_float32_matmul_precision('highest')
    code=ROOT/'effective_code/effective_loss_and_scoring.py'
    spec=importlib.util.spec_from_file_location('scoring',code);scoring=importlib.util.module_from_spec(spec);spec.loader.exec_module(scoring)
    head=T.nn.Linear(1024,32,bias=True).cuda().float()
    with np.load(ROOT/'cache/head_original.npz') as d:head.load_state_dict({k:T.from_numpy(d[k].copy()).cuda() for k in ['weight','bias']})
    data=[]
    for row in meta['meshes']:
        uid=row['uid'];hp=ROOT/'cache'/f'{uid}.npz';assert sha(hp)==row['sha256']
        with np.load(hp) as a:
            h=T.from_numpy(a['hidden'].copy()).cuda();gt=T.from_numpy(a['edges'].copy()).cuda()
        n=len(h);assert h.shape==(n,1024) and not h.requires_grad
        pairs=T.triu_indices(n,n,1,device='cuda').T
        keys=gt[:,0]*n+gt[:,1];q=pairs[:,0]*n+pairs[:,1]
        at=T.searchsorted(keys,q);labels=(at<len(keys))&(keys[at.clamp_max(len(keys)-1)]==q)
        assert len(pairs)==row['pairs'] and int(labels.sum())==row['gt_edges']
        data.append(dict(uid=uid,hidden=h,pairs=pairs,labels=labels,reference=row))
    assert len(data)==100 and sum(len(d['hidden']) for d in data)==106325
    assert sum(len(d['pairs']) for d in data)==84669234
    return cfg,meta,head,data,scoring
def forward(head,d,scoring,scale):
    raw=head(d['hidden']);center=raw-raw.mean(0,keepdim=True);pair=d['pairs']
    logits=scoring.first_order_interval(center[pair[:,0]],center[pair[:,1]])*scale
    ns,ms=scoring.soft4_sums(logits,d['labels']);means=ns/(ms+1e-8)
    return raw,center,logits,means.mean()
def metrics(d,values):
    raw,center,s,loss=values;y=d['labels'];pred=s>0
    tp=int((pred&y).sum());fp=int((pred&~y).sum());fn=int((~pred&y).sum());tn=int((~pred&~y).sum())
    return dict(uid=d['uid'],tp=tp,fp=fp,fn=fn,tn=tn,perfect=fp==fn==0,f1=2*tp/max(2*tp+fp+fn,1),
        edge_soft4=float(loss.detach()),min_margin_gt=float(s[y].min().detach()),min_margin_non_gt=float((-s[~y]).min().detach()))
def summary(rows,step):
    return dict(step=step,meshes=rows,edge_perfect=sum(r['perfect'] for r in rows),all100_perfect=all(r['perfect'] for r in rows),
        objective=sum(r['edge_soft4'] for r in rows)/100,total_fp=sum(r['fp'] for r in rows),total_fn=sum(r['fn'] for r in rows))
def cycle(head,data,scoring,scale,backward):
    rows=[]
    for d in data:
        with T.set_grad_enabled(backward):
            values=forward(head,d,scoring,scale)
            rows.append(metrics(d,values))
            if backward:(values[3]/100).backward()
        del values
    return rows
