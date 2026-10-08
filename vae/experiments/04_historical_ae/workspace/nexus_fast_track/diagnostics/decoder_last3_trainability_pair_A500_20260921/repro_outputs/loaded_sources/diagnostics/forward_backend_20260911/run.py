"""Read-only, same-GPU, fixed-epsilon backend repeatability diagnostic."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION','python')
import sys,inspect,json,time,fcntl,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'encoder_gradient_linesearch_20260911'))
import common as c
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
import mini_nexus.flash_varlen_topology as flash
WATCH=[[385,387,426],[677,730,734],[1737,1861,1862],[481,482,496]]

def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))

class Recorder:
    def __init__(self,out):
        self.out=out;self.state={};self.current={};self.order=[]
    def begin(self,k):self.k=k;self.current={}
    def add(self,name,tensor):
        assert name not in self.current, 'duplicate trace name '+name
        x=tensor.detach().float().cpu().numpy().copy()
        if not np.isfinite(x).all():raise ValueError('nonfinite '+name)
        if self.k==0:
            self.order.append(name)
            self.state[name]=dict(first=x.copy(),lo=x.copy(),hi=x.copy(),mean=x.astype(np.float64),m2=np.zeros_like(x,dtype=np.float64),rel=[],absolute=[])
        v=self.state[name];delta=x.astype(np.float64)-v['first']
        rel=float(np.linalg.norm(delta.ravel())/(np.linalg.norm(v['first'].astype(np.float64).ravel())+1e-30))
        absolute=float(np.abs(delta).max());v['rel'].append(rel);v['absolute'].append(absolute)
        if self.k:
            np.minimum(v['lo'],x,out=v['lo']);np.maximum(v['hi'],x,out=v['hi'])
            d=x-v['mean'];v['mean']+=d/(self.k+1);v['m2']+=d*(x-v['mean'])
        self.current[name]=dict(relative_l2=rel,max_absolute_delta=absolute)
    def finish(self):
        result={}
        for name in self.order:
            v=self.state[name];ran=v['hi'].astype(np.float64)-v['lo'];std=np.sqrt(v['m2']/(self.k+1))
            result[name]=dict(shape=list(v['first'].shape),range_max=float(ran.max()),range_rms=float(np.sqrt(np.mean(ran**2))),std_max=float(std.max()),std_rms=float(np.sqrt(np.mean(std**2))),relative_l2=v['rel'],max_absolute_delta=v['absolute'],changed_elements=int(np.count_nonzero(ran)))
            if name.startswith(('mu/','logvar/','edge_embedding/','face_embedding/','edge_logits/','face_logits/')):
                np.savez_compressed(self.out/(name.replace('/','__')+'.npz'),first=v['first'],minimum=v['lo'],maximum=v['hi'],std=std)
        write(self.out/'repeatability.json',dict(order=self.order,layers=result,first_nonidentical=next((n for n in self.order if result[n]['range_max']>0),None)))


def math_attention(attention,tokens,cu,maximum_length):
    assert tokens.dtype==torch.float32
    h=attention.num_heads;d=attention.embed_dim//h
    qkv=F.linear(tokens,attention.in_proj_weight,attention.in_proj_bias).reshape(len(tokens),3,h,d)
    bounds=cu.cpu().tolist();out=[]
    with sdpa_kernel(SDPBackend.MATH):
        for start,end in zip(bounds[:-1],bounds[1:]):
            q,k,v=[qkv[start:end,j].transpose(0,1).unsqueeze(0) for j in range(3)]
            out.append(F.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=False).squeeze(0).transpose(0,1).reshape(end-start,h*d))
    return F.linear(torch.cat(out),attention.out_proj.weight,attention.out_proj.bias)


def instrument(model,rec):
    # Packed implementation bypasses composite module.forward, so ordinary block hooks would miss it.
    source=flash._diagnostic_original_forward_source
    source=source.replace('.clamp(-10.0, 10.0)', '.clamp(autoencoder.diagnostic_logvar_min, 10.0)')
    source=source.replace('            latent_rows.append(', '            autoencoder._diagnostic_eps.append(noise.detach())\n            latent_rows.append(')
    substitutions=[
      ('    nodes = torch.cat(node_rows, dim=0)', '    nodes = torch.cat(node_rows, dim=0)\n    _record("encoder_input", nodes)'),
      ('for block in autoencoder.encoder_blocks:', 'for encoder_index, block in enumerate(autoencoder.encoder_blocks):'),
      ('        nodes = nodes + block.graph_activation(graph_update)', '        _record(f"E{encoder_index:02d}/graph_update", graph_update)\n        nodes = nodes + block.graph_activation(graph_update)\n        _record(f"E{encoder_index:02d}/after_graph", nodes)'),
      ('    # Encoder 序列', '        _record(f"E{encoder_index:02d}/block_output", nodes)\n\n    # Encoder 序列'),
      ('for block in autoencoder.decoder_blocks:', 'for decoder_index, block in enumerate(autoencoder.decoder_blocks):'),
      ('        hidden = hidden + update', '        hidden = hidden + update\n        _record(f"D{decoder_index:02d}/block_output", hidden)')]
    for old,new in substitutions:
        assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
    flash._record=rec.add
    exec(compile(source,str(ROOT/'instrumented_forward.py'),'exec'),flash.__dict__)
    (rec.out/'instrumented_forward.py').write_text(source)
    hooks=[]
    for name in ['encoder_output_norm','mu','log_variance','latent_input','decoder_output_norm','edge_embedding','face_embedding']:
        hooks.append(getattr(model.autoencoder,name).register_forward_hook(lambda mod,args,out,n=name:rec.add('module/'+n,out)))
    # Initial packed graph input is recorded once per forward, before any aggregation.
    hooks.append(model.autoencoder.encoder_blocks[0].graph.register_forward_pre_hook(lambda mod,args:rec.add('E00/graph_input',args[0])))
    original=flash._flash_self_attention
    names={id(b.transformer.self_attn):f'E{i:02d}' for i,b in enumerate(model.autoencoder.encoder_blocks)}
    names.update({id(b.attention):f'D{i:02d}' for i,b in enumerate(model.autoencoder.decoder_blocks)})
    def traced(attention,tokens,cu,maximum_length):
        key=names[id(attention)];rec.add(key+'/attention_input',tokens)
        out=original(attention,tokens,cu,maximum_length);rec.add(key+'/attention_output',out);return out
    flash._flash_self_attention=traced
    return hooks


def main(backend):
    out=ROOT/backend;out.mkdir(exist_ok=True)
    assert not (out/'trace.jsonl').exists(),'Never overwrite an existing run'
    cp,model,batch,enc,data=c.setup(0);model.requires_grad_(False)
    if backend=='B_fp32_math':
        torch.use_deterministic_algorithms(True)
        flash._flash_self_attention=math_attention
    else:assert backend=='A_original'
    rec=Recorder(out);hooks=instrument(model,rec)
    scales=model.scoring_contract();candidates=[]
    for uid,d in zip(c.UIDS,data):
        # Freeze exact candidate IDs from original successful B16100 sampled baseline, shared A/B.
        path=c.ROOT/'baseline'/f'{uid}.npz';t=np.load(path)
        ft=t['face_triples'];fa=t['face_actual'];ft=ft[fa]
        fy=t['face_labels'][fa];ep=t['edge_pairs'];ey=t['edge_labels']
        assert all(any(np.array_equal(w,f) for f in ft) for w in WATCH) if uid==c.UIDS[1] else True
        candidates.append((ep,ey,ft,fy))
        np.savez_compressed(out/(uid+'_candidates.npz'),edge_pairs=ep,edge_labels=ey,face_triples=ft,face_labels=fy)
    write(out/'manifest.json',dict(backend=backend,gpu=torch.cuda.get_device_name(0),gpu_index=0,checkpoint_sha256=c.m.START_SHA,source_sha256=cp['diagnostic_run']['source_sha256'],flash_file=flash.__file__,flash_sha256=c.m.probe.digest(flash.__file__),mode='sample',seeds=c.SEEDS,repeats=50,threshold=0,optimizer=False,deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),torch=torch.__version__,candidate_scope='fixed original B16100 sampled baseline actual cliques; metrics are fixed-candidate classification, not re-enumerated reconstruction',precision='A: original FP32 except BF16 Flash IO; B: FP32 SDPA math, deterministic algorithms enabled including supported graph reductions',epsilon_note='Each repeat explicitly checks equal generated epsilon; same seeds/dtype/shapes for A and B'))
    start=time.monotonic();eps_first=None
    with (out/'trace.jsonl').open('w',buffering=1) as f,torch.no_grad():
        for k in range(50):
            rec.begin(k);rows,stats,zs=c.m.forward(model,batch,'sample',c.SEEDS)
            eps=[x.cpu().clone() for x in model.autoencoder._diagnostic_eps]
            if eps_first is None:eps_first=eps;np.savez(out/'epsilon.npz',**{f'mesh{i}':e.numpy() for i,e in enumerate(eps)})
            else:assert all(torch.equal(a,b) for a,b in zip(eps,eps_first))
            metrics=[];watch=[]
            for i,uid in enumerate(c.UIDS):
                for j,name in enumerate(['mu','logvar','edge_embedding','face_embedding']):rec.add(name+'/'+uid,rows[j][i])
                ep,ey,ft,fy=candidates[i];logits=[]
                for begin in range(0,len(ep),65536):logits.append(c.edge_logits(rows[2][i],torch.as_tensor(ep[begin:begin+65536],device='cuda'),scales))
                el=torch.cat(logits);fl=c.face_logits(rows[3][i],torch.as_tensor(ft,device='cuda'),scales)
                rec.add('edge_logits/'+uid,el);rec.add('face_logits/'+uid,fl)
                metrics.append(dict(uid=uid,edge=c.metrics(ey,el.cpu().numpy()),face_fixed_candidates=c.metrics(fy,fl.cpu().numpy())))
                if i==1:
                    for w in WATCH:
                        j=np.flatnonzero((ft==w).all(1))[0];watch.append(dict(vertices=w,label=int(fy[j]),logit=float(fl[j]),margin=float(fl[j])*(2*int(fy[j])-1)))
            f.write(json.dumps(dict(repeat=k,elapsed=time.monotonic()-start,metrics=metrics,watch=watch,layers=rec.current))+'\n')
            print(backend,k+1,'/50',round(time.monotonic()-start,1),flush=True)
    rec.finish()
    assert all(torch.equal(p.cpu(),cp['model'][n]) for n,p in model.named_parameters())
    write(out/'complete.json',dict(repeats=50,parameters_unchanged=True,no_optimizer=True,seconds=time.monotonic()-start))

if __name__=='__main__':
    lock=(ROOT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    main(sys.argv[1])
