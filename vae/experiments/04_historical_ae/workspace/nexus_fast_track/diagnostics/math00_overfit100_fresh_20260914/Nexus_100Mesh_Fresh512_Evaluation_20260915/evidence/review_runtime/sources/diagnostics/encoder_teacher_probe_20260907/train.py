"""Train encoder mu against verified 64D codes, with the paired decoder frozen."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
MODULE_ROOT=ROOT.parent/'edge_only_ab_20260907'
sys.path.insert(0,str(MODULE_ROOT))
import module_probe as modules
probe=modules.probe
from mini_nexus.flash_varlen_topology import _cu_seqlens, _encoder_transformer_varlen, _split_rows
from mini_nexus.training_2k import paper_edge_loss_all_pairs, _all_pair_chunks, _canonical_positive_edge_keys

TEACHER=MODULE_ROOT/'latent_train_decoder_checkpoint.pt'
START=modules.ab.CHECKPOINT


def inputs(batch,data):
    counts=[len(d['vertices']) for d in data]
    vertices=[batch.vertices[i,:n] for i,n in enumerate(counts)]
    faces=list(batch.faces)
    lengths=[len(v)+len(f) for v,f in zip(vertices,faces)]
    indices=[];source=[];target=[];offset=0
    for n,length,incidence in zip(counts,lengths,batch.incidence_index):
        indices.append(torch.arange(n,device='cuda')+offset)
        source.append(incidence[0]+offset);target.append(incidence[1]+offset);offset+=length
    return dict(vertices=vertices,centroids=[v[f].mean(dim=1) for v,f in zip(vertices,faces)],
        counts=counts,source=torch.cat(source),target=torch.cat(target),indices=torch.cat(indices),
        cu=_cu_seqlens(lengths,'cuda'),maximum=max(lengths))


def encode(a,x):
    h=torch.cat([torch.cat([a.vertex_input(v),a.face_input(c)]) for v,c in zip(x['vertices'],x['centroids'])])
    for block in a.encoder_blocks:
        h=h+block.graph_activation(block.graph(block.graph_norm(h),x['source'],x['target']))
        h=_encoder_transformer_varlen(block.transformer,h,x['cu'],x['maximum'])
    return _split_rows(a.mu(a.encoder_output_norm(h[x['indices']])),x['counts'])


@torch.no_grad()
def evaluate(a,x,targets,data,sc,label,step):
    codes=encode(a,x);edge,face=modules.decode(a,codes);rows=[]
    for i,u in enumerate(probe.UIDS):
        var=(targets[i]-targets[i].mean(0,keepdim=True)).square().mean()
        mse=(codes[i]-targets[i]).square().mean()
        _,stats,candidate_count=probe.graph(edge[i],data[i]['edges'],sc)
        rows.append(dict(uid=u,code_nmse=float(mse/var),code_std=float(codes[i].std(0).square().mean().sqrt()),
            target_std=float(targets[i].std(0).square().mean().sqrt()),edge=stats,candidate_count=candidate_count,
            edge_gate=stats['precision']>=.99 and stats['recall']>=.99))
        np.savez_compressed(ROOT/f'{label}_step{step:04d}_{u}.npz',mu=codes[i].cpu().numpy(),edge=edge[i].cpu().numpy(),face=face[i].cpu().numpy())
    result=dict(step=step,rows=rows,both_meshes_pass=all(r['edge_gate'] for r in rows))
    print(json.dumps(dict(event='evaluation',label=label,**result)),flush=True)
    return result


def setup():
    cp,model,batch=probe.setup_model(START);a=model.autoencoder
    model.eval();model.requires_grad_(False)
    teacher=torch.load(TEACHER,map_location='cpu',mmap=True,weights_only=False)
    assert teacher['intervention']['steps']==1000
    assert teacher['intervention']['variant']=='latent_train_decoder'
    result=a.load_state_dict(teacher['decoder'],strict=False)
    assert not result.unexpected_keys
    for part in [a.vertex_input,a.face_input,a.encoder_blocks,a.encoder_output_norm,a.mu]:
        part.requires_grad_(True)
    targets=[z.cuda() for z in teacher['latents']]
    data=[np.load(modules.ab.PREVIOUS/(u+'_pool.npz')) for u in probe.UIDS]
    x=inputs(batch,data);sc=model.scoring_contract()
    return cp,model,batch,targets,data,x,sc


@torch.no_grad()
def prepare():
    cp,model,batch,targets,data,x,sc=setup()
    mu=encode(model.autoencoder,x);reference=probe.get_rows(model,batch,'mu')[0]
    differences=[float((z-r).abs().max()) for z,r in zip(mu,reference)]
    assert max(differences)<.01,differences
    edge,_=modules.decode(model.autoencoder,targets)
    scores=[];errors=[]
    for i,u in enumerate(probe.UIDS):
        expected=torch.from_numpy(np.load(MODULE_ROOT/('latent_train_decoder_final_'+u+'_direct64.npz'))['edge']).cuda()
        errors.append(float((edge[i]-expected).abs().max()))
        torch.testing.assert_close(edge[i],expected,rtol=0.,atol=1e-5)
        _,e,_=probe.graph(edge[i],data[i]['edges'],sc);scores.append(dict(uid=u,edge=e))
        assert e['precision']>=.99 and e['recall']>=.99
        np.savez_compressed(ROOT/(u+'_teacher.npz'),mu=targets[i].cpu().numpy(),edge=edge[i].cpu().numpy())
    probe.write(ROOT/'preparation.json',dict(start_checkpoint=str(START),start_sha256=probe.digest(START),
        teacher_checkpoint=str(TEACHER),teacher_sha256=probe.digest(TEACHER),teacher_shapes=[list(z.shape) for z in targets],
        encoder_path_max_abs_errors=differences,teacher_decoder_max_abs_errors=errors,teacher_metrics=scores,
        source=str(probe.SOURCE),source_hashes={str(p):probe.digest(p) for p in [probe.SOURCE/'mini_nexus/topology.py',
        probe.SOURCE/'mini_nexus/flash_varlen_topology.py',probe.SOURCE/'mini_nexus/training_2k.py']},
        interpretation='NMSE denominator is per-mesh across-vertex target variance; meshwise constant target mean has NMSE 1.'))
    print(json.dumps(dict(event='prepared',encoder_errors=differences,teacher_errors=errors)),flush=True)


def hard_edge_loss(z,edges,chunk_size,scale):
    keys=_canonical_positive_edge_keys(edges,len(z),z.device)
    positives=[];negatives=[]
    for pairs in _all_pair_chunks(len(z),z.device,chunk_size):
        pair_keys=pairs[:,0]*len(z)+pairs[:,1]
        at=torch.searchsorted(keys,pair_keys)
        truth=(at<len(keys)) & (keys[at.clamp_max(len(keys)-1)]==pair_keys)
        logits=probe.first_order_interval(z[pairs[:,0]],z[pairs[:,1]])*scale
        positives.append(logits[truth]);negatives.append(logits[~truth])
    positives=torch.cat(positives);negatives=torch.cat(negatives)
    hard=negatives.topk(min(len(positives),len(negatives))).values
    return .5*(torch.nn.functional.softplus(-positives).mean()+torch.nn.functional.softplus(hard).mean())


def train(label,lr,steps,resume_from=None,objective='mse'):
    cp,model,batch,targets,data,x,sc=setup();a=model.autoencoder
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(params,lr=lr)
    if resume_from:
        previous=torch.load(resume_from,map_location='cpu',mmap=True,weights_only=False)
        result=a.load_state_dict(previous['encoder'],strict=False)
        assert not result.unexpected_keys
        optimizer.load_state_dict(previous['optimizer'])
        for group in optimizer.param_groups:group['lr']=lr
    variances=[(z-z.mean(0,keepdim=True)).square().mean().detach() for z in targets]
    meta=dict(label=label,lr=lr,steps=steps,objective=objective,
        objective_description={'mse':'normalized code MSE','edge':'original four-group all-pair Edge loss',
            'hard_edge':'0.5 mean positive BCE + 0.5 mean top-K negative BCE; K=#GT edges, all pairs scored'}[objective],
        resume_from=str(resume_from) if resume_from else None,resume_sha256=probe.digest(resume_from) if resume_from else None,
        encoder_trainable=True,decoder_frozen=True,latent_dim=64,kl_weight=0.,sample_posterior=False,
        optimizer='Adam, clip1, no weight decay; restored when resuming',trainable_parameters=sum(p.numel() for p in params),
        teacher_sha256=probe.digest(TEACHER),start_sha256=probe.digest(START),run_sha256=probe.digest(Path(__file__)))
    probe.write(ROOT/(label+'_provenance.json'),meta)
    evaluations=[evaluate(a,x,targets,data,sc,label,0)];trace=[];started=time.monotonic()
    for step in range(1,steps+1):
        optimizer.zero_grad(set_to_none=True)
        codes=encode(a,x)
        parts=[(z-t).square().mean()/v for z,t,v in zip(codes,targets,variances)]
        if objective in ['edge','hard_edge']:
            edge,_=modules.decode(a,codes)
            if objective=='edge':
                edge_parts=[paper_edge_loss_all_pairs(z,torch.as_tensor(d['edges'].T,device='cuda'),
                    pair_chunk_size=cp['args']['pair_chunk_size'],counts_on_device=True,logit_scale=sc['edge_logit_scale'])[0]
                    for z,d in zip(edge,data)]
            else:
                edge_parts=[hard_edge_loss(z,torch.as_tensor(d['edges'].T,device='cuda'),
                    cp['args']['pair_chunk_size'],sc['edge_logit_scale']) for z,d in zip(edge,data)]
            loss=torch.stack(edge_parts).mean()
        else:
            loss=torch.stack(parts).mean()
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
        if step==1 or step%25==0:
            trace.append(dict(step=step,loss=float(loss.detach()),parts=[float(p.detach()) for p in parts],
                gradient_norm=float(norm),seconds=time.monotonic()-started))
            print(json.dumps(dict(event='train',label=label,**trace[-1])),flush=True)
            probe.write(ROOT/(label+'_trace.json'),trace)
        if step%100==0 or step==steps:
            evaluations.append(evaluate(a,x,targets,data,sc,label,step));probe.write(ROOT/(label+'_evaluations.json'),evaluations)
    names={name for name,p in a.named_parameters() if p.requires_grad}
    torch.save(dict(encoder={name:v for name,v in a.state_dict().items() if name in names},
        optimizer=optimizer.state_dict(),intervention=meta),ROOT/(label+'_checkpoint.pt'))
    probe.write(ROOT/(label+'_complete.json'),dict(steps=steps,seconds=time.monotonic()-started,
        passed_steps=[e['step'] for e in evaluations if e['both_meshes_pass']],
        final_two_evaluations_pass=all(e['both_meshes_pass'] for e in evaluations[-2:]),intervention=meta))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train']);p.add_argument('--label')
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--resume-from',type=Path);p.add_argument('--objective',choices=['mse','edge','hard_edge'],default='mse')
    args=p.parse_args();torch.set_num_threads(1);torch.manual_seed(20260907);torch.cuda.set_device(0)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.action=='prepare':prepare()
    else:train(args.label,args.lr,args.steps,args.resume_from,args.objective)
