"""Bypass encoder: fit per-vertex 64D codes with frozen or trainable decoder."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

import run as ab
probe=ab.probe
from mini_nexus.flash_varlen_topology import _cu_seqlens, _flash_self_attention, _split_rows
from mini_nexus.training_2k import paper_edge_loss_all_pairs


def decode(a, latents):
    lengths=[len(z) for z in latents]
    h=a.latent_input(torch.cat(list(latents)))
    cu=_cu_seqlens(lengths,h.device)
    for block in a.decoder_blocks:
        h=h+_flash_self_attention(block.attention,block.norm(h),cu,max(lengths))
    h=a.decoder_output_norm(h)
    edge=_split_rows(a.edge_embedding(h),lengths)
    face=_split_rows(a.face_embedding(h),lengths)
    return tuple(x-x.mean(dim=0,keepdim=True) for x in edge),tuple(x-x.mean(dim=0,keepdim=True) for x in face)


@torch.no_grad()
def prepare():
    cp,model,batch=probe.setup_model(ab.CHECKPOINT)
    assert not model.scoring_contract()['normalize_spacetime_embeddings']
    rows=probe.get_rows(model,batch,'mu')
    decoded=decode(model.autoencoder,rows[0])
    errors=[float((a-b).abs().max()) for j in range(2) for a,b in zip(decoded[j],rows[j+2])]
    assert max(errors)==0.,errors
    torch.save(dict(latents=[z.cpu() for z in rows[0]], checkpoint_sha256=probe.digest(ab.CHECKPOINT),
        uids=probe.UIDS, decoder_equivalence_errors=errors),ab.ROOT/'module_start.pt')
    probe.write(ab.ROOT/'module_preparation.json',dict(checkpoint=str(ab.CHECKPOINT),
        checkpoint_sha256=probe.digest(ab.CHECKPOINT),latent_shapes=[list(z.shape) for z in rows[0]],
        decoder_equivalence_errors=errors))
    print(json.dumps(dict(event='prepared',equivalence_errors=errors)),flush=True)


@torch.no_grad()
def evaluate(a,latents,data,sc,variant,step):
    edge,face=decode(a,latents);records=[]
    for i,u in enumerate(probe.UIDS):
        _,metrics,nt=probe.graph(edge[i],data[i]['edges'],sc)
        records.append(dict(uid=u,edge=metrics,candidate_count=nt,
            edge_gate=metrics['precision']>=.99 and metrics['recall']>=.99))
        np.savez_compressed(ab.ROOT/f'{variant}_step{step:04d}_{u}_mu.npz',edge=edge[i].cpu().numpy(),face=face[i].cpu().numpy())
    result=dict(step=step,rows=records,both_meshes_pass=all(r['edge_gate'] for r in records))
    print(json.dumps(dict(event='evaluation',variant=variant,**result)),flush=True)
    return result


def train(variant,steps,resume=False):
    cp,model,batch=probe.setup_model(ab.CHECKPOINT)
    model.eval();model.requires_grad_(False);a=model.autoencoder;sc=model.scoring_contract()
    start=torch.load(ab.ROOT/'module_start.pt',weights_only=False,map_location='cpu')
    assert start['checkpoint_sha256']==probe.digest(ab.CHECKPOINT)
    assert start['uids']==probe.UIDS
    previous=torch.load(ab.ROOT/(variant+'_checkpoint.pt'),weights_only=False,map_location='cpu') if resume else None
    base_step=previous['intervention']['steps'] if previous else 0
    latent_values=previous['latents'] if previous else start['latents']
    latents=torch.nn.ParameterList([torch.nn.Parameter(z.cuda()) for z in latent_values])
    decoder_params=[]
    if variant=='latent_train_decoder':
        for module in [a.latent_input,a.decoder_blocks,a.decoder_output_norm,a.edge_embedding]:
            module.requires_grad_(True);decoder_params.extend(module.parameters())
        if previous:
            result=a.load_state_dict(previous['decoder'],strict=False)
            assert not result.unexpected_keys
    groups=[dict(params=list(latents),lr=.01)]
    if decoder_params:groups.append(dict(params=decoder_params,lr=1e-4))
    optimizer=torch.optim.Adam(groups)
    if previous:optimizer.load_state_dict(previous['optimizer'])
    if previous:
        with torch.no_grad():
            resumed_edge,_=decode(a,latents)
            errors=[]
            for uid,e in zip(probe.UIDS,resumed_edge):
                expected=torch.from_numpy(np.load(ab.ROOT/(variant+'_final_'+uid+'_direct64.npz'))['edge']).cuda()
                errors.append(float((e-expected).abs().max()))
                torch.testing.assert_close(e,expected,rtol=0.,atol=1e-5)
            probe.write(ab.ROOT/(variant+'_resume_equivalence.json'),dict(from_step=base_step,edge_max_abs_errors=errors))
    data=[np.load(ab.PREVIOUS/(u+'_pool.npz')) for u in probe.UIDS]
    edges=[torch.as_tensor(d['edges'].T,device='cuda') for d in data]
    meta=dict(variant=variant,steps=base_step+steps,additional_steps=steps,resumed_from_step=base_step,
        latent_dim=64,encoder_bypassed=True,decoder_trainable=bool(decoder_params),
        latent_lr=.01,decoder_lr=1e-4 if decoder_params else None,optimizer='Adam, zero weight decay; fresh at step0, preserved on continuation',
        objective='original four-group Edge loss only',kl_weight=0.,sample_posterior=False,
        source_checkpoint=str(ab.CHECKPOINT),checkpoint_sha256=start['checkpoint_sha256'],
        module_start_sha256=probe.digest(ab.ROOT/'module_start.pt'),run_source_sha256=probe.digest(Path(__file__)),
        limitation='Independent per-mesh codes bypass encoder and need not follow original posterior distribution.')
    probe.write(ab.ROOT/(variant+'_provenance.json'),meta)
    print(json.dumps(dict(event='start',**meta)),flush=True)
    trace=json.loads((ab.ROOT/(variant+'_trace.json')).read_text()) if previous else []
    evaluations=json.loads((ab.ROOT/(variant+'_evaluations.json')).read_text()) if previous else []
    elapsed_offset=trace[-1]['seconds'] if trace else 0.
    started=time.monotonic()
    if not previous:evaluations.append(evaluate(a,latents,data,sc,variant,0))
    for step in range(base_step+1,base_step+steps+1):
        optimizer.zero_grad(set_to_none=True)
        out,_=decode(a,latents)
        parts=[paper_edge_loss_all_pairs(z,e,pair_chunk_size=cp['args']['pair_chunk_size'],counts_on_device=True,
            logit_scale=sc['edge_logit_scale']) for z,e in zip(out,edges)]
        loss=torch.stack([p[0] for p in parts]).mean();loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(list(latents)+decoder_params,1.,error_if_nonfinite=True)
        optimizer.step()
        if step==base_step+1 or step%10==0:
            trace.append(dict(step=step,loss=float(loss.detach()),gradient_norm=float(grad),seconds=elapsed_offset+time.monotonic()-started,
                counts=[{k:int(v) for k,v in p[1].items()} for p in parts]))
            probe.write(ab.ROOT/(variant+'_trace.json'),trace)
            print(json.dumps(dict(event='train',variant=variant,**trace[-1])),flush=True)
        if step%50==0 or step==base_step+steps:
            evaluations.append(evaluate(a,latents,data,sc,variant,step))
            probe.write(ab.ROOT/(variant+'_evaluations.json'),evaluations)
    with torch.no_grad():
        edge,face=decode(a,latents)
        probe.evaluate(ab.ROOT,variant+'_final',{'direct64':(None,None,edge,face)},sc)
    saved=dict(latents=[z.detach().cpu() for z in latents],optimizer=optimizer.state_dict(),intervention=meta)
    if decoder_params:
        saved['decoder']={k:v for k,v in a.state_dict().items() if k.startswith(('latent_input.','decoder_blocks.','decoder_output_norm.','edge_embedding.'))}
    torch.save(saved,ab.ROOT/(variant+'_checkpoint.pt'))
    probe.write(ab.ROOT/(variant+'_complete.json'),dict(steps=base_step+steps,seconds=elapsed_offset+time.monotonic()-started,
        passed_steps=[r['step'] for r in evaluations if r['both_meshes_pass']],
        final_two_evaluations_pass=all(r['both_meshes_pass'] for r in evaluations[-2:]),intervention=meta))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train'])
    p.add_argument('--variant',choices=['latent_frozen_decoder','latent_train_decoder']);p.add_argument('--steps',type=int,default=500)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();torch.set_num_threads(1);torch.manual_seed(20260907);torch.cuda.set_device(0)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.action=='prepare':prepare()
    else:train(args.variant,args.steps,args.resume)
