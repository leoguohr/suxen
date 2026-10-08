"""Freeze original epoch900 features and verify exact original-head readout on all 100."""
import json,shutil,time
from pathlib import Path
from runtime import ROOT,T,np,b,c,setup,sha,write,tensor_hash

cfg=json.loads((ROOT/'config.json').read_text());source=Path(cfg['source_checkpoint'])
assert sha(source)==cfg['source_sha256']
cache=ROOT/'cache';cache.mkdir(exist_ok=False)
shutil.copytree(cfg['scoring_reference'],ROOT/'effective_code')
model,opt,groups,uids,pools,forward,capture,args=setup();del opt,groups
T.use_deterministic_algorithms(True)
cp=T.load(source,map_location='cpu',mmap=True,weights_only=False)
assert cp['epoch']==900 and cp['completed_updates']==22500 and uids==cp['manifest']['uids']
model.load_state_dict(cp['model'],strict=True);del cp
state0=tensor_hash((k,v) for k,v in model.state_dict().items() if T.is_tensor(v))
head=model.autoencoder.edge_embedding;scales=model.scoring_contract()
np.savez(cache/'head_original.npz',**{k:v.detach().cpu().numpy() for k,v in head.state_dict().items()})
archived={r['uid']:r for r in map(json.loads,(ROOT/'source_archived_eval.jsonl').read_text().splitlines())}
grab={};hook=head.register_forward_hook(lambda m,x,y:grab.update(hidden=x[0].detach(),raw=y.detach()))
records=[];start=time.monotonic();g=c.m.h.soft4_loss.__globals__;old=g['soft4_sums'];g['soft4_sums']=b.full_sums
try:
    for i,uid in enumerate(uids):
        rows=forward(uid);h=grab['hidden'];raw=grab['raw'];center=rows[2][0];n=len(h)
        assert h.shape==(n,1024) and h.dtype==T.float32 and T.equal(h,capture['hidden'])
        edges=np.unique(np.sort(pools[uid]['edges'].T,axis=1),axis=0);keys=T.as_tensor(edges@np.array([n,1]),device='cuda')
        loss,_=c.m.h.soft4_loss(center,keys,args['pair_chunk_size'],scales['edge_logit_scale'])
        original_gradient=T.autograd.grad(loss/100,tuple(head.parameters()))
        independent=head(h);z=independent-independent.mean(0,keepdim=True)
        loss2,_=c.m.h.soft4_loss(z,keys,args['pair_chunk_size'],scales['edge_logit_scale'])
        grad2=T.autograd.grad(loss2/100,tuple(head.parameters()))
        assert T.equal(raw,independent) and T.equal(center,z) and T.equal(loss,loss2)
        assert all(T.equal(a,bv) for a,bv in zip(original_gradient,grad2))
        assert float(loss)==archived[uid]['parts']['edge']
        pair=T.triu_indices(n,n,1,device='cuda').T
        assert len(pair)<=args['pair_chunk_size']
        with T.no_grad():
            s=c.edge_logits(center,pair,scales);q=pair[:,0]*n+pair[:,1]
            at=T.searchsorted(keys,q);y=(at<len(keys))&(keys[at.clamp_max(len(keys)-1)]==q);p=s>0
            counts=dict(tp=int((p&y).sum()),fp=int((p&~y).sum()),fn=int((~p&y).sum()),tn=int((~p&~y).sum()))
        assert all(counts[k]==archived[uid]['edge'][k] for k in counts)
        path=cache/f'{uid}.npz'
        np.savez(path,hidden=h.cpu().numpy(),vertices=pools[uid]['vertices'],edges=edges,faces=pools[uid]['positive'],
                 original_edge_raw=raw.cpu().numpy(),original_edge_centered=center.detach().cpu().numpy(),
                 original_weight_gradient=original_gradient[0].cpu().numpy(),original_bias_gradient=original_gradient[1].cpu().numpy())
        record=dict(uid=uid,vertices=n,pairs=len(pair),gt_edges=len(edges),gt_faces=len(pools[uid]['positive']),
            sha256=sha(path),hidden_bytes=h.numel()*h.element_size(),edge_loss=float(loss),counts=counts,
            raw_center_loss_gradient_bitwise=True,logits_sha256=tensor_hash([('logits',s)]))
        records.append(record);write(ROOT/'status.json',dict(state='exporting',completed=i+1,uid=uid,optimizer_updates=0))
        print('EXPORT',i+1,uid,counts,flush=True)
        del rows,h,raw,center,loss,independent,z,loss2,original_gradient,grad2,pair,s,q,y,p,at
        capture.clear();grab.clear()
finally:g['soft4_sums']=old;hook.remove()
assert state0==tensor_hash((k,v) for k,v in model.state_dict().items() if T.is_tensor(v))
assert sha(source)==cfg['source_sha256']
manifest=dict(meshes=records,source_sha256=cfg['source_sha256'],scales=scales,edge_chunk=args['pair_chunk_size'],
    hidden_total_bytes=sum(r['hidden_bytes'] for r in records),total_vertices=sum(r['vertices'] for r in records),
    total_pairs=sum(r['pairs'] for r in records),head_sha256=sha(cache/'head_original.npz'),
    source_unchanged=True,optimizer_updates=0,seconds=time.monotonic()-start)
write(cache/'manifest.json',manifest);write(ROOT/'export_complete.json',manifest)
print('EXPORTED_ALL100',manifest['hidden_total_bytes'],manifest['total_pairs'],flush=True)
