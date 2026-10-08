"""Read-only export from the round2 epoch900 network; no optimizer update."""
import hashlib
import inspect
import json
from pathlib import Path
from runtime import ROOT, T, np, b, c, setup, sha, write, tensor_hash

PARENT=ROOT.parent/'math00_overfit100_face_hardneg_round2_20260917'
SOURCE=PARENT/'run/checkpoint-update22500.pt'
SOURCE_SHA='428aeddbc6ea03ae166ba22aa431fe40f4298ca83e0e5008302a3c5eed3ceb79'
SELECTION=[('nexus_2k_000446',1519,1337,1),('nexus_2k_001093',1964,7063,0),('nexus_2k_000898',2547,13303,0)]

def main():
    assert not (ROOT/'snapshots').exists(), 'Preserve existing exports'
    assert sha(SOURCE)==SOURCE_SHA
    model,opt,groups,uids,pools,forward,capture,args=setup()
    cp=T.load(SOURCE,map_location='cpu',mmap=True,weights_only=False)
    model.load_state_dict(cp['model'],strict=True)
    del cp,opt
    original_hash=tensor_hash((k,v) for k,v in model.state_dict().items() if T.is_tensor(v))
    original_metadata=json.dumps({k:v for k,v in model.state_dict().items() if not T.is_tensor(v)},sort_keys=True)
    scales=model.scoring_contract()
    assert scales==json.loads((PARENT/'run/manifest.json').read_text())['scales']
    baseline={r['uid']:r for r in map(json.loads,(PARENT/'run/eval-epoch900.jsonl').read_text().splitlines())}
    h=c.m.h
    # Capture the currently installed effective functions, including the full-diff patch.
    interval=c.m.probe.first_order_interval
    loss_globals=h.soft4_loss.__globals__
    functions=[interval.__globals__['_split_spacetime'],interval,
               loss_globals['teacher']._all_pair_chunks]
    source='import math\nimport torch\nfrom torch import Tensor\nfrom torch.nn import functional as F\nfrom types import SimpleNamespace\nEPS=1e-8\nGROUPS=("tp","tn","fp","fn")\n\n'
    source+='\n\n'.join(inspect.getsource(f) for f in functions)
    source+='\n\n'+b.src+'\n\nteacher=SimpleNamespace(_all_pair_chunks=_all_pair_chunks)\nprobe=SimpleNamespace(first_order_interval=first_order_interval)\n\n'
    source+=inspect.getsource(h.soft4_loss)
    assert '.detach()' not in b.src and 's.sigmoid()' in b.src
    effective={}
    exec(compile(source,'effective_loss_and_scoring.py','exec'),effective)
    old_sums=loss_globals['soft4_sums']
    records=[]
    raw_capture={}
    handle=model.autoencoder.edge_embedding.register_forward_hook(lambda m,i,o:raw_capture.update(raw=o))
    try:
        for uid,n,fp,fn in SELECTION:
            assert uid in uids and len(pools[uid]['vertices'])==n
            assert baseline[uid]['edge']['fp']==fp and baseline[uid]['edge']['fn']==fn
            snap=ROOT/'snapshots'/uid; (snap/'effective_code').mkdir(parents=True)
            (snap/'effective_code/effective_loss_and_scoring.py').write_text(source)
            rows=forward(uid);raw=raw_capture['raw'];centered=rows[2][0]
            assert raw.shape==(n,32) and T.equal(raw-raw.mean(0,keepdim=True),centered)
            pair=T.triu_indices(n,n,1,device='cuda').T
            edges=np.asarray(pools[uid]['edges']).T
            edgekeys=np.unique(np.sort(edges,axis=1)@np.array([n,1]))
            faces=np.asarray(pools[uid]['positive'])
            derived=np.concatenate([faces[:,[0,1]],faces[:,[0,2]],faces[:,[1,2]]])
            derivedkeys=np.unique(np.sort(derived,axis=1)@np.array([n,1]))
            assert np.array_equal(edgekeys,derivedkeys), 'Pool edges differ from GT face edges'
            keys=T.as_tensor(edgekeys,device='cuda',dtype=T.long)
            ids=pair[:,0]*n+pair[:,1];at=T.searchsorted(keys,ids)
            y=(at<len(keys))&(keys[at.clamp_max(len(keys)-1)]==ids)
            logits=c.edge_logits(centered,pair,scales)
            loss_globals['soft4_sums']=b.full_sums
            loss,stats=h.soft4_loss(centered,keys,c.PAIR_CHUNK,scales['edge_logit_scale'])
            grad=T.autograd.grad(loss/4,raw)[0]
            loss_globals['soft4_sums']=old_sums
            pred=logits.detach()>0
            counts={k:int(mask.sum()) for k,mask in dict(tp=pred&y,fp=pred&~y,fn=~pred&y,tn=~pred&~y).items()}
            assert all(counts[k]==baseline[uid]['edge'][k] for k in counts)
            assert float(loss)==baseline[uid]['parts']['edge']
            # Independently verify the exported standalone functions against runtime.
            independent=raw.detach().clone().requires_grad_()
            z=independent-independent.mean(0,keepdim=True)
            lref,_=effective['soft4_loss'](z,keys,c.PAIR_CHUNK,scales['edge_logit_scale'])
            gref=T.autograd.grad(lref/4,independent)[0]
            assert T.equal(loss,lref) and T.equal(grad,gref)
            with T.no_grad():
                lref_logits=effective['first_order_interval'](z[pair[:,0]],z[pair[:,1]])*scales['edge_logit_scale']
            assert T.equal(logits,lref_logits)
            np.savez_compressed(snap/'representations_and_gradients.npz',edge_head_raw=raw.detach().cpu().numpy(),
                edge_embedding_scoring=centered.detach().cpu().numpy(),grad_edge_head_raw=grad.cpu().numpy(),
                vertices=pools[uid]['vertices'],gt_faces=faces,gt_edges=np.sort(edges,axis=1),local_vertex_id=np.arange(n))
            np.savez_compressed(snap/'edge_all_pairs.npz',pairs=pair.cpu().numpy(),labels=y.cpu().numpy(),
                logits=logits.detach().cpu().numpy(),in_training=np.ones(len(pair),dtype=bool))
            summary=dict(uid=uid,vertices=n,gt_edges=len(keys),gt_faces=len(faces),pairs=len(pair),checkpoint=str(SOURCE),
                checkpoint_sha256=SOURCE_SHA,parts=dict(edge=float(loss)),actual_reconstruction=dict(edge=counts),
                model_updates=0,standalone_loss_logits_gradient_bitwise_match=True,gt_edges_match_faces=True)
            write(snap/'summary.json',summary)
            write(snap/'source_evaluation.json',baseline[uid])
            write(snap/'effective_code/runtime_contract.json',dict(scales=scales,edge_chunk=c.PAIR_CHUNK,
                membership_detach=False,soft4_epsilon=1e-8,tau=1,threshold=0,space_time_dims=[16,16],
                objective_outer_coefficient=.25,reduction_dtype='float32',backend_export='math00 deterministic Graph',
                function_sources='Extracted current runtime, including backend00 full-differentiable patch',
                stale_source_comment='soft4_sums inherited comment says no membership gradient; actual captured source has NO detach'))
            records.append(summary); print('EXPORTED',json.dumps(summary),flush=True)
            del rows,raw,centered,pair,keys,y,logits,loss,grad,independent,z,lref,gref,lref_logits
            raw_capture.clear();capture.clear();T.cuda.empty_cache()
    finally:
        handle.remove();loss_globals['soft4_sums']=old_sums
    assert tensor_hash((k,v) for k,v in model.state_dict().items() if T.is_tensor(v))==original_hash
    assert json.dumps({k:v for k,v in model.state_dict().items() if not T.is_tensor(v)},sort_keys=True)==original_metadata
    assert sha(SOURCE)==SOURCE_SHA
    write(ROOT/'export_complete.json',dict(records=records,model_unchanged=True,checkpoint_unchanged=True,
          model_state_sha256=original_hash,optimizer_updates=0))

if __name__=='__main__':main()
