"""Mine round2 at the unchanged epoch800 checkpoint; no optimizer updates."""
import json,time,fcntl
from runtime import ROOT,BASE,T,np,b,setup,write,sha,tensor_hash
from evaluate import evaluate_mesh
from mining_core import mine_faces

PARENT=BASE/'diagnostics/math00_overfit100_low_lr_continue100ep_20260916'
SOURCE=PARENT/'run/checkpoint-update20000.pt'
SOURCE_SHA='38087aeefbc12444cb51ba35dbe3715cc0f785c30cd6ed53e093a0ee7b929f80'

def main():
    lock=(ROOT/'mining.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'mining_complete.json').exists(),'Refuse remine completed fixed pool'
    assert sha(SOURCE)==SOURCE_SHA
    cp=T.load(SOURCE,map_location='cpu',mmap=True,weights_only=False)
    assert cp['epoch']==800 and cp['completed_updates']==20000
    model,opt,groups,uids,base_pools,forward,capture,args=setup()
    model.load_state_dict(cp['model'],strict=True)
    assert uids==cp['manifest']['uids']
    scales=model.scoring_contract();assert scales==cp['manifest']['scales']
    assert sha(ROOT/'parent_mining_complete.json')==cp['manifest']['mining_manifest_sha256']
    pools={}
    for u in uids:
        p=ROOT/'common_pools'/f'{u}_pool.npz'
        assert sha(p)==cp['manifest']['augmented_pool_sha256'][u]
        with np.load(p) as d:pools[u]={k:d[k] for k in ['vertices','edges','positive','mixed']}
        assert all(np.array_equal(pools[u][k],base_pools[u][k]) for k in ['vertices','edges','positive'])
        assert np.array_equal(pools[u]['mixed'][:len(base_pools[u]['mixed'])],base_pools[u]['mixed'])
    before=tensor_hash(model.named_parameters());del cp
    old={r['uid']:r for r in map(json.loads,(PARENT/'run/eval-epoch800.jsonl').read_text().splitlines())}
    assert list(old)==uids
    (ROOT/'augmented_pools').mkdir(exist_ok=True);(ROOT/'mined').mkdir(exist_ok=True)
    records=[]
    with (ROOT/'mining.jsonl').open('x',buffering=1) as f:
        for i,u in enumerate(uids):
            write(ROOT/'mining_status.json',dict(state='mining',completed_meshes=i,uid=u))
            t=time.monotonic();rows=forward(u)
            detached=tuple(tuple(x.detach() for x in rr) for rr in rows);del rows
            _,parts,_=b.full_objective(detached,[pools[u]],scales)
            actual=evaluate_mesh(detached,pools[u],scales)
            assert actual['face']['complete'],u
            for key in ['edge','gt_face_candidates','missing_gt_face_candidates','margins','joint_perfect']:
                assert actual[key]==old[u][key],(u,key)
            # Only outside-pool bookkeeping changes when evaluating with the round1 pool.
            for key,value in actual['face'].items():
                if not key.startswith('actual_fp_outside_training_pool'):assert value==old[u]['face'][key],(u,key)
            assert parts[0]==old[u]['parts'],u
            added,logits,counts=mine_faces(detached,pools[u],scales);del detached
            assert counts['eligible_outside_pool_fp']==actual['face']['actual_fp_outside_training_pool'],u
            assert counts['actual_candidates_scored']==actual['face']['scored_candidates'],u
            merged=dict(pools[u],mixed=np.concatenate([pools[u]['mixed'],added]))
            assert np.array_equal(merged['mixed'][:len(pools[u]['mixed'])],pools[u]['mixed'])
            p=ROOT/'augmented_pools'/f'{u}_pool.npz';np.savez_compressed(p,**merged)
            q=ROOT/'mined'/f'{u}.npz';np.savez_compressed(q,ids=added,logits=logits,labels=np.zeros(len(added),dtype=np.bool_))
            record=dict(uid=u,vertices=len(pools[u]['vertices']),gt_faces=len(pools[u]['positive']),
                old_negative_count=len(pools[u]['mixed']),new_negative_count=len(merged['mixed']),
                old_pool_sha256=sha(ROOT/'common_pools'/f'{u}_pool.npz'),new_pool_sha256=sha(p),
                selected_npz_sha256=sha(q),parent_actual_counts_exact=True,old_pool_parts=parts[0],
                old_pool_and_round1_prefix_preserved=True,**counts,seconds=time.monotonic()-t)
            records.append(record);f.write(json.dumps(record,allow_nan=False)+'\n')
            print('MINED_ROUND2',i+1,u,'added',len(added),'eligible',counts['eligible_outside_pool_fp'],flush=True)
    assert len(records)==100 and tensor_hash(model.named_parameters())==before and not opt.state
    write(ROOT/'mining_complete.json',dict(state='completed',round=2,parent_checkpoint=str(SOURCE),parent_sha256=SOURCE_SHA,
        parent_mining_manifest_sha256=sha(ROOT/'parent_mining_complete.json'),
        optimizer_updates=0,model_unchanged=True,all100_parent_actual_baselines_exact=True,
        selection_rule='actual predicted-edge Face, logit>0, GT negative, outside entire epoch800 round1 pool; descending logit then ascending canonical ID; cap=GT Face count',
        fixed_pool_no_refresh=True,records=records,total_added=sum(x['selected'] for x in records),
        total_eligible=sum(x['eligible_outside_pool_fp'] for x in records)))
    write(ROOT/'mining_status.json',dict(state='completed',completed_meshes=100,total_added=sum(x['selected'] for x in records)))

if __name__=='__main__':main()
