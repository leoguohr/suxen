"""Read-only full-network reintegration of one shared head; no backward or optimizer step."""
import fcntl,json,time,traceback
from pathlib import Path
from runtime import ROOT,T,np,b,c,setup,write,sha,tensor_hash
import evaluate

def detach(rows):
    return tuple(tuple(x.detach() for x in group) for group in rows)

def main():
    lock=(ROOT/'execution.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=ROOT/'run';out.mkdir(exist_ok=False)
    cfg=json.loads((ROOT/'config.json').read_text());source=Path(cfg['source_checkpoint']);hp=Path(cfg['head_checkpoint'])
    write(out/'status.json',dict(state='verifying_sources',optimizer_updates=0))
    assert sha(source)==cfg['source_sha256'] and sha(hp)==cfg['head_sha256']
    model,opt,groups,uids,base_pools,forward,capture,args=setup()
    del opt,groups  # The archived construction helper creates an unused optimizer; never loaded or stepped.
    T.use_deterministic_algorithms(True)
    cp=T.load(source,map_location='cpu',mmap=True,weights_only=False)
    assert cp['completed_updates']==22500 and cp['epoch']==900 and cp['manifest']['uids']==uids
    assert len(uids)==len(set(uids))==100
    model.load_state_dict(cp['model'],strict=True)
    head=model.autoencoder.edge_embedding
    original={k:v.detach().clone() for k,v in head.state_dict().items()}
    head_cp=T.load(hp,map_location='cpu',weights_only=False)
    assert head_cp['step']==2000 and set(head_cp['head'])=={'weight','bias'}
    new={k:v.cuda() for k,v in head_cp['head'].items()}
    assert new['weight'].shape==(32,1024) and new['bias'].shape==(32,)
    prefix='autoencoder.edge_embedding.'
    changed=[k for k,v in model.state_dict().items() if k.startswith(prefix)]
    assert sorted(changed)==[prefix+'bias',prefix+'weight']
    def non_head_hash():
        state=model.state_dict()
        return dict(tensors=tensor_hash((k,v) for k,v in state.items() if T.is_tensor(v) and not k.startswith(prefix)),
                    metadata=json.dumps({k:v for k,v in state.items() if not T.is_tensor(v)},sort_keys=True))
    source_other=non_head_hash();scales=model.scoring_contract()
    assert scales==cp['manifest']['scales']
    pools={};poolhash={}
    for uid in uids:
        p=ROOT/'augmented_pools'/f'{uid}_pool.npz';poolhash[uid]=sha(p)
        assert poolhash[uid]==cp['manifest']['augmented_pool_sha256'][uid]
        with np.load(p) as d:pools[uid]={k:d[k] for k in ['vertices','edges','positive','mixed']}
    # Save a separate inference-only network copy. Historical Adam would not match the new head.
    head.load_state_dict(new,strict=True)
    assert non_head_hash()==source_other
    installed=out/'model-source22500-head2000-inference.pt'
    T.save(dict(model={k:v.detach().cpu() if T.is_tensor(v) else v for k,v in model.state_dict().items()},args=cp['args'],
                inference_only=True,source_checkpoint=str(source),source_sha256=cfg['source_sha256'],
                head_checkpoint=str(hp),head_sha256=cfg['head_sha256'],optimizer_updates=0),installed)
    reloaded=T.load(installed,map_location='cpu',mmap=True,weights_only=False)
    diff=[]
    for key,value in cp['model'].items():
        other=reloaded['model'][key]
        equal=T.equal(value,other) if T.is_tensor(value) else value==other
        if not equal:diff.append(key)
    assert sorted(diff)==sorted(changed)
    model.load_state_dict(reloaded['model'],strict=True)
    assert non_head_hash()==source_other
    del reloaded,cp,head_cp
    write(out/'installation_verification.json',dict(source_sha256=cfg['source_sha256'],head_sha256=cfg['head_sha256'],
        installed_checkpoint=str(installed),installed_sha256=sha(installed),changed_state_keys=diff,
        other_parameters_buffers_metadata_bitwise_unchanged=True,optimizer_updates=0,backward_calls=0,
        reload_verified=True,inference_only_no_optimizer_state=True))
    np.savez(out/'shared_head_before_after.npz',**{'before_'+k:v.cpu().numpy() for k,v in original.items()},**{'after_'+k:v.cpu().numpy() for k,v in new.items()})
    archived={r['uid']:r for r in map(json.loads,(ROOT/'source_archived_eval.jsonl').read_text().splitlines())}
    evaluate.FACE_SECONDS=float('inf')  # Same enumeration/scoring; finish all candidates instead of a timed lower bound.
    results={'before':{},'after':{}};feature_hashes={};checks={};three_checks=[]
    saved_raw={}
    hook=head.register_forward_hook(lambda module,inputs,output:saved_raw.update(raw=output.detach()))
    def run_forward(uid):
        # Preserve the source's grad-mode kernel dispatch, then drop the graph. Never backward.
        rows=detach(forward(uid));hidden=capture['hidden'].detach();raw=saved_raw['raw']
        capture.clear();saved_raw.clear()
        return rows,hidden,raw
    def features(rows,hidden):
        return {name:tensor_hash([(name,t)]) for name,t in [('mu',rows[0][0]),('logvar',rows[1][0]),('hidden',hidden),('face_embedding',rows[3][0])]}
    def evaluate_one(uid,branch,rows,hidden):
        start=time.monotonic()
        with T.no_grad():
            _,parts,saved=b.full_objective(rows,[pools[uid]],scales)
            metrics=evaluate.evaluate_mesh(rows,base_pools[uid],scales)
        assert metrics['face']['complete']
        y=np.r_[np.ones(len(pools[uid]['positive']),dtype=bool),np.zeros(len(pools[uid]['mixed']),dtype=bool)]
        row=dict(uid=uid,branch=branch,vertices=len(pools[uid]['vertices']),gt_edges=len(pools[uid]['edges'].T),
                 gt_faces=len(pools[uid]['positive']),parts=parts[0],face_training_pool=c.metrics(y,saved[0]['face_train_logits']),
                 face_train_logits_sha256=tensor_hash([('face',T.from_numpy(saved[0]['face_train_logits']))]),
                 feature_hashes=features(rows,hidden),actual_fp_outside_pool_reference='original_base_pool',**metrics)
        row['evaluation_seconds']=time.monotonic()-start
        if branch=='before':
            old=archived[uid]
            for key in ['edge','face_training_pool','parts','gt_face_candidates','missing_gt_face_candidates','margins','edge_perfect','face_perfect','joint_perfect']:
                assert row[key]==old[key],(uid,key,'baseline mismatch')
            for key in ['tp','fp','fn','tn','complete','scored_candidates','actual_fp_outside_training_pool']:
                assert row['face'][key]==old['face'][key],(uid,'face',key)
            feature_hashes[uid]=row['feature_hashes']
        else:
            assert row['feature_hashes']==feature_hashes[uid],uid
            assert row['face_train_logits_sha256']==results['before'][uid]['face_train_logits_sha256'],uid
            assert row['parts']['face']==results['before'][uid]['parts']['face']
        results[branch][uid]=row
        write(out/f'{branch}-{uid}.json',row)
        with (out/f'{branch}.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        write(out/'status.json',dict(state='evaluating',branch=branch,completed=len(results[branch]),uid=uid,
             edge_fp_fn=[row['edge']['fp'],row['edge']['fn']],face_fp_fn=[row['face']['fp'],row['face']['fn']],optimizer_updates=0))
        print('EVAL',branch,len(results[branch]),uid,'edge',row['edge']['fp'],row['edge']['fn'],'face',row['face']['fp'],row['face']['fn'],flush=True)
        return row
    # Gate: actual Encoder -> mu -> Decoder, both original and installed heads, on the locked three.
    for uid in cfg['three_uids']:
        head.load_state_dict(original,strict=True);r0,h0,raw0=run_forward(uid)
        evaluate_one(uid,'before',r0,h0)
        head.load_state_dict(new,strict=True);r1,h1,raw1=run_forward(uid)
        record=dict(uid=uid,mu_bitwise=T.equal(r0[0][0],r1[0][0]),hidden_bitwise=T.equal(h0,h1),
                    face_embedding_bitwise=T.equal(r0[3][0],r1[3][0]),logvar_bitwise=T.equal(r0[1][0],r1[1][0]))
        snap=hp.parent.parent/'snapshots'/uid
        assert np.array_equal(h1.cpu().numpy(),np.load(snap/'hidden.npy'))
        expected=np.load(hp.parent/f'checkpoint-step2000-{uid}.npz')
        record['raw_matches_probe']=np.array_equal(raw1.cpu().numpy(),expected['edge_head_raw'])
        record['center_matches_probe']=np.array_equal(r1[2][0].cpu().numpy(),expected['edge_embedding_scoring'])
        pairs=np.load(snap/'edge_all_pairs.npz')['pairs']
        with T.no_grad():logits=c.edge_logits(r1[2][0],T.as_tensor(pairs,device='cuda'),scales).cpu().numpy()
        record['all_logits_match_probe']=np.array_equal(logits,expected['logits'])
        r2,h2,raw2=run_forward(uid)
        record['repeated_forward_bitwise']=all(T.equal(a[0],bb[0]) for a,bb in zip(r1,r2)) and T.equal(h1,h2) and T.equal(raw1,raw2)
        assert all(v for k,v in record.items() if k!='uid'),record
        row=evaluate_one(uid,'after',r1,h1);assert row['edge_perfect']
        record['actual_edge_perfect']=True
        np.savez_compressed(out/f'three-features-{uid}.npz',mu=r1[0][0].cpu().numpy(),hidden=h1.cpu().numpy(),face_embedding=r1[3][0].cpu().numpy(),
            original_edge_embedding=r0[2][0].cpu().numpy(),installed_edge_embedding=r1[2][0].cpu().numpy())
        three_checks.append(record);write(out/'three_verification.json',dict(meshes=three_checks,passed=len(three_checks)==3))
        del r0,h0,raw0,r1,h1,raw1,r2,h2,raw2,logits,expected
    # Evaluate each checkpoint consistently over the remaining fixed 97 UIDs; never combine per-mesh heads.
    for branch,state in [('before',original),('after',new)]:
        head.load_state_dict(state,strict=True);headhash=tensor_hash(head.state_dict().items())
        for uid in uids:
            if uid in results[branch]:continue
            assert tensor_hash(head.state_dict().items())==headhash
            rows,hidden,raw=run_forward(uid);evaluate_one(uid,branch,rows,hidden)
            del rows,hidden,raw
    hook.remove()
    assert non_head_hash()==source_other and all(p.grad is None for p in model.parameters())
    assert sha(source)==cfg['source_sha256'] and sha(hp)==cfg['head_sha256']
    summaries={}
    for branch in results:
        rows=[results[branch][u] for u in uids]
        summaries[branch]=dict(meshes=len(rows),edge_perfect=sum(r['edge_perfect'] for r in rows),
            face_perfect=sum(r['face_perfect'] for r in rows),joint_perfect=sum(r['joint_perfect'] for r in rows),
            **{kind+'_'+k:sum(r[kind][k] for r in rows) for kind in ['edge','face'] for k in ['tp','fp','fn']})
    retention={}
    for key in ['edge_perfect','face_perfect','joint_perfect']:
        old={u for u in uids if results['before'][u][key]};newset={u for u in uids if results['after'][u][key]}
        retention[key]=dict(retained=sorted(old&newset),lost=sorted(old-newset),gained=sorted(newset-old),before=sorted(old),after=sorted(newset))
    complete=dict(summaries=summaries,retention=retention,all100_baseline_matches_archived=True,
        all100_mu_hidden_logvar_face_embedding_unchanged=True,all100_face_training_logits_bitwise_unchanged=True,
        all200_actual_face_evaluations_complete=True,three_gate_passed=True,optimizer_updates=0,backward_calls=0,
        source_and_head_files_unchanged=True,source_nonhead_state_unchanged=True,trainable_parameters_updated=0,
        inference_copy=str(installed),inference_copy_sha256=sha(installed),source_config=cfg)
    write(out/'complete.json',complete);write(out/'attention_audit.json',b.AUDIT)
    write(out/'status.json',dict(state='complete',optimizer_updates=0,evaluations=200))
    print('COMPLETE',json.dumps(summaries),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        (ROOT/'failure.txt').write_text(traceback.format_exc());raise
