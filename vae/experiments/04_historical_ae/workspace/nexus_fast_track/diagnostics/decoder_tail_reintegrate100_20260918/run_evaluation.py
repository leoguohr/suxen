"""Install trained tail into a source-model copy; all100 actual reconstruction, zero updates."""
import fcntl,json,shutil,time,traceback
from pathlib import Path
from runtime import ROOT,T,np,b,c,setup,write,sha,tensor_hash
import evaluate

def detach(rows):return tuple(tuple(x.detach() for x in group) for group in rows)
def feature_hashes(rows,hidden):
    return {name:tensor_hash([(name,x)]) for name,x in [('mu',rows[0][0]),('logvar',rows[1][0]),('hidden',hidden),('edge_embedding',rows[2][0]),('face_embedding',rows[3][0])]}
def delta(x,y):
    dx=(y-x).double();den=x.double().norm()
    return dict(bitwise=T.equal(x,y),max_abs=float(dx.abs().max()),relative_l2=float(dx.norm()/den) if den>0 else None)
def summary(rows):
    return dict(meshes=len(rows),edge_perfect=sum(r['edge_perfect'] for r in rows),face_perfect=sum(r['face_perfect'] for r in rows),joint_perfect=sum(r['joint_perfect'] for r in rows),
        **{kind+'_'+k:sum(r[kind][k] for r in rows) for kind in ['edge','face'] for k in ['tp','fp','fn']},
        missing_gt_face_candidates=sum(r['missing_gt_face_candidates'] for r in rows),all_face_complete=all(r['face']['complete'] for r in rows))

def main():
    lock=(ROOT/'execution.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=json.loads((ROOT/'config.json').read_text());source=Path(cfg['source_checkpoint']);tp=Path(cfg['tail_checkpoint']);probe=Path(cfg['tail_root'])
    assert sha(source)==cfg['source_sha256'] and sha(tp)==cfg['tail_sha256']
    parent=source.parent.parent
    for name in ['READY.json','data_manifest.csv','overfit100_manifest.csv','selection.json','pool_provenance.json','data_validation.json','pools','source_archive','review_runtime','augmented_pools','mining_complete.json']:
        target=(parent/name).resolve();assert target.exists(),target
        dest=ROOT/name
        if not dest.exists():dest.symlink_to(target)
        assert dest.resolve()==target
    shutil.copyfile(parent/'run/eval-epoch900.jsonl',ROOT/'source_archived_eval.jsonl')
    shutil.copyfile(parent/'run/manifest.json',ROOT/'source_manifest.json')
    shutil.copyfile(probe/'run/checkpoint-step0500.json',ROOT/'tail_expected_edge_step500.json')
    out=ROOT/'run';out.mkdir(exist_ok=False);features=out/'features';features.mkdir()
    write(out/'status.json',dict(stage='loading',optimizer_updates=0))
    model,opt,groups,uids,base_pools,forward,capture,args=setup();assert opt is None;del groups
    T.use_deterministic_algorithms(True)
    cp=T.load(source,map_location='cpu',mmap=True,weights_only=False)
    assert cp['completed_updates']==22500 and cp['epoch']==900 and cp['manifest']['uids']==uids and len(set(uids))==100
    model.load_state_dict(cp['model'],strict=True);a=model.autoencoder
    tailcp=T.load(tp,map_location='cpu',weights_only=False)
    assert tailcp['completed_updates']==500 and tailcp['config']['source_sha256']==cfg['source_sha256']
    tail=T.nn.ModuleDict(dict(block=a.decoder_blocks[15],final_norm=a.decoder_output_norm,head=a.edge_embedding))
    assert set(tail.state_dict())==set(tailcp['tail'])
    original={k:v.detach().clone() for k,v in tail.state_dict().items()};new={k:v.cuda() for k,v in tailcp['tail'].items()}
    mapping={}
    for key in new:
        prefix=next(p for p in cfg['mapping'] if key.startswith(p));mapping[key]=cfg['mapping'][prefix]+key[len(prefix):]
    assert len(mapping)==10
    def rest_hash():
        state=model.state_dict()
        return dict(tensors=tensor_hash((k,v) for k,v in state.items() if T.is_tensor(v) and k not in mapping.values()),metadata=json.dumps({k:v for k,v in state.items() if not T.is_tensor(v)},sort_keys=True))
    source_rest=rest_hash();scales=model.scoring_contract();assert scales==cp['manifest']['scales']
    tail.load_state_dict(new,strict=True);assert rest_hash()==source_rest
    installed=out/'model-source22500-tail500-inference.pt'
    T.save(dict(model={k:v.detach().cpu() if T.is_tensor(v) else v for k,v in model.state_dict().items()},args=cp['args'],inference_only=True,source_checkpoint=str(source),source_sha256=cfg['source_sha256'],tail_checkpoint=str(tp),tail_sha256=cfg['tail_sha256'],optimizer_updates=0,mapping=mapping),installed)
    reloaded=T.load(installed,map_location='cpu',mmap=True,weights_only=False)
    changed=[]
    for key,value in cp['model'].items():
        other=reloaded['model'][key];same=T.equal(value,other) if T.is_tensor(value) else value==other
        if not same:changed.append(key)
        if key not in mapping.values():assert same,key
    for key,destination in mapping.items():assert T.equal(reloaded['model'][destination],tailcp['tail'][key]),destination
    assert changed and set(changed)<=set(mapping.values())
    model.load_state_dict(reloaded['model'],strict=True);assert rest_hash()==source_rest
    pools={}
    for uid in uids:
        path=ROOT/'augmented_pools'/f'{uid}_pool.npz';assert sha(path)==cp['manifest']['augmented_pool_sha256'][uid]
        with np.load(path) as f:pools[uid]={k:f[k] for k in ['vertices','edges','positive','mixed']}
    del cp,reloaded,tailcp
    installation=dict(source_sha256=cfg['source_sha256'],tail_sha256=cfg['tail_sha256'],mapping=mapping,changed_state_keys=changed,non_target_parameters_buffers_metadata_bitwise_unchanged=True,face_head_weights_unchanged=True,inference_copy=str(installed),inference_sha256=sha(installed),reloaded=True,optimizer_constructed=False,optimizer_updates=0,backward_calls=0)
    write(out/'installation_verification.json',installation)
    expected={r['uid']:r for r in json.loads((ROOT/'tail_expected_edge_step500.json').read_text())['meshes']}
    archived={r['uid']:r for r in map(json.loads,(ROOT/'source_archived_eval.jsonl').read_text().splitlines())}
    grabs={}
    hooks=[a.decoder_blocks[15].norm.register_forward_pre_hook(lambda m,x:grabs.update(last_input=x[0].detach())),a.edge_embedding.register_forward_hook(lambda m,x,y:grabs.update(raw=y.detach()))]
    def run_forward(uid):
        # Retain the validated source grad-mode kernel dispatch; detach outputs, never backward.
        rows=detach(forward(uid));h=capture['hidden'].detach();raw=grabs['raw'];x=grabs['last_input'];capture.clear();grabs.clear()
        return rows,h,raw,x
    # First gate ALL100 Edge results before starting actual-Face evaluation.
    gate=[];reference_hashes={'before':{},'after':{}}
    for i,uid in enumerate(uids):
        tail.load_state_dict(original,strict=True);r0,h0,raw0,x0=run_forward(uid)
        tail.load_state_dict(new,strict=True);r1,h1,raw1,x1=run_forward(uid)
        assert T.equal(r0[0][0],r1[0][0]) and T.equal(r0[1][0],r1[1][0]) and T.equal(x0,x1),uid
        with np.load(probe/'cache'/f'{uid}.npz') as cache:
            assert np.array_equal(x1.cpu().numpy(),cache['last_block_input']) and np.array_equal(h0.cpu().numpy(),cache['original_hidden']) and np.array_equal(raw0.cpu().numpy(),cache['original_edge_raw']),uid
        n=len(h1);pairs=T.triu_indices(n,n,1,device='cuda').T
        with T.no_grad():s=c.edge_logits(r1[2][0],pairs,scales).cpu().numpy()
        with np.load(probe/'run/final_outputs'/f'{uid}.npz') as saved:
            assert np.array_equal(raw1.cpu().numpy(),saved['edge_head_raw']) and np.array_equal(r1[2][0].cpu().numpy(),saved['edge_embedding']) and np.array_equal(s,saved['logits']),uid
        edges=np.unique(np.sort(base_pools[uid]['edges'].T,axis=1),axis=0);keys=edges@np.array([n,1]);pairnp=pairs.cpu().numpy();labels=np.isin(pairnp@np.array([n,1]),keys);pred=s>0
        counts=dict(tp=int((pred&labels).sum()),fp=int((pred&~labels).sum()),fn=int((~pred&labels).sum()),tn=int((~pred&~labels).sum()))
        assert all(expected[uid][k]==v for k,v in counts.items()),uid
        reference_hashes['before'][uid]=feature_hashes(r0,h0);reference_hashes['after'][uid]=feature_hashes(r1,h1)
        record=dict(uid=uid,vertices=n,counts=counts,perfect=counts['fp']==counts['fn']==0,all_pair_logits_bitwise_match_cached=True,mu_logvar_last_block_input_bitwise_unchanged=True,hidden_change=delta(h0,h1),face_embedding_change=delta(r0[3][0],r1[3][0]))
        if uid in cfg['seven_uids']:
            rr,hh,rawr,xx=run_forward(uid)
            assert all(T.equal(x[0],y[0]) for x,y in zip(rr,r1)) and T.equal(hh,h1) and T.equal(rawr,raw1)
            record['repeat_forward_bitwise']=True
            del rr,hh,rawr,xx
        np.savez_compressed(features/f'{uid}.npz',mu=r1[0][0].cpu().numpy(),logvar=r1[1][0].cpu().numpy(),hidden_before=h0.cpu().numpy(),hidden_after=h1.cpu().numpy(),edge_before=r0[2][0].cpu().numpy(),edge_after=r1[2][0].cpu().numpy(),face_before=r0[3][0].cpu().numpy(),face_after=r1[3][0].cpu().numpy(),edge_logits_after=s,vertices=base_pools[uid]['vertices'],edges=edges,faces=base_pools[uid]['positive'])
        record['features_sha256']=sha(features/f'{uid}.npz');gate.append(record)
        write(out/'edge_gate.json',dict(meshes=gate,complete=len(gate)==100))
        write(out/'status.json',dict(stage='edge_gate',completed=i+1,uid=uid,optimizer_updates=0))
        print('EDGE_GATE',i+1,uid,counts,flush=True)
        del r0,r1,h0,h1,raw0,raw1,x0,x1,pairs,s
    totals=dict(perfect=sum(x['perfect'] for x in gate),fp=sum(x['counts']['fp'] for x in gate),fn=sum(x['counts']['fn'] for x in gate));assert totals==cfg['expected_installed_edge'],totals
    write(out/'edge_gate_summary.json',dict(complete=True,meshes=100,**totals,all_logits_bitwise_match_cached=True,upstream_outputs_unchanged=True))
    write(out/'feature_hashes.json',reference_hashes)
    evaluate.FACE_SECONDS=float('inf')
    results={'before':{},'after':{}}
    for branch,state in [('before',original),('after',new)]:
        tail.load_state_dict(state,strict=True);startstate=tensor_hash(tail.state_dict().items())
        for i,uid in enumerate(uids):
            started=time.monotonic();rows,hidden,raw,x=run_forward(uid)
            assert feature_hashes(rows,hidden)==reference_hashes[branch][uid],(branch,uid)
            with T.no_grad():
                _,parts,saved=b.full_objective(rows,[pools[uid]],scales)
                actual=evaluate.evaluate_mesh(rows,base_pools[uid],scales)
            assert actual['face']['complete'],uid
            y=np.r_[np.ones(len(pools[uid]['positive']),dtype=bool),np.zeros(len(pools[uid]['mixed']),dtype=bool)]
            row=dict(uid=uid,branch=branch,vertices=len(pools[uid]['vertices']),gt_edges=len(pools[uid]['edges'].T),gt_faces=len(pools[uid]['positive']),parts=parts[0],face_training_pool=c.metrics(y,saved[0]['face_train_logits']),feature_hashes=reference_hashes[branch][uid],actual_fp_outside_pool_reference='original_base_pool',**actual)
            if branch=='before':
                old=archived[uid]
                for key in ['edge','face_training_pool','parts','gt_face_candidates','missing_gt_face_candidates','margins','edge_perfect','face_perfect','joint_perfect']:assert row[key]==old[key],(uid,key)
                for key in ['tp','fp','fn','tn','complete','scored_candidates','actual_fp_outside_training_pool']:assert row['face'][key]==old['face'][key],(uid,key)
            else:
                assert all(row['edge'][k]==expected[uid][k] for k in ['tp','fp','fn','tn'])
            row['evaluation_seconds']=time.monotonic()-started;results[branch][uid]=row
            write(out/f'{branch}-{uid}.json',row)
            with (out/f'{branch}.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            write(out/'status.json',dict(stage='actual_face',branch=branch,completed=i+1,uid=uid,edge_fp_fn=[row['edge']['fp'],row['edge']['fn']],face_fp_fn=[row['face']['fp'],row['face']['fn']],optimizer_updates=0))
            print('EVAL',branch,i+1,uid,'edge',row['edge']['fp'],row['edge']['fn'],'face',row['face']['fp'],row['face']['fn'],flush=True)
            del rows,hidden,raw,x,saved
        assert tensor_hash(tail.state_dict().items())==startstate
    for hook in hooks:hook.remove()
    assert rest_hash()==source_rest and all(p.grad is None for p in model.parameters())
    assert sha(source)==cfg['source_sha256'] and sha(tp)==cfg['tail_sha256']
    retention={}
    for key in ['edge_perfect','face_perfect','joint_perfect']:
        old={u for u in uids if results['before'][u][key]};newset={u for u in uids if results['after'][u][key]}
        retention[key]=dict(before=sorted(old),after=sorted(newset),retained=sorted(old&newset),lost=sorted(old-newset),gained=sorted(newset-old))
    seven=[dict(uid=uid,before=results['before'][uid],after=results['after'][uid]) for uid in cfg['seven_uids']]
    write(out/'seven_large_meshes.json',seven)
    complete=dict(summaries={name:summary([rr[u] for u in uids]) for name,rr in results.items()},retention=retention,all100_edge_logits_bitwise_reproduced=True,all100_baseline_matches_archived=True,all200_actual_face_enumerations_complete=True,mu_logvar_and_pre_tail_features_bitwise_unchanged=True,face_embedding_changed_meshes=sum(not r['face_embedding_change']['bitwise'] for r in gate),face_head_parameters_unchanged=True,optimizer_constructed=False,optimizer_updates=0,backward_calls=0,all_parameter_grads_none=True,source_and_tail_files_unchanged=True,non_target_state_unchanged=True,inference_copy=str(installed),inference_copy_sha256=sha(installed))
    write(out/'complete.json',complete);write(out/'attention_audit.json',b.AUDIT)
    write(out/'status.json',dict(stage='complete',optimizer_updates=0,evaluations=200));print('COMPLETE',json.dumps(complete['summaries']),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        (ROOT/'failure.txt').write_text(traceback.format_exc());raise
