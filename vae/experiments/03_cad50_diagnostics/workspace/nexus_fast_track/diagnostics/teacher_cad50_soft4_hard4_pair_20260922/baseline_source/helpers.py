import json,random,time
from runtime import ROOT,BASE,T,np,c,write,sha,tensor_hash
from evaluate import evaluate_mesh
PARENT=BASE/'diagnostics/teacher_cad50_lr03_pair_20260921'
SOURCE=PARENT/'B_lr03/checkpoint-new0500-step2500.pt'
SOURCE_SHA='4c67709dcd3bacb6a09181b034512469b1e7f38463f1aa7b41affc8bcf435a66'
SELECTED=[]
def rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=T.get_rng_state(),cuda=T.cuda.get_rng_state_all())


def restore_rng(x):
    random.setstate(x['python']);np.random.set_state(x['numpy']);T.set_rng_state(x['torch']);T.cuda.set_rng_state_all(x['cuda'])


def equal(a,b):
    if T.is_tensor(a):return T.equal(a.detach().cpu(),b.detach().cpu())
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def group_steps(opt):
    ans={}
    for g in opt.param_groups:
        values={int(opt.state[p]['step']) for p in g['params']}
        assert len(values)==1
        ans[g['name']]=values.pop()
    return ans


@T.no_grad()
def edge_observation(rows,pool,scales,arrays=False):
    n=len(pool['vertices']);keys=T.as_tensor(np.unique(pool['edges'].T@np.array([n,1])),device='cuda')
    ids=T.triu_indices(n,n,offset=1,device='cuda').T
    logits=c.edge_logits(rows[2][0].detach(),ids,scales)
    assert T.isfinite(logits).all()
    y=T.isin(ids[:,0]*n+ids[:,1],keys);positive=logits>0
    tp=int((positive&y).sum());fp=int((positive&~y).sum());fn=int((~positive&y).sum())
    result=dict(tp=tp,fp=fp,fn=fn,edge_perfect=fp==fn==0)
    if arrays:return result,ids.cpu().numpy().astype(np.int32),logits.cpu().numpy(),y.cpu().numpy()
    return result


def keep_encoder_output(model,captured):
    def hook(module,inputs,output):
        captured['encoder_vertex_hidden']=output.detach().clone()
        return None
    return model.autoencoder.encoder_output_norm.register_forward_hook(hook)


def summarize(rows,parent_success):
    totals={kind:{k:sum(x[kind][k] for x in rows) for k in ['tp','fp','fn','tn']} for kind in ['edge','face']}
    for d in totals.values():d['micro_f1']=2*d['tp']/max(2*d['tp']+d['fp']+d['fn'],1)
    edge=[x['uid'] for x in rows if x['edge_perfect']];face=[x['uid'] for x in rows if x['face_perfect']]
    joint=[x['uid'] for x in rows if x['joint_perfect']]
    return dict(counts=totals,edge_perfect=len(edge),face_perfect=len(face),joint_perfect=len(joint),
        edge_perfect_uids=edge,face_perfect_uids=face,perfect_uids=joint,
        retained=sorted(set(joint)&set(parent_success)),lost=sorted(set(parent_success)-set(joint)),new=sorted(set(joint)-set(parent_success)),
        losses={k:sum(x['parts'][k] for x in rows)/len(rows) for k in ['edge','face']},
        face_fn_missing=sum(x['missing_gt_face_candidates'] for x in rows),
        face_fn_present=sum(x['face']['fn_present_but_negative'] for x in rows),
        face_fp_inside_pool=sum(x['face']['actual_fp_inside_training_pool'] for x in rows),
        face_fp_outside_pool=sum(x['face']['actual_fp_outside_training_pool'] for x in rows))


def evaluate(model,uids,pools,objective,capture,batches,out,new_step,checkpoint,parent_eval,representations=False):
    params_before=tensor_hash(model.named_parameters());rng_before=rng();scales=model.scoring_contract()
    dest=out/f'predictions-new{new_step:04d}';dest.mkdir(exist_ok=False)
    repdir=out/f'representations-new{new_step:04d}'
    if representations:repdir.mkdir(exist_ok=False)
    records=[];rep_index=[];started=time.monotonic()
    for uid in uids:
        captured={};hook=None
        if representations and uid in SELECTED:hook=keep_encoder_output(model,captured)
        rows,loss,parts=objective(uid)
        if hook:hook.remove()
        edge_check=edge_observation(rows,pools[uid],scales)
        detached=tuple(tuple(x.detach() for x in group) for group in rows)
        if captured:
            batch=batches[uid];n=len(pools[uid]['vertices'])
            _,pair,edge_logits,labels=edge_observation(rows,pools[uid],scales,True)
            gt_faces=np.sort(pools[uid]['positive'],axis=1).astype(np.int32)
            with T.no_grad():fl=c.face_logits(detached[3][0],T.as_tensor(gt_faces,device='cuda'),scales).cpu().numpy()
            tensors=dict(vertices=batch.vertices[0,:n].detach().cpu().numpy(),vertex_mask=batch.vertex_mask[0].cpu().numpy(),
                local_vertex_indices=np.arange(n,dtype=np.int32),encoder_vertex_hidden=captured['encoder_vertex_hidden'].cpu().numpy(),
                mu=detached[0][0].cpu().numpy(),decoder_hidden=capture['hidden'].detach().cpu().numpy(),
                edge_embedding=detached[2][0].cpu().numpy(),face_embedding=detached[3][0].cpu().numpy(),
                edge_pair_ids=pair,edge_pair_logits=edge_logits,edge_pair_gt=labels,gt_face_ids=gt_faces,gt_face_logits=fl)
            assert np.array_equal(tensors['vertices'],pools[uid]['vertices'])
            p=repdir/f'{uid}.npz';np.savez_compressed(p,**tensors)
            rep_index.append(dict(uid=uid,path=str(p.relative_to(ROOT)),sha256=sha(p),shapes={k:list(v.shape) for k,v in tensors.items()},
                positions=dict(encoder_vertex_hidden='encoder_output_norm output: vertex nodes only, before mu projection',
                    mu='mu head output; exactly decoder latent input',decoder_hidden='decoder_output_norm output, after all16 blocks',
                    edge_embedding='actual per-mesh centered scoring representation',face_embedding='actual per-mesh centered scoring representation'),
                coordinates='unchanged source FP32; original cached local node numbering',mask='microbatch1, all real vertices true'))
        del rows,loss;capture.clear();captured.clear()
        p=dest/f'{uid}.npz';m=evaluate_mesh(detached,pools[uid],scales,p);del detached
        assert m['face']['complete']
        assert all(edge_check[k]==m['edge'][k] for k in ['tp','fp','fn'])
        m['face']['actual_fp_inside_training_pool']=m['face']['fp']-m['face']['actual_fp_outside_training_pool']
        m['face']['fn_missing_candidate']=m['missing_gt_face_candidates']
        m['face']['fn_present_but_negative']=m['face']['fn']-m['missing_gt_face_candidates']
        records.append(dict(uid=uid,vertices=len(pools[uid]['vertices']),parts=parts,
            prediction_path=str(p.relative_to(ROOT)),prediction_sha256=sha(p),**m))
    summary=summarize(records,parent_eval['perfect_uids'])
    summary.update(new_step=new_step,completed_updates=2500+new_step,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        timing='frozen state after completed_updates optimizer updates',seconds=time.monotonic()-started,meshes=records)
    summary['size_groups']={}
    for name,lo,hi in [('8_vertices',8,8),('12_to_16',12,16),('66_to_274',66,274)]:
        subset=[x for x in records if lo<=x['vertices']<=hi]
        subset_uids={x['uid'] for x in subset}
        subset_parent=[u for u in parent_eval['perfect_uids'] if u in subset_uids]
        summary['size_groups'][name]=dict(meshes=len(subset),**summarize(subset,subset_parent))
    assert sum(x['meshes'] for x in summary['size_groups'].values())==50
    if new_step==0:
        assert summary['counts']==parent_eval['counts']
        assert summary['perfect_uids']==parent_eval['perfect_uids']
        assert summary['edge_perfect']==summary['face_perfect']==summary['joint_perfect']==32
        for actual,reference in zip(records,parent_eval['meshes']):
            assert actual['uid']==reference['uid'] and actual['parts']==reference['parts']
            for kind in ['edge','face']:
                for k in ['tp','fp','fn','tn']:assert actual[kind][k]==reference[kind][k]
            with np.load(PARENT/reference['prediction_path']) as old,np.load(ROOT/actual['prediction_path']) as now:
                assert old.files==now.files and all(np.array_equal(old[k],now[k]) for k in old.files)
    assert tensor_hash(model.named_parameters())==params_before and equal(rng_before,rng())
    assert all(p.grad is None for p in model.autoencoder.log_variance.parameters())
    write(out/f'eval-new{new_step:04d}.json',summary)
    if representations:write(repdir/'manifest.json',rep_index)
    print('EVAL',out.name,new_step,summary['joint_perfect'],summary['counts'],flush=True)
    return summary


def save_checkpoint(path,model,opt,config,cp,new_step,updates_applied=None):
    tmp=path.with_suffix('.tmp')
    T.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng(),config=config,
        completed_updates=2500+new_step,new_updates=new_step,participation={u:2500+new_step for u in cp['participation']},
        parent_checkpoint=str(SOURCE),parent_sha256=SOURCE_SHA,updates_applied=updates_applied),tmp)
    tmp.replace(path)
    return path

