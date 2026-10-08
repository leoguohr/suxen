"""Fresh-process full network and exhaustive predicted-edge clique evaluation."""
import argparse
import itertools
import time
import traceback
from run_support import *
from native_models import Config, NativeTopologyAE, Graph
from data_objective import load_dataset, edge_logits, face_logits


def aggregate(meshes):
    counts={task:{k:sum(x[task][k] for x in meshes) for k in ['tp','fp','fn','tn']} for task in ['edge','face']}
    for value in counts.values():value['micro_f1']=2*value['tp']/max(2*value['tp']+value['fp']+value['fn'],1)
    return dict(counts=counts,joint_perfect=sum(x['joint_perfect'] for x in meshes),
        perfect_uids=[x['uid'] for x in meshes if x['joint_perfect']],
        edge_perfect_uids=[x['uid'] for x in meshes if x['edge']['fp']==x['edge']['fn']==0],
        face_perfect_uids=[x['uid'] for x in meshes if x['face']['fp']==x['face']['fn']==0],
        face_fn_missing=sum(x['face_fn_missing'] for x in meshes),
        face_fn_present=sum(x['face_fn_present'] for x in meshes))


@torch.no_grad()
def evaluate_mesh(model,item,destination,account):
    n=len(item['vertices']);device='cuda'
    data=move_item(item,device);graph=Graph.from_faces(data['faces'],n)
    rows=model(data['vertices'],data['faces'],sample_latent=False,graph=graph)
    assert rows['latent'] is rows['mu'] and all(torch.isfinite(v).all() for v in rows.values())
    pair_ids=item['pairs'].numpy();edge_parts=[]
    for start in range(0,len(pair_ids),32768):edge_parts.append(edge_logits(rows['edge'],data['pairs'][start:start+32768]).cpu().numpy())
    edge_scores=np.concatenate(edge_parts);labels=item['edge_labels'].numpy();positive=edge_scores>0
    edge=dict(tp=int((positive&labels).sum()),fp=int((positive&~labels).sum()),fn=int((~positive&labels).sum()),tn=int((~positive&~labels).sum()))
    adjacency=np.zeros((n,n),dtype=bool);predicted_pairs=pair_ids[positive]
    adjacency[predicted_pairs[:,0],predicted_pairs[:,1]]=True
    gt=item['gt_faces'].numpy();gt_keys=(gt[:,0]*n+gt[:,1])*n+gt[:,2]
    covered=adjacency[gt[:,0],gt[:,1]]&adjacency[gt[:,0],gt[:,2]]&adjacency[gt[:,1],gt[:,2]]
    gt_logits=face_logits(rows['face'],data['gt_faces']).cpu().numpy()
    triangles=[];scores=[];face_labels=[];pending=[]
    def score(part):
        if account.update()>=MAX_GPU_SECONDS:raise RuntimeError('Global GPU resource limit reached during complete evaluation')
        ids=np.asarray(part,dtype=np.int64).reshape(-1,3)
        logits=face_logits(rows['face'],torch.as_tensor(ids,device=device)).cpu().numpy()
        assert np.isfinite(logits).all()
        keys=(ids[:,0]*n+ids[:,1])*n+ids[:,2]
        triangles.append(ids.astype(np.int32));scores.append(logits);face_labels.append(np.isin(keys,gt_keys))
    for i in range(n):
        for j in np.flatnonzero(adjacency[i]):
            ks=np.flatnonzero(adjacency[i]&adjacency[j])
            pending.extend((i,int(j),int(k)) for k in ks)
            while len(pending)>=32768:score(pending[:32768]);pending=pending[32768:]
    if pending:score(pending)
    ids=np.concatenate(triangles) if triangles else np.empty((0,3),np.int32)
    fl=np.concatenate(scores) if scores else np.empty(0,np.float32)
    fy=np.concatenate(face_labels) if face_labels else np.empty(0,bool)
    fp=int(((fl>0)&~fy).sum());tp=int(((fl>0)&fy).sum())
    face=dict(tp=tp,fp=fp,fn=len(gt)-tp,tn=int(((fl<=0)&~fy).sum()))
    assert tp==int(((gt_logits>0)&covered).sum())
    missing=int((~covered).sum());present=int(((gt_logits<=0)&covered).sum())
    assert face['fn']==missing+present
    np.savez_compressed(destination,vertices=item['vertices'].numpy(),original_vertex_indices=item['original_vertex_indices'].numpy(),gt_edges=item['edges'].numpy().astype(np.int32),
        gt_faces=gt.astype(np.int32),all_edge_pair_ids=pair_ids.astype(np.int32),all_edge_pair_logits=edge_scores,
        all_edge_pair_gt=labels,predicted_edge_ids=predicted_pairs.astype(np.int32),
        actual_face_candidate_ids=ids,actual_face_candidate_logits=fl,actual_face_candidate_gt=fy,
        gt_face_covered=covered,gt_face_logits=gt_logits)
    return dict(uid=item['uid'],vertices=n,edge=edge,face=face,face_fn_missing=missing,face_fn_present=present,
        joint_perfect=edge['fp']==edge['fn']==face['fp']==face['fn']==0,
        actual_face_candidates=len(ids),complete=True,prediction_path=str(destination.relative_to(ROOT)),prediction_sha256=sha(destination),
        representation=dict(mu_rms=float(rows['mu'].square().mean().sqrt()),
            decoder_hidden_rms=float(rows['decoder_hidden'].square().mean().sqrt()),
            edge_rms=float(rows['edge'].square().mean().sqrt()),face_rms=float(rows['face'].square().mean().sqrt())))


def main(request_path):
    request=read(request_path);variant=request['variant'];step=request['step'];kind=request.get('kind','evaluation')
    branch=ROOT/variant
    account=ResourceAccount(f'{kind}-{variant}-{step:05d}')
    started=time.monotonic()
    try:
        configure();torch.cuda.set_device(0)
        cp_path=Path(request['checkpoint']['path'])
        assert sha(cp_path)==request['checkpoint']['sha256']
        cp=torch.load(cp_path,map_location='cpu',mmap=True,weights_only=False)
        assert cp['model_variant']==variant and cp['completed_updates']==step
        for name,digest in cp['config']['code_sha256'].items():assert sha(ROOT/name)==digest,name
        model=NativeTopologyAE(Config(**cp['model_config'])).cuda().float()
        model.load_state_dict(cp['model'],strict=True);model.eval();restore_rng(cp['rng'])
        assert tensor_hash(model.state_dict())==tensor_hash(cp['model'])
        assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
        state_before=tensor_hash(model.state_dict())
        items,manifest=load_dataset(ROOT/'data',ROOT/'pools')
        assert manifest==cp['config']['data']
        folder=branch/'evaluations'/f'{kind}-{step:05d}';folder.mkdir(parents=True,exist_ok=False)
        meshes=[]
        for uid in manifest['uids']:
            if account.update()>=MAX_GPU_SECONDS:raise RuntimeError('Global GPU resource limit')
            meshes.append(evaluate_mesh(model,items[uid],folder/(uid+'.npz'),account))
        assert tensor_hash(model.state_dict())==state_before
        summary=aggregate(meshes)
        large=[m for m in meshes if 66<=m['vertices']<=274]
        assert len(large)==16
        result=dict(complete=True,variant=variant,step=step,checkpoint=request['checkpoint'],
            kind=kind,optimizer_updates=0,from_fresh_process=True,native_forward=True,model_unchanged=True,
            **summary,large16=aggregate(large),meshes=meshes,seconds=time.monotonic()-started,
            gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],process_id=os.getpid())
        if kind=='cold':
            previous=read(branch/'evaluations'/f'eval-{step:05d}.json')
            assert result['counts']==previous['counts'] and result['perfect_uids']==previous['perfect_uids']
            for old,new in zip(previous['meshes'],meshes):
                with np.load(ROOT/old['prediction_path']) as a,np.load(ROOT/new['prediction_path']) as b:
                    assert a.files==b.files and all(a[k].dtype==b[k].dtype and a[k].tobytes()==b[k].tobytes() for k in a.files)
            result['all50_arrays_bitwise_equal']=True
            output=branch/'evaluations'/f'cold-{step:05d}.json'
        else:output=branch/'evaluations'/f'eval-{step:05d}.json'
        write(output,result)
        if kind=='evaluation':
            for name,value in [('face_f1',summary['counts']['face']['micro_f1']),('strict',summary['joint_perfect'])]:
                pointer=branch/f'best_{name}.json'
                if not pointer.exists() or value>read(pointer)['value']:
                    write(pointer,dict(value=value,step=step,checkpoint=request['checkpoint'],evaluation=str(output)))
            if summary['counts']['face']['micro_f1']>=.997:
                write(branch/'TARGET_REACHED.json',dict(step=step,face_f1=summary['counts']['face']['micro_f1'],checkpoint=request['checkpoint']))
                write(ROOT/'queue'/f'{step:05d}-{variant}-cold.json',{**request,'kind':'cold'})
        print('EVAL_COMPLETE',variant,step,result['counts'],result['joint_perfect'],flush=True)
    except BaseException as error:
        write(branch/'evaluation_failure.json',dict(step=step,kind=kind,error=str(error),traceback=traceback.format_exc()))
        raise
    finally:account.update('finish')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('request');main(parser.parse_args().request)
