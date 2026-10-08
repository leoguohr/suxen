"""Independent FP32 finite displacements of an already saved Adam candidate.

No optimizer is constructed. No backward or parameter update is performed.
Attention keeps the verified grad-enabled dispatch; only inference is consumed.
"""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import fcntl, json, math, shutil, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/guohaoran/nexus_fast_track/diagnostics/decoder_last2_joint_fixed100_20260920')
PREVIOUS = SOURCE.parent/'decoder_last2_gradient_groups_20260920'
PARTS = ['success_edge', 'success_face', 'failed_edge', 'failed_face']
GRID = [0., 1/16, -1/16, 1/8, -1/8, 1/4, -1/4, 1/2, -1/2, 1., -1.]
CP_HASH = '30d207d3508429a749885764712c169a85479e4c2673f94f816fa54cfa94e753'
FULL_HASH = '82cf33fbc0151d72332218c5e447f7bf84e8934947e4c70148205c6a308db884'
VECTORS_HASH = '09d1b99493391317fa6661c6626d0282d9bb5ff23602418ab954a7f93749d9b3'


def point_id(value):
    return 'zero' if value == 0 else ('plus' if value > 0 else 'minus')+str(round(1/abs(value)))


def main():
    lock = (ROOT/'execution.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (ROOT/'complete.json').exists(), 'Do not overwrite completed experiment'
    started = time.monotonic()
    for name in ['READY.json', 'overfit100_manifest.csv', 'data_manifest.csv',
                 'selection.json', 'pool_provenance.json', 'pools', 'construction_args.json']:
        target = (SOURCE/name).resolve()
        assert target.exists(), target
        if not (ROOT/name).exists(): (ROOT/name).symlink_to(target)
        assert (ROOT/name).resolve() == target
    if not (ROOT/'effective_code').exists(): shutil.copytree(SOURCE/'effective_code', ROOT/'effective_code')
    from runtime import T, np, setup, sha, write, tensor_hash
    import prior_core as core
    import joint_core as joint
    import evaluate
    evaluate.FACE_SECONDS = float('inf')
    cp_path = SOURCE/'run/checkpoint-new0500-tail1500.pt'
    full_path = SOURCE/'run/model-last2-joint1000-tail1500-block14new500-inference.pt'
    vector_path = PREVIOUS/'gradient_vectors.pt'
    for path, expected in [(cp_path, CP_HASH), (full_path, FULL_HASH), (vector_path, VECTORS_HASH)]:
        assert sha(path) == expected, path
    partition = json.loads((PREVIOUS/'partition.json').read_text())
    assert sha(partition['source']) == partition['source_sha256']
    success, failed = set(partition['success71']), set(partition['failed29'])
    assert len(success) == 71 and len(failed) == 29 and not success & failed
    write(ROOT/'partition.json', partition)
    model, opt, groups, uids, _, forward, capture, args = setup()
    assert opt is None
    del groups
    T.use_deterministic_algorithms(True)
    full = T.load(full_path, map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(full['model'], strict=True)
    del full
    model.requires_grad_(False)
    a = model.autoencoder
    tail = T.nn.ModuleDict(dict(block=a.decoder_blocks[15], final_norm=a.decoder_output_norm,
        head=a.edge_embedding, face=a.face_embedding, penultimate=a.decoder_blocks[14]))
    tail.requires_grad_(True)
    named = list(tail.named_parameters())
    pnames = [n for n,p in named]
    model_before = tensor_hash((n,t) for n,t in model.state_dict().items() if T.is_tensor(t))
    cp = T.load(cp_path, map_location='cpu', weights_only=False)
    theta0 = {n:t.detach().clone() for n,t in cp['tail'].items()}
    assert all(T.equal(p.detach().cpu(), theta0[n]) for n,p in tail.state_dict().items())
    assert all(v.dtype == T.float32 for v in theta0.values())
    vectors = T.load(vector_path, map_location='cpu', weights_only=False)
    assert pnames == vectors['parameter_names']
    candidate = vectors['cloned_adam_actual_fp32_displacement']
    gradients = vectors['gradients']
    assert set(gradients) == set(PARTS)
    assert len(candidate) == len(named) and set(theta0) == set(pnames)

    def dot(xs, ys):
        return sum(float((x.double()*y.double()).sum()) for x,y in zip(xs,ys))
    def norm(xs): return math.sqrt(max(0., dot(xs,xs)))

    parameter_norm = norm([theta0[n] for n in pnames])
    candidate_norm = norm(candidate)
    old_result = json.loads((PREVIOUS/'result.json').read_text())
    for name in PARTS:
        assert math.isclose(dot(gradients[name], candidate),
            old_result['impacts'][name]['adam_actual_displacement_dot'], rel_tol=1e-10, abs_tol=1e-13)
    meta = json.loads((SOURCE/'cache/manifest.json').read_text())
    assert sha(SOURCE/'cache/manifest.json') == cp['cache_manifest_sha256']
    provenance = json.loads((SOURCE/'source_manifest.json').read_text())
    data = []
    for uid,row in zip(uids, meta['meshes']):
        assert uid == row['uid'] and uid in success|failed
        path = SOURCE/'cache'/f'{uid}.npz'
        assert sha(path) == row['sha256']
        with np.load(path) as q:
            x = T.from_numpy(q['block14_input'].copy()).cuda()
            gt = T.from_numpy(q['edges'].copy()).cuda()
        n = len(x)
        pairs = T.triu_indices(n,n,1,device='cuda').T
        keys = gt[:,0]*n+gt[:,1]
        query = pairs[:,0]*n+pairs[:,1]
        at = T.searchsorted(keys,query)
        labels = (at<len(keys)) & (keys[at.clamp_max(len(keys)-1)]==query)
        pool_path = SOURCE/'augmented_pools'/f'{uid}_pool.npz'
        assert sha(pool_path) == provenance['augmented_pool_sha256'][uid]
        with np.load(pool_path) as q: pool = {k:q[k].copy() for k in ['vertices','edges','positive','mixed']}
        tris = T.as_tensor(np.concatenate([pool['positive'],pool['mixed']]),device='cuda')
        yl = T.cat([T.ones(len(pool['positive']),device='cuda'),T.zeros(len(pool['mixed']),device='cuda')])
        data.append(dict(uid=uid,x=x,pairs=pairs,labels=labels,face_tris=tris,face_labels=yl,pool=pool))
    assert len(data) == 100 and sum(len(d['pairs']) for d in data) == 84669234
    sc, scales = core.scoring_module(), model.scoring_contract()
    expected = json.loads((SOURCE/'run/checkpoint-new0500-tail1500.json').read_text())['meshes']
    actual_ref = {r['uid']:r for r in json.loads((SOURCE/'run/final_real_network.json').read_text())['meshes']}
    T.set_rng_state(cp['rng'].cpu())
    T.cuda.set_rng_state_all([s.cpu() for s in cp['cuda_rng']])
    rng0, cuda_rng0 = T.get_rng_state().clone(), T.cuda.get_rng_state_all()
    run = ROOT/'points'; run.mkdir(exist_ok=True)
    manifest = dict(grid=GRID, parts=PARTS, per_mesh_coefficient=.01,
        source_checkpoint=str(cp_path), source_sha256=CP_HASH,
        source_full_model=str(full_path), source_full_sha256=FULL_HASH,
        saved_vectors=str(vector_path), saved_vectors_sha256=VECTORS_HASH,
        cache_manifest_sha256=sha(SOURCE/'cache/manifest.json'),
        pool_provenance_source=str(SOURCE/'source_manifest.json'),
        parameter_names=pnames, parameter_norm=parameter_norm, candidate_norm=candidate_norm,
        construction='CPU FP32: theta0 + FP32(lambda * saved_delta); save then reload; every point independent',
        displacement='Reloaded FP32 values minus theta0, subtraction and dot reductions in FP64',
        forward='Exact frozen block14-input cache; grad-enabled math00 dispatch; no backward; actual predicted-Edge Face enumeration complete',
        optimizer_constructions=0, optimizer_steps=0, backward_calls=0)
    write(ROOT/'manifest.json',manifest)
    print('READY: hashes checked, saved displacement reused, 100 full meshes; no optimizer',flush=True)
    records = []
    for value in GRID:
        pid = point_id(value)
        point = run/pid; point.mkdir(exist_ok=True)
        state_path = point/'tail-fp32.pt'
        state = {n:(theta0[n]+candidate[i]*value).float() for i,n in enumerate(pnames)}
        if state_path.exists():
            saved = T.load(state_path,map_location='cpu',weights_only=False)
            assert all(T.equal(state[n],saved[n]) for n in pnames)
        else:
            T.save(state,state_path)
            saved = T.load(state_path,map_location='cpu',weights_only=False)
        assert all(T.equal(state[n],saved[n]) for n in pnames)
        tail.load_state_dict(saved,strict=True)
        loaded = {n:t.detach().cpu() for n,t in tail.state_dict().items()}
        assert all(T.equal(loaded[n],saved[n]) for n in pnames)
        delta = [loaded[n].double()-theta0[n].double() for n in pnames]
        predictions = {name:dot(gradients[name],delta) for name in PARTS}
        dn = norm(delta)
        geometry = dict(norm=dn,relative_norm=dn/parameter_norm,
            changed_elements=sum(int(T.count_nonzero(d)) for d in delta),
            cosine_with_saved_candidate=dot(delta,candidate)/(dn*candidate_norm) if dn else None,
            relative_rounding_error=norm([d-value*q.double() for d,q in zip(delta,candidate)])/(abs(value)*candidate_norm) if value else 0.,
            predictions=predictions, original_joint_prediction=dot(vectors['original_total_gradient'],delta))
        # Resume only fully completed, hash-matched points; incomplete point recomputed.
        if (point/'result.json').exists():
            record=json.loads((point/'result.json').read_text())
            assert record['lambda'] == value and record['state_sha256'] == sha(state_path)
            assert record['geometry'] == geometry
            records.append(record)
            print('REUSE',pid,flush=True)
            continue
        point_start = time.monotonic()
        real_checks=[]
        for d in [data[0], data[-1]]:
            rows=forward(d['uid'])
            real_hidden=capture['hidden'].clone()
            h,ev,fv=joint.score(tail,d,meta['scale'],sc,scales)
            assert T.equal(h.detach(),real_hidden)
            assert T.equal(ev[1].detach(),rows[2][0]) and T.equal(fv[1].detach(),rows[3][0])
            real_checks.append(d['uid'])
            del h,ev,fv,rows,real_hidden
            capture.clear()
        scalar_rows, actual_rows = [],[]
        losses={name:0. for name in PARTS}
        with (point/'meshes.jsonl').open('w',buffering=1) as log:
            for index,d in enumerate(data):
                h,ev,fv=joint.score(tail,d,meta['scale'],sc,scales)
                row=core.metrics(d,ev); row['face_soft4']=float(fv[3].detach())
                # Bitwise repeat of this identical parameter point, including all training logits.
                h2,ev2,fv2=joint.score(tail,d,meta['scale'],sc,scales)
                assert all(T.equal(x.detach(),y.detach()) for x,y in [(h,h2),(ev[2],ev2[2]),(fv[2],fv2[2]),(ev[3],ev2[3]),(fv[3],fv2[3])])
                del h2,ev2,fv2
                base='success' if d['uid'] in success else 'failed'
                losses[base+'_edge'] += row['edge_soft4']/100
                losses[base+'_face'] += row['face_soft4']/100
                actual=evaluate.evaluate_mesh(((),(),(ev[1].detach(),),(fv[1].detach(),)),d['pool'],scales)
                assert actual['face']['complete']
                assert all(row[k] == actual['edge'][k] for k in ['tp','fp','fn','tn'])
                actual.update(uid=d['uid'],fixed_group=base,vertices=len(d['x']),edge_soft4=row['edge_soft4'],face_soft4=row['face_soft4'])
                if value == 0:
                    assert row == expected[index], (d['uid'],'scalar baseline')
                    for key in ['edge','face','gt_face_candidates','missing_gt_face_candidates','edge_perfect','face_perfect','joint_perfect','margins']:
                        assert actual[key] == actual_ref[d['uid']][key], (d['uid'],key)
                scalar_rows.append(row); actual_rows.append(actual)
                log.write(json.dumps(actual,allow_nan=False)+'\n')
                write(ROOT/'status.json',dict(stage='finite_forward',point=pid,lambda_value=value,completed_meshes=index+1,
                    completed_points=len(records),total_points=len(GRID),optimizer_steps=0,seconds=time.monotonic()-started))
                if (index+1)%20==0: print('POINT',pid,index+1,'/100',flush=True)
                del h,ev,fv; capture.clear()
        if value == 0: assert losses == old_result['component_losses']
        summaries={}
        for group in ['all','success','failed']:
            rs=[r for r in actual_rows if group=='all' or r['fixed_group']==group]
            summaries[group]=dict(meshes=len(rs),edge_perfect=sum(r['edge_perfect'] for r in rs),
                joint_perfect=sum(r['joint_perfect'] for r in rs),
                edge={k:sum(r['edge'][k] for r in rs) for k in ['tp','fp','fn','tn']},
                face={k:sum(r['face'][k] for r in rs) for k in ['tp','fp','fn','tn']},
                missing_gt_face_candidates=sum(r['missing_gt_face_candidates'] for r in rs))
        deltal={name:losses[name]-old_result['component_losses'][name] for name in PARTS}
        comparisons={name:dict(delta_loss=deltal[name],prediction=predictions[name],
            residual=deltal[name]-predictions[name],
            ratio=deltal[name]/predictions[name] if abs(predictions[name])>1e-12 else None) for name in PARTS}
        joint_set={r['uid'] for r in actual_rows if r['joint_perfect']}
        original_joint_set={u for u,r in actual_ref.items() if r['joint_perfect']}
        record=dict(point=pid,lambda_value=value,**{'lambda':value},state_sha256=sha(state_path),geometry=geometry,
            losses=losses,total_loss=sum(losses.values()),comparisons=comparisons,summary=summaries,
            joint_perfect_uids=sorted(joint_set),retained_source72=sorted(joint_set&original_joint_set),
            lost_source72=sorted(original_joint_set-joint_set),gained_over_source72=sorted(joint_set-original_joint_set),
            verification=dict(all100_forward_twice_bitwise=True,full_face_enumeration=True,
                cached_vs_real_network_bitwise_uids=real_checks,loaded_fp32_equal_saved=True,optimizer_steps=0),
            seconds=time.monotonic()-point_start)
        write(point/'result.json',record)
        records.append(record)
        print('DONE',pid,json.dumps(dict(losses=losses,summary=summaries['all'],comparisons=comparisons)),flush=True)
    tail.load_state_dict(theta0,strict=True)
    assert model_before == tensor_hash((n,t) for n,t in model.state_dict().items() if T.is_tensor(t))
    assert all(p.grad is None for p in model.parameters())
    assert T.equal(rng0,T.get_rng_state()) and all(T.equal(x,y) for x,y in zip(cuda_rng0,T.cuda.get_rng_state_all()))
    for path,expected in [(cp_path,CP_HASH),(full_path,FULL_HASH),(vector_path,VECTORS_HASH)]: assert sha(path)==expected
    secants=[]
    byvalue={r['lambda']:r for r in records}
    for size in [1/16,1/8,1/4,1/2,1.]:
        plus,minus=byvalue[size],byvalue[-size]
        for name in PARTS:
            observed=plus['losses'][name]-minus['losses'][name]
            predicted=plus['geometry']['predictions'][name]-minus['geometry']['predictions'][name]
            secants.append(dict(size=size,part=name,loss_difference=observed,actual_displacement_prediction=predicted,
                ratio=observed/predicted if abs(predicted)>1e-12 else None,
                relative_error=abs(observed-predicted)/abs(predicted) if abs(predicted)>1e-12 else None,
                nominal_centered_slope=observed/(2*size),autograd_saved_candidate_slope=dot(gradients[name],candidate)))
    write(ROOT/'result.json',dict(manifest=manifest,points=records,centered_secants=secants,
        verification=dict(source_files_unchanged=True,in_memory_model_restored=True,optimizer_constructions=0,
            optimizer_steps=0,backward_calls=0,all_grad_none=True,rng_unchanged=True),seconds=time.monotonic()-started))
    write(ROOT/'complete.json',dict(state='complete',points=len(records),mesh_evaluations=1100,
        result_sha256=sha(ROOT/'result.json'),optimizer_steps=0,seconds=time.monotonic()-started))
    print('COMPLETE',flush=True)


if __name__=='__main__':
    try: main()
    except Exception:
        (ROOT/'failure.txt').write_text(traceback.format_exc())
        raise
