"""Local four-part gradient/Adam diagnostic; never updates the source model."""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import copy, json, math, time, traceback, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/guohaoran/nexus_fast_track/diagnostics/decoder_last2_joint_fixed100_20260920')
PARTS = ['success_edge', 'success_face', 'failed_edge', 'failed_face']


def main():
    assert not (ROOT/'result.json').exists(), 'Completed diagnostic must not be overwritten'
    started = time.monotonic()
    for name in ['READY.json', 'overfit100_manifest.csv', 'data_manifest.csv',
                 'selection.json', 'pool_provenance.json', 'pools', 'construction_args.json']:
        target = (SOURCE/name).resolve()
        assert target.exists(), target
        link = ROOT/name
        if not link.exists():
            link.symlink_to(target)
    if not (ROOT/'effective_code').exists():
        shutil.copytree(SOURCE/'effective_code', ROOT/'effective_code')
    from runtime import T, np, b, c, setup, sha, write, tensor_hash
    import prior_core as core
    import joint_core as joint

    cp_path = SOURCE/'run/checkpoint-new0500-tail1500.pt'
    full_path = SOURCE/'run/model-last2-joint1000-tail1500-block14new500-inference.pt'
    cp_sha = sha(cp_path)
    assert cp_sha == '30d207d3508429a749885764712c169a85479e4c2673f94f816fa54cfa94e753'
    assert sha(full_path) == '82cf33fbc0151d72332218c5e447f7bf84e8934947e4c70148205c6a308db884'
    parent = json.loads((SOURCE/'parent_actual_baseline.json').read_text())
    success = {r['uid'] for r in parent['meshes'] if r['joint_perfect']}
    failed = {r['uid'] for r in parent['meshes'] if not r['joint_perfect']}
    assert len(success) == 71 and len(failed) == 29 and not success & failed
    write(ROOT/'partition.json', dict(source=str(SOURCE/'parent_actual_baseline.json'),
          source_sha256=sha(SOURCE/'parent_actual_baseline.json'),
          success71=sorted(success), failed29=sorted(failed), per_mesh_coefficient=0.01))
    model, unused, unused_groups, uids, _, forward, capture, args = setup()
    assert unused is None
    del unused_groups
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
    params = tuple(p for _, p in named)
    model_before = tensor_hash((n, p) for n, p in model.state_dict().items() if T.is_tensor(p))
    cp = T.load(cp_path, map_location='cpu', weights_only=False)
    assert all(T.equal(p.detach().cpu(), cp['tail'][n]) for n, p in tail.state_dict().items())
    assert cp['new_updates'] == 500 and cp['new_decoder14_updates'] == 500
    meta = json.loads((SOURCE/'cache/manifest.json').read_text())
    assert sha(SOURCE/'cache/manifest.json') == cp['cache_manifest_sha256']
    provenance = json.loads((SOURCE/'source_manifest.json').read_text())
    data = []
    for uid, row in zip(uids, meta['meshes']):
        assert uid == row['uid'] and uid in success | failed
        path = SOURCE/'cache'/f'{uid}.npz'
        assert sha(path) == row['sha256']
        with np.load(path) as q:
            x = T.from_numpy(q['block14_input'].copy()).cuda()
            gt = T.from_numpy(q['edges'].copy()).cuda()
        n = len(x)
        pairs = T.triu_indices(n, n, 1, device='cuda').T
        keys = gt[:, 0]*n + gt[:, 1]
        query = pairs[:, 0]*n + pairs[:, 1]
        at = T.searchsorted(keys, query)
        labels = (at < len(keys)) & (keys[at.clamp_max(len(keys)-1)] == query)
        path = SOURCE/'augmented_pools'/f'{uid}_pool.npz'
        assert sha(path) == provenance['augmented_pool_sha256'][uid]
        with np.load(path) as q:
            tris = T.as_tensor(np.concatenate([q['positive'], q['mixed']]), device='cuda')
            yl = T.cat([T.ones(len(q['positive']), device='cuda'), T.zeros(len(q['mixed']), device='cuda')])
        data.append(dict(uid=uid, x=x, pairs=pairs, labels=labels, face_tris=tris, face_labels=yl))
    assert len(data) == 100 and sum(len(d['pairs']) for d in data) == 84669234
    scales = model.scoring_contract()
    sc = core.scoring_module()
    expected = json.loads((SOURCE/'run/checkpoint-new0500-tail1500.json').read_text())
    # Reuse the already validated exact cache boundary, and check two real network forwards now.
    real_checks = []
    for d in [data[0], data[-1]]:
        # Keep the original grad-enabled dispatch: its checkpoint wrapper enters
        # the audited precision context even across frozen upstream blocks.
        rows = forward(d['uid'])
        h, ev, fv = joint.score(tail, d, meta['scale'], sc, scales)
        assert T.equal(h.detach(), capture['hidden'])
        assert T.equal(ev[1].detach(), rows[2][0]) and T.equal(fv[1].detach(), rows[3][0])
        real_checks.append(d['uid'])
        del rows, h, ev, fv
        capture.clear()
    T.set_rng_state(cp['rng'].cpu())
    T.cuda.set_rng_state_all([s.cpu() for s in cp['cuda_rng']])
    rng_before = T.get_rng_state().clone()
    cuda_rng_before = T.cuda.get_rng_state_all()
    pnames = [n for n, _ in named]
    modules = {k: [i for i, n in enumerate(pnames) if n.startswith(k+'.')] for k in tail}
    shared = [i for i,n in enumerate(pnames) if n.split('.')[0] in ['block','penultimate','final_norm']]
    scopes = {'all': list(range(len(params))), 'shared_decoder': shared, **modules}

    def dot(x, y, indices=None):
        indices = range(len(x)) if indices is None else indices
        return float(sum((x[i].double()*y[i].double()).sum() for i in indices))

    def norm(x, indices=None):
        return math.sqrt(max(0., dot(x, x, indices)))

    def four_gradients():
        acc = [[T.zeros_like(p) for p in params] for _ in PARTS]
        loss = [0.]*4
        rows = []
        for d in data:
            h, ev, fv = joint.score(tail, d, meta['scale'], sc, scales)
            row = core.metrics(d, ev)
            row['face_soft4'] = float(fv[3].detach())
            rows.append(row)
            base = 0 if d['uid'] in success else 2
            for j, scalar in enumerate([ev[3], fv[3]]):
                loss[base+j] += float(scalar.detach())/100
                grads = T.autograd.grad(scalar/100, params, retain_graph=(j == 0), allow_unused=True)
                for target, g in zip(acc[base+j], grads):
                    if g is not None:
                        target.add_(g)
            del h, ev, fv, grads
        assert rows == expected['meshes'], 'Current scalar/count baseline differs'
        return acc, loss, rows

    print('READY: exact checkpoint/cache/pool restored; fixed71/29; source optimizer untouched', flush=True)
    grads, loss, rows = four_gradients()
    repeated, loss2, _ = four_gradients()
    repeat = {p: dict(bitwise=all(T.equal(x,y) for x,y in zip(g,r)),
              relative_l2=norm([x-y for x,y in zip(g,r)])/max(norm(g),1e-300))
              for p,g,r in zip(PARTS,grads,repeated)}
    assert loss == loss2 and all(v['bitwise'] for v in repeat.values())
    del repeated
    # Original joint backward with the original per-mesh addition and accumulation order.
    tail.zero_grad(set_to_none=True)
    joint_rows = joint.cycle(tail, data, meta['scale'], sc, scales, True)
    assert joint_rows == rows
    total = [p.grad.detach().clone() for p in params]
    tail.zero_grad(set_to_none=True)
    summed = [sum(g[i] for g in grads) for i in range(len(params))]
    sum_relative = norm([x-y for x,y in zip(summed,total)])/norm(total)
    assert sum_relative < 1e-4, sum_relative
    del summed
    matrices = {}
    for scope, ids in scopes.items():
        norms = [norm(g, ids) for g in grads]
        gram = [[dot(g,h,ids) for h in grads] for g in grads]
        cosine = [[gram[i][j]/(norms[i]*norms[j]) if norms[i]*norms[j] else None
                   for j in range(4)] for i in range(4)]
        matrices[scope] = dict(norms=norms, dot=gram, cosine=cosine, total_gradient_norm=norm(total,ids))
    print('GRADIENTS', dict(zip(PARTS, matrices['all']['norms'])), flush=True)
    # Only independent cloned Parameters enter this temporary optimizer.
    copied = {n:T.nn.Parameter(p.detach().clone()) for n,p in named}
    group_names = [['block','final_norm'],['head'],['face'],['penultimate']]
    opt_groups = [dict(params=[copied[n] for n in pnames if n.split('.')[0] in keys], name=g['name'])
                  for keys,g in zip(group_names,cp['optimizer']['param_groups'])]
    opt = T.optim.Adam(opt_groups)
    opt.load_state_dict(copy.deepcopy(cp['optimizer']))
    assert all(id(p) not in {id(q) for q in params} for g in opt.param_groups for p in g['params'])
    original_states = {n: {k:v.detach().clone() if T.is_tensor(v) else v
                          for k,v in opt.state[p].items()} for n,p in copied.items()}
    group_config = [{k:v for k,v in g.items() if k != 'params'} for g in opt.param_groups]
    assert [g['lr'] for g in opt.param_groups] == [1e-5,1e-4,1e-4,1e-5]
    for n,g in zip(pnames,total):
        copied[n].grad = g.clone()
    clip_norm = T.nn.utils.clip_grad_norm_(list(copied.values()), 1., error_if_nonfinite=True)
    coefficient = min(1., 1./(float(clip_norm)+1e-6))
    clipped = [copied[n].grad.detach().clone() for n in pnames]
    # Fixed next-step denominator decomposition, not a changed optimizer or counterfactual run.
    history_direction = [T.zeros_like(p,dtype=T.float64) for p in params]
    current_direction = [T.zeros_like(p,dtype=T.float64) for p in params]
    factors = [None]*len(params)
    name_index = {id(p):i for i,p in enumerate(copied.values())}
    for group in opt.param_groups:
        beta1,beta2 = group['betas']
        assert not group['amsgrad'] and group['weight_decay'] == 0 and not group['maximize']
        for p in group['params']:
            i = name_index[id(p)]
            old = original_states[pnames[i]]
            step = int(old['step'])+1
            v = beta2*old['exp_avg_sq'].double()+(1-beta2)*clipped[i].double().square()
            factor = -group['lr']/((1-beta1**step)*(v.sqrt()/math.sqrt(1-beta2**step)+group['eps']))
            history_direction[i] = factor*beta1*old['exp_avg'].double()
            current_direction[i] = factor*(1-beta1)*clipped[i].double()
            factors[i] = factor*(1-beta1)*coefficient
    opt.step()  # A single hypothetical update of cloned tensors; source Parameters are never stepped.
    delta = [copied[n].detach()-p.detach() for n,p in named]
    nominal = [h+g for h,g in zip(history_direction,current_direction)]
    impacts = {}
    for name,g in zip(PARTS,grads):
        pred = dot(g,delta)
        impacts[name] = dict(loss=loss[PARTS.index(name)], gradient_norm=norm(g),
           adam_actual_displacement_dot=pred, adam_nominal_displacement_dot=dot(g,nominal),
           inherited_moment_dot=dot(g,history_direction), current_gradient_dot=dot(g,current_direction),
           relative_first_order_change=pred/loss[PARTS.index(name)],
           modules={k:dict(gradient_norm=norm(g,ids), adam_actual_displacement_dot=dot(g,delta,ids))
                    for k,ids in modules.items()})
    current_cross = [[sum(float((g[i].double()*factors[i]*h[i].double()).sum()) for i in range(len(params)))
                      for h in grads] for g in grads]
    parameter_norm = norm([p.detach() for p in params])
    adam = dict(groups=group_config, old_steps={n:int(v['step']) for n,v in original_states.items()},
       clip_norm=float(clip_norm), clip_coefficient=coefficient,
       parameter_norm=parameter_norm, actual_displacement_norm=norm(delta),
       actual_relative_displacement=norm(delta)/parameter_norm,
       cosine_with_negative_total_gradient=-dot(total,delta)/(norm(total)*norm(delta)),
       total_actual_first_order=dot(total,delta),
       summed_parts_actual_first_order=sum(x['adam_actual_displacement_dot'] for x in impacts.values()),
       current_component_cross_dot=current_cross,
       nominal_vs_fp32_displacement_relative=norm([d-n for d,n in zip(delta,nominal)])/norm(nominal))
    assert model_before == tensor_hash((n,p) for n,p in model.state_dict().items() if T.is_tensor(p))
    assert all(p.grad is None for p in model.parameters())
    assert T.equal(rng_before,T.get_rng_state()) and all(T.equal(x,y) for x,y in zip(cuda_rng_before,T.cuda.get_rng_state_all()))
    assert sha(cp_path) == cp_sha and sha(full_path) == '82cf33fbc0151d72332218c5e447f7bf84e8934947e4c70148205c6a308db884'
    T.save(dict(parameter_names=pnames, gradients={n:[x.cpu() for x in g] for n,g in zip(PARTS,grads)},
       original_total_gradient=[x.cpu() for x in total],
       clipped_total_gradient=[x.cpu() for x in clipped],
       cloned_adam_actual_fp32_displacement=[x.cpu() for x in delta]), ROOT/'gradient_vectors.pt')
    write(ROOT/'per_mesh_baseline.json',dict(meshes=rows, original_actual_metrics_source=str(SOURCE/'run/final_real_network.json')))
    result = dict(source_checkpoint=str(cp_path), source_sha256=cp_sha, source_full_model=str(full_path),
       source_full_sha256=sha(full_path), partition='fixed joint-success71/failed29 at common parent; never reassigned',
       parts=PARTS, scope='decoder14,decoder15,final LayerNorm,Edge head,Face head only',
       per_mesh_coefficient=0.01, component_losses=dict(zip(PARTS,loss)), total_loss=sum(loss),
       matrices=matrices, repeat_backward=repeat, sum_parts_vs_original_joint_relative_l2=sum_relative,
       impacts=impacts, adam_candidate=adam, seconds=time.monotonic()-started,
       verification=dict(original_model_parameters_buffers_unchanged=True,source_files_unchanged=True,
         source_optimizer_never_stepped=True,source_optimizer_updates=0,cloned_optimizer_steps=1,
         rng_unchanged=True,all100_scalar_and_edge_counts_match_endpoint=True,
         real_network_checked_uids=real_checks,cache_and_face_pool_hashes_checked=True,
         first_order_predictions_only=True,no_displaced_forward_or_hard_count_prediction=True))
    write(ROOT/'result.json',result)
    write(ROOT/'complete.json',dict(state='complete',source_optimizer_updates=0,seconds=result['seconds'],
                                   result_sha256=sha(ROOT/'result.json'),vectors_sha256=sha(ROOT/'gradient_vectors.pt')))
    print('COMPLETE',json.dumps(impacts),flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        (ROOT/'failure.txt').write_text(traceback.format_exc())
        raise
