"""Continue branch B step11100 and all Adam states for 1000 updates; all settings unchanged."""
import os
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'encoder_teacher_probe_20260907'))
import train as teacher
probe = teacher.probe

OLD = ROOT.parent/'from_scratch_2x2_20260910'
PREVIOUS = ROOT.parent/'soft4_edge_face_lr600_20260910'
PHASES = {'continue': (11100, 12100, 1e-8, 1e-7)}
FACE_EVAL_PERIOD = 50
GROUPS = ('tp', 'tn', 'fp', 'fn')
EPS = 1e-8
SEED = 20260910


# Use precisely the loss implementations validated in the small-only experiment.
import importlib.util
SOFT_ROOT = ROOT.parent/'soft4_small_20260910'
spec = importlib.util.spec_from_file_location('archived_small_soft4', SOFT_ROOT/'run.py')
archived = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archived)
balanced_loss, soft4_loss = archived.balanced_loss, archived.soft4_loss


def norm(x):
    return float(x.double().norm())


@torch.no_grad()
def full_reconstruction(rows,data,scales,step):
    results=[]
    for i,uid in enumerate(probe.UIDS):
        keys,edge,nc = probe.graph(rows[2][i],data[i]['edges'],scales)
        r=dict(uid=uid,edge=edge,candidate_count=nc,recovery_complete=keys is not None)
        if keys is not None:
            scores=probe.score_faces(rows[3][i],probe.triples(keys,len(rows[2][i])),scales)
            truth=np.isin(keys,probe.keys(data[i]['positive'],len(rows[2][i])))
            tp=int(((scores>0)&truth).sum());fp=int(((scores>0)&~truth).sum())
            fn=len(data[i]['positive'])-tp
            r['face']=dict(tp=tp,fp=fp,fn=fn,f1=2*tp/max(2*tp+fp+fn,1),
                           gt_faces_missing_from_edge_candidates=int(len(data[i]['positive'])-truth.sum()))
        r['joint_perfect']=r['recovery_complete'] and edge['fp']==edge['fn']==0 and r['face']['fp']==r['face']['fn']==0
        results.append(r)
    result=dict(step=step,rows=results,both_edge_and_face_perfect=all(r['joint_perfect'] for r in results))
    print(json.dumps(dict(event='full_reconstruction',**result)),flush=True)
    return result


def train(phase):
    start_step, steps, encoder_lr, decoder_lr = PHASES[phase]
    out = ROOT/phase
    out.mkdir(exist_ok=True)
    assert not (out/'trace.jsonl').exists()
    prep = json.loads((OLD/'preparation.json').read_text())
    assert probe.digest(OLD/'initial.pt') == prep['initial_sha256']
    for path, digest in prep['source_sha256'].items():
        assert probe.digest(path) == digest
    assert probe.digest(OLD/'run.py') == prep['script_sha256']
    small_meta = json.loads((SOFT_ROOT/'soft4/provenance.json').read_text())
    assert probe.digest(SOFT_ROOT/'run.py') == small_meta['script_sha256']
    assert small_meta['initial_sha256'] == prep['initial_sha256']
    assert EPS == archived.EPS == 1e-8
    probe.UIDS = ['nexus_2k_000387', 'nexus_2k_001849']
    start_path = PREVIOUS/'B/checkpoint-11100.pt'
    start_sha = probe.digest(start_path)
    assert start_sha == '64bf0d75765353c3bc03859b8dfa1ff241fd99aec71afda4870d70dac302a2a9'
    cp, model, batch = probe.setup_model(start_path)
    assert len(batch.edge_index) == len(batch.vertices) == 2
    assert [int(x.sum()) for x in batch.vertex_mask] == [386,2575]
    assert cp['diagnostic_run']['phase'] == 'B' and cp['diagnostic_run']['completed_steps'] == 11100
    assert cp['diagnostic_run']['script_sha256'] == probe.digest(PREVIOUS/'run.py')
    model.eval(); model.requires_grad_(True)
    a = model.autoencoder
    a.log_variance.requires_grad_(False)
    enc_prefix = ('vertex_input.', 'face_input.', 'encoder_blocks.', 'encoder_output_norm.', 'mu.')
    enc = [p for n, p in a.named_parameters() if p.requires_grad and n.startswith(enc_prefix)]
    dec = [p for n, p in a.named_parameters() if p.requires_grad and not n.startswith(enc_prefix) and not n.startswith('face_embedding.')]
    face_params = list(a.face_embedding.parameters())
    params = enc+dec+face_params
    frozen = {n: p.detach().clone() for n, p in a.named_parameters() if not p.requires_grad}
    for n, p in model.named_parameters():
        assert torch.equal(p.cpu(), cp['model'][n]), n
    objective_name = 'soft4'
    import mini_nexus.topology as topology
    optimizer = torch.optim.Adam([dict(params=enc, lr=encoder_lr), dict(params=dec, lr=decoder_lr),dict(params=face_params,lr=decoder_lr)], weight_decay=0.)
    assert not optimizer.state
    optimizer.load_state_dict(cp['optimizer'])
    assert len(optimizer.state) == len(params)
    for group_idx, (group, saved_group) in enumerate(zip(optimizer.param_groups, cp['optimizer']['param_groups'])):
        assert len(group['params']) == len(saved_group['params'])
        assert {k:v for k,v in group.items() if k != 'params'} == {k:v for k,v in saved_group.items() if k != 'params'}
        for param, idx in zip(group['params'], saved_group['params']):
            state, saved = optimizer.state[param], cp['optimizer']['state'][idx]
            assert float(state['step']) == [11100,11100,4600][group_idx]
            for k in saved:
                assert torch.equal(state[k].cpu(), saved[k]), k
    # All group fields, including LR and moments, remain exactly as saved.
    assert [g['lr'] for g in optimizer.param_groups] == [encoder_lr,decoder_lr,decoder_lr]
    assert all(g['weight_decay'] == 0 and tuple(g['betas']) == (.9,.999) and g['eps'] == 1e-8 for g in optimizer.param_groups)
    scale = model.scoring_contract()['edge_logit_scale']
    keys = [teacher._canonical_positive_edge_keys(e, int(batch.vertex_mask[i].sum()), 'cuda')
            for i, e in enumerate(batch.edge_index)]
    face_data, face_triplets, face_labels = [], [], []
    for i, uid in enumerate(probe.UIDS):
        path = teacher.modules.ab.PREVIOUS/(uid+'_pool.npz')
        assert probe.digest(path) == prep['previous_configuration']['pool_sha256'][uid]
        d = np.load(path)
        assert np.array_equal(d['vertices'], batch.vertices[i, :len(d['vertices'])].cpu().numpy())
        assert np.array_equal(d['positive'], batch.face_set[i].cpu().numpy())
        face_data.append(d)
        tri = torch.as_tensor(np.concatenate([d['positive'], d['mixed']]),device='cuda',dtype=torch.long)
        labels = torch.cat([torch.ones(len(d['positive']),device='cuda'),torch.zeros(len(d['mixed']),device='cuda')])
        assert not np.intersect1d(probe.keys(d['positive'],len(d['vertices'])),probe.keys(d['mixed'],len(d['vertices']))).size
        face_triplets.append(tri); face_labels.append(labels)

    chunk = cp['args']['pair_chunk_size']
    meta = dict(phase=phase, objective=objective_name, encoder_lr=encoder_lr, decoder_lr=decoder_lr,
        steps=steps, soft4_tau=1., soft4_epsilon=EPS, soft4_membership='sigmoid(logits).detach()',
        soft4_group_reduction='FP32; sums over all pairs before dividing; fixed divisor 4',
        threshold=0., selected_uids=probe.UIDS, initial_checkpoint=str(OLD/'initial.pt'),
        loss_reduction='mean_over_two_meshes(EdgeSoft4 + 1.0 * FaceSoft4)',
        face_weight=1., face_loss='same archived.soft4_sums as Edge; detached sigmoid tau1; FP32 sums; epsilon1e-8; fixed divisor4',
        face_negatives='same frozen mixed arrays as previous edge+face diagnostic; no resampling',
        face_head_parameters=sum(p.numel() for p in face_params),face_head_lr=decoder_lr,
        face_head_optimizer='restored face Adam step4600 and moments; unchanged LR',
        full_face_evaluation_period=FACE_EVAL_PERIOD,
        initial_sha256=prep['initial_sha256'], seed=SEED, gpu=torch.cuda.current_device(),
        trainable_encoder_parameters=sum(p.numel() for p in enc), trainable_decoder_parameters=sum(p.numel() for p in dec),
        optimizer='restored all three Adam groups exactly; old params step11100, face head step4600; unchanged LR',
        clip_norm=1., weight_decay=0., start_step=start_step, start_checkpoint=str(start_path), start_checkpoint_sha256=start_sha,
        loaded_weights_verified_tensor_equal=True, optimizer_restored_exactly=True, optimizer_fields_except_lr_unchanged=True,
        early_stop=False, updates=1000, perfect_counting='edge counts for updates11101..12100; complete edge+face recovery checked every50 updates',
        hardware=torch.cuda.get_device_name(), torch_version=torch.__version__, cuda_version=torch.version.cuda,
        strict_acceptance='both meshes FP=FN=0 at the same model parameters',
        loss_script_sha256=small_meta['script_sha256'],
        mode='mu, eval disables dropout; Face=1, KL=0, no data/sampling randomness',
        timing='step t: after t updates; both losses, F1 and mu from the same forward; gradients lead to update t+1',
        mu_drift='L2(mu_t-mu_(t-1))/(L2(mu_(t-1))+1e-12), separately for each full mesh',
        numerical_note='Original FP32/BF16 Flash and CUDA reductions; RNG unchanged does not imply bitwise determinism.',
        source_sha256=prep['source_sha256'], script_sha256=probe.digest(Path(__file__)))
    for k in ['soft4_tau', 'soft4_epsilon', 'soft4_membership', 'soft4_group_reduction', 'threshold']:
        assert meta[k] == small_meta[k], k
    for k in ['source_sha256', 'loss_script_sha256', 'soft4_tau', 'soft4_epsilon',
              'soft4_membership', 'soft4_group_reduction', 'threshold', 'clip_norm', 'weight_decay',
              'face_loss', 'face_weight', 'face_negatives', 'loss_reduction', 'mode', 'full_face_evaluation_period',
              'encoder_lr', 'decoder_lr', 'face_head_lr']:
        assert meta[k] == cp['diagnostic_run'][k], k
    probe.write(out/'provenance.json', meta)
    previous_mu = None
    first_full_perfect = None
    formula_checks = []
    full_evaluations = []
    first_both, both_count, streak, longest = None, 0, 0, 0
    perfect = {uid: dict(first_step=None, count=0, streak=0, longest=0) for uid in probe.UIDS}
    torch.set_rng_state(cp['rng_state']['cpu'])
    torch.cuda.set_rng_state(cp['rng_state']['cuda'])
    started = time.monotonic()
    with (out/'trace.jsonl').open('w', buffering=1) as log:
        for step in range(start_step, steps+1):
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
            with torch.set_grad_enabled(step < steps):
                rows = probe.get_rows(model, batch, 'mu')
                selected, metrics = [], []
                for i, uid in enumerate(probe.UIDS):
                    z = rows[2][i]
                    with torch.set_grad_enabled(step < steps and objective_name == 'fixed4'):
                        four, counts = teacher.paper_edge_loss_all_pairs(z, batch.edge_index[i],
                            pair_chunk_size=chunk, positive_keys=keys[i], counts_on_device=True, logit_scale=scale)
                    with torch.set_grad_enabled(step < steps and objective_name == 'balanced'):
                        balanced = balanced_loss(z, keys[i], chunk, scale)
                    with torch.set_grad_enabled(step < steps):
                        soft, soft_stats = soft4_loss(z, keys[i], chunk, scale)
                    fixed4 = four*(sum(int(counts[g]) > 0 for g in GROUPS)/4.)
                    tri = face_triplets[i]
                    f = rows[3][i]
                    logits = topology.face_interval_logits(f[tri[:,0]], f[tri[:,1]], f[tri[:,2]],
                        logit_scale=model.scoring_contract()['face_logit_scale'],area_factor=model.scoring_contract()['face_interval_factor'])
                    ns,ms = archived.soft4_sums(logits,face_labels[i])
                    assert ns.dtype == ms.dtype == torch.float32 and not ms.requires_grad
                    face_means = ns/(ms+EPS)
                    face_loss = face_means.mean()
                    face_soft_groups = {g:dict(mass=float(ms[j]),numerator=float(ns[j].detach()),mean=float(face_means[j].detach())) for j,g in enumerate(GROUPS)}
                    if step == start_step:
                        # Verify the requested stopped-membership derivative on the actual Face logits.
                        actual = torch.autograd.grad(face_loss,logits,retain_graph=True)[0]
                        y=face_labels[i];p=logits.detach().float().sigmoid()
                        w=torch.stack([y*p,(1-y)*(1-p),(1-y)*p,y*(1-p)])
                        expected=.25*(w/(w.sum(1,keepdim=True)+EPS)).sum(0)*(p-y)
                        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-8)
                        formula_checks.append(dict(uid=uid,gradient_max_absolute_error=float((actual-expected).abs().max()),
                            fp32_reduction=True,membership_detached=True,epsilon=EPS,tau=1.))
                        probe.write(out/'face_formula_verification.json',formula_checks)
                    with torch.no_grad():
                        face_hard,fc = topology.paper_balanced_binary_loss(logits,face_labels[i],counts_on_device=True)

                    selected.append(soft+face_loss)
                    face_counts = {k:int(v) for k,v in fc.items()}
                    with torch.no_grad():
                        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits,face_labels[i],reduction='none')
                        masks = {'tp':(face_labels[i]>0)&(logits>0),'tn':(face_labels[i]==0)&(logits<=0),
                                 'fp':(face_labels[i]==0)&(logits>0),'fn':(face_labels[i]>0)&(logits<=0)}
                        face_groups = {k:dict(count=int(v.sum()),mean_bce=float(bce[v].mean()) if v.any() else None) for k,v in masks.items()}
                        face_balanced = .5*(bce[face_labels[i]>0].mean()+bce[face_labels[i]==0].mean())
                    face_metric = dict(**face_counts,
                        f1=2*face_counts['tp']/max(2*face_counts['tp']+face_counts['fp']+face_counts['fn'],1),
                        loss=float(face_loss.detach()),hard_four_loss=float(face_hard),soft_groups=face_soft_groups,balanced_bce=float(face_balanced),groups=face_groups)

                    c = {k: int(v) for k, v in counts.items()}
                    tp, fp, fn = c['tp'], c['fp'], c['fn']
                    mu = rows[0][i].detach()
                    delta = mu-previous_mu[i] if previous_mu is not None else None
                    metrics.append(dict(uid=uid, four_loss=float(four.detach()), balanced_loss=float(balanced.detach()),
                        fixed4_loss=float(fixed4.detach()), soft4_loss=float(soft.detach()), soft4_groups=soft_stats,
                        face_training=face_metric, **c, is_perfect=(fp == 0 and fn == 0), edge_f1=2*tp/max(2*tp+fp+fn, 1), precision=tp/max(tp+fp, 1), recall=tp/max(tp+fn, 1),
                        mu_l2=norm(mu), mu_delta_l2=norm(delta) if delta is not None else None,
                        mu_relative_delta=norm(delta)/(norm(previous_mu[i])+1e-12) if delta is not None else None))
                assert len(selected) == 2
                loss = torch.stack(selected).mean()
                assert loss.requires_grad == (step < steps)
            both = all(r['is_perfect'] for r in metrics)
            newly_both = both and first_both is None and step > start_step
            newly_large = metrics[1]['is_perfect'] and perfect[probe.UIDS[1]]['first_step'] is None and step > start_step
            if step > start_step:
                if newly_both:
                    first_both = step
                both_count += int(both)
                streak = streak+1 if both else 0
                longest = max(longest, streak)
                for r in metrics:
                    stat = perfect[r['uid']]
                    if r['is_perfect'] and stat['first_step'] is None:
                        stat['first_step'] = step
                    stat['count'] += int(r['is_perfect'])
                    stat['streak'] = stat['streak']+1 if r['is_perfect'] else 0
                    stat['longest'] = max(stat['longest'], stat['streak'])
            record = dict(per_mesh_perfect={k:dict(v) for k,v in perfect.items()}, both_meshes_in_backward=True, step=step, phase=phase, objective=float(loss.detach()), rows=metrics, I_t=int(both),
                first_both_step=first_both, both_count=both_count, both_streak=streak, longest_both_streak=longest,
                encoder_lr=optimizer.param_groups[0]['lr'], decoder_lr=optimizer.param_groups[1]['lr'])
            if step == start_step:
                probe.write(out/'resume_verification.json', dict(model_and_adam_restored=True,
                    previous_step11100=cp['diagnostic_metrics'], replayed_step11100=record,
                    boundary_counting='1000 new updates11101..12100; starting11100 replay separately'))
            newly_full_perfect = False
            if step == start_step or (step-start_step) % FACE_EVAL_PERIOD == 0 or step == steps:
                full = full_reconstruction(rows,face_data,model.scoring_contract(),step)
                full_evaluations.append(full)
                probe.write(out/'full_reconstruction.json',full_evaluations)
                record['full_reconstruction'] = full
                if full['both_edge_and_face_perfect'] and first_full_perfect is None:
                    first_full_perfect = step
                    newly_full_perfect = True
                record['first_full_perfect_evaluation_step'] = first_full_perfect
            if step == start_step or (step-start_step) % 200 == 0 or step == steps or newly_full_perfect:

                for i, uid in enumerate(probe.UIDS):
                    np.savez_compressed(out/f'step{step:04d}_{uid}.npz', mu=rows[0][i].detach().cpu().numpy(),
                        edge=rows[2][i].detach().cpu().numpy(),face=rows[3][i].detach().cpu().numpy())
                if step > start_step:
                    saved = dict(cp, model=model.state_dict(), diagnostic_run=dict(meta, completed_steps=step))
                    saved['optimizer'] = optimizer.state_dict()
                    saved['diagnostic_metrics'] = record
                    saved['rng_state'] = dict(cpu=cpu_rng, cuda=gpu_rng)
                    if (step-start_step) % 200 == 0 or step == steps:
                        name = f'checkpoint-{step:04d}.pt'
                    else:
                        tag = 'first-edge-face-perfect'
                        name = f'checkpoint-{tag}-{step:04d}.pt'
                    record['saved_checkpoint'] = str(out/name)
                    torch.save(saved, out/name)
            previous_mu = tuple(mu.detach().clone() for mu in rows[0])
            if step < steps:
                loss.backward()
                record['next_update_encoder_grad_l2'] = float(torch.stack([p.grad.square().sum() for p in enc if p.grad is not None]).sum().sqrt())
                record['next_update_face_head_grad_l2'] = float(torch.stack([p.grad.square().sum() for p in face_params if p.grad is not None]).sum().sqrt())
                record['next_update_decoder_grad_l2'] = float(torch.stack([p.grad.square().sum() for p in dec if p.grad is not None]).sum().sqrt())
                record['next_update_preclip_grad_l2'] = float(torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True))
                optimizer.step()
            assert all(torch.equal(dict(a.named_parameters())[n], v) for n, v in frozen.items())
            assert all(dict(a.named_parameters())[n].grad is None for n in frozen)
            assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(gpu_rng, torch.cuda.get_rng_state())
            record.update(seconds=time.monotonic()-started, iteration_seconds=time.monotonic()-tick,
                          frozen_unchanged=True, rng_unchanged=True)
            log.write(json.dumps(record, allow_nan=False)+'\n')
            if step <= start_step+5 or (step-start_step) % 100 == 0 or step == steps:
                print(json.dumps(dict(event='train', **record)), flush=True)
    assert probe.digest(OLD/'initial.pt') == prep['initial_sha256']
    assert probe.digest(OLD/'run.py') == prep['script_sha256']
    for path, digest in prep['source_sha256'].items():
        assert probe.digest(path) == digest
    assert probe.digest(SOFT_ROOT/'run.py') == small_meta['script_sha256']
    assert probe.digest(start_path) == start_sha
    assert all(float(optimizer.state[p]['step']) == step for p in enc+dec)
    assert all(float(optimizer.state[p]['step']) == step-6500 for p in face_params)
    assert step == steps
    probe.write(out/'complete.json', dict(first_both_step=first_both, both_count=both_count, longest_both_streak=longest, optimizer_final_step=step, steps=step, max_steps=steps,
        additional_updates=step-start_step, early_stop=False, per_mesh_perfect=perfect, stop_reason='1000 additional updates with Edge and Face Soft4 completed', first_full_perfect_evaluation_step=first_full_perfect, face_head_optimizer_step=step-6500, full_reconstruction_final=full_evaluations[-1],
        seconds=time.monotonic()-started, final=record,
        frozen_and_rng_checks_passed=True, original_sources_unchanged=True))
    print(json.dumps(dict(event='complete', phase=phase, steps=step)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=PHASES, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.manual_seed(SEED); torch.cuda.set_device(args.gpu)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    train(args.phase)
