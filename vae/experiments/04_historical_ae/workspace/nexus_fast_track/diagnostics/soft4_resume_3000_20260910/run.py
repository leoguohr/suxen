"""Resume the archived small-only Soft4 model and Adam at update 1200, unchanged to 3000."""
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
CASES = ('soft4',)
GROUPS = ('tp', 'tn', 'fp', 'fn')
EPS = 1e-8
SEED = 20260910


# Reuse the exact archived Soft4 and diagnostic losses.
import importlib.util
SOFT_ROOT = ROOT.parent/'soft4_small_20260910'
spec = importlib.util.spec_from_file_location('archived_soft4_training', SOFT_ROOT/'run.py')
archived = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archived)
balanced_loss, soft4_loss = archived.balanced_loss, archived.soft4_loss
START = SOFT_ROOT/'soft4/checkpoint-1200.pt'


def norm(x):
    return float(x.double().norm())


def train(case, steps):
    out = ROOT/case
    out.mkdir(exist_ok=True)
    assert not (out/'trace.jsonl').exists()
    prep = json.loads((OLD/'preparation.json').read_text())
    assert probe.digest(OLD/'initial.pt') == prep['initial_sha256']
    for path, digest in prep['source_sha256'].items():
        assert probe.digest(path) == digest
    assert probe.digest(OLD/'run.py') == prep['script_sha256']
    probe.UIDS = ['nexus_2k_000387']
    old_meta = json.loads((SOFT_ROOT/'soft4/provenance.json').read_text())
    assert probe.digest(SOFT_ROOT/'run.py') == old_meta['script_sha256']
    start_sha = probe.digest(START)
    cp, model, batch = probe.setup_model(START)
    assert cp['diagnostic_run']['completed_steps'] == 1200
    for key in ['objective', 'encoder_lr', 'decoder_lr', 'soft4_tau', 'soft4_epsilon', 'soft4_membership', 'soft4_group_reduction', 'mode', 'selected_uids']:
        assert cp['diagnostic_run'][key] == old_meta[key]
    assert EPS == archived.EPS == 1e-8
    assert len(batch.edge_index) == len(batch.vertices) == 1
    assert int(batch.vertex_mask[0].sum()) == 386
    model.eval(); model.requires_grad_(True)
    a = model.autoencoder
    a.log_variance.requires_grad_(False); a.face_embedding.requires_grad_(False)
    enc_prefix = ('vertex_input.', 'face_input.', 'encoder_blocks.', 'encoder_output_norm.', 'mu.')
    enc = [p for n, p in a.named_parameters() if p.requires_grad and n.startswith(enc_prefix)]
    dec = [p for n, p in a.named_parameters() if p.requires_grad and not n.startswith(enc_prefix)]
    params = enc+dec
    frozen = {n: p.detach().clone() for n, p in a.named_parameters() if not p.requires_grad}
    for n, p in model.named_parameters():
        assert torch.equal(p.cpu(), cp['model'][n]), n
    objective_name, encoder_lr = case, 1e-5
    optimizer = torch.optim.Adam([dict(params=enc, lr=encoder_lr), dict(params=dec, lr=1e-4)], weight_decay=0.)
    optimizer.load_state_dict(cp['optimizer'])
    assert [g['lr'] for g in optimizer.param_groups] == [1e-5, 1e-4]
    assert all(g['weight_decay'] == 0 and tuple(g['betas']) == (.9, .999) and g['eps'] == 1e-8 for g in optimizer.param_groups)
    assert len(optimizer.state) == len(params)
    for group, saved_group in zip(optimizer.param_groups, cp['optimizer']['param_groups']):
        assert len(group['params']) == len(saved_group['params'])
        for param, idx in zip(group['params'], saved_group['params']):
            state, saved_state = optimizer.state[param], cp['optimizer']['state'][idx]
            assert float(state['step']) == 1200
            for key in saved_state:
                assert torch.equal(state[key].cpu(), saved_state[key]), key
    scale = model.scoring_contract()['edge_logit_scale']
    keys = [teacher._canonical_positive_edge_keys(e, int(batch.vertex_mask[i].sum()), 'cuda')
            for i, e in enumerate(batch.edge_index)]
    for i, uid in enumerate(probe.UIDS):
        path = teacher.modules.ab.PREVIOUS/(uid+'_pool.npz')
        assert probe.digest(path) == prep['previous_configuration']['pool_sha256'][uid]
        d = np.load(path)
        assert np.array_equal(d['vertices'], batch.vertices[i, :len(d['vertices'])].cpu().numpy())
        assert np.array_equal(d['positive'], batch.face_set[i].cpu().numpy())
    chunk = cp['args']['pair_chunk_size']
    meta = dict(case=case, objective=objective_name, encoder_lr=encoder_lr, decoder_lr=1e-4,
        steps=steps, soft4_tau=1., soft4_epsilon=EPS, soft4_membership='sigmoid(logits).detach()',
        soft4_group_reduction='FP32; sums over all pairs before dividing; fixed divisor 4',
        threshold=0., selected_uids=probe.UIDS, initial_checkpoint=str(OLD/'initial.pt'),
        loss_reduction='one full mesh loss, no batch factor 1/2; previous two-mesh objective was their equal mean',
        initial_sha256=prep['initial_sha256'], seed=SEED, gpu=torch.cuda.current_device(),
        trainable_encoder_parameters=sum(p.numel() for p in enc), trainable_decoder_parameters=sum(p.numel() for p in dec),
        optimizer='restored Adam including step/moments and param groups; wd=0, clip=1, no warmup or scheduler',
        resume_checkpoint=str(START), resume_sha256=start_sha, completed_before_resume=1200, additional_updates=steps-1200,
        optimizer_state_restored_exactly=True, previous_script_sha256=old_meta['script_sha256'],
        mode='mu, eval disables dropout; Face=0, KL=0, no data/sampling randomness',
        timing='step t: after t updates; both losses, F1 and mu from the same forward; gradients lead to update t+1',
        mu_drift='L2(mu_t-mu_(t-1))/(L2(mu_(t-1))+1e-12), separately for each full mesh',
        numerical_note='Original FP32/BF16 Flash and CUDA reductions; RNG unchanged does not imply bitwise determinism.',
        source_sha256=prep['source_sha256'], script_sha256=probe.digest(Path(__file__)))
    probe.write(out/'provenance.json', meta)
    previous_mu = None
    first_perfect, perfect_count, streak, longest_streak = None, 0, 0, 0
    started = time.monotonic()
    with (out/'trace.jsonl').open('w', buffering=1) as log:
        for step in range(1200, steps+1):
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
                    with torch.set_grad_enabled(step < steps and objective_name == 'soft4'):
                        soft, soft_stats = soft4_loss(z, keys[i], chunk, scale)
                    fixed4 = four*(sum(int(counts[g]) > 0 for g in GROUPS)/4.)
                    selected.append({'soft4': soft, 'fixed4': fixed4, 'balanced': balanced}[objective_name])
                    c = {k: int(v) for k, v in counts.items()}
                    tp, fp, fn = c['tp'], c['fp'], c['fn']
                    mu = rows[0][i].detach()
                    delta = mu-previous_mu[i] if previous_mu is not None else None
                    metrics.append(dict(uid=uid, four_loss=float(four.detach()), balanced_loss=float(balanced.detach()),
                        fixed4_loss=float(fixed4.detach()), soft4_loss=float(soft.detach()), soft4_groups=soft_stats,
                        **c, is_perfect=(fp == 0 and fn == 0), edge_f1=2*tp/max(2*tp+fp+fn, 1), precision=tp/max(tp+fp, 1), recall=tp/max(tp+fn, 1),
                        mu_l2=norm(mu), mu_delta_l2=norm(delta) if delta is not None else None,
                        mu_relative_delta=norm(delta)/(norm(previous_mu[i])+1e-12) if delta is not None else None))
                loss = torch.stack(selected).mean()
            perfect = metrics[0]['is_perfect']
            newly_perfect = perfect and first_perfect is None
            if newly_perfect:
                first_perfect = step
            perfect_count += int(perfect)
            streak = streak+1 if perfect else 0
            longest_streak = max(longest_streak, streak)
            record = dict(step=step, objective=float(loss.detach()), rows=metrics, is_perfect=perfect,
                first_perfect_step=first_perfect, perfect_evaluations=perfect_count, perfect_streak=streak, longest_perfect_streak=longest_streak,
                encoder_lr=optimizer.param_groups[0]['lr'], decoder_lr=optimizer.param_groups[1]['lr'])
            if step == 1200:
                prior = np.load(SOFT_ROOT/'soft4/step1200_nexus_2k_000387.npz')
                old_final = json.loads((SOFT_ROOT/'soft4/complete.json').read_text())['final']
                replay = dict(old_final=old_final, resumed_before_update=record, same_model_and_optimizer=True,
                    relative_mu_l2=norm(rows[0][0].detach()-torch.as_tensor(prior['mu'],device='cuda'))/(norm(torch.as_tensor(prior['mu'],device='cuda'))+1e-12))
                probe.write(out/'resume_verification.json',replay)
            if step % 200 == 0 or step == steps or newly_perfect:
                for i, uid in enumerate(probe.UIDS):
                    np.savez_compressed(out/f'step{step:04d}_{uid}.npz', mu=rows[0][i].detach().cpu().numpy(),
                        edge=rows[2][i].detach().cpu().numpy())
                if step > 1200:
                    saved = dict(cp, model=model.state_dict(), diagnostic_run=dict(meta, completed_steps=step))
                    saved['optimizer'] = optimizer.state_dict()
                    saved['diagnostic_metrics'] = record
                    saved['rng_state'] = dict(cpu=cpu_rng, cuda=gpu_rng)
                    name = f'checkpoint-{step:04d}.pt' if step % 200 == 0 or step == steps else f'checkpoint-first-perfect-{step:04d}.pt'
                    torch.save(saved, out/name)
            previous_mu = tuple(mu.detach().clone() for mu in rows[0])
            if step < steps:
                loss.backward()
                record['next_update_encoder_grad_l2'] = float(torch.stack([p.grad.square().sum() for p in enc if p.grad is not None]).sum().sqrt())
                record['next_update_decoder_grad_l2'] = float(torch.stack([p.grad.square().sum() for p in dec if p.grad is not None]).sum().sqrt())
                record['next_update_preclip_grad_l2'] = float(torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True))
                optimizer.step()
            assert all(torch.equal(dict(a.named_parameters())[n], v) for n, v in frozen.items())
            assert all(dict(a.named_parameters())[n].grad is None for n in frozen)
            assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(gpu_rng, torch.cuda.get_rng_state())
            record.update(seconds=time.monotonic()-started, iteration_seconds=time.monotonic()-tick,
                          frozen_unchanged=True, rng_unchanged=True)
            log.write(json.dumps(record, allow_nan=False)+'\n')
            if step <= 1205 or step % 100 == 0 or step == steps or newly_perfect:
                print(json.dumps(dict(event='train', case=case, **record)), flush=True)
    assert probe.digest(OLD/'initial.pt') == prep['initial_sha256']
    assert probe.digest(OLD/'run.py') == prep['script_sha256']
    for path, digest in prep['source_sha256'].items():
        assert probe.digest(path) == digest
    assert probe.digest(START) == start_sha
    assert probe.digest(SOFT_ROOT/'run.py') == old_meta['script_sha256']
    assert all(float(v['step']) == steps for v in optimizer.state.values())
    probe.write(out/'complete.json', dict(first_perfect_step=first_perfect, perfect_evaluations=perfect_count, longest_perfect_streak=longest_streak, optimizer_final_step=steps, steps=steps, seconds=time.monotonic()-started, final=record,
        frozen_and_rng_checks_passed=True, original_sources_unchanged=True))
    print(json.dumps(dict(event='complete', case=case, steps=steps)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--steps', type=int, default=3000)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.manual_seed(SEED); torch.cuda.set_device(args.gpu)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    train(args.case, args.steps)
