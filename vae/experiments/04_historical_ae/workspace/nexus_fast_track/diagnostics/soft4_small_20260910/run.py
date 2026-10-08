"""Small-only Soft4 and missing controls; archived original run is reused."""
import os
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'encoder_teacher_probe_20260907'))
import train as teacher
probe = teacher.probe

OLD = ROOT.parent/'from_scratch_2x2_20260910'
CASES = ('soft4', 'fixed4', 'balanced')
GROUPS = ('tp', 'tn', 'fp', 'fn')
EPS = 1e-8
SEED = 20260910


def balanced_loss(z, keys, chunk, scale):
    """All unordered pairs, two fixed GT classes, no prediction-based grouping."""
    sums = torch.zeros(2, device=z.device, dtype=z.dtype)
    counts = torch.zeros(2, device=z.device, dtype=torch.long)
    for pair in teacher._all_pair_chunks(len(z), z.device, chunk):
        k = pair[:, 0]*len(z)+pair[:, 1]
        at = torch.searchsorted(keys, k)
        y = (at < len(keys)) & (keys[at.clamp_max(len(keys)-1)] == k)
        logits = probe.first_order_interval(z[pair[:, 0]], z[pair[:, 1]])*scale
        bce = F.binary_cross_entropy_with_logits(logits, y.to(logits), reduction='none')
        sums = sums+torch.stack([bce[y].sum(), bce[~y].sum()])
        counts = counts+torch.stack([y.sum(), (~y).sum()])
    return .5*(sums/counts).sum()


def soft4_sums(logits, labels):
    s = logits.float()
    y = labels.to(s)
    p = s.sigmoid().detach()  # Membership must not contribute to the gradient.
    weights = torch.stack([y*p, (1-y)*(1-p), (1-y)*p, y*(1-p)])
    assert not weights.requires_grad
    bce = F.binary_cross_entropy_with_logits(s, y, reduction='none')
    return (weights*bce).sum(dim=1, dtype=torch.float32), weights.sum(dim=1, dtype=torch.float32)


def soft4_loss(z, keys, chunk, scale):
    numerator = torch.zeros(4, device=z.device, dtype=torch.float32)
    mass = torch.zeros_like(numerator)
    for pair in teacher._all_pair_chunks(len(z), z.device, chunk):
        k = pair[:, 0]*len(z)+pair[:, 1]
        at = torch.searchsorted(keys, k)
        y = (at < len(keys)) & (keys[at.clamp_max(len(keys)-1)] == k)
        logits = probe.first_order_interval(z[pair[:, 0]], z[pair[:, 1]])*scale
        ns, ms = soft4_sums(logits, y)
        numerator = numerator+ns
        mass = mass+ms
    means = numerator/(mass+EPS)
    stats = {g: dict(mass=float(mass[j]), numerator=float(numerator[j].detach()), mean=float(means[j].detach()))
             for j,g in enumerate(GROUPS)}
    return means.mean(), stats


def validate_soft4():
    # Analytic stop-gradient derivative, including a threshold crossing and saturation.
    records = []
    for values in [[-3., -.01, 0., .01, 2., 5.], [-100., -30., -1., 1., 30., 100.]]:
        s = torch.tensor(values, device='cuda', requires_grad=True)
        y = torch.tensor([1., 1., 0., 1., 0., 0.], device='cuda')
        ns, ms = soft4_sums(s, y)
        actual = (ns/(ms+EPS)).mean()
        grad = torch.autograd.grad(actual, s, retain_graph=True)[0]
        p = s.detach().sigmoid()
        w = torch.stack([y*p, (1-y)*(1-p), (1-y)*p, y*(1-p)])
        expected_grad = .25*(w/(w.sum(1, keepdim=True)+EPS)).sum(0)*(p-y)
        torch.testing.assert_close(grad, expected_grad, rtol=1e-5, atol=1e-7)
        n1,m1 = soft4_sums(s[:2], y[:2]); n2,m2 = soft4_sums(s[2:], y[2:])
        split = ((n1+n2)/(m1+m2+EPS)).mean()
        torch.testing.assert_close(actual, split, rtol=1e-6, atol=1e-6)
        assert ns.dtype == ms.dtype == torch.float32
        assert not ms.requires_grad and torch.isfinite(grad).all()
        records.append(dict(value=float(actual.detach()), gradient_max_error=float((grad-expected_grad).abs().max())))
    # Check the complete all-pair path against dense formula and its z gradient.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(916)
        z = torch.randn(7, 32, device='cuda', requires_grad=True)
        pairs = torch.triu_indices(7, 7, 1, device='cuda').T
        edges = pairs[[0, 3, 8]]
        keys = teacher._canonical_positive_edge_keys(edges, 7, z.device)
        s = probe.first_order_interval(z[pairs[:,0]], z[pairs[:,1]])*.93
        y = torch.isin(pairs[:,0]*7+pairs[:,1], keys)
        ns, ms = soft4_sums(s,y)
        expected = (ns/(ms+EPS)).mean()
        actual,_ = soft4_loss(z, keys, 3, .93)
        torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
        ga = torch.autograd.grad(actual,z,retain_graph=True)[0]
        ge = torch.autograd.grad(expected,z)[0]
        torch.testing.assert_close(ga,ge,rtol=1e-5,atol=1e-6)
    print(json.dumps(dict(event='soft4_validation', checks=records, full_pair_value_gradient_passed=True)), flush=True)


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
    cp, model, batch = probe.setup_model(OLD/'initial.pt')
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
    assert not optimizer.state
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
        optimizer='fresh Adam, default betas/eps, wd=0, global clip_norm=1, no warmup',
        mode='mu, eval disables dropout; Face=0, KL=0, no data/sampling randomness',
        timing='step t: after t updates; both losses, F1 and mu from the same forward; gradients lead to update t+1',
        mu_drift='L2(mu_t-mu_(t-1))/(L2(mu_(t-1))+1e-12), separately for each full mesh',
        numerical_note='Original FP32/BF16 Flash and CUDA reductions; RNG unchanged does not imply bitwise determinism.',
        source_sha256=prep['source_sha256'], script_sha256=probe.digest(Path(__file__)))
    probe.write(out/'provenance.json', meta)
    previous_mu = None
    started = time.monotonic()
    with (out/'trace.jsonl').open('w', buffering=1) as log:
        for step in range(steps+1):
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
                        **c, edge_f1=2*tp/max(2*tp+fp+fn, 1), precision=tp/max(tp+fp, 1), recall=tp/max(tp+fn, 1),
                        mu_l2=norm(mu), mu_delta_l2=norm(delta) if delta is not None else None,
                        mu_relative_delta=norm(delta)/(norm(previous_mu[i])+1e-12) if delta is not None else None))
                loss = torch.stack(selected).mean()
            record = dict(step=step, objective=float(loss.detach()), rows=metrics)
            if step == 0 or step % 200 == 0 or step == steps:
                for i, uid in enumerate(probe.UIDS):
                    np.savez_compressed(out/f'step{step:04d}_{uid}.npz', mu=rows[0][i].detach().cpu().numpy(),
                        edge=rows[2][i].detach().cpu().numpy())
                if step > 0:
                    saved = dict(cp, model=model.state_dict(), diagnostic_run=dict(meta, completed_steps=step))
                    if step == steps:
                        saved['optimizer'] = optimizer.state_dict()
                    torch.save(saved, out/f'checkpoint-{step:04d}.pt')
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
            if step <= 5 or step % 100 == 0 or step == steps:
                print(json.dumps(dict(event='train', case=case, **record)), flush=True)
    assert probe.digest(OLD/'initial.pt') == prep['initial_sha256']
    assert probe.digest(OLD/'run.py') == prep['script_sha256']
    for path, digest in prep['source_sha256'].items():
        assert probe.digest(path) == digest
    probe.write(out/'complete.json', dict(steps=steps, seconds=time.monotonic()-started, final=record,
        frozen_and_rng_checks_passed=True, original_sources_unchanged=True))
    print(json.dumps(dict(event='complete', case=case, steps=steps)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--steps', type=int, default=1200)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.manual_seed(SEED); torch.cuda.set_device(args.gpu)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    validate_soft4()
    train(args.case, args.steps)
