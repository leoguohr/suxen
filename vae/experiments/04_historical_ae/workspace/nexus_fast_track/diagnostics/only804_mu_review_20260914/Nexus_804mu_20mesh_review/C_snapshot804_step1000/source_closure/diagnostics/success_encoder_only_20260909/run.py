"""Freeze the successful decoder and continue only its paired encoder."""
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


def emit(event, **values):
    print(json.dumps(dict(event=event, **values)), flush=True)


def main(steps):
    checkpoint = ROOT.parent/'encoder_teacher_probe_20260907/hard_edge_switch_merged_inference.pt'
    expected = '4d0eeca9e08e50108dcdd0243e3233afa4a121a6075da08c361537c88e10b922'
    assert probe.digest(checkpoint) == expected
    previous = json.loads((ROOT.parent/'success_e2e_resume_20260909/provenance.json').read_text())
    for path, expected_hash in previous['source_sha256'].items():
        assert probe.digest(path) == expected_hash
    cp, model, batch = probe.setup_model(checkpoint)
    model.eval()
    model.requires_grad_(False)
    a = model.autoencoder
    for module in [a.vertex_input, a.face_input, a.encoder_blocks, a.encoder_output_norm, a.mu]:
        module.requires_grad_(True)
    enc_prefixes = ('vertex_input.', 'face_input.', 'encoder_blocks.', 'encoder_output_norm.', 'mu.')
    dec_prefixes = ('latent_input.', 'decoder_blocks.', 'decoder_output_norm.', 'edge_embedding.')
    groups = {'encoder': [], 'decoder': []}
    for name, p in a.named_parameters():
        if name.startswith(enc_prefixes):
            groups['encoder'].append((name, p))
        elif name.startswith(dec_prefixes):
            groups['decoder'].append((name, p))
    params = [p for group in groups.values() for _, p in group if p.requires_grad]
    initial = {name: p.detach().clone() for name, p in a.named_parameters()}
    frozen = {name: p for name, p in a.named_parameters() if not p.requires_grad}
    assert all(not p.requires_grad for _, p in groups['decoder'])
    assert set(map(id, params)).isdisjoint(map(id, frozen.values()))
    with torch.no_grad():
        initial_mu = tuple(z.detach().clone() for z in probe.get_rows(model, batch, 'mu')[0])
    optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=0.)
    data = [np.load(teacher.modules.ab.PREVIOUS/(u+'_pool.npz')) for u in probe.UIDS]
    for i, d in enumerate(data):
        assert np.array_equal(d['vertices'], batch.vertices[i, :len(d['vertices'])].cpu().numpy())
        assert np.array_equal(d['positive'], batch.face_set[i].cpu().numpy())
    sc = model.scoring_contract()
    assert a.mu.out_features == 64 and a.edge_embedding.out_features == 32
    teacher.modules.ab.ROOT = ROOT
    source_files = [Path(__file__), teacher.MODULE_ROOT/'run.py',
                    teacher.modules.ab.PREVIOUS/'experiment.py',
                    *[probe.SOURCE/'mini_nexus'/f for f in
                      ['topology.py', 'training_2k.py', 'flash_varlen_topology.py']]]
    hashes = {str(p): probe.digest(p) for p in source_files}
    meta = dict(steps=steps, checkpoint=str(checkpoint), checkpoint_sha256=expected,
                model='archived 64 latent / 32 output / attention-only decoder / no XYZ',
                objective='original all-pair nonempty TP/TN/FP/FN group means, averaged over two meshes',
                original_forward_and_loss=True, encoder_trainable=True, decoder_trainable=False,
                face_head_frozen=True, log_variance_head_frozen=True,
                mode='mu', face_weight=0., kl_weight=0., dropout='disabled by eval mode',
                optimizer='fresh Adam, lr=1e-4, weight_decay=0, clip_norm=1, no warmup',
                optimizer_limitation='Merged success checkpoint has no unified optimizer state.',
                backend='original FP32 projections / BF16 FlashAttention; original nondeterministic reductions retained',
                seed=20260907, source_sha256=hashes,
                uids=probe.UIDS, scoring=sc,
                pool_sha256={u: probe.digest(teacher.modules.ab.PREVIOUS/(u+'_pool.npz')) for u in probe.UIDS},
                trainable_parameters={g: sum(p.numel() for _, p in ps if p.requires_grad) for g, ps in groups.items()},
                frozen_decoder_parameters=sum(p.numel() for _, p in groups['decoder']),
                comparison='Same successful checkpoint and original edge-only objective as success_e2e_resume_20260909 and success_decoder_only_20260909; decoder frozen',
                forward_policy='Original complete encoder-decoder forward each step; no latent substitution or new sampling')
    probe.write(ROOT/'provenance.json', meta)
    emit('start', **meta)
    started = time.monotonic()
    evaluations = []

    @torch.no_grad()
    def evaluate(step):
        result = teacher.modules.ab.evaluate_mu(model, batch, data, sc, 'encoder_only', step)
        result['parameter_delta_l2'] = {
            g: float(torch.stack([(p-initial[n]).square().sum() for n, p in ps]).sum().sqrt())
            for g, ps in groups.items()}
        assert all(torch.equal(p, initial[name]) for name, p in frozen.items())
        result['all_frozen_parameters_unchanged'] = True
        result['both_exact'] = all(r['edge']['fp'] == 0 and r['edge']['fn'] == 0 for r in result['rows'])
        evaluations.append(result)
        probe.write(ROOT/'evaluations.json', evaluations)
        return result

    for _ in range(2):
        baseline = evaluate(0)
        assert baseline['both_exact'], 'The loaded starting point is not perfect; do not train.'
    trace_path = ROOT/'trace.jsonl'
    assert not trace_path.exists(), 'Use a new output directory for another run.'
    schedule = {1, 2, 5, 10, 20, 50, 100, 200, 300, 400, steps}
    with trace_path.open('w', buffering=1) as log:
        for step in range(1, steps+1):
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
            rows = probe.get_rows(model, batch, 'mu')
            parts = [teacher.paper_edge_loss_all_pairs(z, batch.edge_index[i],
                     pair_chunk_size=cp['args']['pair_chunk_size'], counts_on_device=True,
                     logit_scale=sc['edge_logit_scale']) for i, z in enumerate(rows[2])]
            loss = torch.stack([v[0] for v in parts]).mean()
            loss.backward()
            assert all(p.grad is None for p in frozen.values())
            assert all(z.requires_grad for z in rows[0]+rows[2])
            gradient_norms = {g: float(sum((p.grad.square().sum() for _, p in ps if p.grad is not None),
                              torch.zeros((), device='cuda')).sqrt()) for g, ps in groups.items()}
            assert gradient_norms['decoder'] == 0.
            assert np.isfinite(gradient_norms['encoder'])
            if step == 1:
                assert gradient_norms['encoder'] > 0.
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step()
            assert all(torch.equal(p, initial[name]) for name, p in frozen.items())
            assert torch.equal(cpu_rng, torch.get_rng_state())
            assert torch.equal(gpu_rng, torch.cuda.get_rng_state())
            record = dict(step=step, metrics_timing='before this optimizer update', loss=float(loss.detach()),
                          parts=[float(v[0].detach()) for v in parts],
                          counts=[{k: int(v) for k, v in item[1].items()} for item in parts],
                          gradient_norm=float(norm), gradient_norms=gradient_norms,
                          mu_max_abs_difference_from_initial=[float((z-initial_mu[i]).abs().max()) for i, z in enumerate(rows[0])],
                          frozen_parameters_unchanged=True,
                          update_seconds=time.monotonic()-tick, seconds=time.monotonic()-started)
            log.write(json.dumps(record)+'\n')
            if step <= 5 or step % 25 == 0:
                emit('train', **record)
            if step in schedule:
                evaluate(step)
            if step == 1 or step % 200 == 0 or step == steps:
                out = dict(cp)
                out['model'] = model.state_dict()
                out['optimizer'] = optimizer.state_dict()
                out['diagnostic_intervention'] = dict(meta, completed_steps=step)
                torch.save(out, ROOT/f'checkpoint-{step:04d}.pt')
    assert {str(p): probe.digest(p) for p in source_files} == hashes
    assert probe.digest(checkpoint) == expected
    # Reload the final saved weights and repeat the same complete inference path.
    final_cp = torch.load(ROOT/f'checkpoint-{steps:04d}.pt', map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(final_cp['model'], strict=True)
    reload_result = evaluate(steps)
    completion = dict(steps=steps, seconds=time.monotonic()-started,
                      final_reload=reload_result, original_sources_and_checkpoint_unchanged=True,
                      rng_unchanged_all_updates=True, decoder_gradients_absent_all_updates=True,
                      all_frozen_parameters_unchanged_all_updates=True)
    probe.write(ROOT/'complete.json', completion)
    emit('complete', steps=steps, seconds=completion['seconds'], both_exact=reload_result['both_exact'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=500)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20260907)
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    main(args.steps)
