"""Fit only the exported 804x32 raw Edge embeddings; never load the network."""
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
# This server's ONNX package uses legacy protobuf descriptors during Adam import.
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
import numpy as np
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def l2(value):
    return float(value.detach().double().norm())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'updates.jsonl').exists():
        raise FileExistsError('Use a new output directory; existing training is preserved.')

    torch.set_num_threads(4)
    torch.manual_seed(20260914)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(0)
    snap = args.snapshot
    spec = importlib.util.spec_from_file_location('snapshot_scoring', snap / 'effective_code/effective_loss_and_scoring.py')
    scoring = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scoring)
    contract = json.loads((snap / 'effective_code/runtime_contract.json').read_text())
    summary = json.loads((snap / 'summary.json').read_text())
    arrays = np.load(snap / 'representations_and_gradients.npz')
    edges = np.load(snap / 'edge_all_pairs.npz')
    initial = torch.from_numpy(arrays['edge_head_raw'].copy()).cuda()
    raw = torch.nn.Parameter(initial.clone())
    pairs = torch.from_numpy(edges['pairs'].copy()).cuda()
    labels = torch.from_numpy(edges['labels'].copy()).cuda()
    assert raw.shape == (804, 32) and raw.dtype == torch.float32
    assert pairs.shape == (322806, 2)
    assert torch.equal(pairs, torch.triu_indices(804, 804, 1, device='cuda').T)
    assert bool(edges['in_training'].all()) and int(labels.sum()) == 2406
    scale = contract['scales']['edge_logit_scale']
    assert contract['membership_detach'] is False and contract['soft4_epsilon'] == 1e-8
    assert contract['scales']['embedding_normalization'] == 'per_mesh_center_only'
    keys = (pairs[:, 0] * 804 + pairs[:, 1])[labels]
    checks = {0, 50, 100, 200, 500, 1000, 2000}

    def forward():
        centered = raw - raw.mean(dim=0, keepdim=True)
        logits = scoring.first_order_interval(centered[pairs[:, 0]], centered[pairs[:, 1]]) * scale
        numerator, mass = scoring.soft4_sums(logits, labels)
        parts = numerator / (mass + 1e-8)
        edge_loss = parts.mean()
        return centered, logits, edge_loss, parts

    def metrics(logits, edge_loss, parts, step):
        pred = logits > 0
        tp = int((pred & labels).sum())
        fp = int((pred & ~labels).sum())
        fn = int((~pred & labels).sum())
        tn = int((~pred & ~labels).sum())
        return dict(step=step, tp=tp, fp=fp, fn=fn, tn=tn,
                    f1=2*tp/(2*tp+fp+fn), perfect=(fp == 0 and fn == 0),
                    edge_soft4=float(edge_loss), objective=float(edge_loss/4),
                    soft_group_means=parts.tolist(),
                    min_margin_gt=float(logits[labels].min()),
                    min_margin_non_gt=float((-logits[~labels]).min()))

    source_files = ['representations_and_gradients.npz', 'edge_all_pairs.npz',
                    'summary.json', 'effective_code/runtime_contract.json',
                    'effective_code/effective_loss_and_scoring.py']
    hashes = {name: digest(snap / name) for name in source_files}
    manifest = dict(experiment='804-only free Edge embeddings', snapshot=str(snap),
                    checkpoint=summary['checkpoint'], checkpoint_sha256=summary['checkpoint_sha256'],
                    uid=summary['uid'], parameter_shape=[804,32], parameters=804*32,
                    initialization='edge_head_raw before centering; center once per forward',
                    source_sha256=hashes, script_sha256=digest(__file__),
                    objective='fully-differentiable Edge Soft4 / 4; no Face, no KL',
                    membership_detach=False, tau=1, epsilon=1e-8, threshold=0,
                    space_time_dims=[16,16], edge_logit_scale=scale, pairs=322806,
                    optimizer='fresh torch.optim.Adam', lr=1e-3, betas=[0.9,0.999],
                    adam_eps=1e-8, weight_decay=0, clip=1, maximum_updates=2000,
                    all_pairs_metrics='every update after optimizer.step', checks=sorted(checks),
                    torch=torch.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0), python=platform.python_version(),
                    dtype='float32', tf32=False, autocast=False,
                    protobuf_implementation=os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'],
                    deterministic_algorithms=True, upstream_network_loaded=False,
                    preflight_only=args.preflight_only)
    write(args.output / 'manifest.json', manifest)

    # Validate the raw-head initialization before any optimizer exists.
    centered, logits, edge_loss, parts = forward()
    baseline = metrics(logits.detach(), edge_loss.detach(), parts.detach(), 0)
    expected_center = torch.from_numpy(arrays['edge_embedding_scoring']).cuda()
    expected_logits = torch.from_numpy(edges['logits']).cuda()
    reference_loss, _ = scoring.soft4_loss(centered, keys, contract['edge_chunk'], scale)
    grad0 = torch.autograd.grad(edge_loss/4, raw)[0]
    with torch.no_grad():
        center1, logits1, loss1, parts1 = forward()
    c2, l2_tensor, loss2, p2 = forward()
    grad1 = torch.autograd.grad(loss2/4, raw)[0]
    expected_grad = torch.from_numpy(arrays['grad_edge_head_raw']).cuda()
    verification = dict(
        raw_initialization_exact=torch.equal(raw, initial),
        centered_exact=torch.equal(centered, expected_center),
        centered_max_abs=float((centered-expected_center).abs().max()),
        logits_exact=torch.equal(logits, expected_logits),
        logits_max_abs=float((logits-expected_logits).abs().max()),
        edge_loss=float(edge_loss), expected_edge_loss=summary['parts']['edge'],
        reference_full_pool_loss=float(reference_loss),
        loss_matches_runtime=torch.equal(edge_loss, reference_loss),
        repeated_forward_exact=torch.equal(logits, logits1) and torch.equal(edge_loss, loss1),
        repeated_backward_exact=torch.equal(grad0, grad1),
        exported_raw_gradient_relative_l2=l2(grad0-expected_grad)/max(l2(expected_grad),1e-30),
        baseline=baseline, optimizer_updates=0)
    write(args.output / 'baseline_verification.json', verification)
    assert verification['centered_exact'] and verification['logits_exact'], verification
    assert verification['loss_matches_runtime'] and float(edge_loss) == summary['parts']['edge'], verification
    assert verification['repeated_forward_exact'] and verification['repeated_backward_exact'], verification
    assert all(baseline[k] == summary['actual_reconstruction']['edge'][k] for k in ['tp','fp','fn','tn']), verification
    assert verification['exported_raw_gradient_relative_l2'] < 1e-4, verification
    print('BASELINE', json.dumps(verification), flush=True)
    del centered, logits, edge_loss, parts, reference_loss, grad0, grad1, c2, l2_tensor, loss2, p2
    if args.preflight_only:
        return

    optimizer = torch.optim.Adam([raw], lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0)
    assert len(optimizer.state) == 0
    history = [baseline]
    first_perfect = None
    streak = longest = perfect_count = 0
    best_errors = baseline['fp'] + baseline['fn']
    best_step = 0
    started = time.monotonic()

    def save(step, record, name):
        with torch.no_grad():
            centered, logits, _, _ = forward()
        torch.save(dict(completed_updates=step, edge_head_raw=raw.detach().cpu(),
                        edge_embedding_scoring=centered.cpu(), optimizer=optimizer.state_dict(),
                        metrics=record, manifest=manifest), args.output / (name + '.pt'))
        np.savez_compressed(args.output / (name + '.npz'),
                            edge_head_raw=raw.detach().cpu().numpy(),
                            edge_embedding_scoring=centered.cpu().numpy(),
                            pairs=edges['pairs'], labels=edges['labels'], logits=logits.cpu().numpy(),
                            local_vertex_id=arrays['local_vertex_id'], vertices=arrays['vertices'])
        write(args.output / (name + '.json'), record)

    save(0, baseline, 'checkpoint-step0000')
    with (args.output / 'updates.jsonl').open('x', buffering=1) as log:
        log.write(json.dumps(baseline, allow_nan=False) + '\n')
        for step in range(1, 2001):
            optimizer.zero_grad(set_to_none=True)
            centered, logits, edge_loss, parts = forward()
            (edge_loss/4).backward()
            grad_norm = l2(raw.grad)
            torch.nn.utils.clip_grad_norm_([raw], 1, error_if_nonfinite=True)
            before = raw.detach().clone()
            center_before = centered.detach().clone()
            objective_before = float((edge_loss/4).detach())
            optimizer.step()
            with torch.no_grad():
                center_after, logits_after, loss_after, parts_after = forward()
                record = metrics(logits_after, loss_after, parts_after, step)
                delta = raw - before
                record.update(objective_before_update=objective_before,
                              gradient_norm_preclip=grad_norm,
                              clip_coefficient=min(1., 1./(grad_norm+1e-6)),
                              raw_update_l2=l2(delta), raw_relative_update=l2(delta)/l2(before),
                              raw_changed_elements=int((delta != 0).sum()),
                              centered_update_l2=l2(center_after-center_before),
                              centered_relative_update=l2(center_after-center_before)/l2(center_before),
                              elapsed_seconds=time.monotonic()-started)
            history.append(record)
            log.write(json.dumps(record, allow_nan=False) + '\n')
            if record['perfect']:
                perfect_count += 1
                streak += 1
                longest = max(longest, streak)
                if first_perfect is None:
                    first_perfect = step
                    save(step, record, 'first-perfect')
                    print('FIRST_PERFECT', step, flush=True)
            else:
                streak = 0
            errors = record['fp'] + record['fn']
            if errors < best_errors:
                best_errors, best_step = errors, step
                # Compact best state; fixed checkpoints contain full pair logits.
                torch.save(dict(completed_updates=step, edge_head_raw=raw.detach().cpu(),
                                optimizer=optimizer.state_dict(), metrics=record), args.output / 'best.pt')
            if step in checks:
                save(step, record, f'checkpoint-step{step:04d}')
            if step == 1 or step % 50 == 0:
                print('UPDATE', step, json.dumps({k:record[k] for k in ['fp','fn','f1','edge_soft4','elapsed_seconds']}), flush=True)

    final_hashes = {name:digest(snap / name) for name in source_files}
    assert final_hashes == hashes
    result = dict(completed_updates=2000, baseline=baseline, final=history[-1],
                  first_perfect_step=first_perfect, perfect_updates=perfect_count,
                  longest_consecutive_perfect=longest, best_step=best_step,
                  minimum_total_errors=best_errors, source_files_unchanged=True,
                  every_update_nonzero=all(x['raw_changed_elements'] > 0 for x in history[1:]),
                  clipped_updates=sum(x['clip_coefficient'] < 1 for x in history[1:]),
                  wall_seconds=time.monotonic()-started)
    for size in [200,500]:
        tail=history[-size:]
        result[f'last_{size}'] = dict(perfect=sum(x['perfect'] for x in tail),
            f1_min=min(x['f1'] for x in tail), f1_median=float(np.median([x['f1'] for x in tail])),
            f1_max=max(x['f1'] for x in tail), fp_min=min(x['fp'] for x in tail),
            fp_max=max(x['fp'] for x in tail), fn_min=min(x['fn'] for x in tail), fn_max=max(x['fn'] for x in tail))
    write(args.output / 'complete.json', result)
    print('COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
