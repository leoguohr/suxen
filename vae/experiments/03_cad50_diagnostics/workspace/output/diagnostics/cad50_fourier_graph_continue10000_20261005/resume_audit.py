"""Exact full-state restoration, startup evaluation and Fourier execution gates."""
from dataclasses import replace
import hashlib
import numpy as np
import torch
from protocol import ADAM, PARENTS, ROOT, require, read, sha, write
from support import rng_state, restore_rng, tensor_hash
from native_models import Config
from data_objective import negative_faces, objective


def bits(value):
    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def equal_state(actual, expected, label):
    if torch.is_tensor(expected):
        require(torch.is_tensor(actual) and actual.dtype == expected.dtype and actual.shape == expected.shape
                and torch.equal(bits(actual), bits(expected)), f'{label}: tensor bits differ')
    elif isinstance(expected, np.ndarray):
        require(isinstance(actual, np.ndarray) and actual.dtype == expected.dtype
                and actual.shape == expected.shape and actual.tobytes() == expected.tobytes(), f'{label}: array bits differ')
    elif isinstance(expected, dict):
        require(actual.keys() == expected.keys(), f'{label}: state keys differ')
        for key in expected:
            equal_state(actual[key], expected[key], f'{label}.{key}')
    elif isinstance(expected, (tuple, list)):
        require(type(actual) is type(expected) and len(actual) == len(expected), f'{label}: sequence differs')
        for i, (a, b) in enumerate(zip(actual, expected)):
            equal_state(a, b, f'{label}.{i}')
    else:
        require(actual == expected, f'{label}: value differs')


def scientific_cfg(value):
    result = Config(**value).to_dict()  # Missing historical coordinate_encoding defaults to Fourier.
    result.pop('activation_checkpointing')
    return result


def validate_parent(cp, arm):
    cfg = cp['config']
    parent = PARENTS[arm]
    expected = Config(model_variant='B_v2_teacher_blocks',
        coordinate_encoding='fourier' if arm.startswith('Fourier') else 'xyz_only',
        graph_norm_position='pre_aggregation' if arm.endswith('_pre') else 'post_projection')
    require(cfg['arm'] == parent['arm'], 'Wrong parent arm')
    require(scientific_cfg(cfg['model_config']) == scientific_cfg(expected.to_dict()), 'Parent scientific model configuration differs')
    require(scientific_cfg(cp['model_config']) == scientific_cfg(expected.to_dict()), 'Parent saved model configuration differs')
    require(cp['model_config']['activation_checkpointing'] is arm.startswith('Fourier'), 'Unexpected parent activation execution mode')
    for key, value in dict(seed=0, lr=1e-4, betas=[.9, .999], eps=1e-8, weight_decay=.01,
            clip=1, sampling=False, KL=0, logvar_frozen=True, dropout=0, microbatch=1,
            meshes_per_update=5, negative_ratio=1.5, neighbor_projection_bias=False,
            trainable_parameters=246575680,
            objective='mean5(Edge Hard4 + Face Hard4); whole-mesh groups; empty0; internal /4',
            ordering='unchanged seed0 epoch/UID SHA256 -> PCG64').items():
        require(cfg[key] == value, f'Parent scientific recipe mismatch: {key}')
    provenance = read(ROOT/'SOURCE_PROVENANCE.json')
    original = provenance['old_control' if arm.startswith('Fourier') else 'factorial']['files']
    for name in ('native_models.py', 'data_objective.py', 'evaluation.py'):
        require(cfg['source_code'][name] == original[name], f'Parent effective source differs: {name}')
    require(len(cp['model']) == 415 and len(cp['optimizer_parameter_names']) == 412, 'Parent model/parameter entry count differs')
    require(cp['completed_updates'] == 2000 and cp['next_epoch'] == 200 and cp['next_batch'] == 0,
            'Parent is not the authorized 2000-update endpoint')
    require(set(cp['participation'].values()) == {200} and len(cp['participation']) == 50, 'Parent participation differs')
    require(set(cp['rng']) == {'python', 'numpy', 'torch', 'cuda'} and len(cp['rng']['cuda']) == 1, 'Incomplete parent RNG')
    require(cp['optimizer_parameter_names'] == cfg['optimizer_parameter_names'], 'Parent config parameter map differs')
    return expected


def validate_adam(state, names, model_state, done):
    require(type(names) is list and len(names) == len(set(names)) == 412, 'Expected flat unique 412-name optimizer map')
    require(len(state['param_groups']) == 1, 'Expected one AdamW group')
    group = state['param_groups'][0]
    require(set(group) == set(ADAM) | {'params'}, 'AdamW group schema differs')
    for key, expected in ADAM.items():
        equal_state(group[key], expected, f'AdamW.{key}')
    ids = group['params']
    require(len(ids) == len(set(ids)) == 412 and set(state['state']) == set(ids), 'Adam state/parameter mapping incomplete')
    for pid, name in zip(ids, names):
        values = state['state'][pid]
        require(set(values) == {'step', 'exp_avg', 'exp_avg_sq'}, f'Adam state schema differs: {name}')
        require(values['step'].numel() == 1 and values['step'].item() == done, f'Adam step differs: {name}')
        for key in ('exp_avg', 'exp_avg_sq'):
            value = values[key]
            require(value.dtype == model_state[name].dtype and value.shape == model_state[name].shape,
                    f'Adam tensor shape/dtype differs: {name}.{key}')
            require(bool(torch.isfinite(value).all()), f'Nonfinite Adam tensor: {name}.{key}')


def audit_restore(model, opt, cp, uids, epoch_batches, items):
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    require(len(model.state_dict()) == 415 and names == cp['optimizer_parameter_names'], 'Restored model/optimizer parameter names differ')
    require(sum(p.numel() for p in model.parameters() if p.requires_grad) == 246575680, 'Trainable parameter count differs')
    validate_adam(cp['optimizer'], names, cp['model'], cp['completed_updates'])
    equal_state(model.state_dict(), cp['model'], 'model')
    # Serialized ids must still refer to the same flat name order after AdamW load.
    equal_state(opt.state_dict(), cp['optimizer'], 'optimizer')
    equal_state(rng_state(), cp['rng'], 'rng.python_numpy_torch_cuda')
    epoch, cursor = cp['next_epoch'], cp['next_batch']
    next_uids = epoch_batches(uids, epoch)[cursor]
    return dict(passed=True, model_entries=415, trainable_parameter_names=names,
        trainable_parameter_count=412, model_bits_equal=True, Adam_all_tensors_and_groups_equal=True,
        Adam_all_steps=cp['completed_updates'], all_four_rng_states_equal=True,
        participation=cp['participation'], next_epoch=epoch, next_batch=cursor,
        next_uids=next_uids, next_negative_hashes={u: negative_faces(items[u], epoch)[1] for u in next_uids},
        model_hash=tensor_hash(model.state_dict()))


def compare_evaluations(actual, expected, uids):
    for value in (actual, expected):
        require(value['complete'] and [r['uid'] for r in value['meshes']] == uids, 'Incomplete/misordered CAD50 evaluation')
    for a, b in zip(actual['meshes'], expected['meshes']):
        for key in ('uid', 'vertices', 'edge', 'face', 'face_fn_missing', 'face_fn_present',
                    'actual_face_candidates', 'face_shards', 'joint_strict', 'complete'):
            require(a[key] == b[key], f'Endpoint evaluation mismatch: {a["uid"]}.{key}')
    for key in ('counts', 'joint_strict_uids', 'joint_strict', 'edge_strict', 'face_fn_missing', 'face_fn_present', 'large16'):
        require(actual[key] == expected[key], f'Endpoint aggregate mismatch: {key}')


def backward_batch(model, cpu_items, gpu_items, graphs, batch, epoch):
    parts, losses = [], []
    for uid in batch:
        item = gpu_items[uid]
        negative, digest = negative_faces(cpu_items[uid], epoch)
        outputs = model(item['vertices'], item['faces'], sample_latent=False, graph=graphs[uid])
        require(outputs['latent'] is outputs['mu'], 'Latent sampling unexpectedly enabled')
        loss, detail = objective(outputs, item, negative)
        require(bool(torch.isfinite(loss)), 'Nonfinite loss')
        losses.append(bits(loss).numpy().tobytes().hex())
        (loss/5).backward()
        parts.append(dict(uid=uid, negative_sha256=digest, **detail))
        del outputs, loss
    require(all(p.grad is None for p in model.log_variance.parameters()), 'Frozen logvar received a gradient')
    require(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                for p in model.parameters() if p.requires_grad), 'Disconnected/nonfinite trainable gradient')
    return parts, losses


def fourier_gate(model, opt, items, gpu_items, graphs, batch, epoch, path):
    """No clip/Adam step. Compare actual accumulated 5-mesh gradients as raw bytes."""
    saved_rng = rng_state()
    original_cfg = model.cfg
    cases = {}
    references = {}
    mismatches = []
    try:
        for recompute in (True, False):
            restore_rng(saved_rng)
            model.cfg = replace(original_cfg, activation_checkpointing=recompute)
            opt.zero_grad(set_to_none=True)
            parts, loss_bits = backward_batch(model, items, gpu_items, graphs, batch, epoch)
            gradients = {}
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                value = param.grad.detach().cpu().contiguous()
                gradients[name] = hashlib.sha256(bits(value).numpy().tobytes()).hexdigest()
                if recompute:
                    references[name] = value.clone()
                elif not torch.equal(bits(value), bits(references[name])):
                    mismatches.append(name)
            equal_state(rng_state(), saved_rng, 'Fourier gate RNG unchanged')
            cases['recompute' if recompute else 'retain'] = dict(loss_fp32_bits=loss_bits,
                loss_before=sum(p['edge_loss']+p['face_loss'] for p in parts)/5, gradients_sha256=gradients,
                negative_hashes={p['uid']: p['negative_sha256'] for p in parts})
        passed = not mismatches and cases['recompute']['loss_fp32_bits'] == cases['retain']['loss_fp32_bits']
        write(path, dict(passed=passed, optimizer_updates=0, epoch=epoch, uids=batch,
            all_412_gradient_raw_bytes_equal=not mismatches, mismatched_parameters=mismatches, cases=cases,
            scope='next complete five-mesh FP32 forward/backward accumulation; before clip/Adam'))
        require(passed, 'Fourier next-batch bitwise equivalence failed; stopping, no fallback')
    finally:
        model.cfg = original_cfg
        opt.zero_grad(set_to_none=True)
        restore_rng(saved_rng)


def xyz_gate(arm):
    path = ROOT/'evidence/XYZ_EQUIVALENCE.json'
    evidence = read(path)
    require(sha(path) == read(ROOT/'SOURCE_PROVENANCE.json')['xyz_equivalence_sha256'], 'XYZ evidence hash differs')
    require(evidence['all_gradients_bitwise_equal'] and evidence['optimizer_updates'] == 0, 'XYZ execution gate missing')
    require({row['batch'] for row in evidence['cases'] if row['arm'] == arm} == {'next_actual_batch', 'maximum_mesh_batch'},
            'XYZ evidence lacks required cases')
    return dict(passed=True, reused_evidence=str(path), sha256=sha(path),
        limitation='Historical loss/gradient equivalence; current checkpoint startup inference separately gated')
