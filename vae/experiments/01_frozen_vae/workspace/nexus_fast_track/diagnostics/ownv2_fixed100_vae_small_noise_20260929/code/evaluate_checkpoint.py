"""Real fixed100 network evaluation with resumable, sharded actual Face candidates."""
import time
import numpy as np
from pathlib import Path
from run_support import torch, sha, write, read, rng_state, restore_rng, tensor_hash, move_item
from native_models import Graph
from data_objective import materialize_item, edge_logits, face_logits
from stream_faces import stream_faces, aggregate, atomic_npz
from vae_protocol import posterior_stats, uid_noise_seed, evaluation_generator


@torch.no_grad()
def evaluate(model, items, data_manifest, entry, run, budget, sample_seed=None, device='cuda'):
    assert sha(entry['path']) == entry['sha256']
    assert tensor_hash(model.state_dict()) == entry['model_state_sha256']
    group = 'mu' if sample_seed is None else f'noise-{sample_seed}'
    folder = Path(run)/'evaluations'/f"update-{entry['completed_updates']:08d}"/group
    folder.mkdir(parents=True, exist_ok=True)
    identity = dict(checkpoint_sha256=entry['sha256'], manifest_sha256=data_manifest['manifest_sha256'],
                    group=group, evaluation_seed=sample_seed)
    complete_file = folder/'complete.json'
    if complete_file.exists():
        result = read(complete_file); assert result['identity'] == identity
        return result
    mode = model.training; state = rng_state(); model.eval(); started = time.monotonic()
    meshes = []
    try:
        for uid in data_manifest['uids']:
            mesh_dir = folder/uid; mesh_dir.mkdir(exist_ok=True)
            binding = dict(identity, uid=uid)
            result_path = mesh_dir/'result.json'
            if result_path.exists():
                m = read(result_path); assert m['identity'] == binding and m['complete']
                meshes.append(m); continue
            if budget.stop(): break
            base_path, base_meta = mesh_dir/'network-output.npz', mesh_dir/'network-output.json'
            if not base_meta.exists():
                item = materialize_item(items[uid]); device_item = move_item(item, device)
                graph = Graph.from_faces(device_item['faces'], len(item['vertices']))
                generator = None
                if sample_seed is not None:
                    generator = evaluation_generator(device, sample_seed, uid)
                rows = model(device_item['vertices'], device_item['faces'], sample_latent=generator is not None,
                             graph=graph, generator=generator)
                assert (sample_seed is not None or rows['latent'] is rows['mu'])
                assert all(torch.isfinite(v).all() for v in rows.values())
                stats = posterior_stats(rows)
                stats['evaluation_seed'] = sample_seed
                stats['uid_noise_seed'] = None if sample_seed is None else uid_noise_seed(sample_seed, uid)
                pairs = item['pairs'].numpy(); scores = np.empty(len(pairs), dtype=np.float32)
                for start in range(0, len(pairs), 32768):
                    if budget.stop(): return incomplete(folder, identity, meshes, entry)
                    scores[start:start+32768] = edge_logits(rows['edge'], device_item['pairs'][start:start+32768]).cpu().numpy()
                gt_scores = face_logits(rows['face'], device_item['gt_faces']).cpu().numpy()
                assert np.isfinite(scores).all() and np.isfinite(gt_scores).all()
                labels = item['edge_labels'].numpy(); predicted = scores > 0
                edge = dict(tp=int((predicted & labels).sum()), fp=int((predicted & ~labels).sum()),
                            fn=int((~predicted & labels).sum()), tn=int((~predicted & ~labels).sum()))
                atomic_npz(base_path, vertices=item['vertices'].numpy(), gt_faces=item['gt_faces'].numpy(),
                    gt_edges=item['edges'].numpy(), local_vertex_indices=np.arange(len(item['vertices'])),
                    all_edge_pairs=pairs.astype(np.int32), edge_logits=scores, edge_labels=labels,
                    predicted_edges=pairs[predicted].astype(np.int32), face_embedding=rows['face'].cpu().numpy(),
                    gt_face_logits=gt_scores)
                write(base_meta, dict(identity=binding, sha256=sha(base_path), edge=edge, posterior=stats,
                    native_forward=True, frozen_checkpoint_cache=True,
                    note='Inference-only cache from this immutable checkpoint; never used for training'))
                del rows, device_item, graph, item
            meta = read(base_meta); assert meta['identity'] == binding and sha(base_path) == meta['sha256']
            with np.load(base_path, allow_pickle=False) as a:
                n = len(a['vertices']); adj = np.zeros((n, n), dtype=bool)
                e = a['predicted_edges']; adj[e[:, 0], e[:, 1]] = True
                gt = a['gt_faces'].copy(); gt_scores = a['gt_face_logits'].copy()
                embedding = torch.from_numpy(a['face_embedding'].copy()).to(device)
            def score(ids):
                return face_logits(embedding, torch.as_tensor(ids, dtype=torch.long, device=device)).cpu().numpy()
            fs = stream_faces(adj, gt, score, mesh_dir/'face_shards', binding, budget.stop)
            del embedding
            if not fs['complete']: break
            covered = adj[gt[:,0],gt[:,1]] & adj[gt[:,0],gt[:,2]] & adj[gt[:,1],gt[:,2]]
            assert fs['tp'] == int(((gt_scores > 0) & covered).sum())
            assert fs['fn_present'] == int(((gt_scores <= 0) & covered).sum())
            face = {k:fs[k] for k in ('tp','fp','fn','tn')}
            m = dict(uid=uid, vertices=n, edge=meta['edge'], face=face, posterior=meta['posterior'],
                face_fn_missing=fs['fn_missing'], face_fn_present=fs['fn_present'],
                actual_face_candidates=fs['candidates'], face_shards=fs['shards'],
                identity=binding, complete=True, native_forward=True,
                prediction_directory=str(mesh_dir))
            write(result_path, m); meshes.append(m)
            write(folder/'progress.json', dict(identity=identity, complete=False,
                evaluated_uids=[x['uid'] for x in meshes], required=100))
        if len(meshes) != 100: return incomplete(folder, identity, meshes, entry)
        assert [m['uid'] for m in meshes] == data_manifest['uids']
        assert tensor_hash(model.state_dict()) == entry['model_state_sha256']
        result = dict(identity=identity, checkpoint=entry, complete=True, native_forward=True,
            optimizer_updates_during_evaluation=0, meshes=meshes, **aggregate(meshes),
            posterior_mean={k:sum(m['posterior'][k] for m in meshes)/100 for k in
                ('kl','k_mu','k_sigma','sigma_mean','sigma_rms','sigma_p01','sigma_p50','sigma_p99',
                 'logvar_mean','clamp_low_fraction','clamp_high_fraction','clamp_fraction',
                 'perturbation_rms','perturbation_max')},
            elapsed_this_attempt_seconds=time.monotonic()-started)
        write(complete_file, result)
        if (folder/'incomplete.json').exists(): (folder/'incomplete.json').unlink()
        for label, value in [(group+'_face_f1',result['face']['micro_f1']),(group+'_strict',result['joint_perfect'])]:
            path = Path(run)/f'best_{label}.json'
            if not path.exists() or value > read(path)['value']:
                write(path, dict(value=value, checkpoint=entry, evaluation=str(complete_file)))
        return result
    finally:
        model.train(mode); restore_rng(state)


def incomplete(folder, identity, meshes, entry):
    result = dict(identity=identity, checkpoint=entry, complete=False,
                  evaluated_uids=[m['uid'] for m in meshes], required=100,
                  full100_metrics=None, reason='evaluation interrupted; resume from committed shards')
    write(folder/'incomplete.json', result)
    return result
