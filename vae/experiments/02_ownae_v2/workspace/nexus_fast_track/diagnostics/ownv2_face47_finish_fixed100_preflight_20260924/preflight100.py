"""Original fixed100 identity and largest complete mesh F/B; no optimizer."""
import csv
import sys
import time
import traceback
from pathlib import Path
from run_support import configure, read, write, sha, tensor_hash, move_item, torch, np
from native_models import NativeTopologyAE, Config, Graph
from data_objective import negative_faces, objective

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'fixed100_preflight'
SOURCE = Path('/guohaoran/nexus_fast_track/diagnostics/shared_edge_head_fixed100_20260917')


def main():
    OUT.mkdir(exist_ok=False)
    try:
        selection = read(SOURCE / 'selection.json')
        records = list(csv.DictReader((SOURCE / 'overfit100_manifest.csv').open()))
        assert [r['uid'] for r in records] == selection['uids']
        assert len(records) == len(set(selection['uids'])) == 100
        largest = max(records, key=lambda r: int(r['vertices']))
        totals = dict(vertices=0, edges=0, faces=0, pairs=0)
        verified = []
        item = None
        for r in records:
            for kind in ['mesh', 'topology']:
                assert sha(r[kind + '_path']) == r[kind + '_sha256']
            with np.load(r['mesh_path'], allow_pickle=False) as a, np.load(r['topology_path'], allow_pickle=False) as b:
                v = a['vertices_norm'].copy(); f = a['faces'].copy()
                gt = np.unique(np.sort(f, axis=1), axis=0)
                e = np.unique(np.sort(b['edge_index'].T, axis=1), axis=0)
                assert v.dtype == np.float32 and f.dtype == np.int64 and np.isfinite(v).all()
                assert f.min() >= 0 and f.max() < len(v) and len(gt) == len(f)
                assert np.array_equal(gt, np.unique(np.sort(b['face_set'], axis=1), axis=0))
                derived = np.unique(np.sort(np.concatenate([f[:,[0,1]], f[:,[0,2]], f[:,[1,2]]]), axis=1), axis=0)
                assert np.array_equal(e, derived)
                with np.load(r['pool_path'], allow_pickle=False) as pool:
                    assert sha(r['pool_path']) == r['pool_sha256']
                    assert np.array_equal(v, pool['vertices'])
                    assert np.array_equal(gt, np.unique(np.sort(pool['positive'], axis=1), axis=0))
                counts = dict(vertices=len(v), edges=len(e), faces=len(f), pairs=len(v)*(len(v)-1)//2)
                assert counts == dict(vertices=int(r['vertices']), edges=int(r['gt_edges']), faces=int(r['gt_faces']), pairs=int(r['edge_pairs']))
                for k, val in counts.items(): totals[k] += val
                verified.append(dict(uid=r['uid'], **counts, mesh_sha256=r['mesh_sha256'], topology_sha256=r['topology_sha256']))
                if r['uid'] == largest['uid']:
                    item = dict(uid=r['uid'], vertices=torch.from_numpy(v), faces=torch.from_numpy(f), gt_faces=torch.from_numpy(gt), edges=torch.from_numpy(e))
        assert totals['vertices'] == 106325 and totals['pairs'] == 84669234
        write(OUT / 'DATA_AUDIT.json', dict(passed=True, meshes=100, totals=totals, largest=largest,
            records=verified, source_manifest_sha256=sha(SOURCE/'overfit100_manifest.csv'),
            selection_sha256=sha(SOURCE/'selection.json'), new_quantization_or_reordering=False, optimizer_updates=0))
        print('DATA_AUDIT_PASSED', largest['uid'], totals, flush=True)
        configure(0); torch.cuda.set_device(0)
        model = NativeTopologyAE(Config(model_variant='B_v2_teacher_blocks')).cuda().float().train()
        initial_hash = tensor_hash(model.state_dict())
        n = len(item['vertices']); item['pairs'] = torch.triu_indices(n, n, 1).T.contiguous()
        keys = item['edges'].numpy() @ np.array([n,1])
        item['edge_labels'] = torch.from_numpy(np.isin(item['pairs'].numpy() @ np.array([n,1]), keys))
        neg, neg_hash = negative_faces(item, epoch=0)
        data = move_item(item, 'cuda'); graph = Graph.from_faces(data['faces'], n)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated(); start = time.monotonic()
        outputs = model(data['vertices'], data['faces'], sample_latent=False, graph=graph)
        torch.cuda.synchronize(); forward_seconds = time.monotonic()-start
        assert outputs['latent'] is outputs['mu']
        loss, parts = objective(outputs, data, neg, pair_chunk=32768, face_chunk=32768)
        assert torch.isfinite(loss)
        torch.cuda.synchronize(); objective_seconds = time.monotonic()-start-forward_seconds
        tic = time.monotonic(); (loss/5).backward(); torch.cuda.synchronize()
        backward_seconds = time.monotonic()-tic
        grad_finite = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        assert grad_finite and all(p.grad is None for p in model.log_variance.parameters())
        peak = torch.cuda.max_memory_allocated(); reserved = torch.cuda.max_memory_reserved()
        assert tensor_hash(model.state_dict()) == initial_hash
        parameter_bytes = sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)
        result = dict(passed=True, uid=item['uid'], vertices=n, graph_nodes=len(graph.degree),
            edge_pairs=len(item['pairs']), gt_faces=len(item['gt_faces']), random_negatives=len(neg),
            negative_sha256=neg_hash, forward_seconds=forward_seconds, objective_seconds=objective_seconds,
            backward_seconds=backward_seconds, total_seconds=forward_seconds+objective_seconds+backward_seconds,
            peak_allocated_bytes=peak, peak_reserved_bytes=reserved, before_forward_allocated_bytes=before,
            trainable_parameter_bytes=parameter_bytes, estimated_extra_adam_moments_bytes=2*parameter_bytes,
            all_trainable_gradients_finite=bool(grad_finite), model_hash_before_after=initial_hash,
            random_initialization=True, teacher_or_CAD_weights_loaded=False, optimizer_created=False,
            optimizer_updates=0, pair_chunk=32768, face_chunk=32768,
            numerical='FP32 SDPA MATH; TF32/autocast off; deterministic Graph; original per-block recompute',
            full_candidate_reconstruction_evaluation='not performed: this measures training F/B only',
            limitation='One largest mesh F/B, no optimizer step or five-mesh timing; peak excludes Adam moments and optimizer-step temporary buffers.',
            loss_components=parts, gpu=torch.cuda.get_device_name(), torch=torch.__version__)
        write(OUT/'RESULT.json', result); print('PREFLIGHT_COMPLETE', result, flush=True)
    except BaseException as e:
        write(OUT/'FAILURE.json', dict(error=str(e), traceback=traceback.format_exc(), optimizer_updates=0))
        raise


if __name__ == '__main__': main()
