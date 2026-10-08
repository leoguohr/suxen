"""Explicit, resumable export of the fixed user-selected 50 meshes."""
import argparse
import fcntl
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from _faces_reference import atomic_json, atomic_npz, file_sha
from _fixed100_reference import array_sha
from latent_data import statistics_from_moments, validate_arrays, normalization_sha256
from selected_data import load_selected_dataset, validate_conditions
from topology_flow import math_context, configure_math_backend
from vae_codec import load_frozen_vae, encode_posterior, decode_latents, SOURCE_CHECKPOINT_SHA256


@torch.no_grad()
def export_item(model, item, points, point_mask, roundtrip_path=None):
    device = next(model.parameters()).device
    vertices, faces = item['vertices'].to(device), item['faces'].to(device)
    with math_context(device.type):
        posterior = encode_posterior(model, vertices, faces)
        decoded = decode_latents(model, posterior['mu'])
        original = model(vertices, faces, sample_latent=False)
    errors = {}
    for key in ('edge', 'face', 'decoder_hidden'):
        torch.testing.assert_close(decoded[key], original[key], atol=0, rtol=0)
        errors[key] = float((decoded[key]-original[key]).abs().max())
    arrays = dict(vertices=item['vertices'].numpy(), faces=item['faces'].numpy(), edges=item['edges'].numpy(),
                  vertex_indices=np.arange(len(vertices), dtype=np.int64), points=points, point_mask=point_mask,
                  **{k:v.cpu().numpy() for k,v in posterior.items()})
    validate_arrays(arrays)
    if roundtrip_path is not None:
        atomic_npz(roundtrip_path, **arrays)
        with np.load(roundtrip_path, allow_pickle=False) as saved:
            reread = {k:saved[k].copy() for k in saved.files}
        if set(reread) != set(arrays) or any(array_sha(v) != array_sha(reread[k]) for k,v in arrays.items()):
            raise ValueError('Export save/readback changed cached arrays')
        with math_context(device.type):
            recovered = decode_latents(model, torch.from_numpy(reread['mu']).to(device))
        for key in ('edge', 'face', 'decoder_hidden'):
            torch.testing.assert_close(recovered[key], decoded[key], atol=0, rtol=0)
            errors['readback_'+key] = float((recovered[key]-decoded[key]).abs().max())
    return arrays, errors


def check_saved_record(root, record):
    path = (root/record['file']).resolve()
    if not path.is_relative_to(root.resolve()) or file_sha(path) != record['sha256']:
        raise ValueError('Existing cache record path/hash mismatch')
    with np.load(path, allow_pickle=False) as f:
        arrays = {k:f[k].copy() for k in f.files}
    validate_arrays(arrays)
    if {k:array_sha(v) for k,v in arrays.items()} != record['array_sha256']:
        raise ValueError('Existing cache array hash mismatch')
    mu, lv = torch.from_numpy(arrays['mu']).double(), torch.from_numpy(arrays['logvar']).double()
    return mu.mean(0), (mu.square()+lv.exp()).mean(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-source', required=True)
    parser.add_argument('--data-source-sha256', help='Required for an explicit source JSON manifest')
    parser.add_argument('--conditions', required=True, help='Manifest binding real XYZ+normals and coordinate audit')
    parser.add_argument('--vae-checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--budget-file', help='Shared GPU budget ledger; required on CUDA')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    items, data = load_selected_dataset(args.data_source, args.data_source_sha256)
    condition_path = Path(args.conditions).resolve()
    conditions, audit = validate_conditions(condition_path, items, data['uids'])
    records = {r['uid']:r for r in conditions['records']}
    if file_sha(args.vae_checkpoint) != SOURCE_CHECKPOINT_SHA256:
        raise ValueError('VAE checkpoint differs from frozen OwnAE-v2 step36220')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    with (out/'.export.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = dict(source_checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,
            source_checkpoint=str(Path(args.vae_checkpoint).resolve()), data=data, uids=data['uids'],
            conditions_sha256=file_sha(condition_path), condition_coordinate_frame=conditions['coordinate_frame'],
            coordinate_audit=audit, export_mode='frozen_eval_fp32_mu_logvar',
            effective_code_sha256={p.name:file_sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
        manifest_path = out/'manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if any(manifest.get(k) != value for k,value in identity.items()):
                raise ValueError('Export resume source/config/code binding mismatch')
            if [r['uid'] for r in manifest['records']] != data['uids'][:len(manifest['records'])]:
                raise ValueError('Export resume UID cursor mismatch')
        else:
            manifest = dict(identity, complete=False, records=[], optimizer_updates=0,
                            reconstruction_count_evaluation=False)
            atomic_json(manifest_path, manifest)
        posterior_moments = [check_saved_record(out, r) for r in manifest['records']]
        if manifest['complete']:
            if manifest.get('normalization_sha256') != normalization_sha256(manifest['normalization']):
                raise ValueError('Final cache normalization hash mismatch')
            print(json.dumps(dict(complete=True, manifest=str(manifest_path), sha256=file_sha(manifest_path))))
            return
        # Available bytes are checked before allocating GPU memory; quota checks remain a launch prerequisite.
        remaining = data['uids'][len(manifest['records']):]
        needed = sum(len(items[uid]['vertices'])*(512*8+3*4+8) for uid in remaining)
        needed += sum((condition_path.parent/records[uid]['file']).stat().st_size for uid in remaining)
        if shutil.disk_usage(out).free < needed*2+(1<<30):
            raise RuntimeError('Insufficient free storage for the remaining cache export')
        from gpu_budget import GPUBudget
        with GPUBudget(args.budget_file, phase='cache_export', device=args.device) as budget:
            configure_math_backend()
            model = None
            try:
                if not budget.stop(reserve_seconds=120):
                    model = load_frozen_vae(args.vae_checkpoint, args.device)
                    manifest['model_config'] = model.cfg.to_dict()
                for uid in remaining:
                    if budget.stop(reserve_seconds=120): break
                    path = out/(uid+'.npz')
                    with np.load(condition_path.parent/records[uid]['file'], allow_pickle=False) as c:
                        arrays, errors = export_item(model, items[uid], c['points'].copy(), c['point_mask'].copy(), path)
                    manifest['records'].append(dict(uid=uid, file=path.name, sha256=file_sha(path),
                        vertices_sha256=array_sha(arrays['vertices']), vertices=len(arrays['vertices']),
                        array_sha256={k:array_sha(v) for k,v in arrays.items()},
                        condition_points=int(arrays['point_mask'].sum()), mu_roundtrip_max_abs=errors))
                    posterior_moments.append(check_saved_record(out, manifest['records'][-1]))
                    manifest['budget'] = budget.snapshot()
                    atomic_json(manifest_path, manifest)
            except BaseException as error:
                manifest['failure'] = dict(type=type(error).__name__, message=str(error), next_uid=data['uids'][len(manifest['records'])] if len(manifest['records']) < 50 else None)
                manifest['budget'] = budget.snapshot()
                atomic_json(manifest_path, manifest)
                raise
            if len(manifest['records']) == len(data['uids']):
                manifest['normalization'] = statistics_from_moments([x[0] for x in posterior_moments], [x[1] for x in posterior_moments])
                manifest['normalization_sha256'] = normalization_sha256(manifest['normalization'])
                manifest['complete'] = True
                manifest.pop('failure', None)
            manifest['budget'] = budget.snapshot()
            atomic_json(manifest_path, manifest)
        print(json.dumps(dict(complete=manifest['complete'], exported=len(manifest['records']),
                              expected=50, manifest=str(manifest_path), sha256=file_sha(manifest_path))))


if __name__ == '__main__': main()
