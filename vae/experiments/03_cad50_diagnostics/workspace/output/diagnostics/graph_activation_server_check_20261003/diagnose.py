"""Read-only pretrained-model operation reversions; zero optimizer updates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import torch
from torch.nn import functional as F


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(part)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def aggregate(rows):
    counts = {task: {k: sum(x[task][k] for x in rows) for k in ['tp','fp','fn']}
              for task in ['edge','face']}
    for value in counts.values():
        value['micro_f1'] = 2*value['tp']/max(2*value['tp']+value['fp']+value['fn'], 1)
    return {'counts': counts, 'joint_strict_uids': [x['uid'] for x in rows if x['joint_strict']],
            'joint_strict': sum(x['joint_strict'] for x in rows),
            'edge_strict': sum(x['edge']['fp']==x['edge']['fn']==0 for x in rows),
            'face_fn_missing': sum(x['face_fn_missing'] for x in rows),
            'face_fn_present': sum(x['face_fn_present'] for x in rows)}


@torch.no_grad()
def evaluate_mesh(model, item, folder):
    folder.mkdir()
    n = len(item['vertices'])
    vertices, faces = item['vertices'].cuda(), item['faces'].cuda()
    rows = model(vertices, faces, sample_latent=False,
                 graph=native.Graph.from_faces(faces, n))
    assert rows['latent'] is rows['mu']
    assert all(torch.isfinite(t).all() for t in rows.values())
    pairs = item['pairs'].numpy()
    scores = np.concatenate([objective.edge_logits(rows['edge'], item['pairs'][s:s+32768].cuda()).cpu().numpy()
                             for s in range(0, len(pairs), 32768)])
    truth, positive = item['edge_labels'].numpy(), scores>0
    edge = {'tp': int((positive&truth).sum()), 'fp': int((positive&~truth).sum()),
            'fn': int((~positive&truth).sum())}
    np.savez_compressed(folder/'edge.npz', pair_ids=pairs, logits=scores, gt=truth)
    adjacency = np.zeros((n,n), dtype=bool)
    selected = pairs[positive]
    adjacency[selected[:,0],selected[:,1]] = True
    gt = item['gt_faces'].numpy()
    gt_keys = (gt[:,0]*n+gt[:,1])*n+gt[:,2]
    covered = adjacency[gt[:,0],gt[:,1]] & adjacency[gt[:,0],gt[:,2]] & adjacency[gt[:,1],gt[:,2]]
    gt_scores = objective.face_logits(rows['face'], item['gt_faces'].cuda()).cpu().numpy()
    tp = fp = candidates = shards = 0
    pending = []

    def flush(part):
        nonlocal tp, fp, candidates, shards
        ids = np.asarray(part, dtype=np.int64).reshape(-1,3)
        logits = objective.face_logits(rows['face'], torch.as_tensor(ids,device='cuda')).cpu().numpy()
        assert np.isfinite(logits).all()
        labels = np.isin((ids[:,0]*n+ids[:,1])*n+ids[:,2], gt_keys)
        tp += int(((logits>0)&labels).sum())
        fp += int(((logits>0)&~labels).sum())
        candidates += len(ids)
        np.savez_compressed(folder/f'face-{shards:06d}.npz', ids=ids, logits=logits, gt=labels)
        shards += 1

    # Upper-triangular adjacency emits each i<j<k triangle exactly once.
    for i in range(n):
        for j in np.flatnonzero(adjacency[i]):
            for k in np.flatnonzero(adjacency[i]&adjacency[j]):
                pending.append((i,int(j),int(k)))
                if len(pending)==32768:
                    flush(pending)
                    pending=[]
    if pending:
        flush(pending)
    missing = int((~covered).sum())
    present = int((covered&(gt_scores<=0)).sum())
    face = {'tp': tp, 'fp': fp, 'fn': len(gt)-tp}
    assert tp==int((covered&(gt_scores>0)).sum())
    assert face['fn']==missing+present
    np.savez_compressed(folder/'gt_face.npz', ids=gt, logits=gt_scores, covered=covered)
    return {'uid': item['uid'], 'vertices': n, 'edge': edge, 'face': face,
            'face_fn_missing': missing, 'face_fn_present': present,
            'actual_face_candidates': candidates, 'face_shards': shards,
            'joint_strict': edge['fp']==edge['fn']==face['fp']==face['fn']==0,
            'complete': True}


def main(source, output):
    global native, objective
    source, output = Path(source).resolve(), Path(output).resolve()
    assert not (output/'result.json').exists(), 'Refuse duplicate run'
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source))
    import native_models as native
    import data_objective as objective
    import run_support as support
    support.configure(0)
    assert torch.cuda.device_count()==1
    torch.cuda.set_device(0)

    class ProbeEncoderBlock(native.EncoderBlock):
        mode = 'native'

        def forward(self, h, graph):
            if self.mode=='native':
                return super().forward(h, graph)
            assert self.v2
            message = (self.graph(self.graph_norm(h), graph) if self.mode=='norm_pre'
                       else self.graph_norm(self.graph(h, graph)))
            h = h + (F.silu(message) if self.mode=='graph_silu' else F.gelu(message))
            layer = self.transformer
            h = h + native.math_attention(layer.self_attn, layer.norm1(h))
            activation = F.relu if self.mode=='encoder_relu' else layer.activation
            return h + layer.linear2(activation(layer.linear1(layer.norm2(h))))

    hashes = {f: sha(source/f) for f in ['native_models.py','data_objective.py','run_support.py']}
    items, manifest = objective.load_dataset(source/'data', source/'pools')
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total','--format=csv,noheader'],text=True).strip()
    result = {'scope': 'pretrained inference reversions, NOT fresh-training causal ablation',
              'optimizer_updates': 0, 'gpu': gpu, 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'source': str(source), 'source_sha256': hashes, 'data': manifest,
              'script_sha256': sha(__file__), 'evaluations': {}}
    started = time.monotonic()
    experiments = [('A_v1_recipe_control','native'), ('B_v2_teacher_blocks','native'),
                   ('B_v2_teacher_blocks','norm_pre'), ('B_v2_teacher_blocks','graph_silu'),
                   ('B_v2_teacher_blocks','encoder_relu')]
    expected_sha = {'A_v1_recipe_control': 'df5439ff36535d5aadd35361effa0f96b63c2e7298546de9b8a9d41112544501',
                    'B_v2_teacher_blocks': '6f965d3837c32101dc7efd52252c12387fb3f10c788e6fd0d33ae359a44903a2'}
    model = cp = current_variant = None
    for variant, mode in experiments:
        if current_variant!=variant:
            if model is not None:
                del model, cp
                torch.cuda.empty_cache()
            path = source/variant/'checkpoint-19356.pt'
            digest = sha(path)
            assert digest==expected_sha[variant], 'Checkpoint identity mismatch'
            cp = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
            assert cp['completed_updates']==19356 and cp['model_variant']==variant
            cfg = native.Config(**cp['model_config'])
            model = native.NativeTopologyAE(cfg)
            if variant.startswith('B'):
                model.encoder_blocks = torch.nn.ModuleList(ProbeEncoderBlock(cfg) for _ in model.encoder_blocks)
            model.load_state_dict(cp['model'], strict=True)
            model = model.cuda().float().eval()
            current_variant = variant
            initial_model_hash = support.tensor_hash(model.state_dict())
            assert initial_model_hash==support.tensor_hash(cp['model'])
        for block in model.encoder_blocks:
            if isinstance(block, ProbeEncoderBlock):
                block.mode = mode
        name = variant+'__'+mode
        folder = output/name
        folder.mkdir()
        rows = []
        tic = time.monotonic()
        for uid in manifest['uids']:
            rows.append(evaluate_mesh(model, items[uid], folder/uid))
            write(output/'status.json', {'state': 'evaluating', 'experiment': name,
                                         'uids_complete': len(rows), 'optimizer_updates': 0,
                                         'elapsed_seconds': time.monotonic()-started})
        assert initial_model_hash==support.tensor_hash(model.state_dict())
        summary = aggregate(rows)
        if mode=='native':
            expected = {'A_v1_recipe_control': (1039,954,1629,1808,34),
                        'B_v2_teacher_blocks': (21,3,47,6,38)}[variant]
            actual = (summary['counts']['edge']['fp'],summary['counts']['edge']['fn'],
                      summary['counts']['face']['fp'],summary['counts']['face']['fn'],summary['joint_strict'])
            assert actual==expected, (variant, actual, expected)
        entry = {'variant': variant, 'mode': mode, 'checkpoint_path': str(path),
                 'checkpoint_sha256': digest, 'model_tensor_hash': initial_model_hash,
                 'parameters_and_buffers_unchanged': True, 'complete': True,
                 'seconds': time.monotonic()-tic, **summary, 'meshes': rows}
        entry['large16'] = aggregate([r for r in rows if 66<=r['vertices']<=274])
        result['evaluations'][name] = entry
        write(folder/'evaluation.json', entry)
        print(name, summary, flush=True)
    assert hashes=={f: sha(source/f) for f in hashes}
    result.update(complete=True, seconds=time.monotonic()-started)
    write(output/'result.json', result)
    write(output/'status.json', {'state': 'complete', 'optimizer_updates': 0,
                                'elapsed_seconds': result['seconds']})


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    main(args.source, args.output)
