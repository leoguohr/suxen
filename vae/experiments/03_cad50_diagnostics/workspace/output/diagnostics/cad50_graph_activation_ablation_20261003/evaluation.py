"""Complete predicted-Edge triangle enumeration; bounded Face scoring chunks."""
from pathlib import Path
import numpy as np
import torch
from native_models import Graph
from data_objective import edge_logits, face_logits
from support import write


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
    folder.mkdir(exist_ok=True)
    n = len(item['vertices'])
    vertices, faces = item['vertices'].cuda(), item['faces'].cuda()
    rows = model(vertices, faces, sample_latent=False,
                 graph=Graph.from_faces(faces, n))
    assert rows['latent'] is rows['mu']
    assert all(torch.isfinite(t).all() for t in rows.values())
    pairs = item['pairs'].numpy()
    scores = np.concatenate([edge_logits(rows['edge'], item['pairs'][s:s+32768].cuda()).cpu().numpy()
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
    gt_scores = face_logits(rows['face'], item['gt_faces'].cuda()).cpu().numpy()
    tp = fp = candidates = shards = 0
    pending = []

    def flush(part):
        nonlocal tp, fp, candidates, shards
        ids = np.asarray(part, dtype=np.int64).reshape(-1,3)
        logits = face_logits(rows['face'], torch.as_tensor(ids,device='cuda')).cpu().numpy()
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


@torch.no_grad()
def evaluate(model, items, folder, checkpoint):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    model.eval()
    rows = []
    for uid, item in items.items():
        path = folder/uid/'metrics.json'
        if path.exists():
            import json
            row = json.loads(path.read_text())
            assert row['checkpoint_sha256'] == checkpoint['sha256']
        else:
            row = evaluate_mesh(model, item, folder/uid)
            row['checkpoint_sha256'] = checkpoint['sha256']
            write(path, row)
        rows.append(row)
        write(folder/'progress.json', dict(complete=False, checkpoint=checkpoint, completed_uids=[r['uid'] for r in rows]))
    result = dict(complete=True, checkpoint=checkpoint, **aggregate(rows), meshes=rows,
                  large16=aggregate([r for r in rows if 66<=r['vertices']<=274]))
    write(folder/'evaluation.json', result)
    model.train()
    return result
