"""Independent NumPy audit; original predictions and targets remain read-only."""
from pathlib import Path
import itertools, json, hashlib
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/Users/luthier/Downloads/Nexus_teacher_full_reconstruction')
EVIDENCE = ROOT / 'downloaded_evidence' if (ROOT / 'downloaded_evidence').is_dir() else SOURCE / 'evidence'
DATA = ROOT / 'ground_truth_export'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def rows(a):
    return set(map(tuple, np.sort(np.asarray(a), axis=1).tolist()))

def counts(pred, gt):
    tp, fp, fn = len(pred & gt), len(pred - gt), len(gt - pred)
    return dict(tp=tp, fp=fp, fn=fn, f1=2*tp/max(2*tp+fp+fn, 1))

def candidates(n, edges):
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)
    return {(a,b,c) for a in range(n) for b in adj[a] if b>a for c in adj[a]&adj[b] if c>b}

def matched(pred, gt):
    p, g = pred.astype(np.float64), gt.astype(np.float64)
    assert p.shape == g.shape
    dist = ((p[:,None]-g[None,:])**2).sum(-1)
    # A bijection of independently nearest targets attains the assignment
    # lower bound, hence is also a global minimum of the Hungarian objective.
    mapping = dist.argmin(axis=1)
    assert len(np.unique(mapping)) == len(p), 'Nearest mapping is not bijective'
    delta = p-g[mapping]
    gd = ((g[:,None]-g[None,:])**2).sum(-1); np.fill_diagonal(gd, np.inf)
    ratio = np.linalg.norm(delta, axis=1).max()/np.sqrt(gd.min())
    return mapping, dict(squared_error=float((delta**2).sum()), coordinates=delta.size, rmse=float(np.sqrt((delta**2).mean())), spacing_ratio=float(ratio))

def aggregate(records):
    result = {}
    for task in ['edge','face']:
        c = {k:sum(r[task][k] for r in records) for k in ['tp','fp','fn']}
        c['f1'] = 2*c['tp']/max(2*c['tp']+c['fp']+c['fn'],1); result[task] = c
    result['samples'] = len(records)
    result['face_below_099_ids'] = [r['id'] for r in records if r['face']['f1']<.99]
    return result

def main():
    manifest = json.loads((DATA/'manifest.json').read_text())
    assert manifest['source_cache_sha256'] == sha(ROOT/'teacher_assets/data/point50/training.pt')
    targets = []
    for r in manifest['meshes']:
        path = DATA/r['path']; assert sha(path)==r['sha256']
        with np.load(path, allow_pickle=False) as a:
            targets.append({k:a[k].copy() for k in a.files})
    outputs = {}
    for directory, mode in [('final_ae/spacetime','topology'),('topology_final','topology'),('topology_final_seed23456','topology'),('cascade','cascade'),('cascade_seed23456','cascade'),('cold_start_full_cascade','cascade'),*[(f'early_three_ae/{m}','topology') for m in ['cosine','euclidean','spacetime']]]:
        recs = []
        for i, gt in enumerate(targets):
            path = EVIDENCE/directory/(f'{i:02d}_mesh.npz' if mode=='cascade' else f'{i:02d}.npz')
            with np.load(path, allow_pickle=False) as a:
                n = len(gt['vertices'])
                assert np.isfinite(a['edge_logits']).all() and np.isfinite(a['face_logits']).all()
                ep = a['edge_pairs']; pe = rows(a['pred_edges']); pf = rows(a['pred_faces']); fc = rows(a['face_candidates'])
                assert rows(ep)==set(itertools.combinations(range(n),2))
                assert pe==rows(ep[a['edge_logits']>0])
                assert pf==rows(a['face_candidates'][a['face_logits']>0])
                assert fc==candidates(n,pe)
                if mode=='cascade':
                    mapping, point = matched(a['vertices'], gt['vertices'])
                    gf = rows(np.argsort(mapping)[gt['faces']])
                else:
                    old = gt['original_vertex_indices']
                    assert np.array_equal(a['vertices'], gt['vertices'][old.argsort()])
                    gf = rows(old[gt['faces']]); point = None
                ge = {e for f in gf for e in itertools.combinations(f,2)}
                recs.append(dict(id=i, edge=counts(pe,ge), face=counts(pf,gf), missing_face_candidates=len(gf-fc), point=point, prediction_sha256=sha(path)))
        outputs[directory] = dict(total=aggregate(recs),per_mesh=recs)
        print(directory,outputs[directory]['total'],flush=True)
    for directory in ['point_final','point_final_seed98765']:
        recs = []
        for i, gt in enumerate(targets):
            path = EVIDENCE/directory/f'{i:02d}.npy'
            _, point = matched(np.load(path, allow_pickle=False),gt['vertices'])
            recs.append(dict(id=i,**point,prediction_sha256=sha(path)))
        total = dict(samples=50,count_correct=50,coordinate_rmse=float(np.sqrt(sum(r['squared_error'] for r in recs)/sum(r['coordinates'] for r in recs))),max_spacing_ratio=max(r['spacing_ratio'] for r in recs))
        total['teacher_gate_pass'] = total['coordinate_rmse']<1e-4 and total['max_spacing_ratio']<.1
        outputs[directory] = dict(total=total,per_mesh=recs)
        print(directory,total,flush=True)
    report = dict(scope='This audit recomputes saved arrays; it is not fresh neural-network execution.', optimizer_updates=0, target_cache_sha256=manifest['source_cache_sha256'], arrays_checked=550, all_pair_and_complete_triangle_checks_passed=True, results=outputs)
    (ROOT/'repro_outputs/SAVED_PREDICTION_AUDIT.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
