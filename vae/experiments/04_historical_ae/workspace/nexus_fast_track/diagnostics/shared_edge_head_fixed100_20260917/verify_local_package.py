"""Independent local ZIP and exhaustive saved-logit audit; no training or network forward."""
import hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
p=Path('/Users/luthier/Downloads/Nexus_SharedEdgeHead_Fixed100_2000Updates_FullEvaluation_20260917.zip')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
expected=json.loads((ROOT/'package_verification.json').read_text())
assert sha(p)==expected['sha256']
with zipfile.ZipFile(p) as z:
    assert z.testzip() is None
    hashes={}
    for line in z.read('SHA256SUMS.txt').decode().splitlines():
        h,n=line.split('  ',1);assert hashlib.sha256(z.read(n)).hexdigest()==h,n;hashes[n]=h
    plan=json.loads(z.read('training_plan.json'));cache=json.loads(z.read('cache/manifest.json'))
    manifest=json.loads(z.read('run/manifest.json'));complete=json.loads(z.read('run/complete.json'))
    for name,h in manifest['code_sha256'].items():assert hashlib.sha256(z.read(name)).hexdigest()==h,name
    budget=plan['updates'];uids=[r['uid'] for r in cache['meshes']]
    assert len(uids)==len(set(uids))==100 and manifest['head_instances']==1 and manifest['trainable_parameters']==32800
    assert sum(r['vertices'] for r in cache['meshes'])==106325 and sum(r['pairs'] for r in cache['meshes'])==84669234
    traces=[json.loads(s) for s in z.read('run/evaluations.jsonl').decode().splitlines()]
    updates=[json.loads(s) for s in z.read('run/updates.jsonl').decode().splitlines()]
    assert [r['step'] for r in traces]==list(range(budget+1))
    assert [r['update'] for r in updates]==list(range(1,budget+1))
    for r in traces:
        assert [m['uid'] for m in r['meshes']]==uids
        # Reproduce the training Python's sequential mesh accumulation explicitly.
        loss_sum=0.0
        for m in r['meshes']:loss_sum+=m['edge_soft4']
        assert r['objective']==loss_sum/100
        assert r['edge_perfect']==sum(m['fp']==m['fn']==0 for m in r['meshes'])
        assert r['total_fp']==sum(m['fp'] for m in r['meshes']) and r['total_fn']==sum(m['fn'] for m in r['meshes'])
    assert all(u['meshes']==100 and u['mesh_coefficient']==.01 for u in updates)
    assert all(u['parameter_updates']['weight']['changed_elements']>0 for u in updates)
    original=json.loads(z.read('baseline_verification.json'))['baseline']
    assert traces[0]['meshes']==original['meshes']
    final=[]
    for ref,expected_mesh in zip(cache['meshes'],traces[-1]['meshes']):
        uid=ref['uid'];n=ref['vertices']
        with np.load(io.BytesIO(z.read('cache/'+uid+'.npz'))) as d:
            edges=d['edges'];faces=d['faces'];assert d['hidden'].shape==(n,1024)
            generated=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[0,2]]]),axis=1),axis=0)
            assert np.array_equal(generated,edges)
        with np.load(io.BytesIO(z.read('run/final_outputs/'+uid+'.npz'))) as d:
            logits=d['logits'];assert d['edge_embedding_scoring'].shape==(n,32)
            assert d['edge_head_raw'].shape==(n,32) and np.isfinite(logits).all()
        i,j=np.triu_indices(n,1);assert len(logits)==len(i)==ref['pairs']
        y=np.isin(i*n+j,edges[:,0]*n+edges[:,1]);pred=logits>0
        counts=dict(tp=int((pred&y).sum()),fp=int((pred&~y).sum()),fn=int((~pred&y).sum()),tn=int((~pred&~y).sum()))
        assert all(counts[k]==expected_mesh[k] for k in counts),uid
        assert counts['tp']+counts['fn']==ref['gt_edges']
        assert float(logits[y].min())==expected_mesh['min_margin_gt']
        assert float((-logits[~y]).min())==expected_mesh['min_margin_non_gt']
        final.append(dict(uid=uid,**counts))
    old={m['uid'] for m in traces[0]['meshes'] if m['perfect']};new={m['uid'] for m in traces[-1]['meshes'] if m['perfect']}
    assert complete['initial_success_retained']==sorted(old&new)
    assert complete['initial_success_lost']==sorted(old-new)
    assert complete['new_final_success']==sorted(new-old)
    all_steps=[r['step'] for r in traces if r['all100_perfect']]
    assert complete['first_all100_perfect_step']==(min(all_steps) if all_steps else None)
    for name in ['REPORT.md','summary.json','curves.png','per_mesh_final.csv','per_mesh_summary.json','benchmark.json','run/complete.json','run/verification.json']:
        (ROOT/Path(name).name).write_bytes(z.read(name))
    audit=dict(zip_sha256=expected['sha256'],crc_passed=True,files_verified=len(hashes),continuous_updates=budget,
        all100_same_head_per_update=True,equal_mesh_objective=True,all84669234_final_pair_labels_counts_margins_recomputed=True,
        gt_edges_rederived_from_faces=True,retention_sets_verified=True,local_network_reexecution=False,
        checkpoint_reload_execution='Verified on server; local audit uses exported tensors and records',final_counts=final)
    (ROOT/'local_independent_verification.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='final_counts'},indent=2))
