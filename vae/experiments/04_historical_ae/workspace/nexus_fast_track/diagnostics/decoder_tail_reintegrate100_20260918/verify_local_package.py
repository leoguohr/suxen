"""Independent archive/log/retention verification, with saved Edge logits when present."""
import hashlib,io,json,sys,zipfile
from pathlib import Path
import numpy as np
p=Path(sys.argv[1]);z=zipfile.ZipFile(p);assert z.testzip() is None
entries={}
for line in z.read('SHA256SUMS.txt').decode().splitlines():
    h,n=line.split('  ',1);entries[n]=h
    with z.open(n) as f:
        d=hashlib.sha256()
        for b in iter(lambda:f.read(4*1024*1024),b''):d.update(b)
    assert d.hexdigest()==h,n
read=lambda n:json.loads(z.read(n))
complete=read('run/complete.json');cfg=read('config.json');uids=read('selection.json')['uids']
assert complete['optimizer_updates']==complete['backward_calls']==0 and not complete['optimizer_constructed']
installation=read('run/installation_verification.json')
assert len(installation['mapping'])==len(installation['changed_state_keys'])==10
assert set(installation['changed_state_keys'])==set(installation['mapping'].values())
assert installation['source_sha256']==cfg['source_sha256']=='428aeddbc6ea03ae166ba22aa431fe40f4298ca83e0e5008302a3c5eed3ceb79'
assert installation['tail_sha256']==cfg['tail_sha256']=='23522f92141dd10857edf38340ab0962cc0a2f6a653ec3227ecd50848d10686a'
assert complete['non_target_state_unchanged'] and complete['face_head_parameters_unchanged']
branches={}
for br in ['before','after']:
    rows=[json.loads(x) for x in z.read(f'run/{br}.jsonl').splitlines()];assert len(rows)==100 and [r['uid'] for r in rows]==uids
    for r in rows:
        assert r['face']['complete'] and r['edge']['tp']+r['edge']['fn']==r['gt_edges']
        assert r['face']['tp']+r['face']['fn']==r['gt_faces']
        assert r['gt_face_candidates']+r['missing_gt_face_candidates']==r['gt_faces']
        assert r['face']['fn']>=r['missing_gt_face_candidates']
        assert r['edge_perfect']==(r['edge']['fp']==r['edge']['fn']==0)
        assert r['face_perfect']==(r['face']['fp']==r['face']['fn']==0)
        assert r['joint_perfect']==(r['edge_perfect'] and r['face_perfect'])
    for k in ['edge_perfect','face_perfect','joint_perfect']:assert sum(r[k] for r in rows)==complete['summaries'][br][k]
    for kind in ['edge','face']:
        for k in ['tp','fp','fn']:assert sum(r[kind][k] for r in rows)==complete['summaries'][br][kind+'_'+k]
    branches[br]={r['uid']:r for r in rows}
for k,record in complete['retention'].items():
    a={u for u in uids if branches['before'][u][k]};b={u for u in uids if branches['after'][u][k]}
    assert set(record['before'])==a and set(record['after'])==b
    assert set(record['retained'])==a&b and set(record['lost'])==a-b and set(record['gained'])==b-a
assert all(branches['after'][uid]['edge_perfect'] for uid in cfg['seven_uids'])
assert {k:complete['summaries']['after'][v] for k,v in [('perfect','edge_perfect'),('fp','edge_fp'),('fn','edge_fn')]}==cfg['expected_installed_edge']
for uid in uids:
    for k in ['mu','logvar']:assert branches['before'][uid]['feature_hashes'][k]==branches['after'][uid]['feature_hashes'][k]
assert sum(branches['before'][u]['feature_hashes']['face_embedding']!=branches['after'][u]['feature_hashes']['face_embedding'] for u in uids)==complete['face_embedding_changed_meshes']
checked_pairs=0
if f'run/features/{uids[0]}.npz' in z.namelist():
    for uid in uids:
        with np.load(io.BytesIO(z.read(f'run/features/{uid}.npz'))) as f:
            s=f['edge_logits_after'];gt=f['edges'];n=len(f['vertices']);faces=f['faces']
            derived=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[0,2]],faces[:,[1,2]]]),axis=1),axis=0);assert np.array_equal(gt,derived)
        a,b=np.triu_indices(n,1);keys=gt[:,0]*n+gt[:,1];q=a*n+b;at=np.searchsorted(keys,q);y=(at<len(keys))&(keys[np.minimum(at,len(keys)-1)]==q);pred=s>0
        m=branches['after'][uid]['edge'];counts=dict(tp=int((pred&y).sum()),fp=int((pred&~y).sum()),fn=int((~pred&y).sum()),tn=int((~pred&~y).sum()))
        assert all(m[k]==v for k,v in counts.items()),uid
        checked_pairs+=len(s)
    assert checked_pairs==84669234
result=dict(package=str(p),hashes_verified=len(entries),evaluations=200,source_and_installed_meshes_each=100,optimizer_updates=0,retention_sets_verified=True,saved_edge_pairs_recounted=checked_pairs,scope='Read-only hashes, logs, set arithmetic and optional saved Edge logits; did not run the network or re-enumerate Face on CPU')
p.with_suffix('.local-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
