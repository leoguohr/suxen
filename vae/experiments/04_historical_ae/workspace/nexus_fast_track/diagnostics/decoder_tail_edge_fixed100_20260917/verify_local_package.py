"""Independent read-only verification; no model inference or optimizer update."""
import hashlib,io,json,sys,zipfile
from pathlib import Path
import numpy as np
p=Path(sys.argv[1]);z=zipfile.ZipFile(p)
assert z.testzip() is None
expected={}
for line in z.read('SHA256SUMS.txt').decode().splitlines():
    h,n=line.split('  ',1);expected[n]=h
    with z.open(n) as f:
        digest=hashlib.sha256()
        for block in iter(lambda:f.read(4*1024*1024),b''):digest.update(block)
    assert digest.hexdigest()==h,n
read=lambda n:json.loads(z.read(n))
trace=[json.loads(s) for s in z.read('run/evaluations.jsonl').splitlines()]
updates=[json.loads(s) for s in z.read('run/updates.jsonl').splitlines()]
meta=read('cache/manifest.json');uids=[m['uid'] for m in meta['meshes']]
assert len(set(uids))==100 and [r['step'] for r in trace]==list(range(501))
assert [r['update'] for r in updates]==list(range(1,501))
for r in trace:
    assert [m['uid'] for m in r['meshes']]==uids
    loss=0.
    for m in r['meshes']:
        loss+=m['edge_soft4'];assert m['perfect']==(m['fp']==m['fn']==0)
    assert loss/100==r['objective']
    assert sum(m['perfect'] for m in r['meshes'])==r['edge_perfect']
    assert sum(m['fp'] for m in r['meshes'])==r['total_fp']
    assert sum(m['fn'] for m in r['meshes'])==r['total_fn']
for u in updates:
    assert u['mesh_count']==100 and u['lrs']=={'decoder_tail':1e-5,'edge_head':1e-4}
    assert all(np.isfinite(v['delta_norm']) and v['delta_norm']>0 for v in u['actual_updates'].values())
verified_pairs=0
if f'cache/{uids[0]}.npz' in z.namelist():
    for i,uid in enumerate(uids):
        with np.load(io.BytesIO(z.read(f'cache/{uid}.npz'))) as f:
            gt=f['edges'];n=len(f['vertices']);x=f['last_block_input'];faces=f['faces']
            assert x.shape==(n,1024) and x.dtype==np.float32
            face_edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[0,2]],faces[:,[1,2]]]),axis=1),axis=0)
            assert np.array_equal(face_edges,gt)
        with np.load(io.BytesIO(z.read(f'run/final_outputs/{uid}.npz'))) as f:s=f['logits']
        a,b=np.triu_indices(n,1);keys=gt[:,0]*n+gt[:,1];q=a*n+b;at=np.searchsorted(keys,q)
        y=(at<len(keys))&(keys[np.minimum(at,len(keys)-1)]==q);pred=s>0
        m=trace[-1]['meshes'][i]
        counts=dict(tp=int((pred&y).sum()),fp=int((pred&~y).sum()),fn=int((~pred&y).sum()),tn=int((~pred&~y).sum()))
        assert all(m[k]==v for k,v in counts.items()),uid
        assert float(s[y].min())==m['min_margin_gt'] and float((-s[~y]).min())==m['min_margin_non_gt']
        verified_pairs+=len(s)
    assert verified_pairs==84669234
result=dict(package=str(p),hashes_verified=len(expected),updates=500,states=501,meshes=100,all_final_logits_counts_independently_recomputed_pairs=verified_pairs,scope='ZIP hashes, logs and saved outputs; no local torch/network execution')
out=p.with_suffix('.local-verification.json');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
