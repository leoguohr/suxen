"""Independent local archive/GT/full-pair sign check; no server model or training."""
import hashlib
import io
import json
import zipfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
ARCHIVE=Path('/Users/luthier/Downloads/Nexus_FreeEdge_Large3_2000Updates_FullEvaluation_20260917.zip')
record=json.loads((ROOT/'package_verification.json').read_text())
actual=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
assert actual==record['sha256']
results=[]
with zipfile.ZipFile(ARCHIVE) as z:
    assert z.testzip() is None
    lines=z.read('SHA256SUMS.txt').decode().splitlines()
    for line in lines:
        expected,name=line.split('  ',1)
        assert hashlib.sha256(z.read(name)).hexdigest()==expected,name
    for uid in ['nexus_2k_000446','nexus_2k_001093','nexus_2k_000898']:
        root='runs/'+uid+'/'
        trace=[json.loads(s) for s in z.read(root+'updates.jsonl').decode().splitlines()]
        assert [r['step'] for r in trace]==list(range(2001))
        complete=json.loads(z.read(root+'complete.json'))
        perfect=[r['step'] for r in trace[1:] if r['fp']==r['fn']==0]
        assert perfect==list(range(complete['first_perfect_step'],2001))
        source=np.load(io.BytesIO(z.read('snapshots/'+uid+'/representations_and_gradients.npz')))
        saved=np.load(io.BytesIO(z.read(root+'checkpoint-step2000.npz')))
        n=len(source['vertices']);faces=source['gt_faces']
        edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[0,2]]]),axis=1)
        keys=np.unique(edges[:,0]*n+edges[:,1])
        pairs=saved['pairs'];assert np.array_equal(pairs,np.stack(np.triu_indices(n,1),axis=1))
        labels=np.isin(pairs[:,0]*n+pairs[:,1],keys)
        assert np.array_equal(labels,saved['labels'])
        scale=json.loads(z.read('snapshots/'+uid+'/effective_code/runtime_contract.json'))['scales']['edge_logit_scale']
        e=saved['edge_embedding_scoring'].astype(np.float64)
        logits=saved['logits'];max_diff=0.;fp=fn=0
        for lo in range(0,len(pairs),65536):
            part=pairs[lo:lo+65536];diff=e[part[:,0]]-e[part[:,1]]
            score=(np.square(diff[:,:16]).sum(1)-np.square(diff[:,16:]).sum(1))*scale
            y=labels[lo:lo+65536];pred=score>0
            fp+=int((pred&~y).sum());fn+=int((~pred&y).sum())
            max_diff=max(max_diff,float(np.abs(score-logits[lo:lo+65536]).max()))
        assert fp==fn==0
        results.append(dict(uid=uid,vertices=n,pairs=len(pairs),gt_edges=len(keys),fp=fp,fn=fn,
            first_perfect_step=perfect[0],consecutive_perfect=len(perfect),
            cpu_float64_scoring_max_difference_from_cuda_float32=max_diff,
            gt_labels_independently_rebuilt=True,all_pairs_complete_and_ordered=True))
    for name in ['REPORT.md','comparison.json','comparison.csv','curves.png','curves.pdf','export_complete.json','result_verification.json']:
        (ROOT/name).write_bytes(z.read(name))
out=dict(zip_sha256=actual,archive_files_verified=len(lines),results=results,
    scope='Independent local archive, labels, logs and final CPU FP64 scoring; no network checkpoint rerun')
(ROOT/'local_independent_verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
