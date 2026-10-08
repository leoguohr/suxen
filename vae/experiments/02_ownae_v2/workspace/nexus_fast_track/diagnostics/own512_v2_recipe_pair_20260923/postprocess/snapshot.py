"""Read-only bounded run status; never launches or resumes a process."""
import json
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
result={}
for name in ['A_v1_recipe_control','B_v2_teacher_blocks']:
    branch=ROOT/name
    row={}
    for filename in ['status.json','training_complete.json','failure.json','evaluation_failure.json']:
        p=branch/filename
        if p.exists():
            x=json.loads(p.read_text())
            row[filename]={k:x[k] for k in ['state','completed_updates','stop_reason','error','seconds_per_update'] if k in x}
    evaluations=[]
    for p in sorted((branch/'evaluations').glob('eval-*.json')):
        x=json.loads(p.read_text())
        evaluations.append(dict(step=x['step'],face_f1=x['counts']['face']['micro_f1'],strict=x['joint_perfect'],
            edge_fp=x['counts']['edge']['fp'],edge_fn=x['counts']['edge']['fn'],
            large16_f1=x['large16']['counts']['face']['micro_f1'],large16_strict=x['large16']['joint_perfect']))
    row['evaluations']=evaluations
    row['target_reached']=(branch/'TARGET_REACHED.json').exists()
    row['cold_verified']=[p.name for p in (branch/'evaluations').glob('cold-*.json')]
    result[name]=row
p=ROOT/'repro_outputs/RESOURCE_ACCOUNT.json'
if p.exists():
    x=json.loads(p.read_text());now=time.time()
    result['charged_gpu_hours']=sum(max(0,j.get('finished',now)-j['started']) for j in x['jobs'].values())/3600
result['worker_complete']=(ROOT/'queue/worker_complete.json').exists()
print(json.dumps(result,separators=(',',':')))
