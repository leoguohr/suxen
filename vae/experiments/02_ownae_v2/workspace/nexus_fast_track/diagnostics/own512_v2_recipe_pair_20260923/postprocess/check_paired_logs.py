"""CPU-only audit of recorded batch order, negatives, participation and LR."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
branches=['A_v1_recipe_control','B_v2_teacher_blocks']
logs=[]
for branch in branches:
    text=(ROOT/branch/'updates.jsonl').read_text()
    lines=text.splitlines()
    if text and not text.endswith('\n'):lines=lines[:-1]
    rows=[json.loads(line) for line in lines]
    for number,row in enumerate(rows,1):
        assert row['update']==row['adam_step']==number
        assert row['epoch']==(number-1)//10 and row['batch_index']==(number-1)%10
        assert len(row['uids'])==5 and len(set(row['uids']))==5
        assert row['lr']==1e-4*min(number/100,1) and row['participations']==number*5
        assert row['clip_coefficient']==min(1.,1/(row['gradient_norm_before_clip']+1e-6))
        assert row['uids']==[m['uid'] for m in row['meshes']]
    for start in range(0,len(rows)-9,10):
        assert sorted(u for row in rows[start:start+10] for u in row['uids'])==[f'teacher_cad50_{i:02d}' for i in range(50)]
    logs.append(rows)
for a,b in zip(*logs):
    assert a['uids']==b['uids']
    assert [m['negative_sha256'] for m in a['meshes']]==[m['negative_sha256'] for m in b['meshes']]
    assert [(m['face_positives'],m['face_negatives']) for m in a['meshes']]==[(m['face_positives'],m['face_negatives']) for m in b['meshes']]
result=dict(passed=True,checked_updates=dict(zip(branches,map(len,logs))),paired_updates=min(map(len,logs)),
    comparison='Recorded UID sequence and actual negative SHA match at every common update',optimizer_updates=0)
output=ROOT/'repro_outputs'/f"PAIRED_LOG_AUDIT_{result['paired_updates']:05d}.json"
output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
