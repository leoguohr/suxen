"""Verify the 71 updates lost at container shutdown were replayed exactly."""
from pathlib import Path
import json

r=Path(__file__).resolve().parent
old=r.with_name(r.name+'_interrupted_update0071')
prior=[json.loads(x) for x in (old/'updates.jsonl').read_text().splitlines()]
current=[json.loads(x) for x in (r/'updates.jsonl').read_text().splitlines()]
assert len(prior)==71 and len(current)==500
for a,b in zip(prior,current):
    assert {k:v for k,v in a.items() if k!='seconds'}=={k:v for k,v in b.items() if k!='seconds'},a['update']
a=json.loads((old/'eval-update0000.json').read_text())
b=json.loads((r/'eval-update0000.json').read_text())
assert {k:v for k,v in a.items() if k!='checkpoint_sha256'}=={k:v for k,v in b.items() if k!='checkpoint_sha256'}
assert json.loads((old/'errors-update0000.json').read_text())==json.loads((r/'errors-update0000.json').read_text())
result=dict(interrupted_archive=str(old),interrupted_updates=71,surviving_updates=500,
            total_executed_including_discarded=571,replayed_first71_exact_excluding_walltime=True,
            baseline_reconstruction_and_error_identities_exact=True,final_cumulative_updates=5000)
(r/'recovery_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
