"""Continue the completed 2000-update MidLR branch for 1000 new updates."""
import hashlib
import sys
from pathlib import Path

base = Path(__file__).resolve().parent
source = (base / 'run.py').read_text()
assert hashlib.sha256(source.encode()).hexdigest() == 'f4cc9bdb5a3b36f4ffdeee5c0bfaf72b574f9bd723850df239d531ac450beb62'
replacements = {
    'BASE=Path(__file__).resolve().parent': 'BASE=Path(__file__).resolve().parent.parent',
    "choices=['control','high10x']": "choices=['continue1000']",
    'STEPS=400;PARENT_UPDATES=1000': 'STEPS=1000;PARENT_UPDATES=3400',
    "START=SOURCE/'checkpoint-update1000.pt'": "START=BASE/'continue2000/checkpoint-update2000.pt'",
    'CHECKS={0,1,10,20,50,100,200,300,400}': 'CHECKS=set(range(0,1001,200))',
    'NOISE_CHECKS={0,200,400}': 'NOISE_CHECKS={0,400,600,800,1000}',
    "assert SEEDS['training']==c.SEEDS": "previous_seeds={s for key in ['evaluation','final_unseen'] for row in SEEDS[key] for s in row}|set(c.SEEDS)\nprevious_seeds.update(s for row in json.loads((BASE/'continue2000/manifest.json').read_text())['final_unseen_seeds'] for s in row)\nSEEDS['final_unseen']=[[2026092000+4*j+i for i in range(4)] for j in range(50)]\nassert not previous_seeds & {s for row in SEEDS['final_unseen'] for s in row}\nassert SEEDS['training']==c.SEEDS",
    "assert cp['completed_updates']==1000": "assert cp['completed_updates']==3400\nprovenance=cp['diagnostic_manifest']\nassert provenance['selected_uids']==c.UIDS",
    "cp['diagnostic_manifest']['source_sha256']": "provenance['source_sha256']",
    'assert DRAW_OFFSET==5200': 'assert DRAW_OFFSET==14800',
    "cp['diagnostic_manifest']['groups']": "provenance['groups']",
    "{'encoder_mu':1e-8,'decoder_body':1e-7,'edge_head':1e-7,'face_head':1e-7,'logvar':1e-4}": "{'encoder_mu':3*1e-8,'decoder_body':3e-7,'edge_head':3e-7,'face_head':3e-7,'logvar':1e-4}",
    "(1600 if group['name']=='logvar' else 1800)": "(4000 if group['name']=='logvar' else 4200)",
    "if args.branch=='high10x':\n    for group in optimizer.param_groups:\n        if group['name']!='logvar':group['lr']*=10": "assert all(g['weight_decay']==0 for g in optimizer.param_groups)",
    "lr_change='none' if args.branch=='control' else '10x encoder_mu, decoder_body, edge_head, face_head; logvar unchanged'": "lr_change='none; continue saved MidLR learning rates'",
    "optimizer='all five Adam states restored exactly from four-mesh step1000'": "optimizer='all five Adam states restored exactly from continue2000 final step2000',groups=provenance['groups'],source_sha256=provenance['source_sha256']",
    "comparison_rule='Both branches independently restore identical parent RNG, so update-wise epsilon hashes must match.'": "comparison_rule='Single continuous run; fresh epsilon from saved continue2000 final RNG; monitoring noise has participated in selection.'",
    "tag='final_unseen' if final else 'independent'": "tag='final_unseen' if final else 'monitoring'",
    "        if step in NOISE_CHECKS:\n            save_checkpoint(step);noise[step]=noise_evaluation(step);record['noise_evaluation_after_update']=noise[step]": "        if step in CHECKS:save_checkpoint(step)\n        if step in NOISE_CHECKS:\n            noise[step]=noise_evaluation(step);record['noise_evaluation_after_update']=noise[step]",
    "print('BEGIN',args.branch,effective_lrs,flush=True)": "import subprocess\nsubprocess.run([__import__('sys').executable,str(BASE/'verify_continue1000.py'),'--start-only'],check=True)\nprint('BEGIN',args.branch,effective_lrs,flush=True)",
}
for old, new in replacements.items():
    assert source.count(old) == 1, old
    source = source.replace(old, new)
target = base / 'continue1000' / 'effective_run.py'
target.parent.mkdir(exist_ok=True)
target.write_text(source)
compile(source, str(target), 'exec')
if '--prepare-only' not in sys.argv:
    exec(compile(source, str(target), 'exec'), {'__file__': str(target), '__name__': '__main__'})
