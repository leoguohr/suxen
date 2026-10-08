"""Continue the frozen MidLR run for exactly 2000 additional updates."""
import hashlib
import sys
from pathlib import Path

base = Path(__file__).resolve().parent
source = (base / 'run.py').read_text()
assert hashlib.sha256(source.encode()).hexdigest() == 'f4cc9bdb5a3b36f4ffdeee5c0bfaf72b574f9bd723850df239d531ac450beb62'
replacements = {
    'BASE=Path(__file__).resolve().parent': 'BASE=Path(__file__).resolve().parent.parent',
    "choices=['control','high10x']": "choices=['continue2000']",
    'STEPS=400;PARENT_UPDATES=1000': 'STEPS=2000;PARENT_UPDATES=1400',
    "START=SOURCE/'checkpoint-update1000.pt'": "START=BASE/'mid3x/checkpoint-update0400.pt'\nassert c.m.probe.digest(START)=='640a981738f96992ba5dd8a6f8a5b0d0306a32560af8edabc195f010d23ba8b1'",
    'CHECKS={0,1,10,20,50,100,200,300,400}': 'CHECKS=set(range(0,2001,200))',
    'NOISE_CHECKS={0,200,400}': 'NOISE_CHECKS={0,400,800,1200,1600,1800,2000}',
    "assert SEEDS['training']==c.SEEDS": "previous_seeds={s for key in ['evaluation','final_unseen'] for row in SEEDS[key] for s in row}|set(c.SEEDS)\nSEEDS['final_unseen']=[[2026091300+4*j+i for i in range(4)] for j in range(50)]\nassert not previous_seeds & {s for row in SEEDS['final_unseen'] for s in row}\nassert SEEDS['training']==c.SEEDS",
    "assert cp['completed_updates']==1000": "assert cp['completed_updates']==1400\n# Only grouping/provenance metadata comes from the verified ancestor.\nancestor_path=SOURCE/'checkpoint-update1000.pt'\nassert c.m.probe.digest(ancestor_path)==cp['diagnostic_manifest']['parent_sha256']\nancestor=torch.load(ancestor_path,map_location='cpu',mmap=True,weights_only=False)\nprovenance=ancestor['diagnostic_manifest']\nassert cp['diagnostic_manifest']['backend_sha']==provenance['backend_sha']\nassert cp['diagnostic_manifest']['selected_uids']==c.UIDS",
    "cp['diagnostic_manifest']['source_sha256']": "provenance['source_sha256']",
    'assert DRAW_OFFSET==5200': 'assert DRAW_OFFSET==6800',
    "cp['diagnostic_manifest']['groups']": "provenance['groups']",
    "{'encoder_mu':1e-8,'decoder_body':1e-7,'edge_head':1e-7,'face_head':1e-7,'logvar':1e-4}": "{'encoder_mu':3*1e-8,'decoder_body':3e-7,'edge_head':3e-7,'face_head':3e-7,'logvar':1e-4}",
    "(1600 if group['name']=='logvar' else 1800)": "(2000 if group['name']=='logvar' else 2200)",
    "if args.branch=='high10x':\n    for group in optimizer.param_groups:\n        if group['name']!='logvar':group['lr']*=10": "assert all(g['weight_decay']==0 for g in optimizer.param_groups)",
    "lr_change='none' if args.branch=='control' else '10x encoder_mu, decoder_body, edge_head, face_head; logvar unchanged'": "lr_change='none; continue saved MidLR learning rates'",
    "optimizer='all five Adam states restored exactly from four-mesh step1000'": "optimizer='all five Adam states restored exactly from MidLR step400',groups=provenance['groups'],source_sha256=provenance['source_sha256']",
    "comparison_rule='Both branches independently restore identical parent RNG, so update-wise epsilon hashes must match.'": "comparison_rule='Single continuous run; fresh epsilon from saved MidLR RNG; monitoring noise has participated in selection.'",
    "tag='final_unseen' if final else 'independent'": "tag='final_unseen' if final else 'monitoring'",
    "        if step in NOISE_CHECKS:\n            save_checkpoint(step);noise[step]=noise_evaluation(step);record['noise_evaluation_after_update']=noise[step]": "        if step in CHECKS:save_checkpoint(step)\n        if step in NOISE_CHECKS:\n            noise[step]=noise_evaluation(step);record['noise_evaluation_after_update']=noise[step]",
}
for old, new in replacements.items():
    assert source.count(old) == 1, old
    source = source.replace(old, new)
target = base / 'continue2000' / 'effective_run.py'
target.parent.mkdir(exist_ok=True)
target.write_text(source)
compile(source, str(target), 'exec')
if '--prepare-only' not in sys.argv:
    exec(compile(source, str(target), 'exec'), {'__file__': str(target), '__name__': '__main__'})
