"""Paired 804-only objective diagnostic; four packed meshes, J804/4."""
import hashlib
import sys
from pathlib import Path

base = Path(__file__).resolve().parent
source = (base / 'run.py').read_text()
assert hashlib.sha256(source.encode()).hexdigest() == 'f4cc9bdb5a3b36f4ffdeee5c0bfaf72b574f9bd723850df239d531ac450beb62'
replacements = {
    'BASE=Path(__file__).resolve().parent': 'BASE=Path(__file__).resolve().parent.parent',
    "choices=['control','high10x']": "choices=['only804']",
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
    "print('BEGIN',args.branch,effective_lrs,flush=True)": "import subprocess\nsubprocess.run([__import__('sys').executable,str(BASE/'verify_only804.py'),'--start-only'],check=True)\nprint('BEGIN',args.branch,effective_lrs,flush=True)",
}
for old, new in replacements.items():
    assert source.count(old) == 1, old
    source = source.replace(old, new)

# The only optimization change is the selected mesh contribution.
extra={'cp,model,batch=c.m.probe.setup_model(START)': "control_root=BASE/'continue1000'\ncontrol_manifest=json.loads((control_root/'manifest.json').read_text())\nassert c.m.probe.digest(START)==control_manifest['parent_sha256']\ncontrol_rows=[json.loads(line) for line in (control_root/'updates.jsonl').read_text().splitlines()]\nassert [r['update'] for r in control_rows]==list(range(1,1001))\nassert c.UIDS[3]=='nexus_2k_001333'\ncp,model,batch=c.m.probe.setup_model(START)", "c.write(ROOT/'manifest.json',manifest)": "manifest.update(loss='(Edge804 + Face804 + 1e-4 KL804)/4; other mesh objectives excluded',objective_mesh_index=3,objective_coefficient=0.25,control=str(control_root),control_updates_sha256=c.m.probe.digest(control_root/'updates.jsonl'),comparison_rule='Same parent weights/Adam/RNG; assert each of four epsilon hashes against Control before every update; compare per-mesh metrics only.')\nc.write(ROOT/'manifest.json',manifest)", "        rec,parts,_=b.full_objective(rows,data,scales);kl,ki=kl_parts(rows);assert abs(float(rec.detach())-sum(x['edge']+x['face'] for x in parts)/M)<1e-5": "        paired=control_rows[step-1]\n        assert eps_hashes==paired['epsilon_sha256'] and tensor_hash(rng_before)==paired['rng_before_sha256'] and tensor_hash(rng_after)==paired['rng_after_sha256']\n        selected=tuple((group[3],) for group in rows)\n        rec804,selected_parts,_=b.full_objective(selected,[data[3]],scales)\n        kl804,selected_ki=kl_parts(selected)\n        with torch.no_grad():\n            _,parts,_=b.full_objective(rows,data,scales)\n            _,ki_all=kl_parts(rows)\n        assert selected_parts[0]==parts[3]\n        rec=rec804/4;kl=kl804/4;ki=dict(selected_mesh=selected_ki,all_meshes_diagnostic_only=ki_all)\n        if step==1:\n            tensors=[t for group in rows for t in group]\n            probes=torch.autograd.grad(rec+1e-4*kl,tensors,allow_unused=True,retain_graph=True)\n            assert all(g is None or not bool(g.count_nonzero()) for j,g in enumerate(probes) if j%4!=3)\n            c.write(ROOT/'objective_scope_check.json',dict(other_mesh_output_gradients_zero=True,coefficient=0.25,selected_uid=c.UIDS[3]))\n            del probes,tensors", 'del value,rec,kl,rows;optimizer.step()': 'del value,rec,kl,rows,selected,rec804,kl804;optimizer.step()', 'record=dict(update=step,beta=1e-4': 'record=dict(update=step,paired_control_epsilon_verified=True,objective_coefficient=0.25,beta=1e-4'}
for old,new in extra.items():
    assert source.count(old)==1,old
    source=source.replace(old,new)
target = base / 'only804' / 'effective_run.py'
target.parent.mkdir(exist_ok=True)
target.write_text(source)
compile(source, str(target), 'exec')
if '--prepare-only' not in sys.argv:
    exec(compile(source, str(target), 'exec'), {'__file__': str(target), '__name__': '__main__'})
