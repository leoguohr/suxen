"""Reuse the executed Control runner; only replace the fixed Face negative pool."""
from pathlib import Path
import ast,hashlib,json,shutil
ROOT=Path(__file__).resolve().parent
CONTROL=ROOT.parent/'math00_overfit100_B_continue100ep_20260915'
old=(CONTROL/'train_continue.py').read_text();src=old
def replace(a,b):
    global src
    assert a in src,a
    src=src.replace(a,b)
replace('"""Resume B epoch500 with unchanged LR and stop at epoch600/update15000."""',
        '"""Fixed actual-Face hard negatives; matched epoch500 to600 Control continuation."""')
replace('CHECK_EPOCHS=[500,525,550,575,600]',"CHECK_EPOCHS=[500,525,550,575,600]\nCONTROL=BASE/'diagnostics/math00_overfit100_B_continue100ep_20260915'")
replace('    old_manifest=cp[\'manifest\']', "    old_manifest=cp['manifest']\n    assert json.loads((CONTROL/'run/complete.json').read_text())['updates']==15000")
replace('    import copy\n    manifest=', '''    old_pools=pools
    mining=json.loads((ROOT/'mining_complete.json').read_text())
    assert mining['parent_sha256']==SOURCE_SHA and mining['optimizer_updates']==0
    assert [r['uid'] for r in mining['records']]==uids
    pools={}
    for u,record in zip(uids,mining['records']):
        p=ROOT/'augmented_pools'/f'{u}_pool.npz'
        assert sha(p)==record['new_pool_sha256'] and sha(ROOT/'pools'/f'{u}_pool.npz')==record['old_pool_sha256']
        with np.load(p) as d:pools[u]={k:d[k] for k in ['vertices','edges','positive','mixed']}
        assert all(np.array_equal(pools[u][k],old_pools[u][k]) for k in ['vertices','edges','positive'])
        count=len(old_pools[u]['mixed']);added=pools[u]['mixed'][count:]
        assert np.array_equal(pools[u]['mixed'][:count],old_pools[u]['mixed'])
        assert len(added)==record['selected']<=len(pools[u]['positive'])
        n=len(pools[u]['vertices']);keys=c.m.probe.keys(added,n)
        assert len(keys)==len(np.unique(keys))
        assert not np.isin(keys,c.m.probe.keys(np.concatenate([old_pools[u]['positive'],old_pools[u]['mixed']]),n)).any()
    import copy
    manifest=''' )
replace("branch='B_continue100ep',", "branch='fixed_actual_face_hardneg',\n        control_directory=str(CONTROL),face_pool_change='append fixed mined negatives; retain old candidate prefix',\n        mining_manifest_sha256=sha(ROOT/'mining_complete.json'),augmented_pool_sha256={r['uid']:r['new_pool_sha256'] for r in mining['records']},\n        old_pool_diagnostic=True,actual_evaluation_pool_reference='old pool for outside-pool FP accounting',")
replace("'train_continue.py','runtime.py','evaluate.py','construction_args.json'", "'train_hardneg.py','runtime.py','evaluate.py','construction_args.json','mine.py'")
replace("metrics=evaluate_mesh(detached,pools[u],scales)", "metrics=evaluate_mesh(detached,old_pools[u],scales)\n                _,old_parts,old_saved=b.full_objective(detached,[old_pools[u]],scales)")
replace("face_training_pool=c.metrics(fy,saved['face_train_logits']),**metrics)", "face_training_pool=c.metrics(fy,saved['face_train_logits']),\n                         old_pool_parts=old_parts[0],\n                         old_face_training_pool=c.metrics(np.r_[np.ones(len(old_pools[u]['positive']),dtype=bool),np.zeros(len(old_pools[u]['mixed']),dtype=bool)],old_saved[0]['face_train_logits']),\n                         actual_fp_outside_pool_reference='old_fixed_pool',**metrics)")
replace("del detached,saved", "del detached,saved,old_saved")
replace("                if old[k]!=new[k]:mismatches.append(dict(uid=old['uid'],field=k,old=old[k],new=new[k]))", "                value=new['old_pool_parts'] if k=='parts' else new['old_face_training_pool'] if k=='face_training_pool' else new[k]\n                if old[k]!=value:mismatches.append(dict(uid=old['uid'],field=k,old=old[k],new=value))")
replace("order=shuffle.permutation(uids).tolist();assert len(set(order))==100", "order=shuffle.permutation(uids).tolist();assert len(set(order))==100\n                assert order==json.loads((CONTROL/'run'/f'epoch-order-{epoch:03d}.json').read_text())['uids'],'Traversal differs from completed Control'")
compile(src,'train_hardneg.py','exec')
unchanged=[]
for name in ['objective','gradnorms','save','rng_snapshot','restore_rng']:
    a=next(x for x in ast.walk(ast.parse(old)) if isinstance(x,ast.FunctionDef) and x.name==name)
    b=next(x for x in ast.walk(ast.parse(src)) if isinstance(x,ast.FunctionDef) and x.name==name)
    assert ast.dump(a)==ast.dump(b),name
    unchanged.append(name)
(ROOT/'train_hardneg.py').write_text(src)
for n in ['runtime.py','evaluate.py','construction_args.json','verify_results.py','run.sh']:shutil.copy2(CONTROL/n,ROOT/n)
p=(CONTROL/'package_results.py').read_text()
p=p.replace("face_fp_outside_training_pool=r['face']['actual_fp_outside_training_pool'],joint_perfect=r['joint_perfect'])", "face_fp_outside_training_pool=r['face']['actual_fp_outside_training_pool'],joint_perfect=r['joint_perfect'],\n                   old_pool_edge_soft4=r['old_pool_parts']['edge'],old_pool_face_soft4=r['old_pool_parts']['face'])")
p=p.replace("face_soft4_mean=statistics.mean(r['face_soft4'] for r in group),", "face_soft4_mean=statistics.mean(r['face_soft4'] for r in group),\n            old_pool_edge_soft4_mean=statistics.mean(r['old_pool_edge_soft4'] for r in group),\n            old_pool_face_soft4_mean=statistics.mean(r['old_pool_face_soft4'] for r in group),")
p=p.replace("files=sorted(set(files))", "files.extend((ROOT/'mined').glob('*.npz'))\nfiles.extend((ROOT/'control_evaluations').glob('*.jsonl'))\nfiles=sorted(set(files))")
(ROOT/'package_results.py').write_text(p)
(ROOT/'code_verification.json').write_text(json.dumps(dict(control_runner=str(CONTROL/'train_continue.py'),
    control_runner_sha256=hashlib.sha256(old.encode()).hexdigest(),new_runner_sha256=hashlib.sha256(src.encode()).hexdigest(),
    ast_identical_functions=unchanged,unchanged_runtime_evaluation_and_construction=True),indent=2)+'\n')
print('Built fixed-pool branch, unchanged objective/optimizer/update arithmetic and actual reconstruction evaluator.')
