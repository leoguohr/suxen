"""Build one second-round fixed-pool intervention from the executed Control."""
from pathlib import Path
import ast,hashlib,json,shutil
ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/'math00_overfit100_low_lr_epoch800_900_20260917'
FIRST=ROOT.parent/'math00_overfit100_face_hardneg_20260915'
old=(PREV/'train_continue.py').read_text();src=old
def change(a,b):
    global src
    assert a in src,a
    src=src.replace(a,b)
change('"""Continue LowLR epoch800 to900 with unchanged optimizer, pools and LR."""',
       '"""Second fixed Face hard-negative round; paired epoch800 to900, same optimizer and LR."""')
change('CHECK_EPOCHS=[800,825,850,875,900]', "CHECK_EPOCHS=[800,825,850,875,900]\nCONTROL=BASE/'diagnostics/math00_overfit100_low_lr_epoch800_900_20260917'")
change('    old_pools=pools', '''    base_pools=pools
    assert json.loads((CONTROL/'run/complete.json').read_text())['updates']==22500
    parent_mining=json.loads((ROOT/'parent_mining_complete.json').read_text())
    assert sha(ROOT/'parent_mining_complete.json')==old_manifest['mining_manifest_sha256']
    old_pools={}
    for u in uids:
        p=ROOT/'common_pools'/f'{u}_pool.npz'
        assert sha(p)==old_manifest['augmented_pool_sha256'][u]
        with np.load(p) as d:old_pools[u]={k:d[k] for k in ['vertices','edges','positive','mixed']}
        assert all(np.array_equal(old_pools[u][k],base_pools[u][k]) for k in ['vertices','edges','positive'])
        assert np.array_equal(old_pools[u]['mixed'][:len(base_pools[u]['mixed'])],base_pools[u]['mixed'])
    parent_mining_by_uid={r['uid']:r for r in parent_mining['records']}''')
change("    assert sha(ROOT/'mining_complete.json')==old_manifest['mining_manifest_sha256']", "    assert mining['parent_sha256']==SOURCE_SHA")
change("    assert {r['uid']:r['new_pool_sha256'] for r in mining['records']}==old_manifest['augmented_pool_sha256']", "    assert {r['uid']:r['old_pool_sha256'] for r in mining['records']}==old_manifest['augmented_pool_sha256']")
change("sha(ROOT/'pools'/f'{u}_pool.npz')==record['old_pool_sha256']", "sha(ROOT/'common_pools'/f'{u}_pool.npz')==record['old_pool_sha256']")
change("branch='low_lr_continue100ep',lr_change='none; preserve parent LR exactly'", "branch='fixed_face_hardneg_round2',control_directory=str(CONTROL),lr_change='none; preserve parent LR exactly'")
change("face_pool_change='none; retain epoch800 augmented pool byte-for-byte'", "face_pool_change='append fixed round2 negatives to entire epoch800 pool; old prefix unchanged',common_pool_sha256=old_manifest['augmented_pool_sha256'],parent_mining_manifest_sha256=sha(ROOT/'parent_mining_complete.json')")
change("actual_evaluation_pool_reference='old pool for outside-pool FP accounting'", "actual_evaluation_pool_reference='original base pool for outside-pool FP accounting; common round1 pool for diagnostic loss'")
change("'hardneg_tracking.py']", "'hardneg_tracking.py','mine.py']")
change('augmented_pool_hashes_equal_parent=True,mining_manifest_equal_parent=True,total_fixed_added_negatives=mining[\'total_added\']',
       "common_pool_hashes_equal_parent=True,parent_mining_manifest_equal=True,old_pool_and_round1_prefix_preserved=True,total_round2_added_negatives=mining['total_added'],total_round1_negatives=parent_mining['total_added']")
change('metrics=evaluate_mesh(detached,old_pools[u],scales)', 'metrics=evaluate_mesh(detached,base_pools[u],scales)')
change("actual_fp_outside_pool_reference='old_fixed_pool',", "actual_fp_outside_pool_reference='original_base_pool',common_pool_diagnostic_reference='epoch800_round1_augmented_pool',\n                         first_round_negative_diagnostic=record_hardneg(ROOT/'first_round_logits',epoch,u,old_pools[u],parent_mining_by_uid[u],old_saved[0]),")
change("            for k in ['uid','parts','old_pool_parts','old_face_training_pool','edge','face_training_pool','gt_face_candidates','missing_gt_face_candidates','margins','joint_perfect','mined_negative_diagnostic']:\n                value=new[k]",
       "            for k in ['uid','parts','edge','face_training_pool','gt_face_candidates','missing_gt_face_candidates','margins','joint_perfect','mined_negative_diagnostic']:\n                value=new['old_pool_parts'] if k=='parts' else new['old_face_training_pool'] if k=='face_training_pool' else new['first_round_negative_diagnostic'] if k=='mined_negative_diagnostic' else new[k]")
change('order=shuffle.permutation(uids).tolist();assert len(set(order))==100', "order=shuffle.permutation(uids).tolist();assert len(set(order))==100\n                assert order==json.loads((CONTROL/'run'/f'epoch-order-{epoch:03d}.json').read_text())['uids'],'Paired Control traversal differs'")
compile(src,'train_continue.py','exec');(ROOT/'train_continue.py').write_text(src)
unchanged=[]
for name in ['objective','gradnorms','save','rng_snapshot','restore_rng']:
    a=next(x for x in ast.walk(ast.parse(old)) if isinstance(x,ast.FunctionDef) and x.name==name)
    b=next(x for x in ast.walk(ast.parse(src)) if isinstance(x,ast.FunctionDef) and x.name==name)
    assert ast.dump(a)==ast.dump(b),name
    unchanged.append(name)
def update_loop(text):
    return next(x for x in ast.walk(ast.parse(text)) if isinstance(x,ast.For) and isinstance(x.target,ast.Name) and x.target.id=='offset')
assert ast.dump(update_loop(old))==ast.dump(update_loop(src))
for n in ['runtime.py','evaluate.py','construction_args.json','hardneg_tracking.py','run.sh','verify_results.py']:
    shutil.copy2(PREV/n,ROOT/n)
shutil.copy2(FIRST/'mine.py',ROOT/'mining_source.py')
source=(FIRST/'mine.py').read_text();tree=ast.parse(source)
fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='mine_faces')
lines=source.splitlines();mine_fn='\n'.join(lines[min(x.lineno for x in fn.decorator_list)-1:fn.end_lineno])+'\n'
(ROOT/'mining_core.py').write_text('"""Unchanged full enumeration and deterministic top-K mining from round1."""\nfrom runtime import T,np,c\nfrom evaluate import FACE_CHUNK\n\n'+mine_fn)
compare=(FIRST/'compare_control.py').read_text()
compare=compare.replace('math00_overfit100_B_continue100ep_20260915',PREV.name).replace('[500,525,550,575,600]','[800,825,850,875,900]')
compare=compare.replace('Control_old_pool','Control_round1_pool').replace('Fixed_hardneg','Fixed_round2_hardneg')
compare=compare.replace("part=r.get('old_pool_parts',r['parts'])", "part=r['parts'] if path==CONTROL else r['old_pool_parts']")
compare=compare.replace("old_pool_diagnostic_comparable=True", "old_pool_diagnostic_comparable=True,common_pool_definition='epoch800 original plus round1 fixed negatives'")
(ROOT/'compare_control.py').write_text(compare)
package=(PREV/'package_results.py').read_text().replace("ROOT/'control_hardneg']", "ROOT/'control_hardneg',ROOT/'first_round_logits']")
package=package.replace("files.extend((ROOT/'mined').glob('*.npz'))", "files.extend((ROOT/'mined').glob('*.npz'))\nfiles.extend((ROOT/'first_round_mined').glob('*.npz'))")
(ROOT/'package_results.py').write_text(package)
job=(PREV/'run_job.py').read_text().replace("    ('train_continue.py'", "    ('mine.py','mining-console.log','mining_exit.json','mining_fixed_round2'),\n    ('train_continue.py'")
job=job.replace("    ('package_results.py'", "    ('compare_control.py','comparison-console.log','comparison_exit.json','comparing_saved_results'),\n    ('package_results.py'")
(ROOT/'run_job.py').write_text(job)
(ROOT/'code_verification.json').write_text(json.dumps(dict(source_runner=str(PREV/'train_continue.py'),
    source_sha256=hashlib.sha256(old.encode()).hexdigest(),runner_sha256=hashlib.sha256(src.encode()).hexdigest(),
    ast_identical_functions=unchanged,optimizer_microbatch_update_loop_identical=True,
    mining_source_sha256=hashlib.sha256(source.encode()).hexdigest(),mining_core_copied_exactly_from=str(FIRST/'mine.py'),
    unchanged_files={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['runtime.py','evaluate.py','construction_args.json','hardneg_tracking.py']}),indent=2)+'\n')
print('Built fixed round2 intervention; training objective/optimizer loop and mining core preserved.')
