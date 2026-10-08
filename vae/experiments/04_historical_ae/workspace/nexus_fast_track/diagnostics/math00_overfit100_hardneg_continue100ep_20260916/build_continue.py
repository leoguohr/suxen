"""Continue the completed hard-negative branch, preserving its fixed pool and optimizer."""
from pathlib import Path
import ast,hashlib,json,re,shutil
ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/'math00_overfit100_face_hardneg_20260915'
old=(PREV/'train_hardneg.py').read_text();src=old
def change(a,b):
    global src
    assert a in src,a
    src=src.replace(a,b)
change('"""Fixed actual-Face hard negatives; matched epoch500 to600 Control continuation."""',
       '"""Continue fixed hard-negative pool from epoch600 to700; no new mining."""')
change("PARENT=BASE/'diagnostics/math00_overfit100_lr03_pair_20260915/B_lr03'", "PARENT=BASE/'diagnostics/math00_overfit100_face_hardneg_20260915'")
change("SOURCE=PARENT/'run/checkpoint-update12500.pt'", "SOURCE=PARENT/'run/checkpoint-update15000.pt'")
change("SOURCE_SHA='dc28a6e72a8e30a84d4536cbd09168d48635d66c5a924a3f52ef907c36fc0038'", "SOURCE_SHA='cf3f780ec9df15c60d49949891d8e5a6ff964940e63a837465e4744c9cd93082'")
change("CONTROL=BASE/'diagnostics/math00_overfit100_B_continue100ep_20260915'\n",'')
change("    assert json.loads((CONTROL/'run/complete.json').read_text())['updates']==15000\n",'')
change("                assert order==json.loads((CONTROL/'run'/f'epoch-order-{epoch:03d}.json').read_text())['uids'],'Traversal differs from completed Control'\n",'')
change("    assert mining['parent_sha256']==SOURCE_SHA and mining['optimizer_updates']==0", "    assert sha(ROOT/'mining_complete.json')==old_manifest['mining_manifest_sha256']\n    assert mining['optimizer_updates']==0 and mining['fixed_pool_no_refresh']\n    assert {r['uid']:r['new_pool_sha256'] for r in mining['records']}==old_manifest['augmented_pool_sha256']")
change("branch='fixed_actual_face_hardneg',", "branch='hardneg_continue100ep',")
change("        control_directory=str(CONTROL),face_pool_change='append fixed mined negatives; retain old candidate prefix',", "        face_pool_change='none; retain epoch600 augmented pool byte-for-byte',")
change("    for key in ['lr_factor','paired_order_plan_sha256']:manifest.pop(key,None)", "    manifest['historical_control_directory']=manifest.pop('control_directory',None)\n    for key in ['lr_factor','paired_order_plan_sha256']:manifest.pop(key,None)")
change("'train_hardneg.py','runtime.py','evaluate.py','construction_args.json','mine.py'", "'train_continue.py','runtime.py','evaluate.py','construction_args.json'")
change("        parent_source_hashes_verified=True,logvar_parameter_hash=freeze_hash))", "        parent_source_hashes_verified=True,logvar_parameter_hash=freeze_hash,\n        augmented_pool_hashes_equal_parent=True,mining_manifest_equal_parent=True,total_fixed_added_negatives=mining['total_added']))")
change("            for k in ['uid','parts','edge','face_training_pool','gt_face_candidates','missing_gt_face_candidates','margins','joint_perfect']:\n                value=new['old_pool_parts'] if k=='parts' else new['old_face_training_pool'] if k=='face_training_pool' else new[k]", "            for k in ['uid','parts','old_pool_parts','old_face_training_pool','edge','face_training_pool','gt_face_candidates','missing_gt_face_candidates','margins','joint_perfect']:\n                value=new[k]")
# Shift only continuation counters; retain the original mining provenance unchanged.
mapping={'12500':'15000','15000':'17500','500':'600','600':'700','60000':'70000','12503':'15003'}
src=re.sub(r'\b(?:12500|15000|500|600|60000|12503)\b',lambda m:mapping[m[0]],src)
src=src.replace('range(501,601)','range(601,701)').replace('epoch500','epoch600')
src=src.replace("out/'eval-summary-epoch600.json').read_text()),stopped_at_budget=True", "out/'eval-summary-epoch700.json').read_text()),stopped_at_budget=True")
src=re.sub(r'^CHECK_EPOCHS=.*$', 'CHECK_EPOCHS=[600,625,650,675,700]',src,flags=re.M)
src=src.replace('COMPLETE_15000_ADDITIONAL2500','COMPLETE_17500_ADDITIONAL2500')
# The path has no word boundary before its numeric suffix and must remain update15000.
assert "SOURCE=PARENT/'run/checkpoint-update15000.pt'" in src
compile(src,'train_continue.py','exec')
unchanged=[]
for name in ['objective','gradnorms','save','evaluate','rng_snapshot','restore_rng']:
    a=next(x for x in ast.walk(ast.parse(old)) if isinstance(x,ast.FunctionDef) and x.name==name)
    b=next(x for x in ast.walk(ast.parse(src)) if isinstance(x,ast.FunctionDef) and x.name==name)
    assert ast.dump(a)==ast.dump(b),name
    unchanged.append(name)
(ROOT/'train_continue.py').write_text(src)
for n in ['runtime.py','evaluate.py','construction_args.json','package_results.py','run.sh']:shutil.copy2(PREV/n,ROOT/n)
shutil.copy2(PREV/'mine.py',ROOT/'mining_source.py')
v=(PREV/'verify_results.py').read_text()
mapping={'12500':'15000','15000':'17500','12501':'15001','15001':'17501','500':'600','600':'700','501':'601','601':'701','525':'625','550':'650','575':'675'}
v=re.sub(r'\b(?:12500|15000|12501|15001|500|600|501|601|525|550|575)\b',lambda m:mapping[m[0]],v)
(ROOT/'verify_results.py').write_text(v)
(ROOT/'code_verification.json').write_text(json.dumps(dict(source_runner=str(PREV/'train_hardneg.py'),
    source_sha256=hashlib.sha256(old.encode()).hexdigest(),runner_sha256=hashlib.sha256(src.encode()).hexdigest(),
    ast_identical_functions=unchanged,unchanged_files={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['runtime.py','evaluate.py','construction_args.json']}),indent=2)+'\n')
print('Continuation created; objective, evaluator, checkpoint and RNG helpers unchanged.')
