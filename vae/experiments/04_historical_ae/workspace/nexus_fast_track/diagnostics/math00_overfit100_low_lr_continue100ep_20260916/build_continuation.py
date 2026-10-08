"""Derive a bounded, unchanged-LR continuation from the executed LowLR runner."""
from pathlib import Path
import ast, hashlib, json, re, shutil

ROOT = Path(__file__).resolve().parent
PREV = ROOT.parent / 'math00_overfit100_hardneg_lr03_from650_20260916'
old = (PREV / 'train_low_lr.py').read_text()
src = old

def change(a, b):
    global src
    assert a in src, a
    src = src.replace(a, b)

change('"""Lower reconstruction LR only, paired epoch650 to700 continuation."""',
       '"""Continue LowLR epoch650 to700 with unchanged optimizer, pools and LR."""')
change("PARENT=BASE/'diagnostics/math00_overfit100_hardneg_continue100ep_20260916'",
       "PARENT=BASE/'diagnostics/PARENT_PLACEHOLDER'")
change("SOURCE_SHA='7c4d4c7539eb4d4a0e483abd9457a44ed7a6700a7d3bff352c6e97fc8723f0e7'",
       "SOURCE_SHA='SHA_PLACEHOLDER'")
change('CHECK_EPOCHS=[650,675,700]', 'CHECK_EPOCHS=CHECKS_PLACEHOLDER')
change("    expected_optimizer=dict(cp['optimizer'],param_groups=[dict(g,lr=9e-7 if g['name']=='encoder_mu' else 9e-6) for g in cp['optimizer']['param_groups']])\n    for pg in opt.param_groups:pg['lr']=9e-7 if pg['name']=='encoder_mu' else 9e-6\n    assert same(opt.state_dict(),expected_optimizer)",
       "    assert same(opt.state_dict(),cp['optimizer'])  # Preserve LR and all Adam state.")
change("(3e-6 if pg['name']=='encoder_mu' else 3e-5)", "(9e-7 if pg['name']=='encoder_mu' else 9e-6)")
change("    assert json.loads((PARENT/'run/complete.json').read_text())['updates']==17500",
       "    assert json.loads((PARENT/'run/complete.json').read_text())['updates']==PARENT_END_PLACEHOLDER")
change("                assert order==json.loads((PARENT/'run'/f'epoch-order-{epoch:03d}.json').read_text())['uids'],'Paired batch order differs'\n", '')
change("branch='hardneg_low_lr_from650',control_directory=str(PARENT),lr_change='3e-6/3e-5 to 9e-7/9e-6 after full Adam restoration'",
       "branch='low_lr_continue100ep',lr_change='none; preserve parent LR exactly'")
change('additional_epochs=50,additional_updates=1250', 'additional_epochs=100,additional_updates=1250')
change('optimizer_equal_except_intended_lr=True', 'optimizer_entire_state_including_lr_equal=True')
change("'train_low_lr.py'", "'train_continue.py'")
change("'margins','joint_perfect'", "'margins','joint_perfect','mined_negative_diagnostic'")
# Replace numerical stage identifiers simultaneously, including filenames.
mapping={'16250':'17500','17500':'20000','16253':'17503','1250':'2500',
         '16251':'17501','17501':'20001','1251':'2501',
         '70000':'80000','650':'700','700':'800','651':'701','701':'801'}
src = re.sub(r'\d+', lambda m: mapping.get(m[0], m[0]), src)
src = src.replace('PARENT_PLACEHOLDER', PREV.name).replace('SHA_PLACEHOLDER',
    '20c75fc55fd0f432e327db25a96c8bc151008107e76b2ba9ac72e144edaa4e12')
src = src.replace('CHECKS_PLACEHOLDER','[700,725,750,775,800]').replace('PARENT_END_PLACEHOLDER','17500')
compile(src, 'train_continue.py', 'exec')
unchanged=[]
for name in ['objective','gradnorms','save','rng_snapshot','restore_rng','evaluate']:
    a=next(x for x in ast.walk(ast.parse(old)) if isinstance(x,ast.FunctionDef) and x.name==name)
    b=next(x for x in ast.walk(ast.parse(src)) if isinstance(x,ast.FunctionDef) and x.name==name)
    assert ast.dump(a)==ast.dump(b),name
    unchanged.append(name)
(ROOT/'train_continue.py').write_text(src)
for name in ['runtime.py','evaluate.py','construction_args.json','hardneg_tracking.py','mining_source.py','package_results.py']:
    shutil.copy2(PREV/name,ROOT/name)
v=(PREV/'verify_results.py').read_text().replace('[650,675,700]','CHECKS_PLACEHOLDER')
v=re.sub(r'\d+',lambda m:mapping.get(m[0],m[0]),v)
v=v.replace('CHECKS_PLACEHOLDER','[700,725,750,775,800]')
v=v.replace('parent_rng_replayed_all50_orders','parent_rng_replayed_all100_orders')
v=v.replace('each_mesh_additional_participations=50','each_mesh_additional_participations=100')
v=v.replace("lrs={g['name']:(9e-7 if g['name']=='encoder_mu' else 9e-6) for g in parent['optimizer']['param_groups']}",
            "lrs={g['name']:g['lr'] for g in parent['optimizer']['param_groups']}")
v=v.replace('fixed_new_lrs=lrs','lrs_unchanged_from_parent=lrs')
compile(v,'verify_results.py','exec');(ROOT/'verify_results.py').write_text(v)
(ROOT/'code_verification.json').write_text(json.dumps(dict(
    source_runner=str(PREV/'train_low_lr.py'),source_sha256=hashlib.sha256(old.encode()).hexdigest(),
    runner_sha256=hashlib.sha256(src.encode()).hexdigest(),ast_identical_functions=unchanged,
    unchanged_files={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
                     ['runtime.py','evaluate.py','construction_args.json','hardneg_tracking.py']}),indent=2)+'\n')
print('Built continuation; computation and evaluation functions unchanged.')
