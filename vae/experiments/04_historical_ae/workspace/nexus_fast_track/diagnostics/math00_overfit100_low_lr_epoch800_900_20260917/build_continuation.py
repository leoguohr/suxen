"""Derive the next fixed-budget continuation without changing training computation."""
from pathlib import Path
import ast, hashlib, json, re, shutil

ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/'math00_overfit100_low_lr_continue100ep_20260916'
old=(PREV/'train_continue.py').read_text()
old_parent="math00_overfit100_hardneg_lr03_from650_20260916"
old_sha='20c75fc55fd0f432e327db25a96c8bc151008107e76b2ba9ac72e144edaa4e12'
new_sha='38087aeefbc12444cb51ba35dbe3715cc0f785c30cd6ed53e093a0ee7b929f80'
mapping={'17500':'20000','17501':'20001','17503':'20003','20000':'22500','20001':'22501',
         '700':'800','701':'801','725':'825','750':'850','775':'875','800':'900','801':'901','80000':'90000'}
def stage(text):
    return re.sub(r'\d+',lambda m:mapping.get(m[0],m[0]),text)
src=old.replace(old_parent,'PARENT_PLACEHOLDER').replace(old_sha,'SHA_PLACEHOLDER')
src=stage(src).replace('PARENT_PLACEHOLDER',PREV.name).replace('SHA_PLACEHOLDER',new_sha)
compile(src,'train_continue.py','exec')
unchanged=[]
for name in ['objective','gradnorms','save','rng_snapshot','restore_rng','evaluate']:
    a=next(x for x in ast.walk(ast.parse(old)) if isinstance(x,ast.FunctionDef) and x.name==name)
    b=next(x for x in ast.walk(ast.parse(src)) if isinstance(x,ast.FunctionDef) and x.name==name)
    assert ast.dump(a)==ast.dump(b),name
    unchanged.append(name)
(ROOT/'train_continue.py').write_text(src)
for name in ['verify_results.py','run_job.py','prepare_remote.py']:
    text=(PREV/name).read_text().replace(old_parent,'PARENT_PLACEHOLDER').replace(old_sha,'SHA_PLACEHOLDER')
    text=stage(text).replace('PARENT_PLACEHOLDER',PREV.name).replace('SHA_PLACEHOLDER',new_sha)
    compile(text,name,'exec');(ROOT/name).write_text(text)
for name in ['runtime.py','evaluate.py','construction_args.json','hardneg_tracking.py','mining_source.py','package_results.py','run.sh']:
    shutil.copy2(PREV/name,ROOT/name)
(ROOT/'code_verification.json').write_text(json.dumps(dict(
    source_runner=str(PREV/'train_continue.py'),source_sha256=hashlib.sha256(old.encode()).hexdigest(),
    runner_sha256=hashlib.sha256(src.encode()).hexdigest(),ast_identical_functions=unchanged,
    unchanged_files={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
        ['runtime.py','evaluate.py','construction_args.json','hardneg_tracking.py']}),indent=2)+'\n')
print('Prepared epoch800 to900; objective, evaluation, checkpoint and RNG functions unchanged.')
