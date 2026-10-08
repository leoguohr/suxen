"""Create the bounded B continuation using the already executed LR-pair runner."""
from pathlib import Path
import ast, hashlib, json, re, shutil

ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/'math00_overfit100_lr03_pair_20260915/B_lr03'
old=(PREV/'train_pair.py').read_text()
src=old
start=src.index("    branch=json.loads")
end=src.index("    for p,digest in old_manifest",start)
src=src[:start]+src[end:]
start=src.index('    # Change LR only after')
end=src.index('    assert all(p not in opt.state',start)
src=src[:start]+src[end:]
src=src.replace("1e-5 if pg['name']=='encoder_mu' else 1e-4", "3e-6 if pg['name']=='encoder_mu' else 3e-5")
src=src.replace("warmup='none; constant per-branch LR',branch=branch['name'],lr_factor=factor,paired_order_plan_sha256=branch['order_plan_sha256'],", "warmup='none; retain B terminal LR without further multiplication',\n        branch='B_continue100ep',lrs={pg['name']:pg['lr'] for pg in opt.param_groups},")
src=src.replace("        optimizer_equal_except_intended_lr=True,lr_factor=factor,paired_order_plan_sha256=branch['order_plan_sha256'],", "        optimizer_entire_state_including_lr_equal=True,")
src=src.replace("                assert order==plan['orders'][str(epoch)],'Paired traversal differs'\n", '')
src=src.replace("PARENT=BASE/'diagnostics/math00_overfit100_continue200ep_20260915_v2'", "PARENT=BASE/'diagnostics/math00_overfit100_lr03_pair_20260915/B_lr03'")
src=src.replace('0804e7dabccd5dfd027215be4a2a8c98281aea8ffd148090913b658209371d14','dc28a6e72a8e30a84d4536cbd09168d48635d66c5a924a3f52ef907c36fc0038')
mapping={'10000':'12500','12500':'15000','400':'500','500':'600','50000':'60000','10003':'12503'}
src=re.sub(r'\b(?:10000|12500|400|500|50000|10003)\b',lambda m:mapping[m[0]],src)
src=src.replace('checkpoint-update10000.pt','checkpoint-update12500.pt')
src=src.replace('range(401,501)','range(501,601)')
src=src.replace('epoch400','epoch500')
src=src.replace("out/'eval-summary-epoch500.json').read_text()),stopped_at_budget=True", "out/'eval-summary-epoch600.json').read_text()),stopped_at_budget=True")
src=re.sub(r'^CHECK_EPOCHS=.*$', 'CHECK_EPOCHS=[500,525,550,575,600]',src,flags=re.M)
src=src.replace('COMPLETE_12500_ADDITIONAL2500','COMPLETE_15000_ADDITIONAL2500')
src=src.replace("'train_pair.py'","'train_continue.py'")
src=src.replace('"""Fixed100 LR pair: each branch resumes update10000 and stops at12500."""', '"""Resume B epoch500 with unchanged LR and stop at epoch600/update15000."""')
src=src.replace('    manifest.update(random_initialization=False', "    for key in ['lr_factor','paired_order_plan_sha256']:manifest.pop(key,None)\n    manifest.update(random_initialization=False")
compile(src,'train_continue.py','exec')
old_ast,new_ast=ast.parse(old),ast.parse(src)
unchanged=[]
for name in ['objective','gradnorms','save','evaluate','rng_snapshot','restore_rng']:
    a=next(x for x in ast.walk(old_ast) if isinstance(x,ast.FunctionDef) and x.name==name)
    b=next(x for x in ast.walk(new_ast) if isinstance(x,ast.FunctionDef) and x.name==name)
    assert ast.dump(a)==ast.dump(b),name
    unchanged.append(name)
(ROOT/'train_continue.py').write_text(src)
for name in ['runtime.py','evaluate.py','construction_args.json','package_results.py']:
    shutil.copy2(PREV/name,ROOT/name)
(ROOT/'code_verification.json').write_text(json.dumps(dict(
    source_runner=str(PREV/'train_pair.py'),source_sha256=hashlib.sha256(old.encode()).hexdigest(),
    new_sha256=hashlib.sha256(src.encode()).hexdigest(),ast_identical_functions=unchanged,
    unchanged_files={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['runtime.py','evaluate.py','construction_args.json']}
),indent=2)+'\n')
print('Created continuation; objective, checkpoint saving, evaluation and RNG helpers unchanged.')
