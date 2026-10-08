"""Reuse the verified continuation; vary only reconstruction learning rates."""
from pathlib import Path
import re,shutil,hashlib,ast
ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/'math00_overfit100_continue200ep_20260915_v2'
src=(PREV/'train_continue.py').read_text()
assert hashlib.sha256(src.encode()).hexdigest()=='5861bb17cbe4f7efccc2331d916e63555e44cb9da5a444340afd044b207909d5'
src='"""Fixed100 LR pair: each branch resumes update10000 and stops at12500."""\n'+src.split('\n',1)[1]
src=src.replace("PARENT=BASE/'diagnostics/math00_overfit100_fresh_20260914'", "PARENT=BASE/'diagnostics/math00_overfit100_continue200ep_20260915_v2'")
src=src.replace("SOURCE=PARENT/'run/checkpoint-update05000.pt'", "SOURCE=PARENT/'run/checkpoint-update10000.pt'")
src=src.replace('6a652045b865b776ca9bfbddd68bf1fc454c1fe29ec5aa83dec6cc3629186457','0804e7dabccd5dfd027215be4a2a8c98281aea8ffd148090913b658209371d14')
mapping={'200':'400','400':'500','5000':'10000','10000':'12500','40000':'50000','5003':'10003'}
src=re.sub(r'\b(?:200|400|5000|10000|40000|5003)\b',lambda m:mapping[m[0]],src)
src=src.replace('additional_epochs=400,additional_updates=10000','additional_epochs=100,additional_updates=2500')
src=src.replace('additional_updates=10000,mesh_participations=50000','additional_updates=2500,mesh_participations=50000')
src=src.replace('range(201,401)','range(401,501)')
src=re.sub(r'^CHECK_EPOCHS=.*$', 'CHECK_EPOCHS=[400,425,450,475,500]',src,flags=re.M)
src=src.replace('epoch200','epoch400')
src=src.replace("out/'eval-summary-epoch400.json').read_text()),stopped_at_budget=True", "out/'eval-summary-epoch500.json').read_text()),stopped_at_budget=True")
src=src.replace('COMPLETE_10000_ADDITIONAL5000','COMPLETE_12500_ADDITIONAL2500')
src=src.replace("'train_continue.py'","'train_pair.py'")
src=src.replace('def main():','def main(verify_only=False):')
src=src.replace('    old_manifest=cp[\'manifest\']', '''    old_manifest=cp['manifest']
    branch=json.loads((ROOT/'branch_config.json').read_text())
    assert branch['name'] in ['A_hold','B_lr03']
    factor=branch['lr_factor'];assert factor==({'A_hold':1.,'B_lr03':.3}[branch['name']])
    plan_path=ROOT.parent/'paired_epoch_orders.json'
    assert sha(plan_path)==branch['order_plan_sha256']
    plan=json.loads(plan_path.read_text());assert plan['parent_sha256']==SOURCE_SHA''')
needle='    assert all(p not in opt.state for p in model.autoencoder.log_variance.parameters())'
src=src.replace(needle,'''    # Change LR only after loading and verifying every historical Adam state.
    expected_optimizer=dict(cp['optimizer'],param_groups=[dict(g,lr=g['lr']*factor) for g in cp['optimizer']['param_groups']])
    for pg in opt.param_groups:pg['lr']*=factor
    assert same(opt.state_dict(),expected_optimizer)
'''+needle)
src=src.replace("warmup='none in continuation; retain terminal parent LR'", "warmup='none; constant per-branch LR',branch=branch['name'],lr_factor=factor,paired_order_plan_sha256=branch['order_plan_sha256']")
src=src.replace('        lrs={pg[\'name\']:pg[\'lr\'] for pg in opt.param_groups},', "        optimizer_equal_except_intended_lr=True,lr_factor=factor,paired_order_plan_sha256=branch['order_plan_sha256'],\n        lrs={pg['name']:pg['lr'] for pg in opt.param_groups},")
src=src.replace("        print('RESUME_BASELINE_VERIFIED',flush=True)", "        print('RESUME_BASELINE_VERIFIED',flush=True)\n        if verify_only:\n            write(out/'status.json',dict(state='baseline_verified_waiting',updates=10000,epoch=400,optimizer_updates_this_process=0))\n            return")
src=src.replace('order=shuffle.permutation(uids).tolist();assert len(set(order))==100','order=shuffle.permutation(uids).tolist();assert len(set(order))==100\n                assert order==plan[\'orders\'][str(epoch)],\'Paired traversal differs\'')
src=src.replace("if __name__=='__main__':\n    main()", "if __name__=='__main__':\n    import argparse\n    parser=argparse.ArgumentParser();parser.add_argument('--verify-only',action='store_true')\n    main(parser.parse_args().verify_only)")
compile(src,'train_pair.py','exec')
original=ast.parse((PREV/'train_continue.py').read_text());new=ast.parse(src)
for name in ['objective','gradnorms','save','evaluate']:
    a=next(x for x in ast.walk(original) if isinstance(x,ast.FunctionDef) and x.name==name)
    b=next(x for x in ast.walk(new) if isinstance(x,ast.FunctionDef) and x.name==name)
    assert ast.dump(a)==ast.dump(b),name
for branch in ['A_hold','B_lr03']:
    folder=ROOT/branch;folder.mkdir(exist_ok=True);(folder/'train_pair.py').write_text(src)
    for name in ['runtime.py','evaluate.py','construction_args.json','package_results.py']:shutil.copy2(PREV/name,folder/name)
print('Generated identical runners; objective, scoring/evaluation and checkpoint saving preserved.')
