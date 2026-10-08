"""Single LR intervention at epoch650, with unchanged fixed-pool training computation."""
from pathlib import Path
import ast,hashlib,json,shutil
ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/'math00_overfit100_hardneg_continue100ep_20260916'
old=(PREV/'train_continue.py').read_text();src=old
def change(a,b):
    global src
    assert a in src,a
    src=src.replace(a,b)
change('"""Continue fixed hard-negative pool from epoch600 to700; no new mining."""','"""Lower reconstruction LR only, paired epoch650 to700 continuation."""')
change('from evaluate import evaluate_mesh,FACE_SECONDS','from evaluate import evaluate_mesh,FACE_SECONDS\nfrom hardneg_tracking import record as record_hardneg')
change("PARENT=BASE/'diagnostics/math00_overfit100_face_hardneg_20260915'", "PARENT=BASE/'diagnostics/math00_overfit100_hardneg_continue100ep_20260916'")
change("SOURCE=PARENT/'run/checkpoint-update15000.pt'", "SOURCE=PARENT/'run/checkpoint-update16250.pt'")
change("SOURCE_SHA='cf3f780ec9df15c60d49949891d8e5a6ff964940e63a837465e4744c9cd93082'", "SOURCE_SHA='7c4d4c7539eb4d4a0e483abd9457a44ed7a6700a7d3bff352c6e97fc8723f0e7'")
change('CHECK_EPOCHS=[600,625,650,675,700]','CHECK_EPOCHS=[650,675,700]')
change("    assert all(p not in opt.state for p in model.autoencoder.log_variance.parameters())", '''    expected_optimizer=dict(cp['optimizer'],param_groups=[dict(g,lr=9e-7 if g['name']=='encoder_mu' else 9e-6) for g in cp['optimizer']['param_groups']])
    for pg in opt.param_groups:pg['lr']=9e-7 if pg['name']=='encoder_mu' else 9e-6
    assert same(opt.state_dict(),expected_optimizer)
    assert all(p not in opt.state for p in model.autoencoder.log_variance.parameters())''')
change("    old_pools=pools", "    old_pools=pools\n    assert json.loads((PARENT/'run/complete.json').read_text())['updates']==17500")
change("    pools={}\n", "    mining_by_uid={r['uid']:r for r in mining['records']}\n    pools={}\n")
change("branch='hardneg_continue100ep',", "branch='hardneg_low_lr_from650',control_directory=str(PARENT),lr_change='3e-6/3e-5 to 9e-7/9e-6 after full Adam restoration',")
change("warmup='none; retain B terminal LR without further multiplication'", "warmup='none; constant 9e-7/9e-6 after restored Adam'")
change("'train_continue.py','runtime.py','evaluate.py','construction_args.json'", "'train_low_lr.py','runtime.py','evaluate.py','construction_args.json','hardneg_tracking.py'")
change("optimizer_entire_state_including_lr_equal=True,", "optimizer_equal_except_intended_lr=True,lr_before={g['name']:g['lr'] for g in cp['optimizer']['param_groups']},")
change("actual_fp_outside_pool_reference='old_fixed_pool',**metrics)", "actual_fp_outside_pool_reference='old_fixed_pool',\n                         mined_negative_diagnostic=record_hardneg(ROOT/'hardneg_logits',epoch,u,pools[u],mining_by_uid[u],saved),**metrics)")
change("order=shuffle.permutation(uids).tolist();assert len(set(order))==100", "order=shuffle.permutation(uids).tolist();assert len(set(order))==100\n                assert order==json.loads((PARENT/'run'/f'epoch-order-{epoch:03d}.json').read_text())['uids'],'Paired batch order differs'")
for a,b in [('==15000 and cp[\'epoch\']==600','==16250 and cp[\'epoch\']==650'),
            ('=={600}','=={650}'),('step=15000;epoch=600','step=16250;epoch=650'),
            ('start_epoch=600,start_update=15000','start_epoch=650,start_update=16250'),
            ('additional_epochs=100,additional_updates=2500','additional_epochs=50,additional_updates=1250'),
            ('start_update=15000,end_update=17500','start_update=16250,end_update=17500'),
            ('updates=15000,epoch=600','updates=16250,epoch=650'),('range(601,701)','range(651,701)'),
            ('step-15000','step-16250'),('step<=15003','step<=16253'),
            ('additional_updates=2500,mesh_participations=70000','additional_updates=1250,mesh_participations=70000'),
            ('COMPLETE_17500_ADDITIONAL2500','COMPLETE_17500_ADDITIONAL1250')]:change(a,b)
src=src.replace('epoch600','epoch650')
compile(src,'train_low_lr.py','exec')
unchanged=[]
for name in ['objective','gradnorms','save','rng_snapshot','restore_rng']:
    a=next(x for x in ast.walk(ast.parse(old)) if isinstance(x,ast.FunctionDef) and x.name==name)
    b=next(x for x in ast.walk(ast.parse(src)) if isinstance(x,ast.FunctionDef) and x.name==name)
    assert ast.dump(a)==ast.dump(b),name
    unchanged.append(name)
(ROOT/'train_low_lr.py').write_text(src)
for n in ['runtime.py','evaluate.py','construction_args.json','run.sh','mining_source.py']:shutil.copy2(PREV/n,ROOT/n)
v=(PREV/'verify_results.py').read_text()
for a,b in [('==15000 and parent[\'epoch\']==600','==16250 and parent[\'epoch\']==650'),
            ('range(15001,17501)','range(16251,17501)'),('range(1,2501)','range(1,1251)'),
            ('range(601,701)','range(651,701)'),('(epoch-601)*25:(epoch-600)*25','(epoch-651)*25:(epoch-650)*25'),
            ('[600,625,650,675,700]','[650,675,700]'),('additional_updates=2500','additional_updates=1250'),
            ('parent_rng_replayed_all100_orders','parent_rng_replayed_all50_orders'),
            ('each_mesh_additional_participations=100','each_mesh_additional_participations=50'),
            ('four_reconstruction_groups_nonzero_all2500_updates','four_reconstruction_groups_nonzero_all1250_updates'),
            ('lrs_unchanged_from_parent=lrs','fixed_new_lrs=lrs')]:
    assert a in v,a;v=v.replace(a,b)
v=v.replace("lrs={g['name']:g['lr'] for g in parent['optimizer']['param_groups']}","lrs={g['name']:(9e-7 if g['name']=='encoder_mu' else 9e-6) for g in parent['optimizer']['param_groups']}")
(ROOT/'verify_results.py').write_text(v)
p=(PREV/'package_results.py').read_text()
p=p.replace("files.extend((ROOT/'control_evaluations').glob('*.jsonl'))", "files.extend((ROOT/'control_evaluations').glob('*.jsonl'))\nfor folder in [ROOT/'hardneg_logits',ROOT/'control_hardneg']:\n    files.extend(p for p in folder.rglob('*') if p.is_file() and p.suffix in ['.npz','.json','.jsonl'])")
(ROOT/'package_results.py').write_text(p)
(ROOT/'code_verification.json').write_text(json.dumps(dict(source_runner=str(PREV/'train_continue.py'),
    source_sha256=hashlib.sha256(old.encode()).hexdigest(),runner_sha256=hashlib.sha256(src.encode()).hexdigest(),
    ast_identical_functions=unchanged,unchanged_files={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['runtime.py','evaluate.py','construction_args.json']}),indent=2)+'\n')
print('Built LR-only intervention; unchanged objective, checkpoint, RNG and actual evaluator.')
