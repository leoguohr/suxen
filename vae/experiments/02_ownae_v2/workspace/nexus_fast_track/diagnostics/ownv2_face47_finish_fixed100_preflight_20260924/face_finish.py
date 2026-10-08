"""Bounded Face-only continuation from the exact B19356 training state."""
import argparse
import copy
import fcntl
import subprocess
import sys
import traceback
from pathlib import Path
from run_support import *
from native_models import NativeTopologyAE, Config, Graph
from data_objective import load_dataset, epoch_batches, negative_faces, hard4_chunks, face_logits, edge_logits, array_sha
from evaluate_checkpoint import evaluate_mesh, aggregate

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'repro_outputs'
OLD = ROOT.parent / 'own512_v2_recipe_pair_20260923_resume_01'
PARENT = OLD / 'B_v2_teacher_blocks/checkpoint-19356.pt'
PARENT_SHA = '6f965d3837c32101dc7efd52252c12387fb3f10c788e6fd0d33ae359a44903a2'
BASE_EVAL = OLD / 'B_v2_teacher_blocks/evaluations/eval-19356.json'
FACE_NAMES = ['face_embedding.weight', 'face_embedding.bias']
LIVE = {}


def equal(a, b):
    if torch.is_tensor(a): assert torch.equal(a.cpu(), b.cpu()) and a.dtype == b.dtype
    elif isinstance(a, np.ndarray): assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal(x, y)
    else: assert a == b, (a, b)


def frozen_hash(model):
    return tensor_hash({k:v for k,v in model.state_dict().items() if k not in FACE_NAMES})


def adam_frozen_hash(opt, names):
    return tensor_hash({f'{i}/{k}':v for i,slot in opt.state_dict()['state'].items()
        if names[i] not in FACE_NAMES for k,v in slot.items()})


def load(path):
    cp = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    assert cp['model_variant'] == 'B_v2_teacher_blocks'
    for name,digest in cp['config']['code_sha256'].items(): assert sha(OLD/name) == digest
    model = NativeTopologyAE(Config(**cp['model_config'])).cuda().float()
    model.load_state_dict(cp['model'], strict=True)
    assert tensor_hash(model.state_dict()) == tensor_hash(cp['model'])
    items,manifest = load_dataset(OLD/'data', OLD/'pools')
    assert manifest == cp['config']['data']
    return cp,model,items,manifest


class EvalAccount:
    def update(self): return 0


def evaluate(path, new, cold=False):
    configure(); cp,model,items,manifest = load(path); model.eval(); restore_rng(cp['rng'])
    assert cp['completed_updates'] == 19356+new
    previous = read(BASE_EVAL)
    name = ('cold' if cold else 'eval') + f'-{new:04d}'
    folder = ROOT/'evaluations'/name; folder.mkdir(parents=True, exist_ok=False)
    rows = []
    for uid,ref in zip(manifest['uids'],previous['meshes']):
        assert uid == ref['uid']
        row = evaluate_mesh(model, items[uid], folder/(uid+'.npz'), EvalAccount())
        with np.load(OLD/ref['prediction_path']) as old, np.load(ROOT/row['prediction_path']) as now:
            for k in ['vertices','gt_edges','gt_faces','all_edge_pair_logits','predicted_edge_ids','actual_face_candidate_ids']:
                assert np.array_equal(old[k],now[k]),(uid,k,'frozen Edge/candidates changed')
            if new == 0:
                assert old.files == now.files
                for k in old.files: assert np.array_equal(old[k],now[k]),(uid,k,'parent mismatch')
        rows.append(row)
    summary = aggregate(rows); large = aggregate([m for m in rows if 66 <= m['vertices'] <= 274])
    assert summary['counts']['edge'] == previous['counts']['edge']
    assert summary['face_fn_missing'] == 6 and len(summary['edge_perfect_uids']) == 40
    assert summary['joint_perfect'] <= 40
    result = dict(complete=True,new_updates=new,completed_updates=19356+new,checkpoint=str(path),
        checkpoint_sha256=sha(path),optimizer_updates_added_by_evaluation=0,full_native_forward=True,
        all50_edges_and_candidates_bitwise_equal_parent=True,**summary,large16=large,meshes=rows)
    if cold:
        earlier = read(ROOT/'evaluations'/f'eval-{new:04d}.json')
        assert result['counts'] == earlier['counts'] and result['perfect_uids'] == earlier['perfect_uids']
        for a,b in zip(earlier['meshes'],rows):
            with np.load(ROOT/a['prediction_path']) as x, np.load(ROOT/b['prediction_path']) as y:
                assert x.files == y.files
                for k in x.files: assert x[k].dtype == y[k].dtype and x[k].tobytes() == y[k].tobytes()
        result['all50_prediction_arrays_bitwise_equal'] = True
    write(ROOT/'evaluations'/(name+'.json'),result)
    print('EVAL_COMPLETE',name,result['counts'],result['joint_perfect'],flush=True)


def full_eval(path, new, cold=False):
    tag = ('cold' if cold else 'eval') + f'-{new:04d}'
    argv = [sys.executable,'-u',str(ROOT/'face_finish.py'),'eval','--checkpoint',str(path),'--new',str(new)]
    if cold: argv.append('--cold')
    before = rng_state()
    write(OUT/(tag+'.command.json'),dict(argv=argv,started=time.time(),optimizer_updates=0))
    with (OUT/(tag+'.log')).open('x') as stream:
        rc = subprocess.run(argv,cwd=ROOT,env=os.environ.copy(),stdout=stream,stderr=subprocess.STDOUT).returncode
    write(OUT/(tag+'.exit.json'),dict(returncode=rc,finished=time.time()))
    assert rc == 0, f'{tag} failed; stop without rollback'
    equal(before,rng_state())
    return read(ROOT/'evaluations'/(tag+'.json'))


def train(resume=None):
    lock=(ROOT/'face_training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'TRAIN_COMPLETE.json').exists()
    assert (ROOT/'updates.jsonl').exists() == (resume is not None)
    configure(); assert sha(PARENT) == PARENT_SHA
    cp,model,items,manifest = load(PARENT if resume is None else resume)
    start_new=cp['completed_updates']-19356
    assert start_new == (0 if resume is None else 50)
    original_names = [n for n,p in model.named_parameters() if p.requires_grad]
    assert original_names == cp['config']['optimizer_parameter_names']
    params = dict(model.named_parameters())
    opt = torch.optim.AdamW([params[n] for n in original_names],lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01,foreach=True)
    opt.load_state_dict(copy.deepcopy(cp['optimizer'])); equal(opt.state_dict(),cp['optimizer'])
    LIVE.update(model=model,optimizer=opt,new_updates=start_new)
    model.requires_grad_(False); model.face_embedding.requires_grad_(True); model.train()
    assert [n for n,p in model.named_parameters() if p.requires_grad] == FACE_NAMES
    active = list(model.face_embedding.parameters()); frozen = frozen_hash(model)
    frozen_adam = adam_frozen_hash(opt,original_names)
    restore_rng(cp['rng']);equal(rng_state(),cp['rng'])
    protocol = dict(parent=str(PARENT),parent_sha256=PARENT_SHA,trainable=FACE_NAMES,
        trainable_parameters=sum(p.numel() for p in active),max_new_updates=500,check_every=50,target_face_f1=.997,
        actual_optimizer='AdamW inherited without reset',param_groups=cp['optimizer']['param_groups'],
        loss='original whole-mesh Hard4; Face plus constant frozen Edge; each mesh/5',
        random_negatives='original sampler continued from global update19356; union with fixed hard negatives',
        cache_boundary='frozen decoder_output_norm output; no trainable output cached',
        frozen_state_sha256=frozen,frozen_adam_sha256=frozen_adam,optimizer_updates_at_restore=0,
        all_adam_slots_and_parameter_mapping_equal=True,rng_equal=True,
        frozen_edge_strict_ceiling=40,source_hashes={p.name:sha(p) for p in ROOT.glob('*.py')})
    if resume is None:
        write(OUT/'PROTOCOL.json',protocol)
        full_eval(PARENT,0)
    else:
        old_protocol=read(OUT/'PROTOCOL.json')
        assert frozen==old_protocol['frozen_state_sha256'] and frozen_adam==old_protocol['frozen_adam_sha256']
        assert read(OUT/'RECOVERY_50_AUDIT.json')['passed']
        assert sha(resume)==read(OUT/'RECOVERY_50_AUDIT.json')['checkpoint']['sha256']
        assert read(ROOT/'evaluations/eval-0050.json')['counts']['face']['micro_f1'] < .997
        write(OUT/'RESUME_RESTORE_AUDIT.json',dict(passed=True,new_updates=start_new,
            full_model_adam_rng_restored=True,source=str(resume),source_sha256=sha(resume),
            frozen_model_and_adam_equal_original_parent=True,additional_optimizer_updates=0))
    reference = read(BASE_EVAL); cache={}; hard={}; hard_records=[]; edge_constants={}
    for uid,ref in zip(manifest['uids'],reference['meshes']):
        with np.load(OLD/ref['prediction_path']) as a:
            assert sha(OLD/ref['prediction_path']) == ref['prediction_sha256']
            negatives = a['actual_face_candidate_ids'][~a['actual_face_candidate_gt']].astype(np.int64)
            assert len(negatives) == ref['face']['fp']
        hard[uid]=torch.from_numpy(negatives)
        hard_records.append(dict(uid=uid,triples=negatives.tolist(),count=len(negatives),sha256=array_sha(negatives)))
        data=move_item(items[uid],'cuda')
        with torch.no_grad():
            rows=model(data['vertices'],data['faces'],sample_latent=False)
            cache[uid]=rows['decoder_hidden'].detach()
            el,_=hard4_chunks(edge_logits,rows['edge'],data['pairs'],data['edge_labels'],32768)
            edge_constants[uid]=float(el)
    assert sum(x['count'] for x in hard_records)==47
    write(OUT/'HARD_NEGATIVES.json',dict(total=47,gt_positives=5576,records=hard_records,refresh=False))

    def face_objective(uid,epoch,real=False):
        item=items[uid];neg,random_sha=negative_faces(item,epoch)
        merged=torch.unique(torch.cat((neg,hard[uid])),sorted=True,dim=0)
        h=cache[uid]
        if real:
            d=move_item(item,'cuda');h=model(d['vertices'],d['faces'],sample_latent=False)['decoder_hidden']
        z=model.face_embedding(h);z=z-z.mean(0,keepdim=True)
        ids=torch.cat((item['gt_faces'],merged)).cuda()
        labels=torch.cat((torch.ones(len(item['gt_faces'])),torch.zeros(len(merged)))).cuda()
        loss,stats=hard4_chunks(face_logits,z,ids,labels,32768)
        return loss,dict(uid=uid,random_negative_sha256=random_sha,negative_union_sha256=array_sha(merged.numpy()),
            random_negatives=len(neg),fixed_hard_negatives=len(hard[uid]),union_negatives=len(merged),
            positives=len(item['gt_faces']),face=stats,edge_loss_constant=edge_constants[uid])

    check_uid=max(manifest['uids'],key=lambda u:len(items[u]['vertices']))
    l,_=face_objective(check_uid,1935,real=True);g=torch.autograd.grad(l,active)
    c,_=face_objective(check_uid,1935);cg=torch.autograd.grad(c,active)
    assert torch.equal(l,c) and all(torch.equal(x,y) for x,y in zip(g,cg))
    restore_rng(cp['rng']); equal(rng_state(),cp['rng'])
    write(OUT/('CACHE_AUDIT.json' if resume is None else 'CACHE_AUDIT_resume0050.json'),dict(passed=True,uid=check_uid,loss_and_face_gradients_bitwise=True,
        frozen_only=True,all50_parent_full_predictions_bitwise=True,optimizer_updates=0))
    participation=cp['participation'].copy();new_participation=cp.get('new_participation',{u:0 for u in manifest['uids']}).copy()
    best_f1=reference['counts']['face']['micro_f1']; best_strict=reference['joint_perfect']
    for name,val in [('face_f1',best_f1),('strict',best_strict)]:
        if not (OUT/f'BEST_{name}.json').exists():
            write(OUT/f'BEST_{name}.json',dict(value=val,new_updates=0,checkpoint=str(PARENT),sha256=PARENT_SHA))
    stop_reason='budget_exhausted';completed=start_new;last=None
    with (ROOT/'updates.jsonl').open('x' if resume is None else 'a',buffering=1) as log:
        for new in range(start_new+1,501):
            step=19356+new;epoch,batch_index=divmod(step-1,10)
            batch=epoch_batches(manifest['uids'],epoch)[batch_index]
            before=[p.detach().clone() for p in active];opt.zero_grad(set_to_none=True);details=[]
            for uid in batch:
                loss,row=face_objective(uid,epoch)
                assert torch.isfinite(loss)
                ((loss+edge_constants[uid])/5).backward();details.append(row)
            assert all(p.grad is None for n,p in model.named_parameters() if n not in FACE_NAMES)
            norm=float(torch.nn.utils.clip_grad_norm_(active,1.,error_if_nonfinite=True,foreach=True))
            opt.step();completed=new
            LIVE['new_updates']=new
            assert all(torch.isfinite(p).all() for p in active)
            assert [int(opt.state[p]['step']) for p in active]==[step,step]
            for uid in batch:participation[uid]+=1;new_participation[uid]+=1
            displacement=sum(float((p.detach()-old).double().square().sum()) for p,old in zip(active,before))**.5
            row=dict(new_update=new,total_update=step,uids=batch,epoch=epoch,batch_index=batch_index,
                meshes=details,gradient_norm_before_clip=norm,clip_coefficient=min(1.,1/(norm+1e-6)),
                lr=opt.param_groups[0]['lr'],adam_face_step=step,parameter_displacement_l2=displacement,
                frozen_edge_fp=21,frozen_edge_fn=3,frozen_edge_strict_count=40,
                timing='loss before update; parameter displacement and Adam step after update')
            log.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
            write(OUT/'TRAIN_STATUS.json',dict(state='training',new_updates=new,max_updates=500))
            if new%50:continue
            # Continuation begins at batch6 of an epoch. The initial/final
            # partial shuffled epochs can differ by one participation per UID.
            assert sum(new_participation.values())==new*5
            assert min(new_participation.values())>=new//10-1 and max(new_participation.values())<=new//10+1
            assert frozen_hash(model)==frozen and adam_frozen_hash(opt,original_names)==frozen_adam
            value=dict(cp);value.update(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),
                completed_updates=step,participation=participation.copy(),completed_epochs=step//10,
                data_generator=dict(cp['data_generator'],next_epoch=step//10,next_batch=step%10),
                finish_protocol=protocol,new_updates=new,new_participation=new_participation.copy())
            path=ROOT/'checkpoints'/f'checkpoint-new{new:04d}-step{step}.pt';path.parent.mkdir(exist_ok=True)
            assert not path.exists();receipt=save_torch(path,value);write(path.with_suffix('.json'),receipt)
            last=receipt;result=full_eval(path,new)
            for name,val in [('face_f1',result['counts']['face']['micro_f1']),('strict',result['joint_perfect'])]:
                p=OUT/f'BEST_{name}.json'
                if val>read(p)['value']:write(p,dict(value=val,new_updates=new,**receipt))
            print('CHECKPOINT',new,result['counts']['face'],result['joint_perfect'],flush=True)
            if result['counts']['face']['micro_f1']>=.997:
                cold=full_eval(path,new,cold=True)
                assert cold['all50_prediction_arrays_bitwise_equal']
                stop_reason='target_reached_and_cold_verified';break
    assert sha(PARENT)==PARENT_SHA
    write(OUT/'TRAIN_COMPLETE.json',dict(completed=True,new_updates=completed,total_updates=19356+completed,
        stop_reason=stop_reason,final_checkpoint=last,frozen_model_and_adam_unchanged=True,
        parent_file_sha256_unchanged=True,new_participations=new_participation,
        no_fixed100_training_launched=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['train','eval'])
    parser.add_argument('--checkpoint');parser.add_argument('--new',type=int);parser.add_argument('--cold',action='store_true')
    parser.add_argument('--resume')
    args=parser.parse_args()
    try:
        if args.mode=='train':train(Path(args.resume) if args.resume else None)
        else:evaluate(Path(args.checkpoint),args.new,args.cold)
    except BaseException as error:
        write(OUT/('TRAIN_FAILURE.json' if args.mode=='train' else f'EVAL_FAILURE_{args.new}_{args.cold}.json'),
            dict(error=str(error),traceback=traceback.format_exc(),mode=args.mode))
        if args.mode=='train' and LIVE.get('new_updates',0)>0:
            save_torch(ROOT/f"failure-state-new{LIVE['new_updates']:04d}.pt",dict(model=LIVE['model'].state_dict(),
                optimizer=LIVE['optimizer'].state_dict(),rng=rng_state(),
                parent=str(PARENT),new_updates=LIVE['new_updates'],not_a_successful_checkpoint=True))
        raise
