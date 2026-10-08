"""CPU evidence audit and small ZIP delivery; never calls an optimizer."""
import collections
import csv
import json
import sys
import zipfile
from pathlib import Path
from run_support import read, write, sha, torch
from data_objective import epoch_batches

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'repro_outputs'


def main():
    complete=read(OUT/'TRAIN_COMPLETE.json');assert complete['new_updates']==500
    baseline=read(ROOT/'evaluations/eval-0000.json')
    evaluations=[read(p) for p in sorted((ROOT/'evaluations').glob('eval-*.json'))]
    assert [x['new_updates'] for x in evaluations]==list(range(0,501,50))
    logs=[json.loads(line) for line in (ROOT/'updates.jsonl').read_text().splitlines()]
    assert [x['new_update'] for x in logs]==list(range(1,501))
    uids=[x['uid'] for x in baseline['meshes']];part=collections.Counter()
    for row in logs:
        step=19356+row['new_update'];epoch,b=divmod(step-1,10)
        assert row['uids']==epoch_batches(uids,epoch)[b]
        assert row['total_update']==row['adam_face_step']==step and row['lr']==1e-4
        part.update(row['uids'])
    assert dict(part)==complete['new_participations'] and sum(part.values())==2500
    for e in evaluations:
        assert e['complete'] and e['optimizer_updates_added_by_evaluation']==0 and len(e['meshes'])==50
        assert e['all50_edges_and_candidates_bitwise_equal_parent']
        for task in ['edge','face']:
            for k in ['tp','fp','fn','tn']:assert e['counts'][task][k]==sum(m[task][k] for m in e['meshes'])
        assert e['counts']['edge']==baseline['counts']['edge']
        assert e['joint_perfect']==sum(m['joint_perfect'] for m in e['meshes'])
        for m in e['meshes']:assert sha(ROOT/m['prediction_path'])==m['prediction_sha256']
    best=max(evaluations,key=lambda x:x['counts']['face']['micro_f1'])
    best_new=max(evaluations[1:],key=lambda x:x['counts']['face']['micro_f1'])
    strict=max(evaluations,key=lambda x:x['joint_perfect'])
    target=any(e['counts']['face']['micro_f1']>=.997 for e in evaluations)
    assert not target
    preflight=read(ROOT/'fixed100_preflight/RESULT.json');assert preflight['passed'] and preflight['optimizer_updates']==0
    manifest=[]
    for p in sorted((ROOT/'checkpoints').glob('checkpoint-*.json')):
        entry=read(p);path=Path(entry['path']);assert path.stat().st_size==entry['bytes']
        entry['identity']='full model + all inherited AdamW slots + RNG + progress + Face finishing protocol'
        if 'new0500' in path.name or 'new0100' in path.name:
            assert sha(path)==entry['sha256'];entry['independently_rechecked_sha256']=True
        manifest.append(entry)
    final=complete['final_checkpoint'];cp=torch.load(final['path'],map_location='cpu',mmap=True,weights_only=False)
    names=cp['config']['optimizer_parameter_names']
    states={n:int(cp['optimizer']['state'][i]['step']) for i,n in enumerate(names)}
    assert all(v==(19856 if n.startswith('face_embedding.') else 19356) for n,v in states.items())
    assert cp['new_updates']==500 and cp['completed_updates']==19856
    write(OUT/'CHECKPOINT_MANIFEST.json',dict(parent=read(OUT/'PROTOCOL.json')['parent'],
        parent_sha256=read(OUT/'PROTOCOL.json')['parent_sha256'],checkpoints=manifest,weights_excluded_from_zip=True))
    write(OUT/'FINAL_AUDIT.json',dict(passed=True,exact_new_optimizer_updates=500,duplicates=0,
        exact_original_sampler_order=True,all50_participated=True,total_new_mesh_participations=2500,
        participation_histogram=dict(collections.Counter(part.values())),all11_full_evaluations_verified=True,
        prediction_hashes_verified=550,final_face_adam_step=19856,frozen_adam_step=19356,
        fixed100_optimizer_updates=0,automatic_extra_training=False,
        recovered_engineering_error='Incorrect equal-per-window participation assertion at update50; saved state resumed without replay',
        cold_target_verification='not triggered; no evaluated checkpoint reached0.997'))
    write(OUT/'RESULTS.json',dict(evaluations=evaluations,baseline=baseline['counts'],best_face_new_updates=best['new_updates'],
        best_new_checkpoint_updates=best_new['new_updates'],best_strict_updates=strict['new_updates'],
        target_reached=target,preflight=preflight,completion=complete))
    write(OUT/'status.json',dict(execution_completed=True,status='partial',target_reached=False,
        optimizer_updates=500,fixed100_optimizer_updates=0,preflight_passed=True,stop_reason='budget_exhausted',
        historical_interruption_recovered=True,no_automatic_follow_on_training=True))
    with (OUT/'PER_MESH.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['new_updates','uid','vertices','edge_fp','edge_fn','face_fp','face_fn','face_fn_missing','face_fn_present','joint_strict'])
        for e in evaluations:
            for m in e['meshes']:w.writerow([e['new_updates'],m['uid'],m['vertices'],m['edge']['fp'],m['edge']['fn'],m['face']['fp'],m['face']['fn'],m['face_fn_missing'],m['face_fn_present'],m['joint_perfect']])
    lines=['# Actual commands','', 'No credentials are included.']
    for n in ['PREFLIGHT_LAUNCH.json','FACE_LAUNCH.json','FACE_RESUME50_LAUNCH.json']:
        x=read(OUT/n);lines+=['',f'## {n}','','```json',json.dumps(x['argv'],indent=2),'```']
    (OUT/'COMMANDS.md').write_text('\n'.join(lines)+'\n')
    dest=ROOT/'delivery';dest.mkdir(exist_ok=False)
    files=[p for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink() and
        not any(s in p.parts for s in ['.git','__pycache__','delivery']) and p.suffix not in ['.pt','.pyc','.tmp','.lock']]
    files.sort();fm=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    write(dest/'FILE_MANIFEST.json',fm)
    groups=[];group=[];size=0
    for p in files:
        if group and size+p.stat().st_size>48*1024*1024:groups.append(group);group=[];size=0
        group.append(p);size+=p.stat().st_size
    if group:groups.append(group)
    archives=[]
    for i,group in enumerate(groups,1):
        p=dest/f'CAD50_Face47_and_fixed100_preflight_{i:02d}.zip'
        with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            z.write(dest/'FILE_MANIFEST.json','FILE_MANIFEST.json')
            for source in group:z.write(source,str(source.relative_to(ROOT)))
        with zipfile.ZipFile(p) as z:assert z.testzip() is None
        archives.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    write(dest/'DELIVERY_INDEX.json',dict(archives=archives,files=len(files),optimizer_updates_added_by_postprocessing=0,
        weights='Excluded; full model/Adam/RNG paths and hashes in CHECKPOINT_MANIFEST.json'))
    print(json.dumps(dict(passed=True,target_reached=False,archives=archives),indent=2))


if __name__=='__main__':main()
