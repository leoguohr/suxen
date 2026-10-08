"""Read existing results into a bounded review archive; never import training code."""
from pathlib import Path
import csv, json, shutil, hashlib, datetime, io
ROOT = Path(__file__).resolve().parent
OUT = ROOT/'package'
DIAG = Path('/guohaoran/nexus_fast_track/diagnostics')
A = DIAG/'math00_four_mesh_lr10x_20260913/only804_mu'
B = DIAG/'math00_twenty_mesh_training_preparation'
RUN = B/'run/from_00000'
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
AA=OUT/'A_only804_mu';BB=OUT/'B_twenty_mesh'
AA.mkdir(parents=True,exist_ok=True);BB.mkdir(parents=True,exist_ok=True)
original_index=[]
def copy(p, dst):
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    original_index.append(dict(source=str(p),archive=str(dst.relative_to(OUT)),sha256=digest(dst)))
for p in A.iterdir():
    if p.is_file() and (p.suffix in ['.json','.py','.jsonl','.log']):copy(p, AA/p.name)
for p in (A/'C_graph_only').iterdir():
    if p.is_file():copy(p,AA/'C_graph_only'/p.name)
updates=[json.loads(line) for line in (AA/'updates.jsonl').read_text().splitlines()]
assert [r['update'] for r in updates]==list(range(1,1001))
for step in [0,400,800,1000]:assert (AA/f'mu_step{step:04d}.json').exists()
for name in ['manifest.json','comparison.json','complete.json','verification.json','start_verification.json','objective_scope_check.json']:
    assert (AA/name).exists()
for name in ['selection.json','pools_ready.json','evaluation_seeds.json','recommended_config.json','READY.json','train.py','run_aistation.sh']:
    copy(B/name, BB/name)
copy(RUN/'manifest.json',BB/'manifest.json')
copy(B/'preflight/result.json',BB/'preflight_result.json')
copy(B/'preflight/manifest.json',BB/'preflight_manifest.json')
manifest=json.loads((BB/'manifest.json').read_text())
assert manifest['logical_batch_size']==20 and all(len(x)==1 for x in manifest['microbatch_uid_order'])
uids=manifest['uids'];sizes={x['uid']:dict(vertices=x['vertices'],gt_faces=x['faces']) for x in manifest['pools']}
eval_files=sorted(RUN.glob('mu_step*.json'))+sorted(RUN.glob('sample_step*.json'))
evals=[]
for p in eval_files:
    result=json.loads(p.read_text());evals.append(result)
    for x in result['meshes']:
        e=x['rec']['edge'];sizes[x['uid']]['gt_edges']=e['tp']+e['fn']
    copy(p,BB/'hard_evaluations'/p.name)
for p in sorted(RUN.glob('*summary_step*.json')):copy(p,BB/'noise_summaries'/p.name)

# Snapshot only complete existing records at a fixed byte offset of the growing log.
up=RUN/'updates.jsonl';byte_limit=up.stat().st_size
with up.open('rb') as f:raw=f.read(byte_limit)
lines=raw.splitlines(keepends=True)
if lines and not lines[-1].endswith(b'\n'):lines.pop()
training=[];all_steps=[]
for line in lines:
    record=json.loads(line);step=record['update'];all_steps.append(step)
    assert [x['uid'] for x in record['meshes']]==uids
    for x in record['meshes']:
        training.append(dict(optimizer_update=step,state_step_before_update=step-1,uid=x['uid'],**sizes[x['uid']],
            completed_participating_updates_after_this_update=step,objective_coefficient=1/20,
            edge_loss=x['parts']['edge'],face_loss=x['parts']['face'],kl_mu=x['kl']['mu'],kl_sigma=x['kl']['sigma'],
            weighted_objective=x['weighted_objective'],sigma_median=x['posterior']['std']['median'],sigma_p95=x['posterior']['std']['p95'],
            global_clip_coefficient=record['clip_coefficient']))
assert all_steps==list(range(1,max(all_steps)+1))
table(BB/'per_mesh_training_updates.csv',training)

hard=[];success={};evaluated={}
def add_hard(result, mode, draw=-1):
    for x in result['meshes']:
        u=x['uid'];r=x['rec'];e=r['edge'];f=r['face'];key=(mode,u)
        passed=all(r[k]['fp']==r[k]['fn']==0 for k in ['edge','face'])
        success[key]=success.get(key,0)+int(passed);evaluated[key]=evaluated.get(key,0)+1
        hard.append(dict(step=result['step'],uid=u,mode=mode,noise_draw=draw,**sizes[u],
            completed_participating_updates_in_20mesh_phase=result['step'],edge_loss=x['parts']['edge'],face_loss=x['parts']['face'],
            edge_tp=e['tp'],edge_fp=e['fp'],edge_fn=e['fn'],face_tp=f['tp'],face_fp=f['fp'],face_fn=f['fn'],
            gt_face_candidates_missing=r['gt_faces_missing_from_edge_candidates'],
            gt_face_candidate_coverage=1-r['gt_faces_missing_from_edge_candidates']/sizes[u]['gt_faces'],
            strict_perfect_this_forward=int(passed),strict_perfect_count_so_far_in_mode=success[key],
            recorded_evaluations_so_far_in_mode=evaluated[key]))
for x in sorted(evals,key=lambda x:(x['mode'],x['step'])):add_hard(x,x['mode'])
monitor=[]
for tag in ['monitoring','final_noise']:
    for p in sorted(RUN.glob(tag+'_step*.jsonl')):
        values=[]
        for j,line in enumerate(p.read_text().splitlines()):
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            add_hard(r,tag,j);values.append(r)
        for i,u in enumerate(uids):
            xs=[v['meshes'][i] for v in values]
            row=dict(step=values[0]['step'],uid=u,mode=tag,**sizes[u],noise_draws=len(xs),
                completed_participating_updates_in_20mesh_phase=values[0]['step'],
                strict_perfect_count=sum(x['perfect'] for x in xs),
                all20_perfect_count=sum(v['all20_perfect'] for v in values),
                mean_edge_loss=sum(x['parts']['edge'] for x in xs)/len(xs),mean_face_loss=sum(x['parts']['face'] for x in xs)/len(xs))
            for part in ['edge','face']:
                for k in ['tp','fp','fn']:
                    z=[x['rec'][part][k] for x in xs]
                    for name,v in [('mean',sum(z)/len(z)),('min',min(z)),('max',max(z))]:row[f'{part}_{k}_{name}']=v
            z=[x['rec']['gt_faces_missing_from_edge_candidates'] for x in xs]
            row.update(gt_face_candidates_missing_mean=sum(z)/len(z),gt_face_candidates_missing_min=min(z),gt_face_candidates_missing_max=max(z))
            monitor.append(row)
table(BB/'per_mesh_hard_reconstruction.csv',hard)
table(BB/'per_mesh_noise_summary.csv',monitor)
write(BB/'execution_and_cutoff.json',dict(captured_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_updates=str(up),source_bytes_snapshot=byte_limit,complete_update_records=len(all_steps),last_update=max(all_steps),
    hard_checkpoints=sorted({v['step'] for v in evals}),noise_checkpoints=sorted({v['step'] for v in monitor}),
    cumulative_participation_definition='Within this 20-mesh phase: every mesh participates once per optimizer update. At checkpoint step t each participated in t updates. Parent histories differ; no identical lifetime count is asserted.',
    objective='sum_j(EdgeSoft4_j+FaceSoft4_j+1e-4*KL_j)/20',logical_meshes_per_update=20,
    microbatches_per_update=20,complete_meshes_per_microbatch=1,backward_calls_per_update=20,global_clip_calls_per_update=1,optimizer_steps_per_update=1,
    loss_reduction='Within each mesh unchanged Soft4 and KL; divide each full mesh objective by20 before backward; accumulate all20 gradients before clipping/Adam.',
    evaluation_layout='20 sequential complete-mesh forwards at unchanged checkpoint with one prescribed 20-noise set; NOT a single packed20 forward.',
    hard_metrics_available_every_update=False,hard_metric_policy='Only actual saved hard evaluations; no interpolation. Training-step table contains loss only.',
    strict_success_count_definition='Counts of recorded evaluation forwards, separately by mode; NOT number of all optimizer steps that would have been perfect.',
    config_status_note='recommended_config.json records preparation time and contains stale not-launched flags. Current run manifest, updates and actual evaluations establish execution.',
    no_candidates_npz_in_twenty_mesh_export=True))
write(OUT/'original_materials_index.json',original_index)
write(AA/'log_completeness.json',dict(update_count=len(updates),first_update=1,last_update=1000,contiguous=True,
    required_mu_checkpoints_present=[0,400,800,1000],source=str(A),log_sha256=digest(AA/'updates.jsonl')))
print('COLLECTED',len(updates),'only804 updates;',max(all_steps),'twenty-mesh updates;',len(hard),'hard rows')
