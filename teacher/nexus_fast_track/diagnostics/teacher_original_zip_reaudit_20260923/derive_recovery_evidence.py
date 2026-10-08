"""Derive evidence tables from fresh original-ZIP metadata; no training."""
import hashlib
import json
from pathlib import Path
import zipfile

OUT=Path(__file__).resolve().parent
CANDIDATE=Path('/Users/luthier/Downloads/Nexus_teacher_full_reconstruction')

def main():
    manifest=json.loads((OUT/'FILE_MANIFEST.json').read_text())
    meta=json.loads((OUT/'ORIGINAL_CHECKPOINT_METADATA.json').read_text())
    payload=lambda name:meta[name]['payload']
    latest=payload('results/minkowski_target_099/latest.pt')
    prior=payload('results/point_prior_candidate/candidate.pt')
    point=payload('results/point_diffusion/latest.pt')
    hashes={e['path']:e['sha256'] for e in manifest['entries']}
    records={}
    with zipfile.ZipFile(manifest['source']) as z:
        for name in z.namelist():
            p=Path(name)
            if (name=='README_交付说明.md' or
                (name.startswith('results/') and (p.suffix in ('.log','.csv','.md') or
                 p.name in ('config.json','status.json','overfit_policy.json','objective_probe.json','fullbatch_probe.json')))):
                records[name]={'sha256':hashes[name],'text':z.read(name).decode('utf-8')}
    (OUT/'ORIGINAL_TRAINING_RECORDS.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    comparisons=[]
    for a,ka,b,kb in [
        ('results/minkowski_target_099/best_ae.pt','state_dict','results/minkowski_target_099/vae.pt','state_dict'),
        ('results/minkowski_target_099/vae.pt','state_dict','results/minkowski_target_099/latest.pt','vae'),
        ('results/minkowski_target_099/best_flow.pt','state_dict','results/minkowski_target_099/latest.pt','flow'),
        ('results/point_diffusion/latest.pt','state_dict','results/point_prior_candidate/candidate.pt','state_dict')]:
        sa,sb=payload(a)[ka],payload(b)[kb]
        different=[k for k in set(sa)|set(sb) if k not in sa or k not in sb or
            sa[k]['tensor_shape']!=sb[k]['tensor_shape'] or sa[k]['dtype']!=sb[k]['dtype'] or
            sa[k]['contiguous_storage_sha256']!=sb[k]['contiguous_storage_sha256']]
        comparisons.append({'a':a+':'+ka,'b':b+':'+kb,'tensors':len(sa),'different_keys':different})
    optimizer=[]
    for (pid,state),(name,tensor) in zip(latest['optimizer']['state'].items(),latest['flow'].items()):
        optimizer.append({'optimizer_id':pid,'flow_parameter':name,
            'parameter_shape':tensor['tensor_shape'],'exp_avg_shape':state['exp_avg']['tensor_shape'],
            'exp_avg_sq_shape':state['exp_avg_sq']['tensor_shape'],
            'shape_matches':tensor['tensor_shape']==state['exp_avg']['tensor_shape']==state['exp_avg_sq']['tensor_shape'],
            'adam_slot_step':state['step']['scalar']})
    prior_names=[n for n in prior['state_dict'] if n.startswith('coordinate_prior.')]+['residual_scale']
    prior_mapping=[]
    for (pid,state),name in zip(prior['optimizer']['state'].items(),prior_names):
        shape=prior['state_dict'][name]['tensor_shape']
        prior_mapping.append({'optimizer_id':pid,'matched_parameter':name,'parameter_shape':shape,
            'exp_avg_shape':state['exp_avg']['tensor_shape'], 'exp_avg_sq_shape':state['exp_avg_sq']['tensor_shape'],
            'shape_matches':shape==state['exp_avg']['tensor_shape']==state['exp_avg_sq']['tensor_shape'],
            'adam_slot_step':state['step']['scalar']})
    stages=[]
    for stage in ('ae','flow'):
        for depth in sorted({x['layers'] for x in latest['history'] if x['stage']==stage}):
            rows=[x for x in latest['history'] if x['stage']==stage and x['layers']==depth]
            stages.append({'stage':stage,'layers':depth,'first_recorded_eval_step':rows[0]['step'],
                           'last_recorded_eval_step':rows[-1]['step'],'evaluations':len(rows)})
    events=[]
    for name,record in records.items():
        if not name.endswith('.log'):continue
        for line_no,line in enumerate(record['text'].splitlines(),1):
            if any(w in line for w in ('DEPTH_INCREASE','PLATEAU_LR_REDUCTION','RESUMED','Resumed flow','Precision continuation','Time coverage fix','pre-change checkpoint')):
                events.append({'member':name,'line':line_no,'text':line})
    baseline=json.loads((CANDIDATE/'evidence/checkpoint_structure.json').read_text())
    baseline_checks=[]
    for name,old in baseline.items():
        if name not in meta or not isinstance(old,dict):continue
        row={'path':name,'archive_member_sha256_matches_yesterday':old.get('sha256')==hashes[name]}
        if 'state_shapes' in old:
            sd=payload(name).get('state_dict')
            if sd is not None:row['state_shapes_match_yesterday']=old['state_shapes']=={k:v['tensor_shape'] for k,v in sd.items()}
        baseline_checks.append(row)
    evidence={'author_model':'gpt-6-astra','reasoning_effort':'xhigh',
        'state_tensor_comparisons':comparisons,'final_flow_optimizer_mapping':optimizer,
        'optimizer_mapping_qualification':'Optimizer stores numerical IDs, not parameter names. Mapping is by complete ordered shapes plus stage/model evidence; repeated equal shapes do not by themselves prove parameter registration identity.',
        'final_flow_rng_keys':list(latest['rng']), 'prior_optimizer_mapping':prior_mapping,
        'prior_optimizer_groups':prior['optimizer']['param_groups'],
        'prior_rng_present':'rng' in prior,'step_semantics':{
            'ae_counter':latest['counters']['ae'],'flow_counter':latest['counters']['flow'],
            'flow_adam_current_slots_step':14000,'prior_cumulative_updates':prior['prior_steps'],
            'prior_adam_current_slots_step':50000,
            'interpretation':'Cumulative stage counter and current Adam slot counter are different. Fresh optimizers on stage/depth changes are documented; mismatch is not itself a bug.',
            'point_source_step_unresolved':{'candidate_source_step':prior['source_step'],
                'inference_frozen_denoiser_source_step':point['frozen_denoiser_source_step']}},
        'recorded_depth_ranges':stages,'original_log_events':events,
        'yesterday_checkpoint_metadata_comparison':baseline_checks,
        'candidate_file_sha256':{n:hashlib.sha256((CANDIDATE/n).read_bytes()).hexdigest() for n in
            ['teacher_ae.py','models.py','indicators.py','objectives.py','replay.py','smoke_test.py','README.md','REPORT.md']},
        'exact_original_code_fragments_in_logs':[{'member':n,'text':r['text']} for n,r in records.items() if n.endswith(('cascade_stderr.log','cascade_overfit_stderr.log'))]}
    (OUT/'DERIVED_RECOVERY_EVIDENCE.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'state_comparisons':comparisons,'flow_slot_count':len(optimizer),
        'flow_all_shapes_match':all(r['shape_matches'] for r in optimizer),
        'prior_slot_count':len(prior_mapping),'prior_all_shapes_match':all(r['shape_matches'] for r in prior_mapping),
        'baseline_comparisons':baseline_checks,'depth_ranges':stages,'training_records':len(records)},indent=2))

if __name__=='__main__':main()
