"""Rebuild metadata-only Topology Flow tables/figures from the immutable snapshot.

No model imports, GPU execution, network access, or writes inside snapshot/.
Missing evaluation rows stay blank; only hash-validated full50 results enter the
full50 aggregate table. Subset aggregates are explicitly named and bounded.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f1(tp, fp, fn):
    denominator = 2*tp+fp+fn
    return 2*tp/denominator if denominator else 1.


def aggregate(metrics):
    if not metrics:
        raise ValueError('No score is defined for an empty observed subset')
    result = {}
    for kind in ('edge', 'face'):
        for field in ('tp', 'fp', 'fn'):
            result[f'{kind}_{field}'] = sum(m[kind][field] for m in metrics)
        result[f'{kind}_micro_f1'] = f1(*(result[f'{kind}_{k}'] for k in ('tp','fp','fn')))
        result[f'{kind}_errors'] = result[f'{kind}_fp']+result[f'{kind}_fn']
    result['edge_strict_count'] = sum(m['edge_strict'] for m in metrics)
    result['joint_strict_count'] = sum(m['joint_strict'] for m in metrics)
    result['face_fn_missing'] = sum(m['face_fn_missing'] for m in metrics)
    result['face_fn_present'] = sum(m['face_fn_present'] for m in metrics)
    return result


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def build(root, plots=True):
    snapshot = root/'snapshot'; out = root/'reports'; out.mkdir(exist_ok=True)
    inputs = {}
    def read(path):
        inputs[str(path.relative_to(root))] = sha(path)
        return json.loads(path.read_text())
    selection = read(snapshot/'initial/configs/selection50.json')
    uids = [r['uid'] for r in selection['records']]
    vertices = {r['uid']:r['vertices'] for r in selection['records']}
    assert len(uids)==len(set(uids))==50
    for path in (snapshot/'c0/configs/selection50.json',snapshot/'candidates/configs/selection50.json'):
        assert read(path)==selection, path
    cache_path = snapshot/'initial/cache/manifest.json'; cache = read(cache_path)
    assert cache['uids']==uids and cache['complete']
    baseline_path = snapshot/'initial/vae_baseline/summary.json'; baseline = read(baseline_path)
    assert baseline['complete'] and baseline['identity']['uids']==uids
    assert baseline['identity']['cache_sha256']==sha(cache_path)
    specs = [
        ('vae_mu','VAE','initial/vae_baseline/mu'),
        ('vae_posterior_seed0','VAE','initial/vae_baseline/posterior_seed0'),
        ('vae_posterior_seed1','VAE','initial/vae_baseline/posterior_seed1'),
        ('c0_step500','C0','c0/evaluations/step500_seed0'),
        ('c0_step1000','C0','c0/evaluations/step1000_seed0'),
        ('c1_step500','C1','candidates/c1/eval500'),
        ('c1_step904','C1','candidates/c1/eval904'),
        ('c2_step500','C2','candidates/c2/eval500')]
    variants = {}; checks=[]; per_uid=[]; coverage=[]; full=[]; partial=[]; visual_inputs=[]
    metric_fields = [f'{kind}_{field}' for kind in ('edge','face') for field in ('tp','fp','fn','micro_f1')]
    metric_fields += ['edge_strict','joint_strict','face_fn_missing','face_fn_present','actual_face_candidates']
    for label, group, relative in specs:
        directory = snapshot/relative
        summary_path = directory/'summary.json'
        summary = read(summary_path) if summary_path.exists() else None
        identity_path = directory/'identity.json'
        identity = read(identity_path) if identity_path.exists() else summary['identity']
        assert identity['uids']==uids and identity['cache_sha256']==sha(cache_path)
        assert identity['vae_sha256']==cache['source_checkpoint_sha256']
        if group!='VAE':
            assert identity['seed']==0 and identity['euler_steps']==50
            assert identity['vae_baseline_summary_sha256']==sha(baseline_path)
        if summary is not None: assert summary['identity']==identity
        if group=='VAE':
            detail=baseline['variants'][identity['variant']]
            assert all(summary[k]==detail[k] for k in ('complete','evaluated_uids','metrics_sha256','metrics'))
        metrics={}; hashes={}
        for path in sorted(directory.glob('nexus_*/metrics.json')):
            uid=path.parent.name; row=read(path)
            assert uid in uids and row['uid']==uid and row['complete']
            assert row['identity']==dict(identity,uid=uid)
            for kind in ('edge','face'):
                assert all(type(row[kind][k]) is int and row[kind][k]>=0 for k in ('tp','fp','fn'))
                assert math.isclose(row[kind]['micro_f1'],f1(*(row[kind][k] for k in ('tp','fp','fn'))),abs_tol=1e-14)
            assert row['edge_strict']==(row['edge']['fp']==row['edge']['fn']==0)
            assert row['joint_strict']==(row['edge_strict'] and row['face']['fp']==row['face']['fn']==0)
            assert row['face_fn_missing']+row['face_fn_present']==row['face']['fn']
            assert row['actual_face_candidates']==row['face']['tp']+row['face']['fp']+row['face'].get('tn',0)+row['face_fn_present']
            for binding_name in ('binding.json','latent.json'):
                binding=read(path.parent/binding_name)
                assert binding['identity']==dict(identity,uid=uid)
            metrics[uid]=row; hashes[uid]=sha(path)
            vp=path.parent/'visualizations.json'
            visual=read(vp) if vp.exists() else None
            if visual:
                assert visual['identity']==dict(identity,uid=uid)
                assert visual['predicted_edges']==row['edge']['tp']+row['edge']['fp']
                assert visual['predicted_faces']==row['face']['tp']+row['face']['fp']
            visual_inputs.append(dict(variant=label,uid=uid,metrics_sha256=hashes[uid],metadata_complete=bool(visual and visual['complete']),
                expected_files=[dict(path=str((path.parent/f['file']).relative_to(root)),sha256=f['sha256'],bytes=f['bytes'],
                    present_when_report_built=(path.parent/f['file']).exists()) for f in (visual['files'] if visual else [])]))
        observed=[u for u in uids if u in metrics]
        complete=bool(summary and summary['complete'] and observed==uids)
        if summary:
            assert summary['evaluated_uids']==observed and summary['metrics_sha256']==hashes
            if summary['complete']: assert observed==uids
            if not complete: assert 'metrics' not in summary
        computed=aggregate([metrics[u] for u in observed]) if observed else None
        if complete:
            saved=summary['metrics']
            for kind in ('edge','face'):
                for field in ('tp','fp','fn','micro_f1'):
                    assert math.isclose(computed[f'{kind}_{field}'],saved[kind][field],abs_tol=1e-14), (label,kind,field)
            for key in ('edge_strict_count','joint_strict_count','face_fn_missing','face_fn_present'):
                assert computed[key]==saved[key],(label,key)
            assert saved['strict50_under_this_noise']==(computed['joint_strict_count']==50)
        info=dict(variant=label,group=group,kind='vae_reconstruction' if group=='VAE' else 'flow_generation',
            completed_updates=identity.get('completed_updates'),flow_sha256=identity.get('flow_sha256'),
            seed=identity.get('seed'),euler_steps=identity.get('euler_steps'),expected_meshes=50,
            evaluated_meshes=len(observed),evaluation_complete=complete,summary_present=summary is not None,
            summary_sha256=sha(summary_path) if summary else None,
            scope='full50' if complete else f'observed_subset_{len(observed)}_of50')
        variants[label]=dict(info=info,metrics=metrics,identity=identity,directory=directory)
        coverage.append(dict(info,missing_uids=';'.join(u for u in uids if u not in metrics)))
        if complete: full.append(dict(info,**computed,strict50_under_this_noise=computed['joint_strict_count']==50))
        elif computed: partial.append(dict(info,**computed))
        checks.append(dict(variant=label,committed_metrics=len(metrics),identity_consistent=True,
            summary_metric_hashes_verified=len(hashes) if summary else None,
            source_summary_aggregate_verified=complete,local_metrics_sha256=hashes,
            limitation=None if summary else 'No source summary; per-UID identities and local SHA recorded, no full50 aggregate'))
        for uid in uids:
            m=metrics.get(uid)
            row=dict(variant=label,group=group,kind=info['kind'],completed_updates=info['completed_updates'],
                flow_sha256=info['flow_sha256'],seed=info['seed'],euler_steps=info['euler_steps'],uid=uid,
                gt_vertices=vertices[uid],evaluation_complete=complete,status='evaluated' if m else 'not_evaluated',
                metrics_sha256=hashes.get(uid),**{k:None for k in metric_fields})
            if m:
                for kind in ('edge','face'):
                    for field in ('tp','fp','fn','micro_f1'):row[f'{kind}_{field}']=m[kind][field]
                for key in metric_fields[8:]:row[key]=m[key]
            per_uid.append(row)
    reference_noise=variants['c0_step500']['identity']
    for v in variants.values():
        if v['info']['group']!='VAE':
            assert all(v['identity'][k]==reference_noise[k] for k in ('noise_recipe','seed','euler_steps','threshold','vertices'))
    # Metrics always refer to the same unmodified GT topology, including partial results.
    mu=variants['vae_mu']['metrics']
    for variant in variants.values():
        for uid,m in variant['metrics'].items():
            for kind in ('edge','face'):assert m[kind]['tp']+m[kind]['fn']==mu[uid][kind]['tp']+mu[uid][kind]['fn']
    assert sum(mu[u]['edge']['tp']+mu[u]['edge']['fn'] for u in uids)==cache['data']['totals']['edges']
    assert sum(mu[u]['face']['tp']+mu[u]['face']['fn'] for u in uids)==cache['data']['totals']['faces']

    checkpoints=[]; training_rows=[]; participation=[]; training_summary=[]; training={}
    for group, run in [('C0',snapshot/'c0/run'),('C1',snapshot/'candidates/c1/run'),('C2',snapshot/'candidates/c2/run')]:
        manifest=read(run/'checkpoint-manifest.json'); latest=read(run/'latest.json'); status=read(run/'status.json')
        for entry in manifest['checkpoints']:
            label=f'{group.lower()}_step{entry["completed_updates"]}'
            checkpoints.append(dict(group=group,completed_updates=entry['completed_updates'],path=entry['path'],
                sha256=entry['sha256'],bytes=entry['bytes'],retained=entry.get('retained',True),
                protection=';'.join(entry.get('protected',[])),evaluation_variant=label if label in variants else None))
            if entry.get('retained',True) and label not in variants:
                info=dict(variant=label,group=group,kind='flow_generation',completed_updates=entry['completed_updates'],
                    flow_sha256=entry['sha256'],seed=None,euler_steps=None,expected_meshes=50,evaluated_meshes=0,
                    evaluation_complete=False,summary_present=False,summary_sha256=None,scope='not_evaluated')
                coverage.append(dict(info,missing_uids=';'.join(uids)))
                for uid in uids:
                    per_uid.append(dict(variant=label,group=group,kind='flow_generation',completed_updates=entry['completed_updates'],
                        flow_sha256=entry['sha256'],seed=None,euler_steps=None,uid=uid,gt_vertices=vertices[uid],evaluation_complete=False,
                        status='not_evaluated',metrics_sha256=None,**{k:None for k in metric_fields}))
        updates={}; counter=collections.Counter(); source_logs=[]
        for log in sorted(run.glob('updates-*.jsonl')):
            inputs[str(log.relative_to(root))]=sha(log);source_logs.append(str(log.relative_to(root)))
            for line in log.read_text().splitlines():
                record=json.loads(line);step=record['completed_updates']
                assert step not in updates,(group,step)
                assert record['forward_at_completed_updates']==step-1 and record['adam_steps']==[step]
                mesh_uids=[m['uid'] for m in record['meshes']]
                assert len(mesh_uids)==len(set(mesh_uids))==5 and set(mesh_uids)<=set(uids)
                assert math.isclose(record['velocity_mse'],statistics.mean(m['velocity_mse'] for m in record['meshes']),abs_tol=1e-12)
                counter.update(mesh_uids);updates[step]=record
                training_rows.append(dict(group=group,completed_updates=step,forward_at_completed_updates=step-1,
                    velocity_mse=record['velocity_mse'],lr=record['lr'],seconds=record['seconds'],elapsed_seconds=record['elapsed_seconds'],
                    peak_allocated_bytes=record['peak_allocated'],peak_reserved_bytes=record['peak_reserved'],
                    uid_group=';'.join(mesh_uids),source_log=str(log.relative_to(root))))
        last=max(updates);assert sorted(updates)==list(range(1,last+1))
        assert latest['completed_updates']==status['completed_updates']==last
        for epoch in range(last//10):
            seen=[m['uid'] for step in range(epoch*10+1,epoch*10+11) for m in updates[step]['meshes']]
            assert sorted(seen)==sorted(uids)
        assert sum(counter.values())==last*5
        for uid in uids:
            participation.append(dict(group=group,uid=uid,gt_vertices=vertices[uid],effective_updates=last,
                direct_participations=counter[uid],complete_epochs=last//10,additional_groups=last%10))
        ordered=[updates[i] for i in range(1,last+1)]
        ts=dict(group=group,completed_updates=last,direct_participations_total=last*5,
            min_per_mesh=min(counter.values()),max_per_mesh=max(counter.values()),complete_epochs=last//10,additional_groups=last%10,
            first10_mean_velocity_mse=statistics.mean(r['velocity_mse'] for r in ordered[:10]),
            last10_mean_velocity_mse=statistics.mean(r['velocity_mse'] for r in ordered[-10:]),
            median_update_seconds=statistics.median(r['seconds'] for r in ordered),
            train_elapsed_seconds=status['elapsed_seconds'],update_seconds_sum=sum(r['seconds'] for r in ordered),
            latest_checkpoint_sha256=latest['sha256'],latest_checkpoint_path=latest['path'],logs_contiguous=True,
            log_sources=';'.join(source_logs),target_reached=status['complete'])
        training_summary.append(ts);training[group]=ordered
    common_training=min(len(v) for v in training.values())
    for step in range(common_training):
        signature=[(m['uid'],m['time']) for m in training['C0'][step]['meshes']]
        assert all([(m['uid'],m['time']) for m in rows[step]['meshes']]==signature for rows in training.values())

    groups={'all50':uids,'N_lt500':[u for u in uids if vertices[u]<500],
        'N_500_to1499':[u for u in uids if 500<=vertices[u]<1500],
        'N_ge1500':[u for u in uids if vertices[u]>=1500],
        'N_ge2000':[u for u in uids if vertices[u]>=2000]}
    groups['vae_mu_error_top10']=sorted(uids,key=lambda u:(-sum(mu[u][k][f] for k in ('edge','face') for f in ('fp','fn')),u))[:10]
    pair_specs=[('c0_step500','c1_step500'),('c0_step500','c0_step1000'),('c1_step500','c1_step904'),
        ('c0_step500','c2_step500'),('c1_step500','c2_step500')]
    comparisons=[];pair_rows=[]
    for left,right in pair_specs:
        a=variants[left]['metrics'];b=variants[right]['metrics'];common=[u for u in uids if u in a and u in b]
        for uid in uids:
            row=dict(left=left,right=right,uid=uid,gt_vertices=vertices[uid],common_available=uid in common)
            for kind in ('edge','face'):
                for field in ('tp','fp','fn','micro_f1'):
                    row[f'delta_{kind}_{field}']=b[uid][kind][field]-a[uid][kind][field] if uid in common else None
            pair_rows.append(row)
        for name,members in groups.items():
            available=[u for u in members if u in common]
            if not available:continue
            aa=aggregate([a[u] for u in available]);bb=aggregate([b[u] for u in available])
            row=dict(left=left,right=right,group=name,group_expected_meshes=len(members),common_meshes=len(available),
                group_complete=len(available)==len(members),source_both_full50=variants[left]['info']['evaluation_complete'] and variants[right]['info']['evaluation_complete'],
                scope='complete_defined_group' if len(available)==len(members) else 'observed_common_subset',uids=';'.join(available))
            for kind in ('edge','face'):
                for field in ('tp','fp','fn','micro_f1','errors'):
                    row[f'left_{kind}_{field}']=aa[f'{kind}_{field}'];row[f'right_{kind}_{field}']=bb[f'{kind}_{field}']
                    row[f'delta_{kind}_{field}']=bb[f'{kind}_{field}']-aa[f'{kind}_{field}']
                deltas=[b[u][kind]['micro_f1']-a[u][kind]['micro_f1'] for u in available]
                row[f'{kind}_f1_improved']=sum(x>1e-14 for x in deltas)
                row[f'{kind}_f1_worsened']=sum(x< -1e-14 for x in deltas)
                row[f'{kind}_f1_unchanged']=sum(abs(x)<=1e-14 for x in deltas)
            comparisons.append(row)
    gaps=[]
    for label,v in variants.items():
        if v['info']['group']=='VAE':continue
        observed=[u for u in uids if u in v['metrics']]
        current=aggregate([v['metrics'][u] for u in observed]);reference=aggregate([mu[u] for u in observed])
        row=dict(variant=label,observed_meshes=len(observed),expected_meshes=50,
            scope='full50' if v['info']['evaluation_complete'] else 'observed_subset_same_UIDs_only',uids=';'.join(observed))
        for key in current:
            row['flow_'+key]=current[key];row['vae_mu_'+key]=reference[key];row['delta_'+key]=current[key]-reference[key]
        gaps.append(row)
    for row in per_uid:
        if row['group']=='VAE':continue
        reference=mu[row['uid']]
        for kind in ('edge','face'):
            for field in ('tp','fp','fn','micro_f1'):
                row[f'vae_mu_{kind}_{field}']=reference[kind][field]
                value=row[f'{kind}_{field}']
                row[f'delta_from_vae_mu_{kind}_{field}']=value-reference[kind][field] if value is not None else None
    write_csv(out/'full50_aggregates.csv',full)
    write_csv(out/'observed_subset_aggregates.csv',partial)
    write_csv(out/'evaluation_coverage.csv',coverage)
    write_csv(out/'per_uid_all_checkpoints.csv',per_uid)
    write_csv(out/'checkpoint_inventory.csv',checkpoints)
    write_csv(out/'training_updates.csv',training_rows)
    write_csv(out/'training_mesh_participation.csv',participation)
    write_csv(out/'training_summary.csv',training_summary)
    write_csv(out/'common_uid_pairwise.csv',pair_rows)
    write_csv(out/'common_uid_group_comparisons.csv',comparisons)
    write_csv(out/'flow_vs_vae_same_uid_gaps.csv',gaps)
    write_json(out/'diagnostic_groups.json',dict(definitions={
        'N_groups':'Original GT vertex count, fixed before comparisons; first three size groups partition50. N_ge2000 overlaps N_ge1500.',
        'vae_mu_error_top10':'Ten highest VAE-mu edge+face FP+FN counts, ties broken by UID; diagnostic group independent of Flow results, not a causal claim.'},groups=groups))
    write_json(out/'visualization_inputs.json',dict(note='Use exact committed gt.obj/predicted.obj and verify these hashes before rendering. No invented geometry or missing-UID substitution.',
        records=visual_inputs,additional_latent_diagnostic_inputs='Each completed UID latent.npz/latent.json plus initial/cache/UID.npz and fixed normalization in manifest.'))
    integrity=dict(passed=True,expected_meshes=50,full50_variants=[r['variant'] for r in full],
        partial_variants=[dict(variant=r['variant'],evaluated_meshes=r['evaluated_meshes']) for r in partial],
        checks=checks,common_training_uid_order_and_logged_times_verified_updates=common_training,
        metadata_only=True,limitations=['No model/Adam tensor or raw edge/face array recomputation is performed.',
        'C2 partial39 has no source summary; local hashes and per-UID identities do not recreate a committed full evaluation summary.',
        'Micro-F1 is computed from summed counts, never mean per-mesh F1. Missing metrics remain blank.',
        'VAE error-count differences are descriptive; overlap/causal attribution requires actual topology sets.'],input_sha256=inputs)
    write_json(out/'metrics_integrity.json',integrity)
    write_json(out/'metrics_summary.json',dict(full50=full,partial_observed_only=partial,coverage=coverage,
        training=training_summary,comparisons=comparisons,gaps=gaps))
    if plots: make_plots(out,training,full,variants,vertices,uids)
    lines=['# Snapshot metrics report','',
        'All tables are rebuilt from snapshot metadata. Full50 aggregates require complete source summaries, exact per-UID SHA matches and recomputed TP/FP/FN agreement. Training MSE, VAE reconstruction and actual Euler50 Gaussian generation are separate.',
        '', '| Variant | Coverage | Edge micro-F1 | Face micro-F1 | Edge strict | Joint strict |', '|---|---:|---:|---:|---:|---:|']
    for r in full:lines.append(f'| {r["variant"]} | 50/50 | {r["edge_micro_f1"]:.9f} | {r["face_micro_f1"]:.9f} | {r["edge_strict_count"]}/50 | {r["joint_strict_count"]}/50 |')
    lines += ['', '## Incomplete evaluations','',
        '- C1 update904: 41/50 committed metrics, source summary incomplete. C2 update500: 39/50 committed metrics, no source summary. Their subset counts live only in `observed_subset_aggregates.csv`; these are not full50 scores.',
        '- C2 retained update1000 and latest1049 have no evaluated metrics. Their 100 per-UID cells remain explicitly not_evaluated with blank metric fields.',
        '- Partial evaluation follows the fixed UID order, mostly omitting larger meshes; observed subsets are not unbiased estimates of full50 performance.', '', '## Training exposure','',
        '| Group | Updates | Direct participations per mesh | Last10 velocity MSE |', '|---|---:|---:|---:|']
    for r in training_summary:lines.append(f'| {r["group"]} | {r["completed_updates"]} | {r["min_per_mesh"]}–{r["max_per_mesh"]} | {r["last10_mean_velocity_mse"]:.6f} |')
    lines += ['', '## Reading the comparisons','',
        '- `common_uid_group_comparisons.csv` gives C0→C1 at500, C0 500→1000, C1 500→904 on the same observed41, and C2 at500 on its same observed39. Every row carries the exact denominator and UID list.',
        '- Large-mesh groups use original GT N≥1500 and N≥2000; overlapping groups are labeled. VAE-worst10 uses only VAE errors, independent of Flow results.',
        '- `flow_vs_vae_same_uid_gaps.csv` recomputes the VAE baseline on the identical observed UIDs for every comparison. Difference is not a causal error decomposition or a reconstruction ceiling.',
        '- The branches share data, evaluation noise, and the logged training UID/time schedule over their common updates. Architecture and execution history still differ; no causal proof is asserted.',
        '- Figures show recorded metadata only. Actual mesh contact sheets must use committed OBJ inputs listed in `visualization_inputs.json`; none are fabricated by this script.',
        '', 'Rebuild: `python tools/build_metrics_report.py --root .` (matplotlib required for PNG/SVG; `--no-plots` rebuilds tables only).']
    (out/'METRICS_README.md').write_text('\n'.join(lines)+'\n')
    generated=[p for p in out.iterdir() if p.is_file() and p.name!='metrics_report_manifest.json']
    write_json(out/'metrics_report_manifest.json',dict(builder_sha256=sha(Path(__file__)),
        input_files=len(inputs),source='snapshot only',files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(generated)]))
    print(json.dumps(dict(passed=True,full50=len(full),partial={r['variant']:r['evaluated_meshes'] for r in partial},
        per_uid_rows=len(per_uid),training_updates={r['group']:r['completed_updates'] for r in training_summary}),indent=2))


def make_plots(out,training,full,variants,vertices,uids):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    colors={'C0':'#225ea8','C1':'#d95f0e','C2':'#238b45'}
    def save(fig,name):
        fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight')
        fig.savefig(out/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
    for group,rows in training.items():
        x=[r['completed_updates'] for r in rows];y=[r['velocity_mse'] for r in rows]
        smooth=[statistics.mean(y[max(0,i-19):i+1]) for i in range(len(y))]
        axes[0].plot(x,y,color=colors[group],alpha=.15,lw=.6)
        axes[0].plot(x,smooth,color=colors[group],label=f'{group}: {len(rows)} updates',lw=1.8)
        axes[1].plot([i/10 for i in x],smooth,color=colors[group],label=group,lw=1.8)
    axes[0].set(xlabel='Effective optimizer update',ylabel='Equal-mesh velocity MSE',title='Training objective: raw + trailing20 mean')
    axes[1].set(xlabel='Average direct participations per mesh',ylabel='Equal-mesh velocity MSE',title='Exposure = updates / 10, not updates per mesh')
    for a in axes:a.grid(alpha=.2);a.legend()
    save(fig,'training_velocity_mse')
    flow=[r for r in full if r['kind']=='flow_generation'];mu=next(r for r in full if r['variant']=='vae_mu')
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
    for ax,kind in zip(axes,('edge','face')):
        labels=[r['variant'] for r in flow]+['VAE mu baseline'];values=[r[kind+'_micro_f1'] for r in flow]+[mu[kind+'_micro_f1']]
        ax.bar(range(len(values)),values,color=['#225ea8','#6baed6','#d95f0e','#777777'])
        ax.set_xticks(range(len(values)),labels,rotation=22,ha='right');ax.set_yscale('log');ax.set_ylim(1e-4,1.6)
        ax.set_ylabel('Micro-F1 (log scale)');ax.set_title(f'{kind.capitalize()}: complete50 only')
        for i,v in enumerate(values):ax.text(i,v*1.12,f'{v:.5f}',ha='center',fontsize=9)
        ax.grid(axis='y',alpha=.2)
    save(fig,'full50_generation_vs_vae')
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),constrained_layout=True)
    for row,(left,right,title) in enumerate([('c0_step500','c1_step500','C1 minus C0 at update500'),('c0_step500','c0_step1000','C0 update1000 minus500')]):
        for col,kind in enumerate(('edge','face')):
            ax=axes[row,col];a=variants[left]['metrics'];b=variants[right]['metrics']
            ax.scatter([vertices[u] for u in uids],[b[u][kind]['micro_f1']-a[u][kind]['micro_f1'] for u in uids],s=26,alpha=.8,color='#225ea8' if col==0 else '#d95f0e')
            ax.axhline(0,color='black',lw=.8);ax.axvline(1500,color='#777',ls='--',lw=.8)
            ax.set(xlabel='Original GT vertices',ylabel=f'Delta {kind} F1',title=title+f' / {kind}');ax.grid(alpha=.2)
    save(fig,'common_uid_f1_changes')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args();build(args.root.resolve(),plots=not args.no_plots)
