"""CPU-only evidence report; does not import or modify frozen experiment runtime.

All 50 UIDs appear in each table/contact sheet. Missing evaluations stay missing.
Geometry uses every committed OBJ face and edge, with one fixed orthographic view.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


TASKS = ('edge', 'face')
COUNTS = ('tp', 'fp', 'fn')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4 << 20), b''): h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def f1(row):
    return 2*row['tp']/max(2*row['tp']+row['fp']+row['fn'], 1)


def csv_write(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def source_identity(directory):
    if (directory/'identity.json').exists(): return read_json(directory/'identity.json')
    if (directory/'summary.json').exists(): return read_json(directory/'summary.json').get('identity')
    return None


def collect_variant(directory, label, kind, cache, cache_sha, expected_identity=None):
    summary = read_json(directory/'summary.json') if (directory/'summary.json').exists() else {}
    identity = expected_identity or source_identity(directory)
    if identity:
        if identity['uids'] != cache['uids'] or identity['cache_sha256'] != cache_sha or identity['vae_sha256'] != cache['source_checkpoint_sha256']:
            raise ValueError('Report input uses a different cache/VAE/selection: '+str(directory))
    metric_hashes = summary.get('metrics_sha256', {})
    rows, raw = [], {}
    for uid in cache['uids']:
        row = dict(kind=kind, variant=label, uid=uid,
            vertices=next((r['vertices'] for r in cache.get('records',[]) if r['uid']==uid),''), status='not_evaluated',
            completed_updates=identity.get('completed_updates', '') if identity else '',
            flow_sha256=identity.get('flow_sha256', '') if identity else '',
            seed=identity.get('seed', '') if identity else '',
            euler_steps=identity.get('euler_steps', '') if identity else '',
            metrics_sha256='', edge_strict='', joint_strict='')
        row.update({task+'_'+key:'' for task in TASKS for key in (*COUNTS, 'micro_f1')})
        row.update(face_fn_missing='', face_fn_present='', actual_face_candidates='')
        path = directory/uid/'metrics.json'
        if path.exists():
            result = read_json(path)
            if not identity or result.get('identity') != dict(identity, uid=uid):
                raise ValueError('Per-UID metric identity mismatch: '+str(path))
            actual_sha = sha(path)
            if uid in metric_hashes and metric_hashes[uid] != actual_sha:
                raise ValueError('Per-UID metric hash mismatch: '+str(path))
            row['metrics_sha256'] = actual_sha
            if result.get('complete'):
                if result.get('uid') != uid: raise ValueError('Metric UID mismatch: '+str(path))
                row['status'] = 'complete' if uid in metric_hashes else 'complete_metric_not_in_summary'
                for task in TASKS:
                    for key in COUNTS:
                        value = result[task][key]
                        if type(value) is not int or value < 0: raise ValueError('Invalid confusion count: '+str(path))
                        row[task+'_'+key] = value
                    row[task+'_micro_f1'] = f1(result[task])
                    if abs(result[task]['micro_f1']-row[task+'_micro_f1']) > 1e-12:
                        raise ValueError('Stored per-UID F1 mismatch: '+str(path))
                row['edge_strict'] = result['edge']['fp'] == result['edge']['fn'] == 0
                row['joint_strict'] = row['edge_strict'] and result['face']['fp'] == result['face']['fn'] == 0
                for key in ('face_fn_missing', 'face_fn_present', 'actual_face_candidates'): row[key] = result.get(key, '')
                raw[uid] = result
            else: row['status'] = 'incomplete'
        rows.append(row)
    complete = summary.get('complete') is True and all(r['status'] == 'complete' for r in rows)
    aggregate = dict(kind=kind, variant=label, complete=complete, evaluated_meshes=len(raw), expected_meshes=len(rows),
                     completed_updates=rows[0]['completed_updates'], flow_sha256=rows[0]['flow_sha256'], seed=rows[0]['seed'])
    aggregate.update({task+'_'+key:'' for task in TASKS for key in (*COUNTS, 'micro_f1')})
    aggregate.update(edge_strict_count='', joint_strict_count='', strict50_under_this_noise='')
    if complete:
        for task in TASKS:
            counts = {key:sum(result[task][key] for result in raw.values()) for key in COUNTS}
            for key,value in counts.items(): aggregate[task+'_'+key] = value
            aggregate[task+'_micro_f1'] = f1(counts)
            for key in (*COUNTS, 'micro_f1'):
                if abs(summary['metrics'][task][key]-aggregate[task+'_'+key]) > 1e-12:
                    raise ValueError('Full-selection summary mismatch: '+str(directory))
        aggregate['edge_strict_count'] = sum(r['edge_strict'] for r in rows)
        aggregate['joint_strict_count'] = sum(r['joint_strict'] for r in rows)
        aggregate['strict50_under_this_noise'] = len(rows) == 50 and aggregate['joint_strict_count'] == 50
        for key in ('edge_strict_count', 'joint_strict_count'):
            if summary['metrics'][key] != aggregate[key]: raise ValueError('Strict-count summary mismatch')
    return dict(directory=directory, label=label, kind=kind, identity=identity, rows=rows, raw=raw, aggregate=aggregate)


def comparisons(variant, baselines):
    rows = []
    for current in variant['rows']:
        for baseline in baselines:
            reference = next(r for r in baseline['rows'] if r['uid'] == current['uid'])
            row = dict(variant=variant['label'], uid=current['uid'], vertices=current['vertices'], baseline=baseline['label'],
                       current_status=current['status'], baseline_status=reference['status'])
            available = current['status'].startswith('complete') and reference['status'].startswith('complete')
            for task in TASKS:
                for key in (*COUNTS, 'micro_f1'):
                    field = task+'_'+key
                    row['delta_'+field] = current[field]-reference[field] if available else ''
            rows.append(row)
    return rows


def obj_vertices(path):
    with path.open() as f:
        vertices = [list(map(float,line.split()[1:])) for line in f if line.startswith('v ')]
    xyz = np.asarray(vertices, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError('Invalid committed OBJ vertices: '+str(path))
    return xyz


def committed_geometry(directory, expected_identity):
    path = directory/'visualizations.json'
    if not path.exists(): return None
    manifest = read_json(path)
    if not manifest.get('complete') or manifest['identity'] != expected_identity:
        raise ValueError('Visualization identity mismatch: '+str(path))
    files = {}
    for record in manifest['files']:
        p = (directory/record['file']).resolve()
        if not p.is_relative_to(directory.resolve()) or sha(p) != record['sha256']:
            raise ValueError('Visualization source path/hash mismatch: '+str(p))
        files[record['file']] = p
    if set(files) != {'gt.obj', 'predicted.obj'}: raise ValueError('Missing committed OBJ pair')
    gt, predicted = obj_vertices(files['gt.obj']), obj_vertices(files['predicted.obj'])
    if not np.array_equal(gt, predicted): raise ValueError('Generated visualization changed GT coordinates/order')
    return dict(vertices=gt, gt=files['gt.obj'], predicted=files['predicted.obj'], manifest=manifest)


def projected(vertices):
    # Fixed azimuth 45 deg, elevation 25 deg; projection only, no mesh modification.
    az, el = np.deg2rad([45.,25.])
    basis = np.array([[np.cos(az),-np.sin(az),0],
                      [np.sin(el)*np.sin(az),np.sin(el)*np.cos(az),np.cos(el)]])
    xy = vertices@basis.T
    center = (xy.max(0)+xy.min(0))/2
    span = max(float(np.ptp(xy,axis=0).max()),1e-12)
    xy = (xy-center)/span*.88
    return xy


def geometry_tile(vertices, geometry, size=420):
    image = Image.new('RGB',(size,size),'white')
    draw = ImageDraw.Draw(image)
    xy = projected(vertices)
    pixels = np.stack(((xy[:,0]+.5)*size,(.5-xy[:,1])*size),axis=1)
    counts = dict(edges=0,faces=0)
    def draw_ids(ids,kind):
        ids = np.asarray(ids, dtype=np.int64)
        if ids.shape != ((3,) if kind=='f' else (2,)) or ids.min()<0 or ids.max()>=len(vertices):
            raise ValueError('OBJ contains invalid local vertex indices')
        coords = [tuple(p) for p in pixels[ids]]
        if kind=='f': draw.polygon(coords,fill='#d6e5f3');counts['faces']+=1
        else: draw.line(coords,fill='#214968',width=1);counts['edges']+=1
    if isinstance(geometry,Path):
        # Two streaming passes draw every face then every edge without retaining a dense graph.
        for kind in ('f','l'):
            with geometry.open() as f:
                for line in f:
                    if line.startswith(kind+' '): draw_ids([int(x)-1 for x in line.split()[1:]],kind)
    else:
        for ids in geometry['faces']: draw_ids(ids,'f')
        for ids in geometry['edges']: draw_ids(ids,'l')
    if not counts['faces'] and not counts['edges']:
        for x,y in pixels: draw.ellipse((x-1,y-1,x+1,y+1),fill='#214968')
    return image, counts


def cache_ground_truth(cache_root, cache, uid):
    record = next(r for r in cache['records'] if r['uid'] == uid)
    path = (cache_root/record['file']).resolve()
    if not path.exists(): return None
    if not path.is_relative_to(cache_root.resolve()) or sha(path) != record['sha256']:
        raise ValueError('GT cache file hash mismatch: '+str(path))
    with np.load(path,allow_pickle=False) as a:
        return dict(vertices=a['vertices'].copy(),faces=a['faces'].copy(),edges=a['edges'].copy())


def contact_sheets(variant, all_variants, cache_root, cache, output, geometries, ground_truth):
    files = []
    rows = variant['rows']
    for offset in range(0,len(rows),10):
        chunk = rows[offset:offset+10]
        fig, axes = plt.subplots(5,4,figsize=(12,14))
        for ax in axes.flat: ax.set_axis_off()
        for index,row in enumerate(chunk):
            uid = row['uid']; pair = [axes[index//2,(index%2)*2+j] for j in (0,1)]
            key = (variant['label'],uid)
            if key not in geometries:
                identity = dict(variant['identity'],uid=uid) if variant['identity'] else None
                geometries[key] = committed_geometry(variant['directory']/uid, identity) if identity else None
            geo = geometries[key]
            if uid not in ground_truth:
                truth = cache_ground_truth(cache_root,cache,uid)
                if truth is None:
                    for source in all_variants:
                        otherkey = (source['label'],uid)
                        if otherkey not in geometries:
                            identity = dict(source['identity'],uid=uid) if source['identity'] else None
                            geometries[otherkey] = committed_geometry(source['directory']/uid,identity) if identity else None
                        other = geometries[otherkey]
                        if other:
                            truth = dict(vertices=other['vertices'],obj=other['gt']);break
                ground_truth[uid] = truth
            truth = ground_truth[uid]
            for j,ax in enumerate(pair):
                title = uid+' | '+('GT' if j==0 else 'predicted')
                if j==0 and truth is not None:
                    tile,counts = geometry_tile(truth['vertices'],truth.get('obj',truth))
                    ax.imshow(tile);title += f"\nE={counts['edges']} F={counts['faces']}"
                elif j==1 and geo is not None:
                    if truth is not None and not np.array_equal(truth['vertices'],geo['vertices']):
                        raise ValueError('OBJ differs from fixed cache GT coordinates')
                    tile,counts = geometry_tile(geo['vertices'],geo['predicted'])
                    if uid in variant['raw']:
                        metric = variant['raw'][uid]
                        if counts != dict(edges=metric['edge']['tp']+metric['edge']['fp'],faces=metric['face']['tp']+metric['face']['fp']):
                            raise ValueError('OBJ prediction counts differ from complete metric')
                    ax.imshow(tile);title += f"\nE={counts['edges']} F={counts['faces']}"
                else:
                    ax.text(.5,.5,'GT geometry unavailable' if j==0 else 'No committed geometry\n'+row['status'],ha='center',va='center',transform=ax.transAxes,color='#777777')
                if j==1 and row['status'].startswith('complete'):
                    title += f"\nE FP/FN={row['edge_fp']}/{row['edge_fn']}  F FP/FN={row['face_fp']}/{row['face_fn']}"
                ax.set_title(title,fontsize=8)
        state = ('50 meshes evaluated; joint strict '+str(variant['aggregate']['joint_strict_count'])+'/50'
                 if variant['aggregate']['complete'] else 'INCOMPLETE - no full-selection score')
        fig.suptitle(variant['kind']+' / '+variant['label']+' / '+state+'\nFixed view; all committed edges/faces; no repair or topology truncation',fontsize=11)
        fig.tight_layout(rect=(0,0,1,.95))
        path = output/f"{variant['label']}-uids-{offset+1:02d}-{offset+len(chunk):02d}.png"
        fig.savefig(path,dpi=130);plt.close(fig);files.append(path.name)
    return files


def training_report(directory,output,uids):
    updates = {}
    sources = []
    for path in sorted(directory.glob('updates-*.jsonl')):
        sources.append(dict(file=str(path),sha256=sha(path)))
        with path.open() as f:
            for line in f:
                record = json.loads(line)
                step = record['completed_updates']
                if step in updates: raise ValueError('Repeated effective update in training logs; inspect restart accounting')
                updates[step]=record
    rows = [dict(completed_updates=step,forward_at_completed_updates=r['forward_at_completed_updates'],velocity_mse=r['velocity_mse'],
                 lr=r['lr'],gradient_norm_preclip=r['gradient_norm_preclip'],seconds=r.get('seconds',''),
                 participating_uids=';'.join(x['uid'] for x in r['meshes']),participating_meshes=len(r['meshes']))
            for step,r in sorted(updates.items())]
    if rows: csv_write(output/'training_velocity_mse.csv',rows)
    participation = {uid:0 for uid in uids}
    for record in updates.values():
        for mesh in record['meshes']: participation[mesh['uid']] += 1
    csv_write(output/'training_mesh_participation.csv',[dict(uid=uid,direct_participation_in_available_logs=count)
        for uid,count in participation.items()])
    fig,ax = plt.subplots(figsize=(9,4))
    if rows: ax.plot([r['forward_at_completed_updates'] for r in rows],[r['velocity_mse'] for r in rows],lw=1,color='#245f9c')
    else: ax.text(.5,.5,'No completed optimizer-update logs available',ha='center',transform=ax.transAxes)
    ax.set(xlabel='Completed optimizer updates at forward pass',ylabel='Equal-mesh velocity MSE',title='Training objective only (separate from reconstruction and generation)')
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(output/'training_velocity_mse.png',dpi=150);plt.close(fig)
    return dict(completed_update_records=len(rows),latest_completed_update=max(updates,default=0),
        logs_contiguous_from_update1=sorted(updates)==list(range(1,max(updates,default=0)+1)),
        direct_mesh_participation=participation,log_sources=sources)


def metric_plot(variants,output):
    complete = [v for v in variants if v['aggregate']['complete']]
    fig,axes = plt.subplots(1,2,figsize=(11,4),sharey=True)
    for task,ax in zip(TASKS,axes):
        if complete:
            bars = ax.bar(range(len(complete)),[v['aggregate'][task+'_micro_f1'] for v in complete],
                   color=['#9baabd' if v['kind']=='vae_reconstruction' else '#2c78b8' for v in complete])
            ax.bar_label(bars,labels=[format(v['aggregate'][task+'_micro_f1'],'.6f') for v in complete],fontsize=8,padding=2)
            ax.set_xticks(range(len(complete)),[v['label'] for v in complete],rotation=25,ha='right')
        else: ax.text(.5,.5,'No complete 50-mesh evaluation',ha='center',transform=ax.transAxes)
        ax.set(title=task.title()+' full-selection micro-F1',ylim=(0,1.02));ax.grid(axis='y',alpha=.2)
    fig.suptitle('VAE reconstruction and Gaussian Flow generation; complete variants only')
    fig.tight_layout();fig.savefig(output/'full_selection_f1.png',dpi=150);plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',required=True,type=Path)
    parser.add_argument('--cache',type=Path)
    parser.add_argument('--vae-baseline',type=Path)
    parser.add_argument('--evaluations',type=Path)
    parser.add_argument('--training-run',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    cache_root = args.cache or args.run_root/'cache'
    baseline_root = args.vae_baseline or args.run_root/'vae_baseline'
    evaluations = args.evaluations or args.run_root/'evaluations'
    training = args.training_run or args.run_root/'run'
    cache = read_json(cache_root/'manifest.json');cache_sha = sha(cache_root/'manifest.json')
    if not cache['complete'] or len(cache['uids']) != 50 or len(set(cache['uids'])) != 50:
        raise ValueError('Expected the complete fixed 50-UID cache')
    selection_path = Path(__file__).resolve().parent.parent/'configs/selection50.json'
    selection = read_json(selection_path)
    if cache['uids'] != [r['uid'] for r in selection['records']] or cache['data']['selection_sha256'] != sha(selection_path):
        raise ValueError('Cache differs from user-selected UID/count binding')
    args.output.mkdir(parents=True,exist_ok=True)
    baseline_identity = source_identity(baseline_root)
    baselines = [collect_variant(baseline_root/name,name,'vae_reconstruction',cache,cache_sha,
        dict(baseline_identity,variant=name) if baseline_identity else None) for name in ('mu','posterior_seed0','posterior_seed1')]
    names = {'step500_seed0','step1000_seed0'}
    if evaluations.exists(): names.update(p.name for p in evaluations.iterdir() if p.is_dir())
    flows = [collect_variant(evaluations/name,name,'flow_generation',cache,cache_sha) for name in sorted(names)]
    variants = baselines+flows
    inventory = []
    geometries,ground_truth = {},{}
    for variant in variants:
        label = variant['label']
        csv_write(args.output/(label+'-per_uid.csv'),variant['rows'])
        if variant['kind']=='flow_generation': csv_write(args.output/(label+'-versus_vae.csv'),comparisons(variant,baselines))
        sheets = contact_sheets(variant,variants,cache_root,cache,args.output,geometries,ground_truth)
        inventory.append(dict(label=label,directory=str(variant['directory']),identity=variant['identity'],
            summary_sha256=sha(variant['directory']/'summary.json') if (variant['directory']/'summary.json').exists() else None,
            aggregate=variant['aggregate'],contact_sheets=sheets))
    csv_write(args.output/'complete_only_aggregates.csv',[v['aggregate'] for v in variants])
    metric_plot(variants,args.output)
    report = dict(cache_manifest_sha256=cache_sha,source_vae_sha256=cache['source_checkpoint_sha256'],selection_sha256=sha(selection_path),
        normalization_sha256=cache.get('normalization_sha256'),uids=cache['uids'],expected_meshes=50,
        training=training_report(training,args.output,cache['uids']),variants=inventory,
        geometry_note='Every committed OBJ edge and triangle drawn in a fixed orthographic projection, azimuth45/elevation25. No repair, winding inference, or topology truncation. Missing data explicitly unavailable.',
        scoring_note='Only a complete single checkpoint/noise variant receives an aggregate. Per-UID deltas are current minus the same UID baseline; errors are not causally attributed by subtraction.',
        reporter_sha256=sha(Path(__file__)))
    (args.output/'report_manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output.resolve()),variants=len(variants),completed_variants=sum(v['aggregate']['complete'] for v in variants),training_updates=report['training']['completed_update_records']),indent=2))


if __name__=='__main__': main()
