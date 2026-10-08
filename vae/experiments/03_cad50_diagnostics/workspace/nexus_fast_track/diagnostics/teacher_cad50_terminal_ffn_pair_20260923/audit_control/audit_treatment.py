"""Bounded final saved-state audit. CPU only; no model forward or training imports."""
import argparse
import datetime
import json
import math
import traceback
from pathlib import Path
from audit_reused_control import sha, same, read, np, torch

STEPS = [0, 25, 50, 75, 100]
OLD_GROUPS = ['encoder_mu', 'decoder', 'edge_head', 'face_head']


def recount(path, pool_path, row):
    with np.load(path) as q, np.load(pool_path) as pool:
        n = len(q['vertices'])
        pe, ge, fi, gf = [q[k] for k in ['predicted_edge_ids', 'gt_edges', 'actual_face_candidate_ids', 'gt_faces']]
        ek = lambda a: a[:, 0].astype(np.int64)*n+a[:, 1]
        fk = lambda a: (a[:, 0].astype(np.int64)*n+a[:, 1])*n+a[:, 2]
        assert np.array_equal(q['vertices'], pool['vertices'])
        assert np.array_equal(ge, np.unique(np.sort(pool['edges'].T, axis=1), axis=0))
        assert np.array_equal(gf, np.sort(pool['positive'], axis=1))
        assert ((pe[:, 0] < pe[:, 1]) & (pe[:, 0] >= 0) & (pe[:, 1] < n)).all()
        assert (fi[:, 0] < fi[:, 1]).all() and (fi[:, 1] < fi[:, 2]).all()
        assert len(np.unique(ek(pe))) == len(pe) and len(np.unique(fk(fi))) == len(fi)
        assert np.isfinite(q['predicted_edge_logits']).all() and (q['predicted_edge_logits'] > 0).all()
        tp = int(np.isin(ek(pe), ek(ge)).sum())
        edge = dict(tp=tp, fp=len(pe)-tp, fn=len(ge)-tp, tn=n*(n-1)//2-len(ge)-(len(pe)-tp))
        assert all(edge[k] == row['edge'][k] for k in edge)
        labels, positive = np.isin(fk(fi), fk(gf)), q['actual_face_candidate_logits'] > 0
        assert np.isfinite(q['actual_face_candidate_logits']).all()
        assert np.isfinite(q['gt_face_logits']).all() and np.array_equal(labels, q['actual_face_candidate_gt'])
        tp, fp = int((positive & labels).sum()), int((positive & ~labels).sum())
        face = dict(tp=tp, fp=fp, fn=len(gf)-tp, tn=int((~positive & ~labels).sum()))
        assert all(face[k] == row['face'][k] for k in face)
        adjacency = [set() for _ in range(n)]
        for i, j in pe:
            adjacency[int(i)].add(int(j))
        triangles = sum(len(adjacency[i] & adjacency[j]) for i in range(n) for j in adjacency[i])
        assert triangles == len(fi) == row['face']['scored_candidates']
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            assert np.isin(fi[:, i].astype(np.int64)*n+fi[:, j], ek(pe)).all()
        covered = np.isin(fk(gf), fk(fi))
        assert np.array_equal(covered, q['gt_face_covered'])
        missing = int((~covered).sum())
        present_negative = int((covered & (q['gt_face_logits'] <= 0)).sum())
        assert missing == row['missing_gt_face_candidates'] == row['face']['fn_missing_candidate']
        assert present_negative == row['face']['fn_present_but_negative']
        assert missing + present_negative == face['fn']
        in_pool = np.isin(fk(fi), fk(np.concatenate([pool['positive'], pool['mixed']])))
        inside, outside = int((positive & ~labels & in_pool).sum()), int((positive & ~labels & ~in_pool).sum())
        assert inside == row['face']['actual_fp_inside_training_pool']
        assert outside == row['face']['actual_fp_outside_training_pool'] and inside + outside == face['fp']
        perfect = all(x[k] == 0 for x in [edge, face] for k in ['fp', 'fn'])
        assert row['joint_perfect'] == perfect
        assert row['edge_perfect'] == (edge['fp'] == edge['fn'] == 0)
        assert row['face_perfect'] == (face['fp'] == face['fn'] == 0)
        return dict(uid=row['uid'], vertices=n, edge=edge, face=face, joint_perfect=perfect,
                    face_fn_missing=missing, face_fn_present=present_negative,
                    face_fp_in_pool=inside, face_fp_out_pool=outside, complete_triangles=triangles)


def audit(args, result):
    root = args.experiment_root.resolve()
    base = root.parent.parent
    out = root / 'H_terminal_ffn'
    reuse = read(root / 'repro_outputs/CONTROL_REUSE_AUDIT.json')
    assert reuse['passed'] and reuse['reuse_control']
    control = Path(reuse['control'])
    source = Path(reuse['source'])
    assert sha(source) == reuse['source_sha256']
    cfg, done = read(out / 'config.json'), read(out / 'complete.json')
    scfg = read(control / 'config.json')
    assert not (out / 'failure.json').exists()
    assert done['new_updates'] == 100 and done['completed_updates'] == 2600 and done['stopped_at_budget']
    assert sha(control / 'config.json') == reuse['control_config_sha256']
    assert sha(control / 'updates.jsonl') == reuse['control_updates_sha256']
    for key in ['source_sha256', 'lr_schedule', 'betas', 'eps', 'weight_decay', 'clip', 'uids', 'totals',
                'data_manifest_sha256', 'pool_manifest_sha256', 'tau', 'soft4_epsilon', 'soft4_reduction',
                'effective_backend', 'model_mode', 'sampling', 'KL', 'logvar_frozen', 'microbatch',
                'meshes_per_update', 'new_updates', 'initial_completed_updates', 'final_completed_updates', 'checkpoints', 'scales']:
        assert cfg[key] == scfg[key], key
    assert cfg['mode'] == 'terminal_ffn' and cfg['stop_weight_grad'] is False
    assert {k: cfg['lrs'][k] for k in OLD_GROUPS} == scfg['lrs'] and cfg['lrs']['terminal_ffn'] == 3e-5
    assert list(cfg['trainable_groups']) == OLD_GROUPS + ['terminal_ffn']
    assert {k: cfg['trainable_groups'][k] for k in OLD_GROUPS} == scfg['trainable_groups']
    assert cfg['terminal_ffn_parameter_count'] == 8395776
    assert sha(out / 'data/manifest.json') == cfg['data_manifest_sha256']
    assert sha(out / 'pool_manifest.json') == cfg['pool_manifest_sha256']
    for row in read(out / 'data/manifest.json')['meshes']:
        assert sha(out / 'data' / row['path']) == row['sha256']
    for row in read(out / 'pool_manifest.json')['records']:
        assert sha(out / row['path']) == row['sha256']
    for name, digest in cfg['code_sha256'].items():
        assert sha(out / name) == digest, name
    for name in ['runtime.py', 'effective_loss_and_scoring.py', 'loader.py', 'evaluate.py', 'construction_args.json', 'pool_manifest.json']:
        assert sha(out / name) == sha(root / 'baseline_source' / name), name
    for p in (out / 'source_archive').rglob('*.py'):
        assert sha(p) == sha(base / p.relative_to(out / 'source_archive')), str(p)
    gate, ready, release = [read(p) for p in [out / 'startup_gate.json', out / 'ready.json', root / 'release.json']]
    assert all(x['passed'] for x in [gate, ready, release])
    assert gate['zero_step_all50_arrays_exact'] and gate['old_parameter_gradients_before_clip_exact']
    assert gate['old_optimizer_state_by_name_exact'] and gate['new_optimizer_state_empty'] and gate['parent_rng_restored']
    assert gate['optimizer_updates'] == ready['optimizer_updates'] == 0
    assert ready['device_check'] == reuse['control_device_check']
    assert release['source_sha256'] == cfg['source_sha256'] and release['new_updates'] == 100
    logs = [json.loads(x) for x in (out / 'updates.jsonl').read_text().splitlines()]
    assert [x['new_update'] for x in logs] == list(range(1, 101))
    for step, row in enumerate(logs, 1):
        assert row['state_before'] == 2499 + step and row['completed_updates'] == 2500 + step
        assert [m['uid'] for m in row['meshes']] == cfg['uids'] and row['participation_per_mesh'] == 2500 + step
        assert row['lrs'] == cfg['lrs'] and row['adam_steps'] == {**{k: 2500+step for k in OLD_GROUPS}, 'terminal_ffn': step}
        assert row['clip_coefficient'] == min(1., 1 / (row['total_grad_norm'] + 1e-6))
        numeric = [row['total_loss_before'], row['total_grad_norm'], row['seconds'], *row['gradient_norms'].values()]
        numeric += [v for m in row['meshes'] for k, v in m.items() if k in ['edge', 'face']]
        numeric += [v for g in row['actual_updates'].values() for v in g.values()]
        assert all(math.isfinite(x) for x in numeric)
        ffn = row['terminal_ffn_gradients_before_clip']
        assert all(math.isfinite(x['norm']) for x in ffn.values())
        if step == 1:
            assert ffn['out_proj.weight']['nonzero'] > 0
            assert all(v['nonzero'] == 0 for n, v in ffn.items() if not n.startswith('out_proj.'))
        if step == 2:
            assert ffn['in_proj.weight']['nonzero'] > 0 and ffn['norm.weight']['nonzero'] > 0
    print('PASS fixed protocol, code/data, startup gate and all100 update logs', flush=True)
    parent = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
    checkpoints, per_mesh, trajectory = [], [], []
    manifest = [dict(role='common parent', path=str(source), bytes=source.stat().st_size,
        sha256=reuse['source_sha256'],model_complete=True,optimizer_present=True,rng_present=True,
        completed_updates=2500,adam_steps={k:2500 for k in OLD_GROUPS},hash_evidence='fresh in this final CPU audit')]
    for old_record in reuse['checkpoint_identity_checks']:
        p=Path(old_record['path']);step=old_record['step']
        assert p.stat().st_size==old_record['bytes']
        if step==100:
            assert sha(p)==old_record['sha256']
        manifest.append(dict(role='reused control resumable',**old_record,model_complete=True,
            optimizer_present=True,rng_present=True,completed_updates=2500+step,
            adam_steps={k:2500+step for k in OLD_GROUPS},
            hash_evidence='fresh in final audit' if step==100 else 'fresh CPU reuse audit immediately before this run'))
    control_done=read(control/'complete.json')
    control_full=Path(control_done['full_model'])
    assert sha(control_full)==control_done['full_model_sha256']
    old_final=torch.load(Path(reuse['checkpoint_identity_checks'][-1]['path']),map_location='cpu',mmap=True,weights_only=False)
    old_inference=torch.load(control_full,map_location='cpu',mmap=True,weights_only=False)
    assert same(old_final['model'],old_inference['model'])
    assert old_final['completed_updates']==old_inference['completed_updates']==2600
    assert 'optimizer' not in old_inference and 'rng' not in old_inference
    manifest.append(dict(role='reused control inference',path=str(control_full),bytes=control_full.stat().st_size,
        sha256=control_done['full_model_sha256'],model_complete=True,optimizer_present=False,rng_present=False,
        completed_updates=2600,hash_evidence='fresh in final audit; model equals final resumable'))
    del old_final,old_inference
    for step in STEPS:
        ev, sev = read(out / f'eval-new{step:04d}.json'), read(control / f'eval-new{step:04d}.json')
        path = Path(ev['checkpoint'])
        assert sha(path) == ev['checkpoint_sha256']
        cp = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        assert same(cp['config'], cfg)
        assert cp['new_updates'] == step and cp['completed_updates'] == 2500 + step
        assert cp['parent_sha256'] == reuse['source_sha256']
        assert list(cp['participation']) == cfg['uids'] and set(cp['participation'].values()) == {2500 + step}
        opt = cp['optimizer']
        assert [g['name'] for g in opt['param_groups']] == OLD_GROUPS + ['terminal_ffn']
        assert same(opt['param_groups'][:4], parent['optimizer']['param_groups'])
        newg = opt['param_groups'][4]
        assert newg['lr'] == 3e-5 and tuple(newg['betas']) == (.9, .999) and newg['eps'] == 1e-8 and newg['weight_decay'] == 0
        assert len(opt['state']) == len(parent['optimizer']['state']) + (len(newg['params']) if step else 0)
        for group in opt['param_groups']:
            names = cfg['trainable_groups'][group['name']]
            assert len(names) == len(group['params'])
            for name, pid in zip(names, group['params']):
                param = cp['model']['autoencoder.' + name]
                if step == 0 and group['name'] == 'terminal_ffn':
                    assert pid not in opt['state']
                else:
                    state = opt['state'][pid]
                    assert int(state['step']) == (step if group['name'] == 'terminal_ffn' else 2500+step)
                    assert state['exp_avg'].shape == state['exp_avg_sq'].shape == param.shape
        for name, value in cp['model'].items():
            if torch.is_tensor(value):
                assert torch.isfinite(value).all(), name
            else:
                assert same(value, parent['model'][name]), name
            if name.startswith('autoencoder.log_variance.'):
                assert same(value, parent['model'][name])
        if step == 0:
            assert same({n:v for n,v in cp['model'].items() if not n.startswith('autoencoder.terminal_ffn.')}, parent['model'])
            assert same({k:v for k,v in opt['state'].items()}, parent['optimizer']['state'])
            assert same(cp['rng'], parent['rng'])
            assert torch.count_nonzero(cp['model']['autoencoder.terminal_ffn.out_proj.weight']) == 0
            assert torch.count_nonzero(cp['model']['autoencoder.terminal_ffn.out_proj.bias']) == 0
        assert ev['new_step'] == step and ev['completed_updates'] == 2500+step
        assert [r['uid'] for r in ev['meshes']] == cfg['uids']
        records = []
        for row in ev['meshes']:
            p = out / row['prediction_path']
            assert sha(p) == row['prediction_sha256'] and row['face']['complete']
            records.append(recount(p, out / 'pools' / (row['uid']+'.npz'), row))
            if step == 0:
                sr = next(r for r in sev['meshes'] if r['uid'] == row['uid'])
                with np.load(p) as a, np.load(control / sr['prediction_path']) as b:
                    assert a.files == b.files and all(a[k].dtype == b[k].dtype and a[k].tobytes() == b[k].tobytes() for k in a.files)
        for name, subset, totals in [('all', ev['meshes'], ev['counts'])] + [(name, [r for r in ev['meshes'] if lo <= r['vertices'] <= hi], ev['size_groups'][name]['counts']) for name, lo, hi in [('8_vertices',8,8), ('12_to_16',12,16), ('66_to_274',66,274)]]:
            for kind in ['edge', 'face']:
                counts = {k:sum(r[kind][k] for r in subset) for k in ['tp','fp','fn','tn']}
                assert all(totals[kind][k] == v for k,v in counts.items())
                assert totals[kind]['micro_f1'] == 2*counts['tp']/max(2*counts['tp']+counts['fp']+counts['fn'],1)
            if name != 'all':
                assert ev['size_groups'][name]['joint_perfect'] == sum(r['joint_perfect'] for r in subset)
        perfect = [r['uid'] for r in records if r['joint_perfect']]
        assert perfect == ev['perfect_uids'] and len(perfect) == ev['joint_perfect']
        parent_success = set(read(source.parent / 'eval-new0500.json')['perfect_uids'])
        assert ev['retained'] == sorted(set(perfect) & parent_success)
        assert ev['lost'] == sorted(parent_success-set(perfect)) and ev['new'] == sorted(set(perfect)-parent_success)
        for key in ['face_fn_missing','face_fn_present','face_fp_inside_pool','face_fp_outside_pool']:
            record_key = {'face_fp_inside_pool':'face_fp_in_pool','face_fp_outside_pool':'face_fp_out_pool'}.get(key,key)
            assert ev[key] == sum(r[record_key] for r in records)
        per_mesh += [dict(step=step, **r) for r in records]
        trajectory.append(dict(step=step, control=dict(counts=sev['counts'],joint=sev['joint_perfect'],perfect_uids=sev['perfect_uids'],large=sev['size_groups']['66_to_274']),
            treatment=dict(counts=ev['counts'],joint=ev['joint_perfect'],perfect_uids=perfect,large=ev['size_groups']['66_to_274'],
                retained=ev['retained'],lost=ev['lost'],new=ev['new'],**{k:ev[k] for k in ['face_fn_missing','face_fn_present','face_fp_inside_pool','face_fp_outside_pool']})))
        checkpoints.append(dict(step=step,path=str(path),sha256=ev['checkpoint_sha256'],bytes=path.stat().st_size))
        manifest.append(dict(role='terminal FFN resumable',**checkpoints[-1],model_complete=True,
            optimizer_present=True,rng_present=True,completed_updates=2500+step,
            adam_steps={**{k:2500+step for k in OLD_GROUPS},'terminal_ffn':step},
            terminal_ffn_hook_required=True,load_entrypoint=str(out/'terminal_ffn.py')+'::load_extended_state',
            hash_evidence='fresh in final audit'))
        print('PASS H checkpoint and all50 prediction recount', step, flush=True)
        if step == 100:
            full = Path(done['full_model'])
            assert sha(full) == done['full_model_sha256']
            inference = torch.load(full, map_location='cpu', mmap=True, weights_only=False)
            assert same(inference['model'], cp['model']) and same(inference['config'], cfg)
            assert inference['completed_updates'] == 2600
            assert 'optimizer' not in inference and 'rng' not in inference
            manifest.append(dict(role='terminal FFN inference',path=str(full),bytes=full.stat().st_size,
                sha256=done['full_model_sha256'],model_complete=True,optimizer_present=False,rng_present=False,
                completed_updates=2600,terminal_ffn_hook_required=True,
                load_entrypoint=str(out/'terminal_ffn.py')+'::load_extended_state',
                hash_evidence='fresh in final audit; model equals final resumable'))
            del inference
        del cp
    for metric in ['face_f1','strict']:
        b = read(out / f'best_{metric}.json')
        scores = [(x['step'],x['treatment']['counts']['face']['micro_f1'] if metric=='face_f1' else x['treatment']['joint']) for x in trajectory]
        value = max(v for s,v in scores)
        assert b['value'] == value and b['new_step'] == next(s for s,v in scores if v==value)
        assert b['sha256'] == next(x['sha256'] for x in checkpoints if x['step']==b['new_step'])
    assert len(per_mesh) == 250 and not torch.cuda.is_initialized()
    result.update(passed=True,optimizer_updates=0,model_forwards=0,gpu_used=False,source_sha256=reuse['source_sha256'],
        original_control_unmodified=True,all100_updates_verified=True,complete_triangle_enumerations=True,
        checkpoint_identity_checks=checkpoints,predictions_hashed_and_recounted=250,inference_matches_final_resumable=True,
        old_final_adam_steps=2600,new_ffn_final_adam_steps=100,per_mesh=per_mesh,trajectory=trajectory,
        startup_gate_sha256=sha(out/'startup_gate.json'),ready_sha256=sha(out/'ready.json'),
        final_config_sha256=sha(out/'config.json'),updates_sha256=sha(out/'updates.jsonl'),code_sha256=cfg['code_sha256'],
        control_audit_sha256=sha(root/'repro_outputs/CONTROL_REUSE_AUDIT.json'))
    assert len(manifest)==13
    (root/'repro_outputs/CHECKPOINT_MANIFEST.json').write_text(json.dumps(dict(
        passed=True,reviewed_by='GPT-6 Astra / xhigh',artifacts=manifest,
        note='Original model files remain on shared server. H requires terminal FFN module and prehook reinstallation before strict load.'),indent=2)+'\n')


def report(result):
    lines = ['# Terminal FFN 最终独立 CPU 审计','',
        '审阅者：GPT-6 Astra / xhigh。已完成保存状态与预测审计；没有模型 forward、GPU 初始化或优化器更新。',
        '','父 B2500、旧328参数 Adam 继承、新增 FFN fresh Adam、固定 Soft4/数据/pool/预算均核对。100 条更新日志、5 个 checkpoint、250 份预测哈希与完整 predicted-Edge triangle 枚举重数通过；最终旧组 Adam=2600，新 FFN=100，推理模型与最终可续训权重一致。',
        '','| 新增步 | S Face F1 | H Face F1 | S strict | H strict | S Edge FP/FN | H Edge FP/FN |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for x in result['trajectory']:
        a,b=x['control'],x['treatment']
        lines.append(f"| {x['step']} | {a['counts']['face']['micro_f1']:.10f} | {b['counts']['face']['micro_f1']:.10f} | {a['joint']} | {b['joint']} | {a['counts']['edge']['fp']}/{a['counts']['edge']['fn']} | {b['counts']['edge']['fp']}/{b['counts']['edge']['fn']} |")
    a,b=result['trajectory'][-1]['control'],result['trajectory'][-1]['treatment']
    lines += ['',f"末尾 H Face FP/FN={b['counts']['face']['fp']}/{b['counts']['face']['fn']}；FN 中缺 Edge 候选={b['face_fn_missing']}、候选内判负={b['face_fn_present']}；Face FP 在训练 pool 内/外={b['face_fp_inside_pool']}/{b['face_fp_outside_pool']}。",
        f"末尾较大16条（66–274点）S/H Face F1={a['large']['counts']['face']['micro_f1']:.10f}/{b['large']['counts']['face']['micro_f1']:.10f}，strict={a['large']['joint_perfect']}/{b['large']['joint_perfect']}。",
        f"H 对父 strict32：保留{len(b['retained'])}，丢失{len(b['lost'])}，新增{len(b['new'])}；丢失UID={b['lost']}；新增UID={b['new']}。",'',
        '这是同一父状态与100次更新预算下的末端非线性分支对照。新增参数、fresh Adam 与全局裁剪的相应变化均属于干预；不能把单次结果外推为从零训练的上限，或认定旧512全100失败只有一个根因。控制S复用合格旧结果，本轮新训练只有H；实际Face只在0/25/50/75/100完整验收。',
        '','本审计未替代独立冷加载前向；其结果另见 COLD_VERIFY.json。逐UID计数和SHA见 TREATMENT_FINAL_AUDIT.json。']
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment-root',type=Path,required=True)
    args=parser.parse_args()
    dest=args.experiment_root/'repro_outputs'
    result=dict(passed=False,reviewed_by='GPT-6 Astra / xhigh',timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),script_sha256=sha(__file__))
    try:
        audit(args,result)
    except BaseException:
        result['error']=traceback.format_exc()
        raise
    finally:
        (dest/'TREATMENT_FINAL_AUDIT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (dest/'TREATMENT_FINAL_REVIEW.md').write_text(report(result))
    print('TREATMENT_FINAL_AUDIT_PASS',flush=True)
