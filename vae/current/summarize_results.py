"""Summarize only complete, immutable-checkpoint evaluations; package evidence."""
import csv
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def hard_outcomes(mesh, pool_row):
    folder = Path(mesh['prediction_directory'])/'face_shards'
    state = read(folder/'progress.json')
    assert state['complete']
    hard = set(map(tuple, pool_row['ids']))
    present = {}
    fps = set()
    for index in range(state['shards']):
        with np.load(folder/f'part-{index:08d}.npz', allow_pickle=False) as z:
            for triple, logit, label in zip(z['ids'], z['logits'], z['labels']):
                key = tuple(map(int, triple))
                if key in hard:
                    assert not label
                    present[key] = float(logit)
                if not label and logit > 0:
                    fps.add(key)
    assert len(fps) == mesh['face']['fp']
    positive = sum(v > 0 for v in present.values())
    return dict(uid=mesh['uid'], hard_total=len(hard), hard_still_fp=positive,
        hard_present_classified_negative=len(present)-positive, hard_absent_candidates=len(hard)-len(present),
        new_fp_outside_fixed1509=len(fps-hard), actual_face_fp=len(fps),
        present_hard_logit_mean=float(np.mean(list(present.values()))) if present else None,
        present_hard_logit_max=max(present.values()) if present else None,
        present_hard_logits=[dict(ids=list(k), logit=v) for k,v in sorted(present.items())])


def main():
    config = read(ROOT/'config.json')
    run = Path(config['run_directory'])
    done = read(run/'complete.json')
    assert done['completed_updates'] == 36220 and done['new_updates'] == 1000 and done['evaluations_complete']
    logs = [json.loads(x) for x in (run/'updates.jsonl').read_text().splitlines()]
    assert [r['update'] for r in logs] == list(range(35221, 36221))
    assert all(len(r['uids']) == len(r['meshes']) == 5 for r in logs)
    counts = {}
    for row in logs:
        for mesh in row['meshes']:
            uid = mesh['uid']; counts[uid] = counts.get(uid, 0)+1
    assert len(counts) == 100 and set(counts.values()) == {50}
    pool = read(config['hard_pool_path'])
    parent = read(config['parent_baseline'])
    parent31 = set(parent['perfect_uids'])
    original28 = set(read(pool['source_baseline'])['perfect_uids'])
    trajectory, per_mesh, hard_rows = [], [], []
    for relative in config['mu_checkpoints']:
        step = 35220+relative
        groups = ['mu']+([f'noise-{s}' for s in config['evaluation_seeds']] if relative in (0, 1000) else [])
        for group in groups:
            folder = run/f'evaluations/update-{step:08d}'/group
            result = read(folder/'complete.json')
            assert result['complete'] and len(result['meshes']) == 100 and result['native_forward']
            success = set(result['perfect_uids'])
            calculated = {m['uid'] for m in result['meshes'] if all(m[t][k] == 0 for t in ('edge','face') for k in ('fp','fn'))}
            assert success == calculated and result['joint_perfect'] == len(success)
            for task in ('edge', 'face'):
                for key in ('tp', 'fp', 'fn', 'tn'):
                    assert sum(m[task][key] for m in result['meshes']) == result[task][key]
            summary = dict(step=step, new_update=relative, condition=group, edge=result['edge'], face=result['face'],
                joint_strict=len(success), strict_uids=sorted(success), face_f1_stage_pass=result['face']['micro_f1'] >= .997,
                retained_parent31=sorted(parent31 & success), lost_parent31=sorted(parent31-success),
                new_vs_parent31=sorted(success-parent31), retained_original28=sorted(original28 & success),
                lost_original28=sorted(original28-success), new_vs_original28=sorted(success-original28),
                reused_parent_evaluation=(folder/'reused_parent_evaluation.json').exists(),
                checkpoint=result['checkpoint'], posterior_mean=result['posterior_mean'])
            for mesh in result['meshes']:
                per_mesh.append(dict(step=step, condition=group, uid=mesh['uid'], vertices=mesh['vertices'],
                    edge_fp=mesh['edge']['fp'], edge_fn=mesh['edge']['fn'], face_fp=mesh['face']['fp'], face_fn=mesh['face']['fn'],
                    face_fn_missing=mesh['face_fn_missing'], face_fn_present=mesh['face_fn_present'],
                    strict_success=mesh['uid'] in success))
                if group == 'mu':
                    hard_rows.append(dict(step=step, new_update=relative, **hard_outcomes(mesh, pool['meshes'][mesh['uid']])))
            if group == 'mu':
                this_hard = [r for r in hard_rows if r['step'] == step]
                summary['fixed_hard_outcomes'] = {key: sum(r[key] for r in this_hard) for key in
                    ('hard_total','hard_still_fp','hard_present_classified_negative','hard_absent_candidates','new_fp_outside_fixed1509')}
            trajectory.append(summary)
    result = dict(complete=True, source_step=35220, final_step=36220, actual_new_updates=1000,
        verified_mesh_participations=5000, primary_target='actual Face micro-F1 >= 0.997, each condition separately',
        trajectory=trajectory, strict_success_is_separate_metric=True,
        parent_checkpoint_sha256=config['parent_sha256'], hard_pool_sha256=config['hard_pool_sha256'])
    (ROOT/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    (ROOT/'hard_negative_trajectory.json').write_text(json.dumps(hard_rows, indent=2)+'\n')
    with (ROOT/'per_mesh_evaluations.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(per_mesh[0])); writer.writeheader(); writer.writerows(per_mesh)
    final = [r for r in trajectory if r['new_update'] == 1000]
    lines = ['# B35220固定难负例VAE续训1000步结果', '',
        '已完成1000次有效更新：35220→36220，到预算停止。主判据为实际Face micro-F1≥0.997，μ和每个采样条件独立报告；100/100严格零错误不是本轮阶段通过的必要条件。', '',
        '| 末尾条件 | 实际Face micro-F1 | 达到0.997 | Face FP/FN | Edge FP/FN | 联合严格成功 |',
        '|---|---:|---|---|---|---:|']
    for r in final:
        lines.append(f"| {r['condition']} | {r['face']['micro_f1']:.9f} | {r['face_f1_stage_pass']} | {r['face']['fp']}/{r['face']['fn']} | {r['edge']['fp']}/{r['edge']['fn']} | {r['joint_strict']}/100 |")
    lines += ['', '完整μ轨迹及原31条、历史28条的保留/丢失/新增UID见results.json。固定1509项负例去向与实际候选logit见hard_negative_trajectory.json；退出Edge候选不等于Face分类修复。', '',
        '起点μ重新运行真实网络并核对；起点五组采样复用同一完整B35220父checkpoint既有的完整真实网络评价，并明确记录来源。其余评价重新执行真实Encoder→μ或sampling→Decoder，Face候选由预测Edge图完整枚举。', '',
        '训练只增加预算：保留固定1509项难负例、1.5F总负例规则、AdamW、RNG、数据游标、LR、β、sampling、Hard4、clip及全网络范围。未比较4F，不能由本结果断言4F有效或无效。', '',
        '模型大文件及network-output.npz保留服务器；包内checkpoint和network-output元数据列出路径、大小（若记录）和SHA。没有自动追加训练。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    files = [p for p in ROOT.rglob('*') if p.is_file() and p.suffix not in ('.pt','.npz','.zip','.pyc')
             and not p.name.startswith('._') and '__pycache__' not in p.parts and not p.name.endswith('.lock')]
    files += list(run.glob('evaluations/**/face_shards/part-*.npz'))
    manifest = {str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(set(files))}
    with zipfile.ZipFile(ROOT/'evaluation.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(set(files)): z.write(p, p.relative_to(ROOT))
        z.writestr('FILES_SHA256.json', json.dumps(manifest, indent=2)+'\n')
    with zipfile.ZipFile(ROOT/'evaluation.zip') as z: assert z.testzip() is None
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
