"""Read completed branch evidence; compare actual structures, never rank different pools' losses."""
import csv
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def fp_candidates(directory, uid, n):
    folder = directory/uid/'face_shards'
    state = read(folder/'progress.json')
    assert state['complete']
    fps, scores = set(), {}
    for index in range(state['shards']):
        with np.load(folder/f'part-{index:08d}.npz',allow_pickle=False) as z:
            ids = z['ids'].astype(np.int64)
            kk = (ids[:,0]*n+ids[:,1])*n+ids[:,2]
            for k,logit in zip(kk[~z['labels']],z['logits'][~z['labels']]):
                scores[int(k)] = float(logit)
                if logit > 0: fps.add(int(k))
    assert len(fps)==state['fp']
    return fps,scores


def main():
    branches = ['A_uniform','B_hard']
    configurations = {b:read(ROOT/f'{b}.json') for b in branches}
    parent = read(Path(configurations['A_uniform']['parent_baseline']))
    parent_success = set(parent['perfect_uids'])
    pool = read(ROOT/'hard_pool.json')
    summaries, details, repairs = {}, [], []
    logs = {}
    restored = {b:read(Path(configurations[b]['run_directory'])/'restore-00034720.json') for b in branches}
    for key in ('model_sha256','optimizer_sha256','rng_sha256','noise_rng_sha256','optimizer_steps','optimizer_group_parameter_names'):
        assert restored['A_uniform'][key] == restored['B_hard'][key], ('parent restore mismatch', key)
    for branch in branches:
        run = Path(configurations[branch]['run_directory'])
        done = read(run/'complete.json')
        assert done['completed_updates']==35220 and done['new_updates']==500 and done['evaluations_complete']
        logs[branch] = [json.loads(x) for x in (run/'updates.jsonl').read_text().splitlines()]
        assert [r['update'] for r in logs[branch]]==list(range(34721,35221))
        trajectory=[]
        for step in (34720,34820,34970,35220):
            groups = ['mu']+([f'noise-{s}' for s in range(861001,861006)] if step in (34720,35220) else [])
            for group in groups:
                directory=run/f'evaluations/update-{step:08d}'/group
                evaluation=read(directory/'complete.json')
                assert evaluation['complete'] and len(evaluation['meshes'])==100
                success=set(evaluation['perfect_uids'])
                row=dict(branch=branch,step=step,group=group,edge=evaluation['edge'],face=evaluation['face'],
                    joint_strict=len(success),edge_strict=len(evaluation['edge_perfect_uids']),
                    retained_parent28=sorted(success & parent_success),lost_parent28=sorted(parent_success-success),
                    new_vs_parent_mu28=sorted(success-parent_success),checkpoint=evaluation['checkpoint'])
                trajectory.append(row)
                for m in evaluation['meshes']:
                    details.append(dict(branch=branch,step=step,group=group,uid=m['uid'],vertices=m['vertices'],
                        edge_fp=m['edge']['fp'],edge_fn=m['edge']['fn'],face_fp=m['face']['fp'],face_fn=m['face']['fn'],
                        face_fn_missing=m['face_fn_missing'],face_fn_present=m['face_fn_present'],
                        joint_strict=m['uid'] in success,parent_mu_strict=m['uid'] in parent_success))
                if group=='mu' and step==35220:
                    for m in evaluation['meshes']:
                        uid,n=m['uid'],m['vertices']
                        ids=np.asarray(pool['meshes'][uid]['ids'],dtype=np.int64).reshape(-1,3)
                        old=set(map(int,(ids[:,0]*n+ids[:,1])*n+ids[:,2]))
                        fps,scores=fp_candidates(directory,uid,n)
                        repairs.append(dict(branch=branch,uid=uid,parent_fp=len(old),remaining_parent_fp=len(old & fps),
                            repaired_still_candidate=sum(k in scores and scores[k]<=0 for k in old),
                            absent_from_end_candidates=sum(k not in scores for k in old),new_fp=len(fps-old),end_fp=len(fps)))
        summaries[branch]=dict(final=next(x for x in trajectory if x['step']==35220 and x['group']=='mu'),trajectory=trajectory)
        summaries[branch]['parent_fp_outcomes']={k:sum(r[k] for r in repairs if r['branch']==branch) for k in
            ('parent_fp','remaining_parent_fp','repaired_still_candidate','absent_from_end_candidates','new_fp','end_fp')}
    for a,b in zip(logs['A_uniform'],logs['B_hard']):
        assert a['update']==b['update'] and a['uids']==b['uids']
        for ma,mb in zip(a['meshes'],b['meshes']):
            assert ma['uid']==mb['uid']
            assert ma['posterior']['epsilon_sha256']==mb['posterior']['epsilon_sha256']
            assert ma['uniform_negative_sha256']==mb['uniform_negative_sha256']
            assert ma['face_negatives']==mb['face_negatives']
    result=dict(complete=True,parent_state_hashes_equal=True,paired_updates_verified=500,paired_mesh_noise_hashes_verified=2500,
        parent_joint_strict=28,source_checkpoint_sha256=configurations['A_uniform']['parent_sha256'],branches=summaries,
        limitation='Different Face pools: compare actual reconstructions; training scalar loss is not a common ranking metric.')
    (ROOT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,rows in [('per_mesh_evaluations.csv',details),('parent_fp_outcomes.csv',repairs)]:
        with (ROOT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    a,b=[summaries[x]['final'] for x in branches]
    report=f'''# 固定100条 VAE：负例来源配对结果

两支各完成500次有效更新，均从34720到35220。末尾μ实际Face FP：A={a['face']['fp']}，B={b['face']['fp']}；FN：A={a['face']['fn']}，B={b['face']['fn']}。联合严格成功：A={a['joint_strict']}/100，B={b['joint_strict']}/100。原28条分别保留{len(a['retained_parent28'])}、{len(b['retained_parent28'])}条。

| 分支 | Edge FP/FN | 实际Face FP/FN | Face micro-F1 | 联合严格成功 | 原28条丢失/新增 |
|---|---|---|---|---|---|
| A均匀负例 | {a['edge']['fp']}/{a['edge']['fn']} | {a['face']['fp']}/{a['face']['fn']} | {a['face']['micro_f1']:.9f} | {a['joint_strict']} | {len(a['lost_parent28'])}/{len(a['new_vs_parent_mu28'])} |
| B来源替换 | {b['edge']['fp']}/{b['edge']['fn']} | {b['face']['fp']}/{b['face']['fn']} | {b['face']['micro_f1']:.9f} | {b['joint_strict']} | {len(b['lost_parent28'])}/{len(b['new_vs_parent_mu28'])} |

500次样本顺序、2500份ε哈希与均匀候选基础哈希逐条匹配，负例数量相同。两支独立继承父AdamW及RNG，没有重置logvar或warmup。全部实际Face由预测Edge图完整枚举。

父1509个FP去向见parent_fp_outcomes.csv：分别记录仍FP、仍在候选但已判负、不再进入候选，以及新增FP；不能把退出候选算作Face分类修复。完整起末五组噪声、μ中间点、原成功UID保留/丢失/新增见comparison.json。

本结果只适用于该起点、预算及替换规则，不证明唯一根因或保证100条全对。两支训练pool不同，不以训练loss排名。原模型与源代码保留，预算结束不延长。

完整权重、Adam/RNG及预测数组仍在各分支服务器目录。evaluation.zip包含配置、有效代码、日志、逐mesh指标、Face预测分片和文件指针，不重复包含大checkpoint或network-output.npz。
'''
    (ROOT/'REPORT.md').write_text(report)
    paths=[p for p in ROOT.rglob('*') if p.is_file() and p.suffix not in ('.pt','.npz','.zip','.pyc') and '__pycache__' not in p.parts]
    paths += list(ROOT.glob('*/evaluations/**/face_shards/part-*.npz'))
    with zipfile.ZipFile(ROOT/'evaluation.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(set(paths)):z.write(p,p.relative_to(ROOT))
    print(report,flush=True)


if __name__=='__main__':main()
