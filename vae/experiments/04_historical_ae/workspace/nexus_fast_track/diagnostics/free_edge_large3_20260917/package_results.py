"""Check complete traces and package code, exports, representations, and evaluations."""
import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
UIDS=['nexus_2k_000446','nexus_2k_001093','nexus_2k_000898']

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()

def main():
    records=[];traces=[]
    for uid in UIDS:
        p=ROOT/'runs'/uid
        complete=json.loads((p/'complete.json').read_text())
        history=list(map(json.loads,(p/'updates.jsonl').read_text().splitlines()))
        assert [x['step'] for x in history]==list(range(2001))
        assert all(math.isfinite(x['objective']) for x in history)
        assert all(x['raw_changed_elements']>0 for x in history[1:])
        perfect=[x['step'] for x in history[1:] if x['fp']==x['fn']==0]
        assert len(perfect)==complete['perfect_updates']
        assert (perfect[0] if perfect else None)==complete['first_perfect_step']
        assert history[-1]==complete['final']
        verified=json.loads((p/'saved_checkpoint_verification.json').read_text())
        assert verified['checkpoint-step2000']['saved_logits_reproduced_exactly']
        summary=json.loads((ROOT/'snapshots'/uid/'summary.json').read_text())
        records.append(dict(uid=uid,vertices=summary['vertices'],gt_edges=summary['gt_edges'],pairs=summary['pairs'],
            initial_fp=history[0]['fp'],initial_fn=history[0]['fn'],final_fp=history[-1]['fp'],final_fn=history[-1]['fn'],
            first_perfect_step=complete['first_perfect_step'],perfect_updates=complete['perfect_updates'],
            longest_consecutive_perfect=complete['longest_consecutive_perfect'],last500_perfect=complete['last_500']['perfect'],
            final_soft4=history[-1]['edge_soft4'],minimum_gt_margin=history[-1]['min_margin_gt'],
            minimum_non_gt_margin=history[-1]['min_margin_non_gt'],wall_seconds=complete['wall_seconds']))
        traces.append(history)
    with (ROOT/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    (ROOT/'comparison.json').write_text(json.dumps(records,indent=2)+'\n')
    (ROOT/'result_verification.json').write_text(json.dumps(dict(uids=UIDS,updates_each=2000,
        contiguous_logs=True,all_updates_nonzero=True,final_reloaded_logits_exact=True,network_updates=0),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for r,h in zip(records,traces):
        steps=[x['step'] for x in h];label=f"{r['uid'][-6:]} ({r['vertices']} vertices)"
        axes[0].plot(steps,[x['fp'] for x in h],label=label)
        axes[1].plot(steps,[x['fn'] for x in h],label=label)
        axes[2].plot(steps,[x['edge_soft4'] for x in h],label=label)
    for ax,title in zip(axes,['Edge FP (all pairs)','Edge FN (all pairs)','Fully differentiable Edge Soft4']):
        ax.set_title(title);ax.set_xlabel('Independent embedding updates');ax.set_yscale('symlog',linthresh=.01 if ax==axes[2] else 1);ax.grid(alpha=.25)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(ROOT/'curves.png',dpi=180);fig.savefig(ROOT/'curves.pdf');plt.close(fig)
    exported=json.loads((ROOT/'export_complete.json').read_text())
    report=['# 三条固定大mesh：自由32维Edge表示诊断','',
        '仅学习每条mesh自己的每顶点32维表示，不更新共享网络。保留原16+16评分、scale、逐mesh中心化、全部无向pair、fully-diff Soft4，目标为 Edge Soft4 / 4；fresh Adam LR=1e-3，clip=1，wd=0，每条独立2000步。',
        '', '| UID | 顶点 | 起点FP/FN | 末尾FP/FN | 首次全对step | 全对更新数 | 最长连续 | 末500全对 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        report.append(f"| {r['uid']} | {r['vertices']} | {r['initial_fp']}/{r['initial_fn']} | {r['final_fp']}/{r['final_fn']} | {r['first_perfect_step']} | {r['perfect_updates']}/2000 | {r['longest_consecutive_perfect']} | {r['last500_perfect']}/500 |")
    report+=['','成功只能证明这些独立32维表示在当前评分下存在并可被本次优化找到；不证明共享网络能够输出它们，不包含Face，也不代表100条联合重建通过。有限预算失败也不是维度不足的证明。',
        '', '## 材料索引', '',
        '- `snapshots/<uid>/`：网络原始head输出、中心化表示、原始顶点/GT面/边、全部pair标签/logits、直接表示梯度、当前有效评分与loss源码、父状态逐mesh评价。',
        '- `runs/<uid>/updates.jsonl`：step0及2000次更新，完整TP/FP/FN/TN、F1、loss、margin、梯度、clip、实际位移。',
        '- `runs/<uid>/checkpoint-*.pt/.npz`：固定检查点的表示、Adam及全部pair logits；若成功包含first-perfect。',
        '- `runs/<uid>/baseline_verification.json`、`saved_checkpoint_verification.json`：网络→导出基线复现及首个成功/最终保存结果重载核验。',
        '- `comparison.csv/json`、`curves.png/pdf`：三条比较。',
        '- `export_complete.json`：原网络参数/缓冲/元数据与源checkpoint不变，零网络更新。',
        '', '## 大文件与边界', '',
        '包内包括三条原始mesh和表示探针checkpoint；不包含上游大网络checkpoint、100条完整数据、其他两条100-mesh分支checkpoint。源网络仍在服务器：',
        '`'+exported['records'][0]['checkpoint']+'`',
        'SHA256: `'+exported['records'][0]['checkpoint_sha256']+'`',
        '', 'Control与第二轮难负例网络均未被覆盖；三条探针是独立变量，不可拼接成共享模型。',
        '', '## 复现', '',
        '`python run_probe.py --snapshot snapshots/<uid> --output <new-output>`，然后 `python verify_saved.py --snapshot snapshots/<uid> --run <new-output>`。需CUDA/PyTorch，具体版本见各manifest。',
        '', '导出阶段曾因state_dict含非Tensor元数据使哈希检查报错；修正检查器后重试。该次错误发生在导出及任何优化前，原日志保留。']
    (ROOT/'REPORT.md').write_text('\n'.join(report)+'\n')
    files=[]
    for name in ['runtime.py','construction_args.json','export.py','run_probe.py','verify_saved.py','run_job.py','package_results.py',
                 'graph_backward.py','sampling_forward.py','launch_verification.json',
                 'export_complete.json','export_console.log','export_preupdate_hashcheck_failure.log','comparison.csv','comparison.json',
                 'result_verification.json','REPORT.md','curves.png','curves.pdf','loss_changes_exact_runtime.txt']:
        if (ROOT/name).is_file():files.append(ROOT/name)
    for folder in ['snapshots','runs']:
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files=sorted(files)
    hashes={str(p.relative_to(ROOT)):digest(p) for p in files}
    (ROOT/'SHA256SUMS.txt').write_text(''.join(f'{v}  {k}\n' for k,v in hashes.items()))
    dest=ROOT/'Nexus_FreeEdge_Large3_2000Updates_FullEvaluation_20260917.zip'
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[ROOT/'SHA256SUMS.txt']:z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        for name,h in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==h,name
    (ROOT/'package_verification.json').write_text(json.dumps(dict(zip=str(dest),sha256=digest(dest),
        bytes=dest.stat().st_size,files_verified=len(hashes),zip_crc_verified=True),indent=2)+'\n')
    print((ROOT/'package_verification.json').read_text(),flush=True)

if __name__=='__main__':main()
