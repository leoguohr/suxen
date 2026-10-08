"""Package fixed features, one-head checkpoints, effective scoring, and all evaluations."""
import csv
import hashlib
import json
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

cfg=json.loads((ROOT/'config.json').read_text())
result=json.loads((ROOT/'run/complete.json').read_text())
verified=json.loads((ROOT/'run/verification.json').read_text())
assert verified['one_head_all_meshes'] and verified['contiguous_updates']==cfg['updates']
rows=[json.loads(s) for s in (ROOT/'run/updates.jsonl').read_text().splitlines()]
with (ROOT/'per_mesh_trace.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['step','uid','tp','fp','fn','tn','f1','edge_soft4','min_margin_gt','min_margin_non_gt','joint_perfect'])
    for r in rows:
        for m in r['meshes']:w.writerow([r['step'],m['uid'],*[m[k] for k in ['tp','fp','fn','tn','f1','edge_soft4','min_margin_gt','min_margin_non_gt']],r['joint_perfect']])
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,3,figsize=(14,4))
for i,uid in enumerate(cfg['uids']):
    for ax,key in zip(axs,['fp','fn','edge_soft4']):ax.plot([r['step'] for r in rows],[r['meshes'][i][key] for r in rows],label=uid[-6:])
for ax,title in zip(axs,['All-pair Edge FP','All-pair Edge FN','Fully differentiable Edge Soft4']):
    ax.set_xlabel('Shared head updates');ax.set_title(title);ax.set_yscale('symlog',linthresh=1e-3 if ax==axs[2] else 1);ax.grid(alpha=.25)
axs[0].legend();fig.tight_layout();fig.savefig(ROOT/'curves.png',dpi=180);fig.savefig(ROOT/'curves.pdf');plt.close(fig)
report=['# 三条固定大mesh：单一共享Edge head诊断','',
    '仅优化同一份32×1024权重与32维bias。三条hidden均固定，来自同一epoch900源checkpoint的Decoder末端LayerNorm输出。没有回归指定成功表示，没有每mesh独立权重，没有Face或KL。',
    '',f"每步三条共同参与，目标为 mean_mesh(Edge Soft4)/4；fresh Adam LR={cfg['lr']}，clip=1，wd=0，预算{cfg['updates']}步。学习率是本轮预定候选设置，不代表最优。",'',
    '| UID | 初始FP/FN | 最终FP/FN | 最终Edge Soft4 |', '|---|---:|---:|---:|']
for a,b in zip(result['baseline']['meshes'],result['final']['meshes']):report.append(f"| {a['uid']} | {a['fp']}/{a['fn']} | {b['fp']}/{b['fn']} | {b['edge_soft4']:.8g} |")
report+=['',f"首次三条同时严格成功：{result['first_joint_perfect_step']}；总共{result['joint_perfect_updates']}次；最长连续{result['longest_joint_perfect']}步；最后500步中{result['last500_joint_perfect']}次。",'',
    '成功仅证明当前固定hidden对这三条存在本次训练找到的共同Edge读出；不代表Face或100条联合任务通过。有限预算失败也不能证明特征或容量不足。',
    '', '## 材料', '',
    '- `snapshots/head_initial.npz`：唯一原始共享head权重与bias。',
    '- `snapshots/<uid>/hidden.npy`：每条固定Decoder hidden；配套全部pair标签、原logits、原head输出、GT结构与有效评分源码。',
    '- `run/updates.jsonl`：逐步三条同一head计数、margin、loss、参数实际更新、梯度、clip和权重范数。',
    '- `run/checkpoint-*.pt`：每个检查点仅一份head和Adam；配套三条NPZ保存实际读出与全部pair logits。',
    '- `run/baseline_verification.json`、`run/verification.json`：基线复现与保存后重载全量验收。',
    '- `export_complete.json`：源网络未更新；未向原网络checkpoint写入新head。',
    '', '不含上游大网络checkpoint或100条完整训练数据。源checkpoint：', '`'+cfg['source_checkpoint']+'`',
    'SHA256: `'+cfg['source_sha256']+'`', '',
    '原网络重载并安装新head的完整推理未纳入本诊断；这里核验固定hidden上的单一共享读出。']
(ROOT/'REPORT.md').write_text('\n'.join(report)+'\n')
files=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.py','.json','.md','.csv','.png','.pdf','.log'] and p.name not in ['package_verification.json','status.json']]
files += [ROOT/'snapshots/head_initial.npz']
for uid in cfg['uids']:
    p=ROOT/'snapshots'/uid
    files += [q for q in p.iterdir() if q.is_file()]
    files += [q for q in (p/'effective_code').rglob('*') if q.is_file() and '__pycache__' not in q.parts]
files += [p for p in (ROOT/'run').rglob('*') if p.is_file()]
files=sorted(set(files));hashes={str(p.relative_to(ROOT)):sha(p) for p in files}
(ROOT/'SHA256SUMS.txt').write_text(''.join(f'{v}  {k}\n' for k,v in hashes.items()))
dest=ROOT/'Nexus_SharedEdgeHead_Large3_2000Updates_FullEvaluation_20260917.zip'
with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in files+[ROOT/'SHA256SUMS.txt']:z.write(p,str(p.relative_to(ROOT)))
with zipfile.ZipFile(dest) as z:
    assert z.testzip() is None
    for n,h in hashes.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
(ROOT/'package_verification.json').write_text(json.dumps(dict(path=str(dest),sha256=sha(dest),bytes=dest.stat().st_size,files_verified=len(hashes)),indent=2)+'\n')
print((ROOT/'package_verification.json').read_text(),flush=True)
