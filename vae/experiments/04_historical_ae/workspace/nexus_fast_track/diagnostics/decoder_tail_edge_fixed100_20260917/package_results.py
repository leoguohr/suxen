"""Package the completed bounded tail experiment and paired step500 Control."""
import csv,hashlib,json,shutil,zipfile
from pathlib import Path
R=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

cfg=read(R/'config.json');meta=read(R/'cache/manifest.json');run=R/'run'
complete=read(run/'complete.json');verification=read(run/'verification.json')
assert complete['completed_updates']==500 and complete['stopped_at_budget']
trace=[json.loads(x) for x in (run/'evaluations.jsonl').read_text().splitlines()]
updates=[json.loads(x) for x in (run/'updates.jsonl').read_text().splitlines()]
assert [r['step'] for r in trace]==list(range(501))
assert [r['update'] for r in updates]==list(range(1,501))
uids=[r['uid'] for r in meta['meshes']]
assert len(set(uids))==100
for r in trace:
    assert [m['uid'] for m in r['meshes']]==uids
    assert r['total_fp']==sum(m['fp'] for m in r['meshes']) and r['total_fn']==sum(m['fn'] for m in r['meshes'])
    assert r['edge_perfect']==sum(m['fp']==m['fn']==0 for m in r['meshes'])
for u in updates:
    assert u['mesh_count']==100 and u['lrs']=={'decoder_tail':1e-5,'edge_head':1e-4}
    assert all(x['delta_norm']>0 for x in u['actual_updates'].values())
control=read(R/'control-checkpoint-step0500.json');initial=trace[0];final=trace[-1]
old_by={r['uid']:r for r in control['meshes']};initial_set={m['uid'] for m in initial['meshes'] if m['perfect']};final_set={m['uid'] for m in final['meshes'] if m['perfect']}
rows=[]
for i,uid in enumerate(uids):
    rows.append(dict(uid=uid,vertices=meta['meshes'][i]['vertices'],baseline=initial['meshes'][i],head_only_step500=old_by[uid],tail_step500=final['meshes'][i]))
write(R/'per_mesh_comparison.json',rows)
with (R/'per_mesh_comparison.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['uid','vertices','baseline_fp','baseline_fn','control_fp','control_fn','tail_fp','tail_fn','control_strict','tail_strict'])
    for r in rows:w.writerow([r['uid'],r['vertices'],r['baseline']['fp'],r['baseline']['fn'],r['head_only_step500']['fp'],r['head_only_step500']['fn'],r['tail_step500']['fp'],r['tail_step500']['fn'],r['head_only_step500']['perfect'],r['tail_step500']['perfect']])
with (R/'per_mesh_trace.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['step','uid','tp','fp','fn','tn','strict','edge_soft4','min_gt_margin','min_negative_margin'])
    for r in trace:
        for m in r['meshes']:w.writerow([r['step'],m['uid'],*[m[k] for k in ['tp','fp','fn','tn','perfect','edge_soft4','min_margin_gt','min_margin_non_gt']]])
groups={}
for name,lo,hi in [('N<=500',0,500),('500<N<=1500',500,1500),('N>1500',1500,100000)]:
    group=[r for r in rows if lo<r['vertices']<=hi]
    groups[name]={'meshes':len(group)}
    for key in ['baseline','head_only_step500','tail_step500']:
        groups[name][key]=dict(fp=sum(r[key]['fp'] for r in group),fn=sum(r[key]['fn'] for r in group),strict=sum(r[key]['perfect'] for r in group))
summarize=lambda r:{k:r[k] for k in ['edge_perfect','total_fp','total_fn','objective']}
result=dict(initial=summarize(initial),control500=summarize(control),tail500=summarize(final),groups=groups,
    retained=sorted(initial_set&final_set),lost=sorted(initial_set-final_set),gained=sorted(final_set-initial_set),
    clip_steps=sum(u['clip_coefficient']<1 for u in updates),face_evaluated=False,
    strict_counts_last100=[r['edge_perfect'] for r in trace[-100:]],
    every_mesh_update_participations=500,source_sha256=cfg['source_sha256'],updates_verified=500)
write(R/'summary.json',result)
# Preserve the exact runtime dependency closure used by the source export.
previous=Path(cfg['control'])
shutil.copytree(previous/'runtime_dependencies',R/'runtime_dependencies',dirs_exist_ok=True)
shutil.copyfile(previous/'runtime_sources.json',R/'runtime_sources.json')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ctrl=[json.loads(x) for x in (previous/'run/evaluations.jsonl').read_text().splitlines()][:501]
fig,axs=plt.subplots(1,3,figsize=(14,4))
for name,seq in [('Head only',ctrl),('Last block + final LN + head',trace)]:
    xs=[x['step'] for x in seq]
    for ax,key in zip(axs,['edge_perfect','total_fp','objective']):ax.plot(xs,[x[key] for x in seq],label=name)
for ax,ylabel in zip(axs,['Strict Edge meshes / 100','All-pair FP','Mean100 Edge Soft4']):ax.set(xlabel='Updates (all100 each)',ylabel=ylabel);ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.tight_layout();fig.savefig(R/'comparison.png',dpi=170);fig.savefig(R/'comparison.pdf');plt.close(fig)
bench=read(R/'benchmark.json')
lines=['# 固定100条：Decoder末块＋最终LayerNorm＋共享Edge head','',
    '从原第二轮难负例epoch900/update22500及其原始head开始。只训练原decoder_blocks.15、decoder_output_norm和edge_embedding；不是从诊断head的step2000继续。',
    '全部100条输入缓存于最后一个Decoder block的LayerNorm之前。完整网络与缓存路径的hidden、中心化表示、loss、全部可训练参数梯度逐条bitwise复现；预检两轮完整100条forward/backward重复一致。',
    '目标mean100(fully-diff Edge Soft4)，全部84,669,234 pairs，原32维16+16评分和scale；每mesh反向/100，100条累计后统一clip=1并更新一次。fresh Adam，末块+LN LR=1e-5，head LR=1e-4；wd=0、μ路径、KL=0。500步后停止。',
    f"实测完整100条评分+反向耗时 {bench['full100_backward_seconds']} 秒；峰值allocated {bench['peak_allocated']/2**30:.3f} GiB、reserved {bench['peak_reserved']/2**30:.3f} GiB。训练与末尾核验/导出合计 {complete['seconds']:.1f}秒。",'',
    '|状态|Edge严格成功|FP|FN|mean Edge Soft4|','|---|---:|---:|---:|---:|']
for name,r in [('原始基线',initial),('Head-only step500',control),('末块+LN+head step500',final)]:lines.append(f"|{name}|{r['edge_perfect']}/100|{r['total_fp']}|{r['total_fn']}|{r['objective']:.9g}|")
lines+=['',f"原50条成功保留{len(result['retained'])}，丢失{len(result['lost'])}，新增{len(result['gained'])}。逐UID清单和>1500组结果在summary.json及per_mesh_comparison.csv。",'',
    '**本次仅Edge诊断。未进行实际Face验收，未将新末块覆盖原源网络。Face head虽冻结，末块适应会改变它的输入；不能据此宣称完整Edge+Face通过。有限预算未全对也不是表示不可能的证明。**','',
    '材料索引：config/manifest/freeze_contract锁定来源与训练范围；export_complete与cache/manifest记录真实网络对齐；benchmark记录成本和重复性；run/updates为500条实际更新；run/evaluations为501个同一模型状态；comparison提供已有Control step500逐mesh配对；checkpoint包含tail和Adam；最终logits按i<j的triu_indices顺序，GT来自cache的edges。',
    '启动预检问题（deterministic安装顺序、state_dict非Tensor元数据、缓存模板中的诊断hook序列化，以及保持attention checkpoint的grad/context约定）均在第一个optimizer update前修复，失败日志保留，没有改变模型公式、loss、LR或预算。',
    'Review包包含完整日志、逐mesh结果、实际代码、核验记录、初始/最终末块与head及最终Adam；大型缓存、全量最终logits、中间checkpoint的路径/哈希见EXCLUDED_FILES.json。FullEvaluation包另含上述缓存、logits与中间checkpoint。源大网络和Face pool不入包；源路径与SHA在config.json。']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
base=[p for p in R.iterdir() if p.is_file() and p.suffix in ['.py','.json','.csv','.md','.png','.pdf','.log','.txt'] and p.name not in ['package_verification.json','EXCLUDED_FILES.json','SHA256SUMS.txt']]
for folder in ['run','cache','effective_code','runtime_dependencies','source_archive','review_runtime']:
    base.extend(p for p in (R/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
base=sorted(set(base));metadata=[]
for kind in ['Review','FullEvaluation']:
    files=base
    if kind=='Review':files=[p for p in base if p.suffix!='.npz' and not (p.name.startswith('checkpoint-step') and p.suffix=='.pt' and p.name not in ['checkpoint-step0000.pt','checkpoint-step0500.pt'])]
    excluded={str(p.relative_to(R)):dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in base if p not in files}
    excluded['source_network']=dict(path=cfg['source_checkpoint'],sha256=cfg['source_sha256'])
    out=R/f'Nexus_DecoderTail_Edge100_500Updates_{kind}_20260917.zip'
    hashes={str(p.relative_to(R)):sha(p) for p in files}
    extra=json.dumps(excluded,indent=2).encode();hashes['EXCLUDED_FILES.json']=hashlib.sha256(extra).hexdigest()
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p in files:z.write(p,str(p.relative_to(R)))
        z.writestr('EXCLUDED_FILES.json',extra);z.writestr('SHA256SUMS.txt',''.join(f'{h}  {n}\n' for n,h in hashes.items()))
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
        for n,h in hashes.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    metadata.append(dict(kind=kind,path=str(out),bytes=out.stat().st_size,sha256=sha(out),files_verified=len(hashes)))
    write(R/'package_verification.json',metadata)
    print('PACKAGED',json.dumps(metadata[-1]),flush=True)
