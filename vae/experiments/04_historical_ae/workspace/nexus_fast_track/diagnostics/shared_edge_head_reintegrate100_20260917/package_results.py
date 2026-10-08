"""Package inference-only reintegration evidence, per-mesh comparisons, and effective code."""
import csv,hashlib,json,shutil,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
out=ROOT/'run'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
complete=json.loads((out/'complete.json').read_text())
cfg=complete['source_config'];uids=json.loads((ROOT/'selection.json').read_text())['uids']
before={r['uid']:r for r in map(json.loads,(out/'before.jsonl').read_text().splitlines())}
after={r['uid']:r for r in map(json.loads,(out/'after.jsonl').read_text().splitlines())}
assert set(before)==set(after)==set(uids) and len(uids)==100
columns=['uid','vertices','gt_edges','gt_faces']
keys=['edge_tp','edge_fp','edge_fn','face_tp','face_fp','face_fn','missing_gt_face_candidates','edge_perfect','face_perfect','joint_perfect','edge_soft4','face_soft4']
columns += [b+'_'+k for b in ['before','after'] for k in keys]
columns += ['edge_fp_delta','edge_fn_delta','face_fp_delta','face_fn_delta']
comparison=[]
for u in uids:
    a,b=before[u],after[u]
    r={k:a[k] for k in columns[:4]}
    for branch,x in [('before',a),('after',b)]:
        assert x['face']['complete']
        for kind in ['edge','face']:
            for k in ['tp','fp','fn']:r[branch+'_'+kind+'_'+k]=x[kind][k]
            r[branch+'_'+kind+'_soft4']=x['parts'][kind]
        for k in ['missing_gt_face_candidates','edge_perfect','face_perfect','joint_perfect']:r[branch+'_'+k]=x[k]
    for kind in ['edge','face']:
        for k in ['fp','fn']:r[kind+'_'+k+'_delta']=b[kind][k]-a[kind][k]
    assert a['feature_hashes']==b['feature_hashes'] and a['face_train_logits_sha256']==b['face_train_logits_sha256']
    comparison.append(r)
with (ROOT/'per_mesh_comparison.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(comparison)
write(ROOT/'comparison.json',dict(summary=complete['summaries'],retention=complete['retention'],meshes=comparison))
ret=complete['retention']['joint_perfect'];s=complete['summaries']
report=['# 共享 Edge head 接回网络：100条无更新验收','',
    '源模型为第二轮固定Face难负例epoch900/update22500；替换为三条共享head诊断step2000的同一份32×1024权重与32维bias。仅修改两个state_dict键，保留所有其他权重、buffer、metadata。新增optimizer update=0，backward=0。',
    '', '先检查指定三条完整Encoder→μ→Decoder前向，再对固定100条逐条完整验收。所有Face都由各自预测Edge图枚举；旧的30秒枚举时间限制取消以完成全部候选，枚举、scoring及chunk顺序未变。',
    '', '| 指标 | 源模型 | 安装共享head后 |','|---|---:|---:|']
for key,label in [('edge_perfect','Edge严格成功'),('joint_perfect','Edge＋实际Face严格成功'),('edge_fp','Edge FP'),('edge_fn','Edge FN'),('face_fp','实际Face FP'),('face_fn','实际Face FN')]:
    report.append(f"| {label} | {s['before'][key]} | {s['after'][key]} |")
report+=['','## 指定三条','', '| UID | Edge FP/FN 前→后 | 实际Face FP/FN 前→后 |','|---|---|---|']
for u in cfg['three_uids']:
    a,b=before[u],after[u]
    report.append(f"| {u} | {a['edge']['fp']}/{a['edge']['fn']} → {b['edge']['fp']}/{b['edge']['fn']} | {a['face']['fp']}/{a['face']['fn']} → {b['face']['fp']}/{b['face']['fn']} |")
report+=['', '三条真实网络Edge logits与探针逐位一致、重复forward逐位一致。全部100条μ、logvar、Decoder hidden、Face embedding和固定Face训练pool的logits在替换前后均逐位不变。原模型重跑的100条loss、硬计数与margin复现归档。',
    '',f"原联合成功{len(ret['before'])}条：保留{len(ret['retained'])}条，丢失{len(ret['lost'])}条，新增{len(ret['gained'])}条；安装后联合成功{len(ret['after'])}/100。具体UID在comparison.json。",'',
    'Face训练pool分数不变不等于实际Face结果不变：新Edge图改变候选进入资格。完整的FP/FN和缺失GT Face候选数保留在逐mesh记录。',
    '', '## 判读边界','',
    '本次验证固定三条的共同读出可以接回真实网络；对其余97条的效果以本次全量记录为准。该head仅在预先指定三条上训练，并非重新优化100条。保留原模型和所有历史分支；本次不自动将派生模型设为主线，不新增训练预算。',
    '', '## 包含与未包含','',
    '- run/before.jsonl、after.jsonl：两份同源模型各100条完整Edge/实际Face计数、loss、margin、候选覆盖和特征哈希。',
    '- run/three_verification.json、installation_verification.json、complete.json：三条接回、模型重载及零更新核验。',
    '- per_mesh_comparison.csv、comparison.json：逐mesh变化、原成功保留/丢失/新增UID。',
    '- run/three-features-*.npz：三条当前μ、hidden、Face embedding以及替换前后Edge embedding。',
    '- head/checkpoint-step2000.pt：同一成功共享head及其诊断历史状态，仅供溯源；本次未使用优化器。',
    '- source_archive、review_runtime、runtime_dependencies、effective_scoring及入口：归档源码、运行时替换与评分公式。',
    '- 不包含完整源网络、派生完整网络、100条原始mesh及Face pool二进制；其路径和哈希在EXCLUDED_FILES.json、manifest和数据清单中。当前head和源网络可重建派生模型。',
    '', '源checkpoint：`'+cfg['source_checkpoint']+'`','源SHA256：`'+cfg['source_sha256']+'`',
    '派生checkpoint：`'+complete['inference_copy']+'`','派生SHA256：`'+complete['inference_copy_sha256']+'`',
    '', '派生checkpoint是仅推理的网络副本，没有拼接不匹配的旧Adam状态；正式续训需要单独约定。']
(ROOT/'REPORT.md').write_text('\n'.join(report)+'\n')
write(ROOT/'EXCLUDED_FILES.json',dict(source_network=dict(path=cfg['source_checkpoint'],sha256=cfg['source_sha256']),
    derived_network=dict(path=complete['inference_copy'],sha256=complete['inference_copy_sha256']),
    data='data_manifest.csv and source_manifest.json contain paths/hashes; binary meshes and pools excluded',
    full_per_candidate_logits_100='not exported in this inference check; exhaustive counts retained'))
(ROOT/'head').mkdir(exist_ok=True)
shutil.copyfile(cfg['head_checkpoint'],ROOT/'head/checkpoint-step2000.pt')
src=Path(cfg['head_checkpoint']).parent.parent
shutil.copytree(src/'snapshots'/cfg['three_uids'][0]/'effective_code',ROOT/'effective_scoring',dirs_exist_ok=True)
dep=ROOT/'runtime_dependencies';dep.mkdir(exist_ok=True)
manifest=json.loads((ROOT/'source_manifest.json').read_text());registry={}
files={Path(p) for p in manifest['source_sha256']}
base=Path('/guohaoran/nexus_fast_track/diagnostics')
files.update(base/p for p in ['forward_backend_20260911/run_C.py','backward_isolation_20260912/run.py',
    'math00_latent512_804_fresh_20260914/sampling_forward.py'])
for p in sorted(files):
    if str(p) in manifest['source_sha256']:assert sha(p)==manifest['source_sha256'][str(p)]
    target=dep/p.relative_to(base);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    registry[str(target.relative_to(ROOT))]=dict(source=str(p),sha256=sha(p))
write(ROOT/'runtime_sources.json',registry)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,2,figsize=(11,4))
for ax,kind in zip(axs,['edge','face']):
    for br,marker in [('before','o'),('after','x')]:
        ax.scatter([r['vertices'] for r in comparison],[r[br+'_'+kind+'_fp']+r[br+'_'+kind+'_fn'] for r in comparison],s=14,marker=marker,label=br,alpha=.75)
    ax.set_yscale('symlog',linthresh=1);ax.set_xlabel('Vertices');ax.set_ylabel('FP + FN');ax.set_title(kind.capitalize()+' actual errors');ax.grid(alpha=.2);ax.legend()
fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
files=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.json','.jsonl','.py','.csv','.md','.png','.log','.txt'] and p.name not in ['package_verification.json','SHA256SUMS.txt']]
for folder in ['run','head','source_archive','review_runtime','runtime_dependencies','effective_scoring']:
    files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and (p.suffix!='.pt' or folder=='head'))
files=sorted(set(files));hashes={str(p.relative_to(ROOT)):sha(p) for p in files}
(ROOT/'SHA256SUMS.txt').write_text(''.join(f'{v}  {k}\n' for k,v in hashes.items()))
dest=ROOT/'Nexus_SharedEdgeHead_Reintegrate100_ReadOnly_20260917.zip'
with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in files+[ROOT/'SHA256SUMS.txt']:z.write(p,str(p.relative_to(ROOT)))
with zipfile.ZipFile(dest) as z:
    assert z.testzip() is None
    for n,h in hashes.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
write(ROOT/'package_verification.json',dict(path=str(dest),bytes=dest.stat().st_size,sha256=sha(dest),files_verified=len(hashes)))
print((ROOT/'package_verification.json').read_text(),flush=True)
