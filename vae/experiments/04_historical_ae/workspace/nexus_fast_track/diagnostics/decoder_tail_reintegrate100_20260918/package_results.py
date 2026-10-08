"""Package the zero-update full-network tail installation, including all100 comparisons."""
import csv,hashlib,json,shutil,zipfile
from pathlib import Path
R=Path(__file__).resolve().parent;out=R/'run'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()
complete=read(out/'complete.json');cfg=read(R/'config.json');uids=read(R/'selection.json')['uids']
assert complete['optimizer_updates']==complete['backward_calls']==0
before={r['uid']:r for r in map(json.loads,(out/'before.jsonl').read_text().splitlines())}
after={r['uid']:r for r in map(json.loads,(out/'after.jsonl').read_text().splitlines())}
assert set(before)==set(after)==set(uids) and len(uids)==100
gate=read(out/'edge_gate.json');assert gate['complete'] and len(gate['meshes'])==100
comparison=[]
for uid in uids:
    a,z=before[uid],after[uid];row={k:a[k] for k in ['uid','vertices','gt_edges','gt_faces']}
    for branch,r in [('before',a),('after',z)]:
        assert r['face']['complete']
        for kind in ['edge','face']:
            for key in ['tp','fp','fn']:row[branch+'_'+kind+'_'+key]=r[kind][key]
            row[branch+'_'+kind+'_soft4']=r['parts'][kind]
        for key in ['missing_gt_face_candidates','gt_face_candidates','edge_perfect','face_perfect','joint_perfect']:row[branch+'_'+key]=r[key]
        row[branch+'_face_candidates']=r['face']['scored_candidates']
        row[branch+'_face_fn_candidate_present']=r['face']['fn']-r['missing_gt_face_candidates']
    comparison.append(row)
with (R/'per_mesh_comparison.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)
write(R/'comparison.json',dict(summary=complete['summaries'],retention=complete['retention'],meshes=comparison))
seven=[r for r in comparison if r['uid'] in cfg['seven_uids']]
with (R/'seven_large_meshes.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(seven[0]));w.writeheader();w.writerows(seven)
s=complete['summaries'];ret=complete['retention']['joint_perfect']
lines=['# Decoder末端接回真实网络：100条Edge＋实际Face无更新验收','',
    '使用原第二轮难负例epoch900/update22500的独立副本，安装Decoder末块＋最终LayerNorm＋Edge head诊断step500权重。仅替换指定10个state_dict张量；其余参数、buffer、metadata逐位不变。Face head权重没有改变。',
    '源SHA256：`'+cfg['source_sha256']+'`；输入末块checkpoint SHA256：`'+cfg['tail_sha256']+'`。没有加载任何Control模型、三条专用head或head-only step2000。',
    '本次没有构造optimizer，没有backward或optimizer update。每次都从真实mesh输入执行Encoder→μ→全部Decoder；没有用缓存hidden代替网络推理。',
    '', '先完成全部100条Edge gate：真实网络的raw/centered Edge表示及全部84,669,234个pair logits逐位复现缓存诊断；71/100、FP156784、FN1全部复现。μ、logvar和末块输入在安装前后逐位不变。7条指定大mesh额外重复forward逐位相同。',
    '随后固定每份模型，逐条完整评估100条；Face从各自当次预测Edge图枚举，不替换成训练pool，不截断候选。替换前100条计数、loss、margin复现源epoch900归档。',
    '', '| 指标 | 源模型 | 安装末块＋LN＋head后 |','|---|---:|---:|']
for key,label in [('edge_perfect','Edge严格成功'),('face_perfect','实际Face严格成功'),('joint_perfect','Edge＋实际Face联合成功'),('edge_fp','Edge FP'),('edge_fn','Edge FN'),('face_fp','实际Face FP'),('face_fn','实际Face FN'),('missing_gt_face_candidates','缺边导致未进入候选的GT Face')]:lines.append(f"| {label} | {s['before'][key]} | {s['after'][key]} |")
lines+=['',f"原联合成功{len(ret['before'])}条，保留{len(ret['retained'])}，丢失{len(ret['lost'])}，新增{len(ret['gained'])}；安装后同一模型联合成功{len(ret['after'])}/100。全部UID在comparison.json。",'',
    '## 7条新成功大mesh的实际Face','', '| UID | 顶点数 | 新Edge FP/FN | 原Face FP/FN | 新实际Face FP/FN | 新联合严格成功 |','|---|---:|---|---|---|---|']
for r in seven:lines.append(f"| {r['uid']} | {r['vertices']} | {r['after_edge_fp']}/{r['after_edge_fn']} | {r['before_face_fp']}/{r['before_face_fn']} | {r['after_face_fp']}/{r['after_face_fn']} | {r['after_joint_perfect']} |")
lines+=['', '## 判读边界','',
    f"100条中有{complete['face_embedding_changed_meshes']}条Face embedding发生变化。冻结Face head参数不等于冻结Face分数：共享Decoder末端变化会影响评分；Edge图变化还会改变Face候选。两种影响同时存在，本轮没有另外做因果拆分。",
    'Edge诊断接回成功不等于Edge＋实际Face成功。联合结论仅以本次同一推理副本、各自当次预测Edge候选的完整计数为准。不拼接不同模型/不同forward的成功。',
    '原源模型与step500诊断checkpoint均保留，派生模型是没有Adam状态的推理副本，本轮没有自动选择新主线或追加训练。',
    '', '## 材料索引','',
    '- run/installation_verification.json：权重映射、实际改变的10个键、其他状态不变、独立副本SHA与重载检查。',
    '- run/edge_gate.json、edge_gate_summary.json：100条真实网络与缓存结果逐位对齐，hidden/Face表示的变化量。',
    '- run/before.jsonl、after.jsonl：两份模型各100条的实际Edge/Face、候选覆盖、FN分解、训练pool诊断loss、margin和特征哈希。',
    '- comparison.json、per_mesh_comparison.csv、seven_large_meshes.csv：汇总、成功保留/丢失/新增UID及7条大mesh结果。',
    '- runtime.py、sampling_forward.py、evaluate.py、run_evaluation.py、effective_scoring及runtime_dependencies：实际执行代码、运行时替换、枚举和评分规则。',
    '- input_tail/checkpoint-step0500.pt：输入三模块权重及此前500步诊断的历史状态；本次只加载其中tail权重，未使用其Adam。',
    '- run/features/*.npz（FullEvidence包）：全部100条μ/logvar、安装前后hidden、Edge/Face表示、全部末尾Edge logits及GT mesh数组。',
    '- 大型源网络、派生完整网络及Face pool二进制不入包；路径与SHA在EXCLUDED_FILES.json和source_manifest.json。Review包另省略大型features，逐文件路径/哈希明确列出。',
    '- 没有导出全部实际Face候选的逐条logit；保留完整枚举计数、代码与评分表示，不能把未导出称为不存在。',
    '', '推理副本：`'+complete['inference_copy']+'`','推理副本SHA256：`'+complete['inference_copy_sha256']+'`']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
(R/'input_tail').mkdir(exist_ok=True);shutil.copyfile(cfg['tail_checkpoint'],R/'input_tail/checkpoint-step0500.pt')
prev=Path(cfg['tail_root'])
for old,new in [('runtime_dependencies','runtime_dependencies'),('effective_code','effective_scoring')]:shutil.copytree(prev/old,R/new,dirs_exist_ok=True)
shutil.copyfile(prev/'runtime_sources.json',R/'runtime_sources.json')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,2,figsize=(11,4))
for ax,kind in zip(axs,['edge','face']):
    for br,marker in [('before','o'),('after','x')]:ax.scatter([r['vertices'] for r in comparison],[r[br+'_'+kind+'_fp']+r[br+'_'+kind+'_fn'] for r in comparison],s=15,marker=marker,label=br,alpha=.75)
    ax.set_yscale('symlog',linthresh=1);ax.set(xlabel='Vertices',ylabel='FP + FN',title=kind.capitalize()+' actual errors');ax.legend();ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(R/'comparison.png',dpi=170);fig.savefig(R/'comparison.pdf');plt.close(fig)
base=[p for p in R.iterdir() if p.is_file() and p.suffix in ['.py','.json','.jsonl','.csv','.md','.png','.pdf','.log','.txt'] and p.name not in ['package_verification.json','SHA256SUMS.txt','EXCLUDED_FILES.json']]
for folder in ['run','input_tail','effective_scoring','runtime_dependencies','source_archive','review_runtime']:
    base.extend(p for p in (R/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='model-source22500-tail500-inference.pt')
base=sorted(set(base));metadata=[]
for kind in ['Review','FullEvidence']:
    files=[p for p in base if kind=='FullEvidence' or 'features' not in p.parts]
    excluded={str(p.relative_to(R)):dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in base if p not in files}
    excluded.update(source_model=dict(path=cfg['source_checkpoint'],sha256=cfg['source_sha256']),installed_model=dict(path=complete['inference_copy'],sha256=complete['inference_copy_sha256']),face_pool='Paths/hashes in source_manifest.json; binary pools excluded',actual_face_candidate_logits='Not exported; complete counts and scoring representations retained')
    extra=json.dumps(excluded,indent=2).encode();hashes={str(p.relative_to(R)):sha(p) for p in files};hashes['EXCLUDED_FILES.json']=hashlib.sha256(extra).hexdigest()
    dest=R/f'Nexus_DecoderTail_Reintegrate100_{kind}_20260918.zip'
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p in files:z.write(p,str(p.relative_to(R)))
        z.writestr('EXCLUDED_FILES.json',extra);z.writestr('SHA256SUMS.txt',''.join(f'{h}  {n}\n' for n,h in hashes.items()))
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        for n,h in hashes.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    metadata.append(dict(kind=kind,path=str(dest),sha256=sha(dest),bytes=dest.stat().st_size,files_verified=len(hashes)))
    write(R/'package_verification.json',metadata);print('PACKAGED',json.dumps(metadata[-1]),flush=True)
