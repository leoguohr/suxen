"""Full fixed-100 shared-head handoff, including cached H and all final pair logits."""
import csv,hashlib,json,shutil,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
out=ROOT/'run';complete=json.loads((out/'complete.json').read_text());verification=json.loads((out/'verification.json').read_text())
assert verification['continuous_updates']==complete['updates'] and verification['one_shared_head']
cfg=json.loads((ROOT/'config.json').read_text());plan=json.loads((ROOT/'training_plan.json').read_text())
cost=json.loads((ROOT/'benchmark.json').read_text());cache=json.loads((ROOT/'cache/manifest.json').read_text())
trace=[json.loads(s) for s in (out/'evaluations.jsonl').read_text().splitlines()]
updates=[json.loads(s) for s in (out/'updates.jsonl').read_text().splitlines()]
uids=[m['uid'] for m in trace[0]['meshes']]
with (ROOT/'per_mesh_trace.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['step','uid','tp','fp','fn','tn','perfect','edge_soft4','min_margin_gt','min_margin_non_gt'])
    for r in trace:
        for m in r['meshes']:w.writerow([r['step'],m['uid'],*[m[k] for k in ['tp','fp','fn','tn','perfect','edge_soft4','min_margin_gt','min_margin_non_gt']]])
permesh=[]
for i,u in enumerate(uids):
    series=[r['meshes'][i] for r in trace];perfect=[r['step'] for r in trace if r['meshes'][i]['perfect']]
    streak=longest=0
    for s in series:
        streak=streak+1 if s['perfect'] else 0;longest=max(longest,streak)
    permesh.append(dict(uid=u,vertices=cache['meshes'][i]['vertices'],pairs=cache['meshes'][i]['pairs'],
        before=series[0],after=series[-1],first_perfect_step=min(perfect) if perfect else None,
        perfect_evaluation_count=len(perfect),longest_perfect_evaluations=longest,
        last500_perfect_evaluations=sum(s['perfect'] for s in series[-500:])))
write(ROOT/'per_mesh_summary.json',permesh)
with (ROOT/'per_mesh_final.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['uid','vertices','initial_fp','initial_fn','final_fp','final_fn','final_perfect','first_perfect_step','last500_perfect','final_edge_soft4'])
    for r in permesh:w.writerow([r['uid'],r['vertices'],r['before']['fp'],r['before']['fn'],r['after']['fp'],r['after']['fn'],r['after']['perfect'],r['first_perfect_step'],r['last500_perfect_evaluations'],r['after']['edge_soft4']])
initial=complete['baseline'];final=complete['final']
groups={}
for name,lo,hi in [('N<=500',0,500),('500<N<=1500',500,1500),('N>1500',1500,100000)]:
    rows=[r for r in permesh if lo<r['vertices']<=hi]
    groups[name]=dict(meshes=len(rows),initial_perfect=sum(r['before']['perfect'] for r in rows),final_perfect=sum(r['after']['perfect'] for r in rows),
        initial_fp=sum(r['before']['fp'] for r in rows),initial_fn=sum(r['before']['fn'] for r in rows),final_fp=sum(r['after']['fp'] for r in rows),final_fn=sum(r['after']['fn'] for r in rows))
summary=dict(initial={k:initial[k] for k in ['edge_perfect','objective','total_fp','total_fn']},
    final={k:final[k] for k in ['edge_perfect','objective','total_fp','total_fn']},groups=groups,
    retained=complete['initial_success_retained'],lost=complete['initial_success_lost'],gained=complete['new_final_success'],
    first_all100_perfect_step=complete['first_all100_perfect_step'],longest_all100_perfect=complete['longest_all100_perfect'],
    last500_all100_perfect=complete['last500_all100_perfect'],
    each_mesh_participations=complete['updates'],total_pair_scores_during_updates=cache['total_pairs']*complete['updates'],
    training_wall_seconds=complete['wall_seconds'],weight_update_nonzero_steps=sum(u['parameter_updates']['weight']['changed_elements']>0 for u in updates),
    clip_steps=sum(u['clip_coefficient']<1 for u in updates))
write(ROOT/'summary.json',summary)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,3,figsize=(14,4))
x=[r['step'] for r in trace]
axs[0].plot(x,[r['edge_perfect'] for r in trace]);axs[0].set_ylabel('Strict Edge-perfect meshes / 100');axs[0].set_ylim(0,101)
axs[1].plot(x,[r['total_fp'] for r in trace],label='FP');axs[1].plot(x,[r['total_fn'] for r in trace],label='FN');axs[1].set_yscale('symlog',linthresh=1);axs[1].legend()
axs[2].plot(x,[r['objective'] for r in trace]);axs[2].set_ylabel('Mean Edge Soft4')
for ax in axs:ax.set_xlabel('Shared-head updates, all 100 meshes each');ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(ROOT/'curves.png',dpi=170);fig.savefig(ROOT/'curves.pdf');plt.close(fig)
report=['# 固定100条 hidden：一个共享 Edge head 的有限拟合','',
    '全部固定特征取自原第二轮难负例epoch900/update22500，同一原Edge head初始化；没有使用三条专用head。仅训练同一个FP32 Linear(1024,32,bias=True)，共32,800个参数。原Encoder、Decoder和其他heads不参与训练循环。',
    '',f"先导出并逐条复现基线，再测量成本并固定预算：{plan['updates']}次实际更新；fresh Adam LR={plan['lr']}，betas=(0.9,0.999)，eps=1e-8，wd=0，global clip=1，不修改LR、不自动延长。",
    '', '每次更新完整遍历原定100条，逐条全量pair loss按1/100反向累积，最后统一clip与一次Adam step。训练目标为mean100(Edge Soft4)，Soft4内部四组均值再除4；不额外乘整目标1/4。没有Face、KL、sampling、MSE或mesh专属head。',
    '', '## 实测成本','',
    f"100条共{cache['total_vertices']:,}个顶点，{cache['total_pairs']:,}个无向pair；hidden原始张量{cache['hidden_total_bytes']/2**20:.2f} MiB。",
    f"A100-80GB实测完整一轮评分{cost['forward_all100_seconds']:.3f}秒，评分加反向平均{cost['mean_forward_backward_seconds']:.3f}秒。预检峰值allocated {cost['peak_allocated_gpu_bytes']/2**30:.2f} GiB，reserved {cost['peak_reserved_gpu_bytes']/2**30:.2f} GiB；预检不更新权重。",
    f"实际训练循环用时{complete['wall_seconds']:.1f}秒，每条直接参与{complete['updates']}次；这是{complete['updates']}次共享head更新，不是100份独立head。",'',
    '| 检查点 | 同一head Edge严格成功 | 全量FP | 全量FN | mean Edge Soft4 |','|---|---:|---:|---:|---:|']
for step in plan['checks']:
    r=trace[step];report.append(f"| {step} | {r['edge_perfect']}/100 | {r['total_fp']} | {r['total_fn']} | {r['objective']:.9g} |")
report+=['',f"末尾保留原成功{len(complete['initial_success_retained'])}条，丢失{len(complete['initial_success_lost'])}条，新增{len(complete['new_final_success'])}条。身份清单在summary.json。",
    f"首次100/100同时严格成功：{complete['first_all100_perfect_step']}；最长连续{complete['longest_all100_perfect']}次更新；末尾500点中{complete['last500_all100_perfect']}次100/100。不同step曾成功的UID不能累加冒充同一head成功。",'',
    '## 判读','',
    ('本轮获得全部100条固定hidden的共同Edge读出，并在预算内记录保持情况。' if final['all100_perfect'] else '有限预算末尾尚未获得全部100条共同Edge读出；这不是特征或32维评分表达不可能的证明。'),
    '本次只评价完整Edge图；没有安装新head到全网络作Face验收，没有宣称Edge＋实际Face或完整AE/VAE通过。原模型checkpoint与历史分支完整保留。',
    '', '## 核验与材料','',
    '- cache/：全部100条FP32 Decoder hidden、GT vertices/edges/faces、原始Edge head输出及基线head梯度；head_original.npz保存原始共享head。',
    '- baseline_verification.json、export_complete.json：缓存读出与真实源网络的raw/center/loss/gradient逐位一致，基线计数复现。',
    '- benchmark.json：完整评分/反向成本、重复性和零更新检查。',
    '- run/evaluations.jsonl：step0到末尾每个同一head状态的100条全量计数、loss、margin。run/updates.jsonl：每次更新的梯度、clip、参数实际位移。',
    '- run/checkpoint-step*.pt：同一head及Adam的检查点；run/verification.json：所有固定检查点重载验收。',
    '- run/final_outputs/：全部100条末尾raw/centered Edge表示及所有pair logits；pair顺序为本地顶点编号i<j的lexicographic上三角顺序，GT由cache中的edges定义。',
    '- per_mesh_trace.csv、per_mesh_final.csv、per_mesh_summary.json、summary.json、curves：逐条轨迹、成功保留与分组汇总。',
    '- 入口、runtime、effective_code、source_archive、review_runtime：实际执行逻辑与有效评分公式。',
    '', '不含原始大网络checkpoint和Face训练pool二进制；其服务器路径/哈希在EXCLUDED_FILES.json及source_manifest.json。所需固定hidden、GT结构、原head、最终head和所有最终pair logits均已入包。',
    '', '启动兼容修复：首次Adam初始化因protobuf环境退出，发生在任何更新前；恢复已有PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python设置后重启。保留失败记录，未更改目标、初始化、LR或预算。']
(ROOT/'REPORT.md').write_text('\n'.join(report)+'\n')
write(ROOT/'EXCLUDED_FILES.json',dict(original_network=dict(path=cfg['source_checkpoint'],sha256=cfg['source_sha256']),
    face_training_pool='Not used in this Edge-only diagnostic; paths/hashes in source_manifest.json'))
# Capture the already-validated runtime dependency closure from the preceding reintegration check.
prev=ROOT.parent/'shared_edge_head_reintegrate100_20260917'
shutil.copytree(prev/'runtime_dependencies',ROOT/'runtime_dependencies',dirs_exist_ok=True)
shutil.copyfile(prev/'runtime_sources.json',ROOT/'runtime_sources.json')
files=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.py','.json','.jsonl','.csv','.md','.png','.pdf','.log','.txt'] and p.name not in ['package_verification.json','SHA256SUMS.txt']]
for folder in ['cache','run','effective_code','source_archive','review_runtime','runtime_dependencies']:
    files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
files=sorted(set(files));hashes={str(p.relative_to(ROOT)):sha(p) for p in files}
(ROOT/'SHA256SUMS.txt').write_text(''.join(f'{h}  {name}\n' for name,h in hashes.items()))
dest=ROOT/'Nexus_SharedEdgeHead_Fixed100_2000Updates_FullEvaluation_20260917.zip'
with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
    for p in files+[ROOT/'SHA256SUMS.txt']:z.write(p,str(p.relative_to(ROOT)))
with zipfile.ZipFile(dest) as z:
    assert z.testzip() is None
    for name,h in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==h,name
write(ROOT/'package_verification.json',dict(path=str(dest),sha256=sha(dest),bytes=dest.stat().st_size,files_verified=len(hashes)))
print((ROOT/'package_verification.json').read_text(),flush=True)
