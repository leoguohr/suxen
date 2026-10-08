"""CPU saved-state/prediction audit and delivery; no forward or optimizer updates."""
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
import csv,json,zipfile,hashlib
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
from loader import sha

R=Path(__file__).resolve().parent
P=R.parent/'teacher_cad50_fresh512_20260921'
SOURCE=R.parent/'teacher_cad50_lr03_pair_20260921/B_lr03/checkpoint-new0500-step2500.pt'
SOURCE_SHA='4c67709dcd3bacb6a09181b034512469b1e7f38463f1aa7b41affc8bcf435a66'
BRANCHES=['A_soft4_full','B_soft4_stopgrad']
assert sha(SOURCE)==SOURCE_SHA
parent=json.loads((R.parent/'teacher_cad50_lr03_pair_20260921/B_lr03/eval-new0500.json').read_text());partition=json.loads((R/BRANCHES[0]/'partition.json').read_text())
assert parent['perfect_uids']==partition['parent_success32']
data_manifest=json.loads((P/'data/manifest.json').read_text());uids=data_manifest['uids']
for x in data_manifest['meshes']:assert sha(P/'data'/x['path'])==x['sha256']
pool_manifest=json.loads((P/'pool_manifest.json').read_text())
for x in pool_manifest['records']:assert sha(P/x['path'])==x['sha256']
artifacts=[dict(path=str(SOURCE),bytes=SOURCE.stat().st_size,sha256=SOURCE_SHA,role='common parent')]
flat=[];trends=[];step_mesh=[];all_eval={};stability={};verified_predictions=0;verified_checkpoints=0

def percentile(values):
    x=np.asarray(values,dtype=np.float64)
    assert np.isfinite(x).all()
    return dict(max=float(x.max()),p95=float(np.quantile(x,.95)),p99=float(np.quantile(x,.99)),median=float(np.median(x)))

def same(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

parent_state=torch.load(SOURCE,map_location='cpu',mmap=True,weights_only=False)
for branch in BRANCHES:
    out=R/branch;done=json.loads((out/'complete.json').read_text());cfg=json.loads((out/'config.json').read_text())
    assert done['stopped_at_budget'] and done['new_updates']==100 and done['completed_updates']==2600
    assert not (out/'failure.json').exists()
    logs=[json.loads(x) for x in (out/'updates.jsonl').read_text().splitlines()]
    assert [x['new_update'] for x in logs]==list(range(1,101))
    for x in logs:
        assert x['completed_updates']==2500+x['new_update'] and x['state_before']==x['completed_updates']-1
        assert [m['uid'] for m in x['meshes']]==uids and x['participation_per_mesh']==x['completed_updates']
        assert x['lrs']==cfg['lrs'] and set(x['adam_steps'].values())=={x['completed_updates']}
        assert np.isfinite(x['total_loss_before']) and all(np.isfinite(v['delta_l2']) for v in x['actual_updates'].values())
        assert x['edge_perfect_uids_before']==[m['uid'] for m in x['meshes'] if m['fp']==m['fn']==0]
        for m in x['meshes']:step_mesh.append(dict(branch=branch,new_update=x['new_update'],state_before=x['state_before'],**m))
    top=sorted(logs,key=lambda x:x['total_loss_before'],reverse=True)[:10]
    spikes=[]
    for i,x in enumerate(logs):
        if i<20:continue
        baseline=float(np.median([v['total_loss_before'] for v in logs[i-20:i]]))
        if x['total_loss_before']>2*baseline:spikes.append(x)
    def detail(x):
        return dict(new_update=x['new_update'],state_before=x['state_before'],total_loss=x['total_loss_before'],
            gradient_norm=x['total_grad_norm'],actual_updates=x['actual_updates'],
            largest_uid_losses=sorted(x['meshes'],key=lambda m:m['edge']+m['face'],reverse=True)[:8])
    stability[branch]=dict(total_loss=percentile([x['total_loss_before'] for x in logs]),
        gradient_norm=percentile([x['total_grad_norm'] for x in logs]),
        top10_total_loss=[detail(x) for x in top],
        spike_definition='pre-update total loss >2x median of previous20 pre-update records; descriptive only',
        spikes=[detail(x) for x in spikes],
        watched_uid_losses={uid:dict(edge=percentile([next(m['edge'] for m in x['meshes'] if m['uid']==uid) for x in logs]),
            face=percentile([next(m['face'] for m in x['meshes'] if m['uid']==uid) for x in logs])) for uid in ['teacher_cad50_12','teacher_cad50_28']})
    es=[]
    for step in [0,25,50,75,100]:
        e=json.loads((out/f'eval-new{step:04d}.json').read_text());es.append(e)
        assert e['new_step']==step and e['completed_updates']==2500+step and [x['uid'] for x in e['meshes']]==uids
        checkpoint=Path(e['checkpoint']);assert sha(checkpoint)==e['checkpoint_sha256']
        cp=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
        assert cp['completed_updates']==2500+step and set(cp['participation'].values())=={2500+step}
        assert all(int(s['step'])==2500+step for s in cp['optimizer']['state'].values())
        assert len(cp['optimizer']['state'])==sum(len(g['params']) for g in cp['optimizer']['param_groups'])
        assert {g['name']:g['lr'] for g in cp['optimizer']['param_groups']}==cfg['lrs']
        for g in cp['optimizer']['param_groups']:
            assert tuple(g['betas'])==(.9,.999) and g['eps']==1e-8 and g['weight_decay']==0
        for name,v in cp['model'].items():
            if name.startswith('autoencoder.log_variance.'):assert same(v,parent_state['model'][name])
        if step==0:
            assert same(cp['model'],parent_state['model']) and same(cp['rng'],parent_state['rng'])
            assert same(cp['optimizer'],parent_state['optimizer'])
            for a,b in zip(cp['optimizer']['param_groups'],parent_state['optimizer']['param_groups']):
                assert {k:v for k,v in a.items() if k!='lr'}=={k:v for k,v in b.items() if k!='lr'}
        artifacts.append(dict(branch=branch,path=str(checkpoint),sha256=e['checkpoint_sha256'],bytes=checkpoint.stat().st_size,new_step=step))
        verified_checkpoints+=1
        for row in e['meshes']:
            path=out/row['prediction_path'];assert sha(path)==row['prediction_sha256']
            with np.load(path) as q:
                n=len(q['vertices']);pe=q['predicted_edge_ids'];ge=q['gt_edges'];fi=q['actual_face_candidate_ids'];gf=q['gt_faces']
                ek=lambda a:a[:,0].astype(np.int64)*n+a[:,1]
                fk=lambda a:(a[:,0].astype(np.int64)*n+a[:,1])*n+a[:,2]
                pred_edges=ek(pe);assert len(np.unique(pred_edges))==len(pe) and (q['predicted_edge_logits']>0).all()
                tp=int(np.isin(pred_edges,ek(ge)).sum());fp=len(pe)-tp;fn=len(ge)-tp
                assert [tp,fp,fn]==[row['edge'][k] for k in ['tp','fp','fn']]
                labels=np.isin(fk(fi),fk(gf));pred=q['actual_face_candidate_logits']>0
                assert len(np.unique(fk(fi)))==len(fi) and np.array_equal(labels,q['actual_face_candidate_gt'])
                tp=int((pred&labels).sum());fp=int((pred&~labels).sum());fn=len(gf)-tp
                assert [tp,fp,fn]==[row['face'][k] for k in ['tp','fp','fn']]
                adj=[set() for _ in range(n)]
                for i,j in pe:adj[int(i)].add(int(j))
                assert sum(len(adj[i]&adj[j]) for i in range(n) for j in adj[i])==len(fi)
                for i,j in [(0,1),(0,2),(1,2)]:assert np.isin(fi[:,i].astype(np.int64)*n+fi[:,j],pred_edges).all()
                covered=np.isin(fk(gf),fk(fi));assert row['missing_gt_face_candidates']==int((~covered).sum())
                assert np.array_equal(covered,q['gt_face_covered'])
                assert row['face']['fn_missing_candidate']+row['face']['fn_present_but_negative']==row['face']['fn']
                with np.load(P/'pools'/f"{row['uid']}.npz") as pool:
                    inpool=np.isin(fk(fi),fk(np.concatenate([pool['positive'],pool['mixed']])))
                assert int((pred&~labels&inpool).sum())==row['face']['actual_fp_inside_training_pool']
                assert int((pred&~labels&~inpool).sum())==row['face']['actual_fp_outside_training_pool']
                assert row['joint_perfect']==(row['edge']['fp']==row['edge']['fn']==row['face']['fp']==row['face']['fn']==0)
            verified_predictions+=1
            flat.append(dict(branch=branch,new_step=step,completed_updates=2500+step,uid=row['uid'],vertices=row['vertices'],
                parent_group='success32' if row['uid'] in partition['parent_success32'] else 'failed18',**row['parts'],
                **{kind+'_'+k:row[kind][k] for kind in ['edge','face'] for k in ['tp','fp','fn']},joint_perfect=row['joint_perfect'],
                face_fn_missing=row['face']['fn_missing_candidate'],face_fn_present=row['face']['fn_present_but_negative'],
                face_fp_in_pool=row['face']['actual_fp_inside_training_pool'],face_fp_out_pool=row['face']['actual_fp_outside_training_pool']))
        perfect=[x['uid'] for x in e['meshes'] if x['joint_perfect']]
        assert e['perfect_uids']==perfect and e['joint_perfect']==len(perfect)
        assert e['retained']==sorted(set(perfect)&set(partition['parent_success32']))
        assert e['lost']==sorted(set(partition['parent_success32'])-set(perfect))
        assert e['new']==sorted(set(perfect)-set(partition['parent_success32']))
        trends.append(dict(branch=branch,new_step=step,edge_f1=e['counts']['edge']['micro_f1'],face_f1=e['counts']['face']['micro_f1'],joint=e['joint_perfect'],retained=len(e['retained']),lost=len(e['lost']),new=len(e['new']),
            large_joint=e['size_groups']['66_to_274']['joint_perfect'],**{kind+'_'+k:e['counts'][kind][k] for kind in ['edge','face'] for k in ['fp','fn']},**e['losses']))
        del cp
        print('AUDIT',branch,step,flush=True)
    all_eval[branch]=es
    full=Path(done['full_model']);assert sha(full)==done['full_model_sha256']
    artifacts.append(dict(branch=branch,path=str(full),sha256=done['full_model_sha256'],bytes=full.stat().st_size,role='full inference model'))

assert verified_predictions==500 and verified_checkpoints==10
assert sha(SOURCE)==SOURCE_SHA
assert json.loads((R/'release.json').read_text())['full_gradient_cross_gpu_bitwise_equal']
assert json.loads((R/'soft4_test.json').read_text())['passed']
configs=[json.loads((R/b/'config.json').read_text()) for b in BRANCHES]
for cfg in configs:
    assert cfg['lrs']==dict(encoder_mu=3e-6,decoder=3e-5,edge_head=3e-5,face_head=3e-5)
    for name,digest in cfg['code_sha256'].items():assert sha(R/cfg['branch']/name)==digest
ignored={'branch','mode','stop_weight_grad','environment'}
assert {k:v for k,v in configs[0].items() if k not in ignored}=={k:v for k,v in configs[1].items() if k not in ignored}
assert configs[0]['mode']=='full' and not configs[0]['stop_weight_grad']
assert configs[1]['mode']=='stopgrad' and configs[1]['stop_weight_grad']
for branch in BRANCHES:
    for metric in ['face_f1','strict']:
        record=json.loads((R/branch/f'best_{metric}.json').read_text())
        key=(lambda e:e['counts']['face']['micro_f1']) if metric=='face_f1' else (lambda e:e['joint_perfect'])
        assert record['value']==max(map(key,all_eval[branch]))
        checkpoint=Path(record['checkpoint']);assert sha(checkpoint)==record['sha256']
        # The best checkpoint already contains the entire model, Adam and RNG.
        record['model_storage']='full resumable checkpoint retained at the referenced path'
        (R/branch/f'best_{metric}.json').write_text(json.dumps(record,indent=2)+'\n')

def csv_write(name,rows):
    with (R/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
csv_write('per_mesh.csv',flat);csv_write('actual_trend.csv',trends);csv_write('step_per_uid.csv',step_mesh)
csv_write('large66_274_per_mesh.csv',[x for x in flat if 66<=x['vertices']<=274])
(R/'stability.json').write_text(json.dumps(stability,indent=2)+'\n')
(R/'MODEL_ARTIFACTS.json').write_text(json.dumps(artifacts,indent=2)+'\n')
audit=dict(branches=2,updates_each=100,final_steps_each=2600,verified_checkpoints=verified_checkpoints,
    prediction_npz_verified=verified_predictions,complete_triangle_enumerations=True,
    original_parent_unchanged=True,model_adam_rng_equal_at_start=True,only_training_change='stop Soft4 weight gradients',
    logvar_unchanged=True,code_hashes_match_start=True,identical_branch_source_files=True,
    no_extra_training=True,scope='CPU saved-state and prediction verification; no additional model or optimizer execution')
(R/'independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
a=all_eval[BRANCHES[0]][-1];b=all_eval[BRANCHES[1]][-1]
af=a['counts']['face']['micro_f1'];bf=b['counts']['face']['micro_f1']
answer=(f"同一B2500起点、各100次全50条更新后：A实际Face micro-F1={af:.10f}，B={bf:.10f}，"
    f"B−A={bf-af:+.10f}（{100*(bf-af):+.4f}个百分点）。"
    f"Edge FP/FN：A={a['counts']['edge']['fp']}/{a['counts']['edge']['fn']}，"
    f"B={b['counts']['edge']['fp']}/{b['counts']['edge']['fn']}。"
    f"同checkpoint联合严格成功A={a['joint_perfect']}/50，B={b['joint_perfect']}/50。")
lines=['# CAD50：Soft4 full vs weight-stopgrad，配对100次更新','',answer,'',
    '两支均从同一完整B2500恢复，继承相同model/Adam/step/RNG；各自累计2600，按100次新增更新预算停止。未触及原固定100条任务，未追加预算。',
    '启动时两张A100-SXM4-80GB均无计算进程，使用独立进程与GPU UUID；每张卡的起点50条预测数组与父记录逐位一致，另在样本20上执行相同full-mode完整梯度，两卡梯度哈希相同。',
    '', '## 同预算末尾','','| 项目 | A full | B stopgrad | B−A |','|---|---:|---:|---:|']
for title,x,y in [('实际Face micro-F1',af,bf),('Edge micro-F1',a['counts']['edge']['micro_f1'],b['counts']['edge']['micro_f1']),
    ('Edge FP',a['counts']['edge']['fp'],b['counts']['edge']['fp']),('Edge FN',a['counts']['edge']['fn'],b['counts']['edge']['fn']),
    ('实际Face FP',a['counts']['face']['fp'],b['counts']['face']['fp']),('实际Face FN',a['counts']['face']['fn'],b['counts']['face']['fn']),
    ('联合严格成功',a['joint_perfect'],b['joint_perfect'])]:
    fmt=lambda v:f'{v:.10f}' if isinstance(v,float) else str(v)
    lines.append(f'| {title} | {fmt(x)} | {fmt(y)} | {fmt(y-x)} |')
for branch,e in [(BRANCHES[0],a),(BRANCHES[1],b)]:
    lines+=['',f"**{branch}**：原32条保留{len(e['retained'])}、丢失{len(e['lost'])}、新增{len(e['new'])}；66—274顶点组严格成功{e['size_groups']['66_to_274']['joint_perfect']}/16。",
        f"丢失UID：{e['lost']}；新增UID：{e['new']}。",
        f"Face阶段目标≥0.997：{'达到' if e['counts']['face']['micro_f1']>=.997 else '未达到'}；全部50条零错误：{'达到' if e['joint_perfect']==50 else '未达到'}。"]
lines+=['','## 实际完整网络检查点','','| 新增步 | A Face F1 | B Face F1 | A联合 | B联合 | A Edge FP/FN | B Edge FP/FN | A Face FP/FN | B Face FP/FN |','|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for x,y in zip(all_eval[BRANCHES[0]],all_eval[BRANCHES[1]]):
    lines.append(f"| {x['new_step']} | {x['counts']['face']['micro_f1']:.8f} | {y['counts']['face']['micro_f1']:.8f} | {x['joint_perfect']} | {y['joint_perfect']} | {x['counts']['edge']['fp']}/{x['counts']['edge']['fn']} | {y['counts']['edge']['fp']}/{y['counts']['edge']['fn']} | {x['counts']['face']['fp']}/{x['counts']['face']['fn']} | {y['counts']['face']['fp']}/{y['counts']['face']['fn']} |")
lines+=['','每次评价都执行真实Encoder→μ→Decoder；实际Face由当时预测Edge图完整枚举，未抽样、截断或使用GT边补候选。预测NPZ含所有输出Edge、全部实际Face候选及logit/GT；Face预测为其中logit>0。缺候选GT计FN。',
    '逐步训练日志为更新前loss和Edge计数，实际参数位移与Adam step为更新后；不能视为逐步实际Face验收。',
    '','## 最佳检查点与辅助稳定性','']
for branch in BRANCHES:
    bestf=json.loads((R/branch/'best_face_f1.json').read_text());bests=json.loads((R/branch/'best_strict.json').read_text())
    s=stability[branch];fmt=lambda d:'/'.join(f'{d[k]:.9g}' for k in ['max','p95','p99'])
    lines += [f"- {branch}：最佳Face F1={bestf['value']:.10f}（新增{bestf['new_step']}）；最佳联合={bests['value']}/50（首次评价于新增{bests['new_step']}）。最高值并列时引用最早已验收checkpoint，所有5份完整checkpoint均保留。",
        f"  更新前总loss Max/P95/P99={fmt(s['total_loss'])}；裁剪前梯度范数={fmt(s['gradient_norm'])}。"]
lines+=['','稳定性仅作辅助。两支均使用100条更新前记录；stability.json保存同口径分位数、最高loss的UID分解与对应实际位移。clip系数是统一梯度缩放，不称作有效LR。',
    '','## 实现与核验','',
    '唯一训练定义差异见MINIMAL_LOSS_DIFF.patch：B仅对Soft4构造权重的sigmoid概率detach，Edge与Face均实际调用该函数；BCE logits仍可微。权重每次重新计算，四组分子/mass跨chunk累积后再归约。两支源文件相同，通过固定启动参数选模式。',
    '小测试证明两种forward相同，stopgrad等于固定当前权重的BCE梯度，梯度差与权重/分母链式项吻合。stopgrad是有意更新规则，不要求它等于重算权重scalar的完整有限差分。',
    '独立CPU审计核对10份可续训checkpoint、500份预测NPZ、200条实际更新日志和数据/pool哈希。每条mesh每次恰参与一次，四组Adam从2500到2600，LR不变，logvar参数不变；父checkpoint哈希保持原值。',
    '','## 解释边界与交付','',
    '末尾差异只支持这个共同起点、继承Adam和100次预算下的干预结果；不把额外训练的共同收益都归于detach，不称作论文hard4，也不据有限结果断言容量或永久有效性。是否增加严格成功须与Face F1变化分开判断。',
    '所有运行配置、日志、逐UID统计、实际预测、数据/pool及有效代码均打包；完整推理模型和全部可续训model/Adam/RNG保留服务器，路径、大小和SHA256见MODEL_ARTIFACTS.json。best_face_f1.json、best_strict.json引用的文件本身包含完整模型。',
    '代码归档中的历史目录仅提供运行时依赖定义；没有载入旧100条模型权重。本轮无特征缓存训练、无新loss/负例/采样/KL。到预算停止。']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
(R/'completion_verified.json').write_text(json.dumps(dict(state='complete_verified',answer=answer,audit=audit),indent=2,ensure_ascii=False)+'\n')
(R/'NEXT_SESSION_PROMPT.md').write_text('\n'.join([
    '# 可发给新会话的上下文','',
    '请只处理老师CAD50实验线，原固定100条由另一个任务负责；不要修改其代码、checkpoint、进程或占用其GPU。',
    '目标仍是固定50条原始无损FP32 CAD、一个共享512维AE、同一个checkpoint，Edge和从预测Edge图完整枚举的实际Face均零错误。Face micro-F1≥0.997是独立阶段目标，不等于50/50严格通过。',
    '数据：50UID、2872顶点、8364GT边、5576GT面、235741全部pair；固定Face pool13946，含5576正例。不得改坐标/顶点编号、焊接、重三角化或自动刷新pool。',
    '已完成fresh2000及各500步LR配对，选取其B2500作为这次共同父状态。父SHA256：'+SOURCE_SHA,
    '本次已实际完成双GPU配对：A fully-differentiable Soft4，B仅detach构造权重的sigmoid；每次重算权重，BCE仍可微。两支各100次全50条累积更新，各自累计2600，均已停止。','',answer,'',
    f"A末尾原32保留/丢失/新增={len(a['retained'])}/{len(a['lost'])}/{len(a['new'])}；B={len(b['retained'])}/{len(b['lost'])}/{len(b['new'])}。",
    f"66—274点组A={a['size_groups']['66_to_274']['joint_perfect']}/16，B={b['size_groups']['66_to_274']['joint_perfect']}/16。",
    '两支完整继承父Adam和RNG；LR E/μ3e-6，D/heads3e-5；完整Encoder+μ+16Decoder+heads训练，logvar冻结；μ、KL0、math00、FP32 SDPA MATH、确定性Graph、TF32/autocast关闭、global clip1。',
    '模型及Adam大文件保留服务器。目录：'+str(R),
    '先读本包REPORT.md、actual_trend.csv、per_mesh.csv、independent_audit.json及MODEL_ARTIFACTS.json。各支best_face_f1.json、best_strict.json明确最佳检查点。',
    '本包有真实全网络5点×2支×50条评价及全部预测；逐步Edge日志不能代替逐步Face评价。停止梯度是有意更新规则，其梯度不是重算权重scalar的完整导数。',
    '当前没有授权自动追加预算或改变loss、LR、pool、架构、Adam、sampling/KL。不要重跑已经完成的实验；先根据本包结果与用户确定后续。SSH凭证不在交付包，需使用用户当时提供的新连接信息。','']) )

# Include current effective dependency sources, also for dynamically loaded helpers.
deps=['diagnostics/math00_four_mesh_lr10x_20260913/backend00.py',
    'diagnostics/encoder_gradient_linesearch_20260911/common.py',
    'diagnostics/forward_backend_20260911/run_C.py','diagnostics/forward_backend_20260911/run.py',
    'diagnostics/backward_isolation_20260912/run.py','diagnostics/sampling_freeze_abc_20260911/run.py',
    'diagnostics/face_freeze_ab_20260910/run.py','diagnostics/soft4_small_20260910/run.py',
    'diagnostics/encoder_teacher_probe_20260907/train.py','diagnostics/edge_only_ab_20260907/run.py',
    'diagnostics/edge_only_ab_20260907/module_probe.py','diagnostics/aggressive_topology_probe_20260907/experiment.py',
    'diagnostics/math00_latent512_804_fresh_20260914/sampling_forward.py']
base=R.parent.parent
files=[]
for p in R.rglob('*'):
    if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts:continue
    if p.suffix in ['.py','.json','.jsonl','.csv','.md','.log','.npz','.patch','.txt'] and p.name!='delivery.json':files.append((p,str(p.relative_to(R))))
for p in (P/'data').rglob('*'):
    if p.is_file():files.append((p,'data/'+str(p.relative_to(P/'data'))))
for p in (P/'pools').glob('*.npz'):files.append((p,'pools/'+p.name))
files.append((P/'pool_manifest.json','pool_manifest.json'))
for name in deps:
    p=base/name;assert p.is_file(),p;files.append((p,'effective_dependencies/'+name))
variant=base/'diagnostics/layernorm_no_rms_20260907_0824/variant_project'
for folder in ['mini_nexus','scripts']:
    for p in (variant/folder).rglob('*.py'):files.append((p,'effective_dependencies/'+str(p.relative_to(base))))
manifest=[];seen=set();archive=R/'CAD50_Soft4_Full_vs_Stopgrad_100_AllResults.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
    for p,name in sorted(files,key=lambda item:item[1]):
        if name in seen:continue
        seen.add(name);z.write(p,name);manifest.append(dict(path=name,bytes=p.stat().st_size,sha256=sha(p)))
    z.writestr('FILE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
assert archive.stat().st_size<900*1024**2
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for item in manifest:assert hashlib.sha256(z.read(item['path'])).hexdigest()==item['sha256']
delivery=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(manifest),zip_verified=True)
(R/'delivery.json').write_text(json.dumps(delivery,indent=2)+'\n')
print(answer,flush=True);print(json.dumps(delivery),flush=True)
