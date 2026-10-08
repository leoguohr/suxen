"""CPU saved-state/prediction audit and delivery; no forward or optimizer updates."""
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
import csv,json,zipfile,hashlib
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

R=Path(__file__).resolve().parent
P=R.parent/'teacher_cad50_fresh512_20260921'
SOURCE=R.parent/'teacher_cad50_lr03_pair_20260921/B_lr03/checkpoint-new0500-step2500.pt'
SOURCE_SHA='4c67709dcd3bacb6a09181b034512469b1e7f38463f1aa7b41affc8bcf435a66'
BRANCHES=['S_soft4_full_control','H_paper_hard4']
O=R/'repro_outputs'
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
                    assert np.array_equal(q['vertices'],pool['vertices'])
                    assert np.array_equal(gf,np.sort(pool['positive'],axis=1))
                    assert np.array_equal(ge,np.unique(np.sort(pool['edges'].T,axis=1),axis=0))
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
        for name,subset,totals in [('all',e['meshes'],e['counts'])]+[(name,[x for x in e['meshes'] if lo<=x['vertices']<=hi],e['size_groups'][name]['counts']) for name,lo,hi in [('8_vertices',8,8),('12_to_16',12,16),('66_to_274',66,274)]]:
            for kind in ['edge','face']:
                sums={k:sum(x[kind][k] for x in subset) for k in ['tp','fp','fn','tn']}
                for k in sums:assert totals[kind][k]==sums[k]
                assert totals[kind]['micro_f1']==2*sums['tp']/max(2*sums['tp']+sums['fp']+sums['fn'],1)
            if name!='all':assert e['size_groups'][name]['joint_perfect']==sum(x['joint_perfect'] for x in subset)
        perfect=[x['uid'] for x in e['meshes'] if x['joint_perfect']]
        assert e['perfect_uids']==perfect and e['joint_perfect']==len(perfect)
        assert e['retained']==sorted(set(perfect)&set(partition['parent_success32']))
        assert e['lost']==sorted(set(partition['parent_success32'])-set(perfect))
        assert e['new']==sorted(set(perfect)-set(partition['parent_success32']))
        trends.append(dict(branch=branch,new_step=step,edge_f1=e['counts']['edge']['micro_f1'],face_f1=e['counts']['face']['micro_f1'],joint=e['joint_perfect'],retained=len(e['retained']),lost=len(e['lost']),new=len(e['new']),
            large_face_f1=e['size_groups']['66_to_274']['counts']['face']['micro_f1'],large_edge_f1=e['size_groups']['66_to_274']['counts']['edge']['micro_f1'],large_joint=e['size_groups']['66_to_274']['joint_perfect'],**{kind+'_'+k:e['counts'][kind][k] for kind in ['edge','face'] for k in ['fp','fn']},**e['losses']))
        del cp
        print('AUDIT',branch,step,flush=True)
    all_eval[branch]=es
    full=Path(done['full_model']);assert sha(full)==done['full_model_sha256']
    final_cp=torch.load(Path(es[-1]['checkpoint']),map_location='cpu',mmap=True,weights_only=False)
    full_cp=torch.load(full,map_location='cpu',mmap=True,weights_only=False)
    assert same(full_cp['model'],final_cp['model'])
    del final_cp,full_cp
    artifacts.append(dict(branch=branch,path=str(full),sha256=done['full_model_sha256'],bytes=full.stat().st_size,role='full inference model'))

assert verified_predictions==500 and verified_checkpoints==10
assert sha(SOURCE)==SOURCE_SHA
configs=[json.loads((R/b/'config.json').read_text()) for b in BRANCHES]
for branch,cfg in zip(BRANCHES,configs):
    assert cfg['source_sha256']==SOURCE_SHA
    for name,digest in cfg['code_sha256'].items():assert sha(R/branch/name)==digest,(branch,name)
locked=['lrs','lr_schedule','betas','eps','weight_decay','clip','trainable_groups','uids','totals',
        'data_manifest_sha256','pool_manifest_sha256','tau','effective_backend','model_mode',
        'sampling','KL','logvar_frozen','microbatch','meshes_per_update','new_updates',
        'initial_completed_updates','final_completed_updates','checkpoints','scales']
for key in locked:assert configs[0][key]==configs[1][key],key
for name in ['loader.py','evaluate.py','construction_args.json']:
    assert sha(R/BRANCHES[0]/name)==sha(R/BRANCHES[1]/name),name
for key in ['python','torch','cuda','cudnn','gpu']:
    assert configs[0]['environment'][key]==configs[1]['environment'][key],key
assert configs[0]['mode']=='full' and not configs[0]['stop_weight_grad']
assert configs[1]['mode']=='paper_hard4' and not configs[1]['soft_membership_used']
assert json.loads((R/'H_paper_hard4/hard4_test.json').read_text())['passed']
gate=json.loads((R/'H_paper_hard4/startup_gate.json').read_text())
assert gate['passed'] and gate['same_weight_full_soft4_gradient_matches_control']
reuse=json.loads((O/'CONTROL_REUSE_AUDIT.json').read_text())
original=Path(reuse['control'])
assert sha(original/'config.json')==reuse['control_config_sha256']
assert sha(original/'updates.jsonl')==reuse['control_updates_sha256']
assert all(sha(original/p.name)==sha(p) for p in (R/'S_soft4_full_control').glob('*.py'))
best={}
for branch in BRANCHES:
    best[branch]={}
    for metric in ['face_f1','strict']:
        record=json.loads((R/branch/f'best_{metric}.json').read_text())
        key=(lambda e:e['counts']['face']['micro_f1']) if metric=='face_f1' else (lambda e:e['joint_perfect'])
        assert record['value']==max(map(key,all_eval[branch]))
        assert sha(Path(record['checkpoint']))==record['sha256']
        best[branch][metric]=record
assert not torch.cuda.is_initialized()

def save(name,obj):
    (O/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csv_write(name,rows):
    with (O/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
csv_write('per_mesh.csv',flat);csv_write('actual_trend.csv',trends)
csv_write('step_per_uid.csv',step_mesh)
csv_write('large66_274_per_mesh.csv',[x for x in flat if 66<=x['vertices']<=274])
save('stability.json',stability);save('CHECKPOINT_MANIFEST.json',artifacts);save('BEST_CHECKPOINTS.json',best)
audit=dict(passed=True,branches=2,comparison_updates_each=100,final_steps_each=2600,
    newly_executed_updates=dict(S_soft4_full_control=0,H_paper_hard4=100),
    verified_checkpoints=verified_checkpoints,prediction_npz_verified=verified_predictions,
    complete_triangle_enumerations=True,original_parent_sha256_unchanged=True,
    model_adam_rng_equal_at_start=True,all_locked_config_fields_equal=locked,
    data_and_pool_hashes_verified=True,logvar_unchanged=True,code_hashes_match_start=True,
    same_weight_Soft4_full_gradient_matches_reused_control=True,
    original_control_files_unchanged=True,final_inference_tensors_match_resumable_model=True,
    audit_optimizer_updates=0,audit_gpu_used=False,
    scope='CPU stored-state and prediction recount; existing original full-network evaluations retained; no extra model forward')
save('ACCEPTANCE.json',audit)
a=all_eval[BRANCHES[0]][-1];b=all_eval[BRANCHES[1]][-1]
af=a['counts']['face']['micro_f1'];bf=b['counts']['face']['micro_f1']
al=a['size_groups']['66_to_274'];bl=b['size_groups']['66_to_274']
answer=(f"同一B2500起点、同样100次更新后，Hard4相对原Soft4：实际Face micro-F1从{af:.10f}变为{bf:.10f}"
    f"（差值{bf-af:+.10f}）；Edge FP/FN从{a['counts']['edge']['fp']}/{a['counts']['edge']['fn']}变为"
    f"{b['counts']['edge']['fp']}/{b['counts']['edge']['fn']}；16条66—274顶点困难CAD的Face micro-F1从"
    f"{al['counts']['face']['micro_f1']:.10f}变为{bl['counts']['face']['micro_f1']:.10f}，严格成功从"
    f"{al['joint_perfect']}/16变为{bl['joint_perfect']}/16；全部50条联合严格成功从{a['joint_perfect']}/50变为{b['joint_perfect']}/50。")
lines=[answer,'','本轮仅新执行H的100次全50条更新；S复用已完成且通过身份、状态、数据、代码、日志与评价核验的Soft4对照，没有给S追加更新。H使用用户确认的GPU1；原固定100条和旧Soft4/stopgrad实验未修改。','',
    '## 同预算末尾','','| 指标 | S Soft4 | H Hard4 | H−S |','|---|---:|---:|---:|']
values=[('实际Face micro-F1',af,bf),('Edge micro-F1',a['counts']['edge']['micro_f1'],b['counts']['edge']['micro_f1']),
        ('Edge FP',a['counts']['edge']['fp'],b['counts']['edge']['fp']),('Edge FN',a['counts']['edge']['fn'],b['counts']['edge']['fn']),
        ('实际Face FP',a['counts']['face']['fp'],b['counts']['face']['fp']),('实际Face FN',a['counts']['face']['fn'],b['counts']['face']['fn']),
        ('Face FN：缺候选',a['face_fn_missing'],b['face_fn_missing']),('Face FN：有候选判负',a['face_fn_present'],b['face_fn_present']),
        ('Face FP：基础pool内',a['face_fp_inside_pool'],b['face_fp_inside_pool']),('Face FP：基础pool外',a['face_fp_outside_pool'],b['face_fp_outside_pool']),
        ('Edge严格成功',a['edge_perfect'],b['edge_perfect']),('Face严格成功',a['face_perfect'],b['face_perfect']),
        ('联合严格成功',a['joint_perfect'],b['joint_perfect']),
        ('困难16条Face micro-F1',al['counts']['face']['micro_f1'],bl['counts']['face']['micro_f1']),
        ('困难16条Edge micro-F1',al['counts']['edge']['micro_f1'],bl['counts']['edge']['micro_f1'])]
for title,x,y in values:
    fmt=lambda v:f'{v:.10f}' if isinstance(v,float) else str(v)
    lines.append(f'| {title} | {fmt(x)} | {fmt(y)} | {fmt(y-x)} |')
lines+=['','## 完整网络评价轨迹','','| 新增步 | S实际Face F1 | H实际Face F1 | S联合 | H联合 | S Edge FP/FN | H Edge FP/FN |','|---:|---:|---:|---:|---:|---:|---:|']
for x,y in zip(all_eval[BRANCHES[0]],all_eval[BRANCHES[1]]):
    lines.append(f"| {x['new_step']} | {x['counts']['face']['micro_f1']:.10f} | {y['counts']['face']['micro_f1']:.10f} | {x['joint_perfect']} | {y['joint_perfect']} | {x['counts']['edge']['fp']}/{x['counts']['edge']['fn']} | {y['counts']['edge']['fp']}/{y['counts']['edge']['fn']} |")
lines+=['','## 保留集合、最高值与目标','']
for branch,e in [(BRANCHES[0],a),(BRANCHES[1],b)]:
    f=best[branch]['face_f1'];s=best[branch]['strict']
    lines.extend([f"- {branch}：父32条保留{len(e['retained'])}，丢失{len(e['lost'])}，新增{len(e['new'])}。丢失UID：{e['lost']}；新增UID：{e['new']}。",
        f"  最高实际Face F1={f['value']:.10f}（新增{f['new_step']}）；最高联合严格成功={s['value']}/50（最早在新增{s['new_step']}检查点）。对应完整model/Adam/RNG路径与哈希见BEST_CHECKPOINTS.json；末尾和全部5份checkpoint均保留。",
        f"  本预算任一已验收checkpoint达到Face F1≥0.997：{f['value']>=.997}；严格50/50：{s['value']==50}。"])
lines+=['','## 实现与证据','',
    'H按logit.detach()>0与GT划分TP/TN/FP/FN，BCE由原始可微logits计算。Edge与Face各自跨全部chunk累计整条mesh组内BCE总和/候选数，再计算四组均值并固定除4；空组贡献0是本次明确约定，不冒称作者官方代码。没有soft membership，没有detach BCE/分子/loss。',
    '实际路径是H_paper_hard4/runtime.py的objective→edge_reconstruction_loss/face_reconstruction_loss→hard4_sums。loss本身使用普通autograd；原Graph自定义反传和attention重计算代码、FP32 MATH上下文保持不变。未只修改历史未调用函数。启动日志记录两条有效路径调用及全网络可微性。',
    '小测试包含四组非空、部分空组、单一非空组、logit=0归负、Edge/Face不同chunk与整块的loss/梯度、固定分组BCE梯度方向及有限差分。FP32不同归约次序仅按记录容差检查；原Soft4的值/梯度另按逐位相等检查。',
    'H的0步全部50条预测数组与B2500逐项一致，且新卡上的同权重Soft4完整梯度与复用S的旧记录逐位一致；检查结束后model/Adam/RNG再验证并恢复原eval模式和父RNG，检查本身没有optimizer更新。H与S不同loss数值不要求相等。',
    'CPU最终审计核对10份完整可续训checkpoint、500份预测NPZ、两个分支的100步日志；重数Edge/Face错误并证明候选完整枚举，分别核对漏候选FN、候选内判负FN、pool内/外FP。最高值、末尾推理权重与可续训状态一致，父SHA和旧S代码/日志保持不变。',
    '逐步loss和Edge错误为更新前；梯度范数为clip前；clip系数、Adam计数与各参数组实际位移有明确记录。逐步Edge不等于逐步实际Face验收。实际Face只在0/25/50/75/100保存完整网络评价。',
    '','## 解释边界与停止','',
    '本实验比较同一B2500、继承同一Adam历史、固定100次更新的损失切换方向。不能按两种训练loss数值排名，不能外推随机初始化训练的最终上限或充分收敛结果，不能把Hard4的变化单独归因于删除Soft4第二项，也不能把它当作detached Soft4实验。',
    '仅恢复用户定义的论文式重建损失；没有恢复论文的全部网络/latent/KL/采样，也没有使用老师原模型权重。100步预算已经停止，不追加训练、不补难负例、不调LR、不扩层、不迁移原100条。',
    '有效代码/diff、配置、命令、日志、逐检查点逐mesh结果及预测打包。完整model/Adam/RNG保留服务器，路径、大小和SHA256见CHECKPOINT_MANIFEST.json；不重复打包大权重。',
    'RigorPilot本轮实际用于读取证据边界、CPU对照复用核验、记录授权loss差异，以及通过run-train持久化H的启动、资源、日志和结束状态。新增证据是Hard4真实路径测试、B2500起点一致性及本次100步实际重建对照；未执行后继训练，也未核验作者官方源码的空组规则。未发现需要修改既有网络或评价定义的明确bug。']
(O/'RUN_REPORT.md').write_text('\n'.join(lines)+'\n')
(O/'SUMMARY.md').write_text(answer+'\n\n'+f"S复用合格旧对照；本轮只新执行H100步，已按预算停止。两支最高Face F1：{best[BRANCHES[0]]['face_f1']['value']:.10f}/{best[BRANCHES[1]]['face_f1']['value']:.10f}。完整报告见RUN_REPORT.md，逐检查点和逐UID数据见CSV。\n")
save('status.json',dict(state='complete_verified',selected_goal='training',outcome='partial',
    execution_complete=True,budget_complete=True,stopped_at_budget=True,control_reused=True,
    stage_results=dict(intake='success',control_qualification='success',hard4_tests='success',training='success',cpu_audit='success',followup_training='not_requested'),
    result_match=dict(status='not_evaluated',reason='Bounded user-defined loss comparison, not a published-paper metric reproduction claim'),
    face_f1_phase_threshold=.997,face_f1_phase_reached=bf>=.997,strict50_reached=b['joint_perfect']==50,
    verification_scope=audit,answer=answer))
print('FINAL_ACCEPTANCE_PASS',answer,flush=True)
