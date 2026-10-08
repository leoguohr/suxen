"""Package all artifacts for this paired experiment after training and CPU acceptance."""
import csv,hashlib,json,os,zipfile
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parent;O=R/'repro_outputs'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
cfg=json.loads((R/'config.json').read_text());accept=json.loads((O/'ACCEPTANCE.json').read_text())
assert accept['cpu_audit_passed'] and accept['total_optimizer_updates']==1000
assert json.loads((O/'recovery_204/train_runner_result.json').read_text())['runtime_status']=='success'
scope='''# 本轮实验完整交付范围

四个ZIP合计包含本轮Last2/Last3固定预算对照的全部结果与代码。

- Review：报告、逐步和逐UID指标、完整日志、运行配置、源代码快照、动态依赖代码、源码Git历史、预检和恢复证据、输入及checkpoint清单。
- Predictions：两支0/100/200/300/400/500共1200份完整预测数组，含GT与实际Face候选。
- Weights：本轮所有完整模型、所有中间/最终/latest续训checkpoint和中断快照，以及共同父A500完整权重与Adam/RNG。
- Inputs_Caches：原100条mesh/topology、实际固定Face pool、block13输入缓存、原始预检与跨容器恢复的预测数组。

不包含其他历史实验或CAD50的大权重，也不复制整个conda/CUDA环境；环境版本与原绝对路径已记录。代码与输入保留路径映射，迁移机器时需重新绑定记录中的绝对路径。本包不宣称已在另一台机器完成从零重放。

最终分卷SHA256以FINAL_DELIVERY.json为准；各卷另带PACKAGE_FILE_MANIFEST.json。旧DELIVERY.json属于自动初版打包记录，不能替代最终交付清单。
'''
(O/'PACKAGE_SCOPE.md').write_text(scope)
checkpoint_manifest=json.loads((O/'CHECKPOINT_MANIFEST.json').read_text())
checkpoint_manifest.update(weights_included_in_zip=True,weights_archive='Nexus_Last2_vs_Last3_A500_Weights.zip')
write(O/'CHECKPOINT_MANIFEST.json',checkpoint_manifest)
metadata=O/'input_metadata';metadata.mkdir(exist_ok=True)
for name in ['READY.json','overfit100_manifest.csv','data_manifest.csv','selection.json','pool_provenance.json','construction_args.json']:
    p=R/name;(metadata/name).write_bytes(p.read_bytes())
source=R.parent/'decoder_last2_joint_fixed100_20260920'
(metadata/'actual_augmented_pool_source_manifest.json').write_bytes((source/'source_manifest.json').read_bytes())
files={'review':[],'predictions':[],'weights':[],'inputs_caches':[]}
for p in sorted(R.rglob('*')):
    if not p.is_file() or p.is_symlink() or any(x in p.parts for x in ['.git','__pycache__']):continue
    if p.suffix in ['.zip','.pyc'] or p.name.endswith('.zip.tmp') or p.name=='FINAL_DELIVERY.json':continue
    rel=p.relative_to(R).as_posix()
    group='weights' if p.suffix=='.pt' else ('predictions' if 'predictions-new' in rel else 'inputs_caches') if p.suffix=='.npz' else 'review'
    files[group].append((p,rel))
for key,label in [('source_full','A500_full.pt'),('source_checkpoint','A500_optimizer_rng.pt')]:
    p=Path(cfg[key]);assert sha(p)==cfg[key+'_sha256'];files['weights'].append((p,'common_parent/'+label))
data_mapping=[]
for row in csv.DictReader((R/'overfit100_manifest.csv').open()):
    uid=row['uid']
    for kind in ['mesh','topology']:
        p=Path(row[kind+'_path']);digest=sha(p);assert digest==row[kind+'_sha256']
        dest='fixed_inputs/'+uid+'/'+kind+'/'+p.name;files['inputs_caches'].append((p,dest));data_mapping.append({'source':str(p),'archive_path':dest,'sha256':digest})
    p=source/'augmented_pools'/(uid+'_pool.npz');dest='fixed_inputs/'+uid+'/face_pool.npz'
    files['inputs_caches'].append((p,dest));data_mapping.append({'source':str(p),'archive_path':dest,'sha256':sha(p)})
write(metadata/'PATH_MAPPING.json',data_mapping)
files['review'].append((metadata/'PATH_MAPPING.json','repro_outputs/input_metadata/PATH_MAPPING.json'))
outputs=[]
names={'review':'Nexus_Last2_vs_Last3_A500_Review.zip','predictions':'Nexus_Last2_vs_Last3_A500_Predictions.zip','weights':'Nexus_Last2_vs_Last3_A500_Weights.zip','inputs_caches':'Nexus_Last2_vs_Last3_A500_Inputs_Caches.zip'}
for group in ['weights','inputs_caches','predictions','review']:
    target=R/names[group];tmp=target.with_suffix('.zip.tmp');manifest=[]
    assert len({name for p,name in files[group]})==len(files[group])
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for p,name in files[group]:
            digest=sha(p);z.write(p,name);manifest.append({'path':name,'source_path':str(p),'bytes':p.stat().st_size,'sha256':digest})
        z.writestr('PACKAGE_FILE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(tmp) as z:assert z.testzip() is None
    os.replace(tmp,target);outputs.append({'group':group,'path':str(target),'files':len(manifest),'bytes':target.stat().st_size,'sha256':sha(target)})
    print('PACKAGED',group,outputs[-1]['bytes'],'bytes',flush=True)
write(O/'FINAL_DELIVERY.json',{'created_at':datetime.now(timezone.utc).isoformat(),'scope':'all artifacts and experimental code for this paired experiment, plus exact common parent and fixed100 inputs','archives':outputs,'all_regular_experiment_files_included_except':['ZIP containers themselves','pyc and __pycache__','final delivery manifest itself'],'optimizer_updates_in_packaging':0,'other_historical_models_and_CAD50_included':False})
print('ALL FOUR PACKAGES COMPLETE',flush=True)
