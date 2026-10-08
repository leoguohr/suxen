"""Verify paired schedules and archive evaluation evidence, excluding large weights."""
import csv
import json
from pathlib import Path
import zipfile
from support import MODES, read, sha, write


def main():
    root=Path(__file__).resolve().parent
    out=root/'repro_outputs'
    results={}
    schedules={}
    flat=[]
    for arm in MODES:
        folder=root/'runs'/arm
        if not (folder/'complete.json').exists():
            results[arm]=dict(complete=False,failure=read(folder/'failure.json') if (folder/'failure.json').exists() else None)
            continue
        with (folder/'updates.jsonl').open() as stream:
            rows=[json.loads(line) for line in stream]
        assert [r['update'] for r in rows]==list(range(1,2001))
        assert all(len(r['uids'])==5 for r in rows)
        assert all(r['meshes'][i]['uid']==r['uids'][i] for r in rows for i in range(5))
        schedules[arm]=[(r['uids'],[m['negative_sha256'] for m in r['meshes']]) for r in rows]
        evaluations=[]
        for path in sorted((folder/'evaluations').glob('step-*/evaluation.json')):
            e=read(path)
            assert e['complete'] and len(e['meshes'])==50
            step=int(path.parent.name.split('-')[1])
            evaluations.append(dict(step=step,counts=e['counts'],joint_strict=e['joint_strict'],
                                    large16_joint_strict=e['large16']['joint_strict'],checkpoint=e['checkpoint']))
            for mesh in e['meshes']:
                flat.append(dict(arm=arm,step=step,uid=mesh['uid'],vertices=mesh['vertices'],
                    edge_fp=mesh['edge']['fp'],edge_fn=mesh['edge']['fn'],face_fp=mesh['face']['fp'],
                    face_fn=mesh['face']['fn'],face_fn_missing=mesh['face_fn_missing'],
                    face_fn_present=mesh['face_fn_present'],joint_strict=mesh['joint_strict']))
        assert [e['step'] for e in evaluations]==[0,500,1000,1500,2000]
        results[arm]=dict(complete=True,completion=read(folder/'complete.json'),evaluations=evaluations,
            best_face_f1=max(evaluations,key=lambda e:e['counts']['face']['micro_f1']),
            best_joint=max(evaluations,key=lambda e:e['joint_strict']))
    if len(schedules)==4:
        assert all(x==schedules['V2_control'] for x in schedules.values())
    write(out/'comparison.json',dict(arms=results,all_4_paired_UID_and_negative_schedules_equal=len(schedules)==4,
        interpretation='One seed, fixed 2000-update budget; isolated training switches within V2, not proof of global cause'))
    if flat:
        with (out/'per_mesh.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(flat[0]))
            writer.writeheader();writer.writerows(flat)
    lines=['# 四支Graph与Encoder激活训练对照','',
           '本轮在V2内逐项撤回操作；随机参数完全相同，各2000次五mesh更新。每条CAD参与200次。', '',
           '| 分支 | 完成更新 | Edge FP/FN | 实际Face FP/FN | Face micro-F1 | 联合严格成功 | 困难16条成功 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for arm,r in results.items():
        if not r['complete']:
            lines.append(f'| {arm} | 未完成 | — | — | — | — | — |');continue
        e=r['evaluations'][-1];edge=e['counts']['edge'];face=e['counts']['face']
        lines.append(f"| {arm} | 2000 | {edge['fp']}/{edge['fn']} | {face['fp']}/{face['fn']} | {face['micro_f1']:.9f} | {e['joint_strict']}/50 | {e['large16_joint_strict']}/16 |")
    lines += ['', '单种子有限预算比较。应结合末尾与各完整检查点判断；有限预算失败不证明容量不足。',
              '原100条、VAE、diffusion及历史模型均未修改。大checkpoint保存在各runs目录，文件大小和SHA见checkpoint JSON；本包不含大权重。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    included=[]
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix not in ('.pt','.zip','.pyc') and '.git' not in path.parts and path.name != 'SHA256SUMS.json':
            if path.stat().st_size > 1000*1024*1024: continue
            included.append(path)
    write(out/'SHA256SUMS.json',{str(p.relative_to(root)):sha(p) for p in included})
    included.append(out/'SHA256SUMS.json')
    archive=root/'cad50_graph_activation_evidence.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for path in included:z.write(path,str(path.relative_to(root)))
    write(root/'DELIVERY.json',dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive)))
    print('PACKAGE',archive,archive.stat().st_size,flush=True)


if __name__=='__main__': main()
