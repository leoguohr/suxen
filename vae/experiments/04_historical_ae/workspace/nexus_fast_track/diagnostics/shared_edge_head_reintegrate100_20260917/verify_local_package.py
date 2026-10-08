"""Independent archive/count audit; does not reexecute the network locally."""
from pathlib import Path
import zipfile,hashlib,json
ROOT=Path(__file__).resolve().parent
p=Path('/Users/luthier/Downloads/Nexus_SharedEdgeHead_Reintegrate100_ReadOnly_20260917.zip')
expected=json.loads((ROOT/'package_verification.json').read_text())['sha256']
actual=hashlib.sha256(p.read_bytes()).hexdigest();assert actual==expected
with zipfile.ZipFile(p) as z:
    assert z.testzip() is None
    hashes={}
    for line in z.read('SHA256SUMS.txt').decode().splitlines():
        h,n=line.split('  ',1);assert hashlib.sha256(z.read(n)).hexdigest()==h,n;hashes[n]=h
    complete=json.loads(z.read('run/complete.json'))
    sets={branch:{v['uid']:v for v in map(json.loads,z.read('run/'+branch+'.jsonl').decode().splitlines())} for branch in ['before','after']}
    uids=json.loads(z.read('selection.json'))['uids']
    archived={v['uid']:v for v in map(json.loads,z.read('source_archived_eval.jsonl').decode().splitlines())}
    for branch,rows in sets.items():
        assert set(rows)==set(uids) and len(rows)==100
        summary=dict(meshes=100,edge_perfect=0,face_perfect=0,joint_perfect=0,**{kind+'_'+k:0 for kind in ['edge','face'] for k in ['tp','fp','fn']})
        for uid,x in rows.items():
            n=x['vertices'];e=x['edge'];f=x['face']
            assert sum(e[k] for k in ['tp','fp','fn','tn'])==n*(n-1)//2
            assert e['tp']+e['fn']==x['gt_edges']
            assert f['complete'] and f['tp']+f['fn']==x['gt_faces']
            assert f['scored_candidates']==f['tp']+f['fp']+f['tn']+f['fn']-x['missing_gt_face_candidates']
            assert x['edge_perfect']==(e['fp']==e['fn']==0)
            assert x['face_perfect']==(f['fp']==f['fn']==0)
            assert x['joint_perfect']==(x['edge_perfect'] and x['face_perfect'])
            for k in ['edge_perfect','face_perfect','joint_perfect']:summary[k]+=x[k]
            for kind in ['edge','face']:
                for k in ['tp','fp','fn']:summary[kind+'_'+k]+=x[kind][k]
        assert summary==complete['summaries'][branch]
    for u in uids:
        a,b=sets['before'][u],sets['after'][u]
        assert a['feature_hashes']==b['feature_hashes']
        assert a['face_train_logits_sha256']==b['face_train_logits_sha256']
        for key in ['edge','face_training_pool','parts','margins','gt_face_candidates','missing_gt_face_candidates','joint_perfect']:assert a[key]==archived[u][key]
        for k in ['tp','fp','fn','tn','complete','scored_candidates']:assert a['face'][k]==archived[u]['face'][k]
    for key,lists in complete['retention'].items():
        old={u for u in uids if sets['before'][u][key]};new={u for u in uids if sets['after'][u][key]}
        for k,val in [('before',old),('after',new),('retained',old&new),('lost',old-new),('gained',new-old)]:assert lists[k]==sorted(val)
    install=json.loads(z.read('run/installation_verification.json'))
    assert sorted(install['changed_state_keys'])==['autoencoder.edge_embedding.bias','autoencoder.edge_embedding.weight']
    three=json.loads(z.read('run/three_verification.json'));assert three['passed']
    assert all(all(v for k,v in row.items() if k!='uid') for row in three['meshes'])
    for name in ['REPORT.md','comparison.json','comparison.png','per_mesh_comparison.csv','run/complete.json','run/installation_verification.json','run/three_verification.json']:
        (ROOT/Path(name).name).write_bytes(z.read(name))
    decomposition={br:{'missing_gt_candidate':sum(x['missing_gt_face_candidates'] for x in rows.values()),'present_but_negative':sum(x['face']['fn']-x['missing_gt_face_candidates'] for x in rows.values())} for br,rows in sets.items()}
    audit=dict(zip_sha256=actual,bytes=p.stat().st_size,zip_crc_passed=True,files_verified=len(hashes),two_complete_100_mesh_evaluations=True,
        gt_pair_face_count_identities_passed=True,baseline_matches_archived=True,aggregate_and_retention_independently_recomputed=True,
        feature_hashes_unchanged=True,changed_model_keys_verified_from_server_record=install['changed_state_keys'],
        local_model_reexecution=False,face_fn_decomposition=decomposition,summaries=complete['summaries'])
    (ROOT/'local_independent_verification.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))
