"""Temporary CPU fixtures for the immutable user50 source/cache contract."""
import copy
import json
import tempfile
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
from _faces_reference import atomic_json, atomic_npz, file_sha
from _fixed100_reference import array_sha
from _vae_reference import NativeTopologyAE, Config
from export_cache import export_item, check_saved_record
from latent_data import LatentCache, posterior_statistics, normalization_sha256
from selected_data import load_selection, load_selected_dataset, validate_conditions, SELECTION_SHA256, COORDINATE_FRAME
from vae_codec import SOURCE_CHECKPOINT_SHA256
import export_cache
import gpu_budget


def rejected(fn):
    try: fn()
    except (ValueError, KeyError): return
    raise AssertionError('Invalid input was accepted')


def main():
    torch.set_num_threads(1)
    torch.manual_seed(1)
    tests = []
    selection = load_selection()
    uids = [r['uid'] for r in selection['records']]
    assert len(uids) == len(set(uids)) == 50
    assert selection['totals'] == dict(meshes=50, vertices=52820, pairs=41866721)
    changed = next(r for r in selection['records'] if r['uid'] == 'nexus_2k_001825')
    assert changed['vertices'] == 1277 and changed['user_reported_d9_vertices'] == 1279
    tests.append('exact user50 order/counts and authorized 1279-to-1277 source discrepancy retained')
    with tempfile.TemporaryDirectory(prefix='topology-user50-cpu-') as tmp:
        root = Path(tmp)
        records, conditions, evidence = [], [], []
        faces = np.array([[0,1,2], [1,2,3]], dtype=np.int64)
        edges = np.unique(np.sort(np.concatenate((faces[:,[0,1]], faces[:,[0,2]], faces[:,[1,2]])), axis=1), axis=0)
        for i,selected in enumerate(selection['records']):
            uid = selected['uid']
            vertices = np.random.default_rng(i).standard_normal((selected['vertices'],3)).astype(np.float32)
            mesh, topo, pc = root/(uid+'-mesh.npz'), root/(uid+'-topology.npz'), root/(uid+'-points.npz')
            atomic_npz(mesh, vertices_norm=vertices, faces=faces)
            atomic_npz(topo, edge_index=edges.T, face_set=faces)
            records.append(dict(uid=uid,mesh_path=mesh.name,mesh_sha256=file_sha(mesh),topology_path=topo.name,topology_sha256=file_sha(topo)))
            points = np.random.default_rng(i+100).standard_normal((1024,6)).astype(np.float32)
            mask = np.ones(1024,dtype=np.bool_)
            atomic_npz(pc,points=points,point_mask=mask)
            conditions.append(dict(uid=uid,file=pc.name,sha256=file_sha(pc),vertices_sha256=array_sha(vertices)))
            evidence.append(dict(uid=uid,vertices_sha256=array_sha(vertices),points_sha256=array_sha(points),point_mask_sha256=array_sha(mask),
                source_kind='existing_pointcloud_xyz_normals',source_file=pc.name,source_sha256=file_sha(pc),
                xyz_transform={'fixture':'identity'},normal_transform={'fixture':'identity'}))
        source = root/'source.json'
        source_data = dict(schema='selected_mesh_source_v1',coordinate_frame=COORDINATE_FRAME,records=records)
        atomic_json(source, source_data)
        items, data = load_selected_dataset(source,file_sha(source))
        assert list(items) == uids and data['selection_sha256'] == SELECTION_SHA256
        assert all(len(items[r['uid']]['vertices']) == r['vertices'] for r in selection['records'])
        rejected(lambda:load_selected_dataset(source,'0'*64))
        malformed = copy.deepcopy(source_data); malformed['records'][1]['uid'] = uids[0]
        atomic_json(source,malformed); rejected(lambda:load_selected_dataset(source,file_sha(source)))
        atomic_json(source,source_data)
        original = (root/records[0]['mesh_path']).read_bytes()
        atomic_npz(root/records[0]['mesh_path'],vertices_norm=np.zeros((19,3),np.float32),faces=faces)
        malformed = copy.deepcopy(source_data); malformed['records'][0]['mesh_sha256'] = file_sha(root/records[0]['mesh_path'])
        atomic_json(source,malformed); rejected(lambda:load_selected_dataset(source,file_sha(source)))
        (root/records[0]['mesh_path']).write_bytes(original); atomic_json(source,source_data)
        tests.append('hashed source contract preserves 50 exact meshes; rejects source hash, duplicate UID and D9 count changes')

        audit = root/'coordinate-audit.json'; condition_path = root/'conditions.json'
        atomic_json(audit,dict(verified=True,coordinate_frame=COORDINATE_FRAME,method='Synthetic fixture binding test, not real coordinate validation',records=evidence))
        condition_data = dict(coordinate_frame=COORDINATE_FRAME,uids=uids,records=conditions,
            coordinate_audit_file=audit.name,coordinate_audit_sha256=file_sha(audit))
        atomic_json(condition_path,condition_data)
        validate_conditions(condition_path,items,uids)
        malformed = json.loads(audit.read_text()); malformed['records'][0]['vertices_sha256']='0'*64
        atomic_json(audit,malformed); condition_data['coordinate_audit_sha256']=file_sha(audit); atomic_json(condition_path,condition_data)
        rejected(lambda:validate_conditions(condition_path,items,uids))
        tests.append('real-condition provenance/coordinate audit binds exact XYZ, normals, masks and mesh arrays')
        atomic_json(audit,dict(verified=True,coordinate_frame=COORDINATE_FRAME,method='Synthetic fixture binding test, not real coordinate validation',records=evidence))
        condition_data['coordinate_audit_sha256']=file_sha(audit);atomic_json(condition_path,condition_data)

        vae = NativeTopologyAE(Config(model_variant='B_v2_teacher_blocks',encoder_width=24,decoder_width=24,heads=3,
            latent_width=512,encoder_composite_blocks=1,decoder_blocks=1,activation_checkpointing=False)).eval().requires_grad_(False)
        with np.load(root/conditions[0]['file']) as p:
            cached = root/'roundtrip.npz'
            arrays, errors = export_item(vae,items[uids[0]],p['points'].copy(),p['point_mask'].copy(),cached)
        assert len(errors) == 6 and all(v == 0 for v in errors.values())
        record = dict(uid=uids[0],file=cached.name,sha256=file_sha(cached),vertices_sha256=array_sha(arrays['vertices']),
                      array_sha256={k:array_sha(v) for k,v in arrays.items()})
        moment = check_saved_record(root,record)
        assert moment[0].shape == moment[1].shape == (512,)
        broken = dict(record,sha256='0'*64); rejected(lambda:check_saved_record(root,broken))
        tests.append('NPZ save/readback arrays and re-decoded Edge/Face/hidden tensors equal original VAE exactly')

        class OneMeshBudget:
            def __init__(self, *args, **kwargs): self.calls = 0
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def stop(self, reserve_seconds=0):
                self.calls += 1
                return self.calls >= 3
            def snapshot(self): return dict(gpu_used=False, synthetic_stop=True)
        checkpoint = root/'synthetic-checkpoint';checkpoint.write_bytes(b'CPU fixture only')
        command = ['export_cache.py','--data-source',str(source),'--data-source-sha256',file_sha(source),
            '--conditions',str(condition_path),'--vae-checkpoint',str(checkpoint),'--output',str(root/'export'), '--device','cpu']
        with patch.object(export_cache,'SOURCE_CHECKPOINT_SHA256',file_sha(checkpoint)), patch.object(export_cache,'load_frozen_vae',return_value=vae), patch.object(gpu_budget,'GPUBudget',OneMeshBudget):
            with patch('sys.argv',command): export_cache.main()
            first = json.loads((root/'export/manifest.json').read_text())
            assert not first['complete'] and len(first['records']) == 1 and 'normalization' not in first
            with patch('sys.argv',command+['--resume']): export_cache.main()
            resumed = json.loads((root/'export/manifest.json').read_text())
        assert not resumed['complete'] and len(resumed['records']) == 2 and 'normalization' not in resumed
        assert resumed['records'][0] == first['records'][0]
        assert [r['uid'] for r in resumed['records']] == uids[:2]
        tests.append('budget-stopped export resumes at the next UID without rewriting previous records or finalizing partial statistics')

        stats = posterior_statistics([(torch.from_numpy(arrays['mu']),torch.from_numpy(arrays['logvar']))]*50)
        manifest = dict(complete=True,uids=uids,source_checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,data=data,
            records=[dict(record,uid=uid) for uid in uids],normalization=stats,normalization_sha256=normalization_sha256(stats),synthetic_fixture=True)
        atomic_json(root/'manifest.json',manifest)
        cache = LatentCache(root)
        assert len(cache.get(uids[0])['vertices']) == selection['records'][0]['vertices']
        bad = copy.deepcopy(manifest);bad['normalization']['mean'][0] += 1
        atomic_json(root/'manifest.json',bad); rejected(lambda:LatentCache(root))
        atomic_json(root/'manifest.json',manifest)
        changed_arrays = dict(arrays);changed_arrays['vertices'] = arrays['vertices'][::-1].copy()
        atomic_npz(cached,**changed_arrays)
        changed_record = dict(record,sha256=file_sha(cached),vertices_sha256=array_sha(changed_arrays['vertices']),
                             array_sha256={k:array_sha(v) for k,v in changed_arrays.items()})
        manifest['records'][0]=changed_record;atomic_json(root/'manifest.json',manifest)
        rejected(lambda:LatentCache(root).get(uids[0]))
        tests.append('production cache validates fixed normalization hash and unchanged source vertex/topology binding')
    result = dict(passed=True,device='cpu',tests=tests,real_source_checkpoint_loaded=False,
                  real_mesh_exported=False,gpu_used=False,temporary_fixtures_removed=True)
    atomic_json(Path(__file__).resolve().parent.parent/'evidence/selected50_cpu_tests.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
