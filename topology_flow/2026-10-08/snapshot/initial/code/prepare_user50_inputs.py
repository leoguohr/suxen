"""Bind existing Stage3 topology and Stage2 point conditions, with a CPU coordinate audit."""
import argparse
import json
from pathlib import Path
import numpy as np
from _faces_reference import atomic_json, atomic_npz, file_sha
from _fixed100_reference import array_sha
from selected_data import load_selected_dataset, validate_conditions, COORDINATE_FRAME


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-source', required=True)
    parser.add_argument('--stages-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    items, data = load_selected_dataset(args.data_source)
    stages = Path(args.stages_root).resolve()
    out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=False)
    audit_records, conditions = [], []
    for record in data['records']:
        uid = record['uid']; stage2 = stages/'stage2_outputs'/uid
        source = stage2/'condition_point.npz'
        transform_path = stage2/'transform.json'
        transform = json.loads(transform_path.read_text())
        quality2_path = stage2/'quality.json'
        quality2 = json.loads(quality2_path.read_text())
        quality3_path = stages/'stage3_outputs'/uid/'quality.json'
        quality3 = json.loads(quality3_path.read_text())
        if transform['normalization'] != 'aabb_uniform_v1' or not transform['roundtrip_ok']:
            raise ValueError('Unexpected coordinate transform: '+uid)
        mesh2_path = stage2/'mesh_normalized.npz'
        if quality3['input_hashes']['stage2_mesh_file_sha256'] != file_sha(mesh2_path):
            raise ValueError('Stage3 does not bind the inspected Stage2 coordinates: '+uid)
        if quality3['output_file_hashes']['mesh_quantized_training_npz'] != record['mesh_sha256']:
            raise ValueError('Stage3 output differs from original frozen target: '+uid)
        if file_sha(source) != quality2['output_file_hashes']['condition_point_npz']:
            raise ValueError('Existing condition is not bound by its Stage2 quality record: '+uid)
        with np.load(source, allow_pickle=False) as z:
            points, normals, face_ids = (z[k].copy() for k in ('points','normals','sampled_face'))
        with np.load(mesh2_path, allow_pickle=False) as z:
            v2, f2 = z['vertices_norm'].copy(), z['faces'].copy()
        if face_ids.dtype != np.int64 or face_ids.shape != (len(points),) or face_ids.min() < 0 or face_ids.max() >= len(f2):
            raise ValueError('Invalid audit-only source face indices: '+uid)
        with np.load(record['mesh_path'], allow_pickle=False) as z:
            q = z['quantized_vertices'].copy()
        vertices = items[uid]['vertices'].numpy()
        centers = (-1+(q.astype(np.float64)+.5)/256).astype(np.float32)
        if not np.array_equal(vertices, centers):
            raise ValueError('Existing D9 targets are not the documented normalized cell centers: '+uid)
        if points.shape != (8192,3) or normals.shape != points.shape or points.dtype != np.float32 or normals.dtype != np.float32:
            raise ValueError('Unexpected existing XYZ/normal arrays: '+uid)
        triangles = v2.astype(np.float64)[f2[face_ids]]
        cross = np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0])
        expected_normals = cross/np.linalg.norm(cross, axis=1, keepdims=True)
        normal_error = float(np.abs(expected_normals-normals).max())
        plane_error = float(np.abs(np.einsum('ij,ij->i', points-triangles[:,0], expected_normals)).max())
        if not np.isfinite([normal_error,plane_error]).all() or normal_error > 1e-6 or plane_error > 1e-6:
            raise ValueError('Existing point XYZ/normals fail their source coordinate check: '+uid)
        # sampled_face is used only for this read-only audit and is never a model input.
        packed = np.concatenate((points,normals), axis=1)
        mask = np.ones(len(points), dtype=np.bool_)
        path = out/'conditions'/(uid+'.npz')
        atomic_npz(path, points=packed, point_mask=mask)
        vertex_sha = array_sha(vertices)
        conditions.append(dict(uid=uid, file=str(path.relative_to(out)), sha256=file_sha(path), vertices_sha256=vertex_sha))
        audit_records.append(dict(uid=uid, vertices_sha256=vertex_sha, points_sha256=array_sha(packed),
            point_mask_sha256=array_sha(mask), source_kind='existing_pointcloud_xyz_normals',
            source_file=str(source), source_sha256=file_sha(source),
            source_xyz_array_sha256=array_sha(points), source_normal_array_sha256=array_sha(normals),
            transform_file=str(transform_path), transform_sha256=file_sha(transform_path),
            stage2_quality_sha256=file_sha(quality2_path), stage3_quality_sha256=file_sha(quality3_path),
            xyz_transform=dict(existing_frame='Stage2 normalized coordinates', packing='byte-preserving concatenate XYZ+normals',
                center_world=transform['center_world'], half_extent_world=transform['half_extent_world'],
                formula=transform['forward_formula'], mesh_coordinates='existing Stage3 D9 cell centers in the same frame'),
            normal_transform=dict(applied='none', reason='existing unit normals; source normalization is positive uniform scaling'),
            observed_normal_max_abs_error=normal_error, observed_point_plane_max_abs_residual=plane_error,
            gt_vertices_unchanged=True, gt_topology_unchanged=True))
    audit = dict(verified=True, coordinate_frame=COORDINATE_FRAME,
        method='Hash-check existing Stage2 point source and Stage3 mesh; bind Stage3 input to Stage2; verify existing D9 center coordinates, point-on-source-face residual and oriented unit normals without generating new points or changing topology.',
        records=audit_records,
        rope=dict(scale=256., input='existing normalized GT XYZ', units='D9 reference-grid units: one cell is one unit',
            formula='rope_positions = vertices_norm * 256',
            basis='Unchanged apply_vertex_rope expects reference-grid units; existing D9 cell spacing is 1/256; local first-round choice, no scale sweep.'))
    atomic_json(out/'coordinate_audit.json', audit)
    condition_manifest = dict(coordinate_frame=COORDINATE_FRAME, uids=data['uids'], records=conditions,
        coordinate_audit_file='coordinate_audit.json', coordinate_audit_sha256=file_sha(out/'coordinate_audit.json'))
    atomic_json(out/'conditions.json', condition_manifest)
    validate_conditions(out/'conditions.json', items, data['uids'])
    source_manifest = dict(schema='selected_mesh_source_v1', coordinate_frame=COORDINATE_FRAME,
        original_source=data['source'], selection_sha256=data['selection_sha256'], records=data['records'])
    atomic_json(out/'data_source.json', source_manifest)
    atomic_json(out/'data_identity.json', data)
    result = dict(complete=True, meshes=50, totals=data['totals'],
        data_source_sha256=file_sha(out/'data_source.json'), conditions_sha256=file_sha(out/'conditions.json'),
        coordinate_audit_sha256=file_sha(out/'coordinate_audit.json'), gpu_used=False, rope_scale=256.)
    atomic_json(out/'preparation.json', result); print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
