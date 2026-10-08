"""Hash-bound user-selected D9 meshes; never requantize or reorder vertices."""
import json
from pathlib import Path
import numpy as np
import torch
from _faces_reference import file_sha
from _fixed100_reference import array_sha, load_dataset


SELECTION_PATH = Path(__file__).resolve().parent.parent/'configs/selection50.json'
SELECTION_SHA256 = 'ee2588bcd212e750893904973b4aee4e3a40d3295f35b2d923f6d92076fa0ac4'
COORDINATE_FRAME = 'same_as_original_vertices_norm'


def load_selection(path=SELECTION_PATH):
    if file_sha(path) != SELECTION_SHA256:
        raise ValueError('Selection differs from the fixed user-supplied 50 D9 meshes')
    return json.loads(Path(path).read_text())


def mesh_arrays(vertices, faces, edges):
    """Canonical sets are labels only; original vertices and faces are retained."""
    n = len(vertices)
    if n == 0 or vertices.dtype != np.float32 or vertices.shape != (n, 3) or not np.isfinite(vertices).all():
        raise ValueError('Source vertices must be finite FP32 [N,3]')
    for name, ids, width in (('faces', faces, 3), ('edges', edges, 2)):
        if ids.dtype != np.int64 or ids.ndim != 2 or ids.shape[1] != width or not len(ids):
            raise ValueError('Invalid source '+name)
        if ids.min() < 0 or ids.max() >= n or (np.diff(np.sort(ids, axis=1), axis=1) <= 0).any():
            raise ValueError('Out-of-range or degenerate source '+name)
    gt_faces = np.unique(np.sort(faces, axis=1), axis=0)
    derived = np.unique(np.sort(np.concatenate((faces[:, [0,1]], faces[:, [0,2]], faces[:, [1,2]])), axis=1), axis=0)
    if len(gt_faces) != len(faces) or not np.array_equal(edges, derived):
        raise ValueError('Source Edge/Face labels are inconsistent')
    return dict(vertices=vertices, faces=faces, edges=edges, gt_faces=gt_faces,
                vertex_indices=np.arange(n, dtype=np.int64))


def read_mesh_record(record, root):
    paths = {kind: (root/record[kind+'_path']).resolve() for kind in ('mesh', 'topology')}
    for kind, path in paths.items():
        if file_sha(path) != record[kind+'_sha256']:
            raise ValueError('Source file hash mismatch: '+record['uid']+' '+kind)
    with np.load(paths['mesh'], allow_pickle=False) as a, np.load(paths['topology'], allow_pickle=False) as b:
        vertices, faces = a['vertices_norm'].copy(), a['faces'].copy()
        raw_edges, raw_faces = b['edge_index'].copy(), b['face_set'].copy()
    if raw_edges.dtype != np.int64 or raw_edges.ndim != 2 or raw_edges.shape[0] != 2:
        raise ValueError('Source edge_index must be int64 [2,E]')
    if raw_faces.dtype != np.int64 or raw_faces.ndim != 2 or raw_faces.shape[1] != 3:
        raise ValueError('Source face_set must be int64 [F,3]')
    edges = np.unique(np.sort(raw_edges.T, axis=1), axis=0)
    arrays = mesh_arrays(vertices, faces, edges)
    if not np.array_equal(arrays['gt_faces'], np.unique(np.sort(raw_faces, axis=1), axis=0)):
        raise ValueError('Mesh/topology Face labels disagree: '+record['uid'])
    provenance = dict(record, **{k+'_path':str(p) for k,p in paths.items()},
        array_sha256={k:array_sha(v) for k,v in arrays.items()},
        source_array_sha256=dict(vertices_norm=array_sha(vertices), faces=array_sha(faces),
                                 edge_index=array_sha(raw_edges), face_set=array_sha(raw_faces)))
    return arrays, provenance


def load_selected_dataset(source, source_sha256=None, selection_path=SELECTION_PATH):
    selection = load_selection(selection_path)
    uids = [r['uid'] for r in selection['records']]
    source = Path(source).resolve()
    if source.is_dir():
        # Preserve the original fixed100 selection/manifest/totals verification.
        _, original = load_dataset(source)
        lookup = {r['uid']:r for r in original['records']}
        if not set(uids).issubset(lookup):
            raise ValueError('Verified original dataset does not contain every selected UID')
        records = [lookup[uid] for uid in uids]
        root = source
        source_identity = dict(kind='subset_of_verified_fixed100', original=original)
    else:
        if not source_sha256 or file_sha(source) != source_sha256:
            raise ValueError('Explicit source manifest needs its expected SHA256')
        manifest = json.loads(source.read_text())
        if manifest.get('schema') != 'selected_mesh_source_v1' or manifest.get('coordinate_frame') != COORDINATE_FRAME:
            raise ValueError('Unsupported selected-data source manifest or coordinate frame')
        records = manifest['records']
        root = source.parent
        source_identity = dict(kind='explicit_hashed_manifest', path=str(source), sha256=source_sha256)
    if [r['uid'] for r in records] != uids:
        raise ValueError('Source records must match the exact user selection order without duplicates')
    items, checked = {}, []
    totals = dict(meshes=len(uids), vertices=0, edges=0, faces=0, pairs=0)
    for selected, record in zip(selection['records'], records):
        arrays, provenance = read_mesh_record(record, root)
        if len(arrays['vertices']) != selected['vertices']:
            raise ValueError('User D9 vertex count mismatch: '+selected['uid'])
        counts = dict(vertices=len(arrays['vertices']), edges=len(arrays['edges']),
                      faces=len(arrays['faces']), pairs=len(arrays['vertices'])*(len(arrays['vertices'])-1)//2)
        for key, value in counts.items(): totals[key] += value
        provenance['counts'] = counts
        checked.append(provenance)
        items[selected['uid']] = dict(uid=selected['uid'], **{k:torch.from_numpy(v) for k,v in arrays.items()})
    if any(totals[k] != value for k,value in selection['totals'].items()):
        raise ValueError('Selected dataset totals mismatch')
    return items, dict(uids=uids, totals=totals, selection_sha256=SELECTION_SHA256,
        selection=selection, source=source_identity, coordinate_frame=COORDINATE_FRAME, records=checked)


def validate_conditions(path, items, uids, min_points=1024):
    """Bind inspected coordinate evidence and preexisting real point-cloud files."""
    path = Path(path).resolve()
    conditions = json.loads(path.read_text())
    if conditions.get('coordinate_frame') != COORDINATE_FRAME or conditions['uids'] != uids:
        raise ValueError('Point cloud coordinate frame or UID order mismatch')
    records = conditions['records']
    if [r['uid'] for r in records] != uids:
        raise ValueError('Point-cloud records mismatch or duplicate UID')
    audit_path = (path.parent/conditions['coordinate_audit_file']).resolve()
    if file_sha(audit_path) != conditions['coordinate_audit_sha256']:
        raise ValueError('Coordinate audit hash mismatch')
    audit = json.loads(audit_path.read_text())
    if (audit.get('verified') is not True or audit.get('coordinate_frame') != COORDINATE_FRAME
            or not audit.get('method') or [r['uid'] for r in audit['records']] != uids):
        raise ValueError('Missing verified, per-UID coordinate audit')
    for record, evidence in zip(records, audit['records']):
        uid = record['uid']
        source_path = (path.parent/record['file']).resolve()
        if file_sha(source_path) != record['sha256']:
            raise ValueError('Condition file hash mismatch: '+uid)
        with np.load(source_path, allow_pickle=False) as a:
            points, mask = a['points'].copy(), a['point_mask'].copy()
        if points.dtype != np.float32 or points.ndim != 2 or points.shape[1] != 6:
            raise ValueError('Conditions must be FP32 [P,6] existing XYZ+normals')
        if mask.dtype != np.bool_ or mask.shape != points.shape[:1] or int(mask.sum()) < min_points:
            raise ValueError('Insufficient valid real points: '+uid)
        if not np.isfinite(points[mask]).all() or (np.linalg.norm(points[mask,3:], axis=1) <= 0).any():
            raise ValueError('Nonfinite condition or missing normals: '+uid)
        expected = dict(vertices_sha256=array_sha(items[uid]['vertices'].numpy()),
                        points_sha256=array_sha(points), point_mask_sha256=array_sha(mask))
        if record['vertices_sha256'] != expected['vertices_sha256'] or any(evidence.get(k) != v for k,v in expected.items()):
            raise ValueError('Coordinate audit does not bind these exact arrays: '+uid)
        if evidence.get('source_kind') != 'existing_pointcloud_xyz_normals' or not evidence.get('xyz_transform') or not evidence.get('normal_transform'):
            raise ValueError('Missing real point-cloud or transform provenance: '+uid)
        raw_path = (audit_path.parent/evidence['source_file']).resolve()
        if file_sha(raw_path) != evidence['source_sha256']:
            raise ValueError('Original point-cloud source hash mismatch: '+uid)
    return conditions, dict(path=str(audit_path), sha256=file_sha(audit_path), method=audit['method'])
