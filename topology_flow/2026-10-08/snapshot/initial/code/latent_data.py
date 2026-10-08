"""Frozen posterior cache and reversible, equal-mesh channel normalization."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from _faces_reference import file_sha
from _fixed100_reference import array_sha
from vae_codec import SOURCE_CHECKPOINT_SHA256
from selected_data import load_selection, SELECTION_SHA256, mesh_arrays


def normalization_sha256(stats):
    return hashlib.sha256(json.dumps(stats, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def posterior_statistics(posteriors, std_floor=1e-6):
    """Analytic E[z], E[z²] for z~q(z|mesh); each mesh has equal mass."""
    means, seconds = [], []
    for mu, logvar in posteriors:
        if mu.shape != logvar.shape or mu.ndim != 2 or mu.shape[1] != 512 or not len(mu):
            raise ValueError('Expected nonempty per-vertex latent512 posteriors')
        mu, logvar = mu.double(), logvar.double()
        if not torch.isfinite(mu).all() or not torch.isfinite(logvar).all():
            raise ValueError('Nonfinite posterior')
        means.append(mu.mean(0))
        seconds.append((mu.square()+logvar.exp()).mean(0))
    return statistics_from_moments(means, seconds, std_floor)


def statistics_from_moments(means, seconds, std_floor=1e-6):
    if not means or len(means) != len(seconds) or not np.isfinite(std_floor) or std_floor <= 0:
        raise ValueError('Invalid posterior moments or normalization floor')
    mean = torch.stack(means).mean(0)
    variance = (torch.stack(seconds).mean(0)-mean.square()).clamp_min(0)
    if mean.shape != (512,) or not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
        raise ValueError('Nonfinite or non-latent512 normalization')
    std = variance.sqrt()
    return dict(mean=mean.float().tolist(), std=std.clamp_min(std_floor).float().tolist(),
                meshes=len(means), weighting='equal_mesh_then_vertex', target='fresh_posterior',
                std_floor=std_floor, floored_channels=int((std < std_floor).sum()))


def transform_latent(value, stats, inverse=False, mask=None):
    mean = value.new_tensor(stats['mean'])
    std = value.new_tensor(stats['std'])
    if value.shape[-1] != len(mean) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError('Invalid latent normalization')
    result = value*std+mean if inverse else (value-mean)/std
    return result if mask is None else result.masked_fill(~mask.unsqueeze(-1), 0)


def validate_arrays(a):
    n = len(a['vertices'])
    mesh_arrays(a['vertices'], a['faces'], a['edges'])
    for key in ('mu', 'logvar'):
        if a[key].dtype != np.float32 or a[key].shape != (n, 512):
            raise ValueError('Posterior cache must be FP32 [N,512]')
    if a['points'].dtype != np.float32 or a['points'].ndim != 2 or a['points'].shape[1] != 6:
        raise ValueError('Conditions must be FP32 [P,6] XYZ+normal, not fabricated fields')
    if a['point_mask'].dtype != np.bool_ or a['point_mask'].shape != a['points'].shape[:1] or not a['point_mask'].any():
        raise ValueError('Invalid point mask')
    for key in ('vertices', 'mu', 'logvar'):
        if not np.isfinite(a[key]).all(): raise ValueError('Nonfinite '+key)
    if not np.isfinite(a['points'][a['point_mask']]).all(): raise ValueError('Nonfinite condition')
    if a['vertex_indices'].dtype != np.int64 or not np.array_equal(a['vertex_indices'], np.arange(n, dtype=np.int64)):
        raise ValueError('Local vertex order changed')


class LatentCache:
    """CPU-only memoization is allowed because the source VAE is entirely frozen."""
    def __init__(self, directory, expected_meshes=50, source_sha256=SOURCE_CHECKPOINT_SHA256):
        self.root = Path(directory)
        self.sha256 = file_sha(self.root/'manifest.json')
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        m = self.manifest
        if not m['complete'] or m['source_checkpoint_sha256'] != source_sha256:
            raise ValueError('Incomplete cache or wrong frozen VAE')
        self.uids = m['uids']
        if len(self.uids) != len(set(self.uids)) or len(self.uids) != expected_meshes:
            raise ValueError('Cache UID count/uniqueness mismatch')
        self.records = {r['uid']: r for r in m['records']}
        if list(self.records) != self.uids or len(m['records']) != len(self.uids):
            raise ValueError('Cache record order/uniqueness mismatch')
        self.selected = None
        if expected_meshes == 50:
            selection = load_selection()
            self.selected = {r['uid']:r for r in selection['records']}
            if self.uids != list(self.selected) or m['data']['selection_sha256'] != SELECTION_SHA256:
                raise ValueError('Cache differs from the fixed user selection')
            if any(m['data']['totals'][k] != v for k,v in selection['totals'].items()):
                raise ValueError('Selected50 cache totals mismatch')
            self.source_records = {r['uid']:r for r in m['data']['records']}
            if list(self.source_records) != self.uids or len(m['data']['records']) != len(self.uids):
                raise ValueError('Cache source record order/uniqueness mismatch')
            if m.get('normalization_sha256') != normalization_sha256(m['normalization']):
                raise ValueError('Fixed normalization hash mismatch')
        elif expected_meshes == 100 and m['data']['totals'] != dict(meshes=100,vertices=106325,edges=309194,faces=204330,pairs=84669234):
            raise ValueError('Original fixed100 totals mismatch')
        self.stats = m['normalization']
        if (self.stats['target'] != 'fresh_posterior' or self.stats['weighting'] != 'equal_mesh_then_vertex'
                or self.stats['meshes'] != expected_meshes):
            raise ValueError('Incompatible normalization definition')
        if len(self.stats['mean']) != 512 or len(self.stats['std']) != 512:
            raise ValueError('Normalization must be latent512')
        transform_latent(torch.zeros(1, 512), self.stats)
        self.items = {}

    def get(self, uid):
        if uid not in self.items:
            r = self.records[uid]
            path = (self.root/r['file']).resolve()
            if not path.is_relative_to(self.root.resolve()) or file_sha(path) != r['sha256']:
                raise ValueError('Cache member path or hash mismatch')
            with np.load(path, allow_pickle=False) as f: a = {k:f[k].copy() for k in f.files}
            validate_arrays(a)
            if array_sha(a['vertices']) != r['vertices_sha256']: raise ValueError('Vertex binding mismatch')
            if self.selected is not None:
                if len(a['vertices']) != self.selected[uid]['vertices']:
                    raise ValueError('Cached D9 vertex count mismatch: '+uid)
                hashes = {k:array_sha(v) for k,v in a.items()}
                if hashes != r['array_sha256']:
                    raise ValueError('Cached array hashes mismatch: '+uid)
                original = self.source_records[uid]['array_sha256']
                if any(hashes[k] != original[k] for k in ('vertices', 'faces', 'edges', 'vertex_indices')):
                    raise ValueError('Cache changed original topology/coordinates/order: '+uid)
            self.items[uid] = {k:torch.from_numpy(v) for k,v in a.items()}
        return self.items[uid]
