"""Read-only state and code identity inspection."""
from pathlib import Path
import hashlib, json, sys
import torch
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'code'))
from models import load_ae, load_points, load_topology

records = {}
for name, path, loader in [
    ('teacher_ae','results/minkowski_target_099/vae.pt',lambda p:load_ae(p)[0]),
    ('teacher_topology_flow','results/minkowski_target_099/best_flow.pt',load_topology),
    ('teacher_point_model','results/point_diffusion/latest.pt',load_points),
]:
    p = ROOT/'teacher_assets'/path
    m = loader(p)
    records[name] = dict(parameters=sum(t.numel() for t in m.parameters()),
        tensor_state_elements=sum(t.numel() for t in m.state_dict().values() if torch.is_tensor(t)),
        strict_load=True, checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    if name == 'teacher_ae':
        records[name].update(graph_blocks=len(m.graph), encoder_transformers=len(m.attn), decoder_blocks=len(m.decode_blocks), latent=64, width=128)
    if name == 'teacher_point_model':
        records[name]['residual_scale'] = float(m.residual_scale.detach())
path = ROOT.parent/'teacher_cad50_lr03_pair_20260921/B_lr03/checkpoint-new0500-step2500.pt'
cp = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
state = cp['model']
records['our_B2500'] = dict(tensor_state_elements=sum(t.numel() for t in state.values() if torch.is_tensor(t)),
    tensor_keys=sum(torch.is_tensor(t) for t in state.values()), completed_updates=cp['completed_updates'],
    source=str(path), config=cp['config'])
(ROOT/'repro_outputs/MODEL_COMPARISON_METADATA.json').write_text(json.dumps(records, indent=2, default=str))
print(json.dumps({k:{a:b for a,b in v.items() if a!='config'} for k,v in records.items()}))
