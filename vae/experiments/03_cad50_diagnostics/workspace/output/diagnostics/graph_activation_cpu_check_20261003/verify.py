"""CPU mechanism check; imports archived code read-only, performs no training."""
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path('/Users/luthier/Documents/sophomore')
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'nexus_fast_track/diagnostics/own512_v2_recipe_pair_20260923/native_models.py'
DATA = ROOT / 'nexus_fast_track/diagnostics/teacher_reverse_audit_20260922/ground_truth_export/meshes/teacher_cad50_00.npz'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(tensor):
    return float(tensor.detach().double().norm())


def compare(a, b):
    a, b = a.detach().double().flatten(), b.detach().double().flatten()
    return {'relative_l2_change': float((b-a).norm()/a.norm()),
            'cosine': float(F.cosine_similarity(a, b, dim=0))}


def explicit_block(block, h, graph, order, graph_activation, ffn_activation):
    """All parameters held fixed; these three switches only change operations."""
    message = (block.graph(block.graph_norm(h), graph) if order == 'pre'
               else block.graph_norm(block.graph(h, graph)))
    h = h + getattr(F, graph_activation)(message)
    layer = block.transformer
    h = h + native.math_attention(layer.self_attn, layer.norm1(h))
    return h + layer.linear2(getattr(F, ffn_activation)(layer.linear1(layer.norm2(h))))


torch.set_num_threads(2)
torch.manual_seed(0)
source_sha, data_sha = sha(SOURCE), sha(DATA)
spec = importlib.util.spec_from_file_location('archived_native', SOURCE)
native = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = native
spec.loader.exec_module(native)

with np.load(DATA, allow_pickle=False) as arrays:
    vertices = torch.from_numpy(arrays['vertices'].copy())
    faces = torch.from_numpy(arrays['faces'].copy())
graph = native.Graph.from_faces(faces, len(vertices))
vertex_input, face_input = nn.Linear(3, 512), nn.Linear(3, 512)
h0 = torch.cat((vertex_input(vertices), face_input(vertices[faces].mean(1)))).detach()
base = native.EncoderBlock(native.Config(model_variant='A_v1_recipe_control'))
initial_state = {k: v.clone() for k, v in base.state_dict().items()}
probe = torch.randn_like(h0) / math.sqrt(h0.numel())

reproduction = {}
for variant, order, ga, fa in [('A_v1_recipe_control', 'pre', 'silu', 'relu'),
                              ('B_v2_teacher_blocks', 'post', 'gelu', 'gelu')]:
    block = base if variant.startswith('A') else native.EncoderBlock(native.Config(model_variant=variant))
    x = h0.clone().requires_grad_(True)
    actual = block(x, graph)
    explicit = explicit_block(block, x, graph, order, ga, fa)
    actual_grad, = torch.autograd.grad((actual*probe).sum(), x)
    explicit_grad, = torch.autograd.grad((explicit*probe).sum(), x)
    torch.testing.assert_close(actual, explicit, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, explicit_grad, rtol=0, atol=0)
    reproduction[variant] = {'output_max_abs_diff': float((actual-explicit).detach().abs().max()),
                            'input_vjp_max_abs_diff': float((actual_grad-explicit_grad).abs().max())}

factorial, values = [], {}
for order, ga, fa in itertools.product(['pre', 'post'], ['silu', 'gelu'], ['relu', 'gelu']):
    x = h0.clone().requires_grad_(True)
    output = explicit_block(base, x, graph, order, ga, fa)
    params = list(base.parameters())
    gradients = torch.autograd.grad((output*probe).sum(), [x]+params)
    assert all(torch.isfinite(t).all() for t in [output]+list(gradients))
    key = f'{order}/{ga}/{fa}'
    flat_param_grad = torch.cat([g.detach().flatten() for g in gradients[1:]])
    values[key] = (output.detach(), gradients[0].detach(), flat_param_grad)
    factorial.append({'operations': key, 'output_norm': norm(output),
                      'input_vjp_norm': norm(gradients[0]), 'parameter_vjp_norm': norm(flat_param_grad)})

contrasts = {}
reference = 'post/gelu/gelu'
for label, other in [('only_norm_position', 'pre/gelu/gelu'),
                     ('only_graph_activation', 'post/silu/gelu'),
                     ('only_encoder_ffn_activation', 'post/gelu/relu')]:
    contrasts[label] = {'from': reference, 'to': other,
                       **{name: compare(a, b) for name, a, b in
                          zip(['output', 'input_vjp', 'parameter_vjp'], values[reference], values[other])}}

scales = []
with torch.no_grad():
    for factor in [0.1, 1.0, 10.0]:
        pre = factor * base.graph(base.graph_norm(h0), graph)
        post = base.graph_norm(factor * base.graph(h0, graph))
        scales.append({'projection_weight_and_bias_multiplier': factor,
                       'pre_norm_mean_node_std_before_activation': float(pre.std(-1, unbiased=False).mean()),
                       'post_norm_mean_node_std_before_activation': float(post.std(-1, unbiased=False).mean()),
                       'pre_norm_gelu_message_norm': norm(F.gelu(pre)),
                       'post_norm_gelu_message_norm': norm(F.gelu(post))})

activation_table = []
for point in [-3., -2., -1., -0.5, 0., 0.5, 1., 2., 3.]:
    row = {'input': point}
    for name in ['relu', 'gelu', 'silu']:
        x = torch.tensor(point, dtype=torch.float64, requires_grad=True)
        y = getattr(F, name)(x)
        grad, = torch.autograd.grad(y, x)
        row[name] = {'value': float(y.detach()), 'derivative': float(grad)}
        if point != 0.:
            step = 1e-5
            finite_difference = (getattr(F, name)(x.detach()+step)-getattr(F, name)(x.detach()-step))/(2*step)
            torch.testing.assert_close(grad, finite_difference, rtol=1e-5, atol=1e-8)
    activation_table.append(row)

assert all(torch.equal(initial_state[k], v) for k, v in base.state_dict().items())
assert source_sha == sha(SOURCE) and data_sha == sha(DATA)
result = {
    'scope': 'CPU local forward/VJP mechanism check, NOT reconstruction-benefit validation',
    'optimizer_updates': 0, 'gpu_used': False, 'trained_weights_loaded': False,
    'torch_version': torch.__version__, 'dtype': 'float32 (activation derivative spot checks float64)',
    'seed': 0, 'model_initialization': 'fresh single EncoderBlock and two input projections',
    'mesh': {'path': str(DATA), 'sha256': data_sha, 'uid': DATA.stem,
             'vertices': len(vertices), 'faces': len(faces), 'graph_nodes': len(h0)},
    'source': {'path': str(SOURCE), 'sha256': source_sha, 'unchanged': True},
    'native_formula_reproduction': reproduction,
    'factorial_definition': 'same input, Graph bias, weights, attention, and scalar linear probe; 2x2x2 operation switches',
    'probe_definition': 'dot(block_output, fixed_random_probe); NOT Edge/Face or training loss',
    'factorial': factorial, 'one_factor_contrasts': contrasts,
    'scale_test': scales,
    'scale_test_scope': 'artificial Graph weight+bias scaling, fresh LN gamma=1,beta=0; not trained-model stability',
    'activation_values_and_derivatives': activation_table,
    'caveats': ['No training or actual Edge/Face reconstruction was performed.',
                'Output/gradient changes do not imply better optimization or lower topology errors.',
                'Archived A/B also changes neighbor projection bias, Fourier input, and Decoder FFNs.',
                'Nonzero GELU derivative can be negative; it is not automatically favorable.',
                'All factorial arms hold the original neighbor bias fixed to isolate operation changes.'],
    'all_checks_passed': True,
}
(OUT/'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({k: result[k] for k in ['native_formula_reproduction', 'one_factor_contrasts', 'scale_test', 'all_checks_passed']}, indent=2))
