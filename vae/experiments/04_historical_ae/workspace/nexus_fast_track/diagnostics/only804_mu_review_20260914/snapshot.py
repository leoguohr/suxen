"""Export the fixed only804_mu checkpoint; no optimizer is constructed."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import sys, json, inspect, shutil, hashlib, datetime, textwrap, ast
from pathlib import Path
BASE = Path('/guohaoran/nexus_fast_track')
REF = BASE/'diagnostics/math00_four_mesh_lr10x_20260913'
BRANCH = REF/'only804_mu'
OUT = Path(__file__).resolve().parent/'package'
SNAP = OUT/'C_snapshot804_step1000'
CODE = SNAP/'effective_code'
CODE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REF))
# Redirect the helper's import-time source export, without touching the live helper.
original_write_text = Path.write_text
def redirected_write(path, *args, **kwargs):
    if path == REF/'loss_changes.txt':
        path = CODE/'loss_changes_exact_runtime.txt'
    return original_write_text(path, *args, **kwargs)
Path.write_text = redirected_write
try:
    import backend00 as b
finally:
    Path.write_text = original_write_text
T, np, c = b.torch, b.np, b.c
def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str)+'\n')
def array(t):
    return t.detach().cpu().numpy()
def digest(path):
    return c.m.probe.digest(path)
def tensor_digest(t):
    return hashlib.sha256(array(t).tobytes()).hexdigest()
def source(fn):
    try:
        return textwrap.dedent(inspect.getsource(fn))
    except TypeError:
        # Dynamically imported helper classes are not always in sys.modules.
        text=Path(fn.forward.__code__.co_filename).read_text()
        node=next(x for x in ast.parse(text).body if isinstance(x,ast.ClassDef) and x.name==fn.__name__)
        return ast.get_source_segment(text,node)+'\n'
T.set_num_threads(1)
T.cuda.set_device(0)
T.backends.cuda.matmul.allow_tf32 = False
T.backends.cudnn.allow_tf32 = False
T.set_float32_matmul_precision('highest')
checkpoint = BRANCH/'checkpoint-update1000.pt'
meta = json.loads((BRANCH/'manifest.json').read_text())
for path, sha in meta['source_sha256'].items():
    assert digest(path) == sha, path
assert digest(REF/'backend00.py') == meta['backend_sha']
c.UIDS = meta['selected_uids']
c.m.UIDS = c.UIDS
c.m.probe.UIDS = c.UIDS
cp, model, batch = c.m.probe.setup_model(checkpoint)
assert cp['additional_updates'] == 1000 and cp['completed_updates'] == 4400
assert c.UIDS[3] == 'nexus_2k_001333'
checkpoint_sha = digest(checkpoint)
b.ROOT = SNAP
b.backend.ROOT = SNAP
(SNAP/'C_graph_only').mkdir(exist_ok=True)
c.m.install_clamp(model, -20.)
# Use the already exported, actually executed branch forward, including its RNG branch.
runtime = (BRANCH/'sampling_forward.py').read_text()
exec(compile(runtime, str(CODE/'sampling_forward.py'), 'exec'), b.backend.b.flash.__dict__)
(CODE/'sampling_forward.py').write_text(runtime)
rng = T.Generator(device='cuda')
rng.set_state(cp['training_rng_state'])
model.autoencoder.diagnostic_train_rng = rng
model.autoencoder.diagnostic_train_draws = cp['training_noise_draws']
b.install_deterministic(model)
for section, blocks in [('encoder', model.autoencoder.encoder_blocks), ('decoder', model.autoencoder.decoder_blocks)]:
    for i, block in enumerate(blocks):
        for name, module in block.named_modules():
            if isinstance(module, T.nn.MultiheadAttention):
                b.ATTENTION_NAMES[id(module)] = f'{section}_{i:02d}/{name}'
assert len(b.ATTENTION_NAMES) == 28
model.requires_grad_(True)
model.autoencoder.log_variance.requires_grad_(False)
data = [np.load(REF/(uid+'_pool.npz')) for uid in c.UIDS]
for i, (uid, d) in enumerate(zip(c.UIDS, data)):
    assert digest(REF/(uid+'_pool.npz')) == meta['pool_sha256'][uid]
    assert np.array_equal(d['vertices'], array(batch.vertices[i, :len(d['vertices'])]))
    assert np.array_equal(d['positive'], array(batch.face_set[i]))
scales = model.scoring_contract()
c.PAIR_CHUNK = cp['args']['pair_chunk_size']
a = model.autoencoder
versions = {n:p._version for n,p in model.named_parameters()}
rng0 = rng.get_state().clone()
cuda_rng0 = T.cuda.get_rng_state().clone()
state_hashes = {n:tensor_digest(p) for n,p in model.named_parameters()}

# Capture actual tensors without replacing projections, normalization, or scoring.
saved, hooks = {}, []
def keep(key, value):
    saved.setdefault(key, []).append(value)
    if value.requires_grad:
        value.retain_grad()
def attach(module, key, pre=False):
    if pre:
        hooks.append(module.register_forward_pre_hook(lambda mod, args: keep(key, args[0])))
    else:
        hooks.append(module.register_forward_hook(lambda mod, args, output: keep(key, output)))
for module, key, pre in [
    (a.vertex_input, 'vertex_xyz', True), (a.vertex_input, 'vertex_projected', False),
    (a.face_input, 'face_centroids', True), (a.face_input, 'face_projected', False),
    (a.mu, 'mu_packed', False), (a.latent_input, 'z_packed', True),
    (a.decoder_output_norm, 'decoder_hidden_before_final_ln', True),
    (a.decoder_output_norm, 'decoder_hidden_after_final_ln', False),
    (a.edge_embedding, 'edge_head_raw', False), (a.face_embedding, 'face_head_raw', False)]:
    attach(module, key, pre)
rows, stats, zs = c.m.forward(model, batch, 'mu')
for hook in hooks:
    hook.remove()
for tensor in [rows[2][3], rows[3][3]]:
    tensor.retain_grad()
assert all(T.equal(x, z) for x,z in zip(rows[0], zs))
assert len(saved['vertex_xyz']) == len(saved['face_centroids']) == 4
print('FORWARD_CAPTURED', flush=True)

# Observe logits of the complete, original training objective, not a hard-error subset.
edge_records, pair_records, face_records = [], [], []
old_sums, old_face = b.full_sums, c.face_logits
pair_owner = c.m.h.soft4_loss.__globals__['teacher']
old_pairs = pair_owner._all_pair_chunks
def record_sums(logits, labels):
    logits.retain_grad()
    edge_records.append((logits, labels.detach()))
    return old_sums(logits, labels)
def record_pairs(*args, **kwargs):
    for pairs in old_pairs(*args, **kwargs):
        pair_records.append(pairs.detach())
        yield pairs
def record_face(*args, **kwargs):
    logits = old_face(*args, **kwargs)
    logits.retain_grad()
    face_records.append(logits)
    return logits
b.full_sums = record_sums
c.face_logits = record_face
pair_owner._all_pair_chunks = record_pairs
try:
    selected = tuple((r[3],) for r in rows)
    rec, parts, _ = b.full_objective(selected, [data[3]], scales)
    loss = rec/4
    loss_value = float(loss.detach())
    loss.backward()
finally:
    b.full_sums, c.face_logits = old_sums, old_face
    pair_owner._all_pair_chunks = old_pairs
assert len(edge_records) == len(pair_records) and len(face_records) == 1
assert all(x.grad is not None for x,y in edge_records)
assert face_records[0].grad is not None
assert all(p.grad is None for p in a.log_variance.parameters())
print('ORIGINAL_OBJECTIVE_BACKWARD', loss_value, parts, flush=True)

d = data[3]
n = len(d['vertices'])
assert n == 804
pairs = np.concatenate([array(x) for x in pair_records])
el = np.concatenate([array(x) for x,y in edge_records])
ey = np.concatenate([array(y).astype(bool) for x,y in edge_records])
eg = np.concatenate([array(x.grad) for x,y in edge_records])
assert len(pairs) == 322806
assert np.array_equal(pairs, np.stack(np.triu_indices(n, 1), axis=1))
edge_ids = pairs[:,0]*n+pairs[:,1]
train_triples = np.concatenate([d['positive'], d['mixed']]).astype(np.int64)
train_labels = np.r_[np.ones(len(d['positive']), bool), np.zeros(len(d['mixed']), bool)]
train_ids = c.m.probe.keys(train_triples, n)
fl, fg = array(face_records[0]), array(face_records[0].grad)
assert len(fl) == len(train_triples)
gt_ids = c.m.probe.keys(d['positive'], n)
assert np.array_equal(train_labels, np.isin(train_ids, gt_ids))

# This enumeration is the unchanged, ordered actual-reconstruction routine.
archived = np.load(BRANCH/'mu_step1000/nexus_2k_001333.npz')
with T.no_grad():
    selected_detached = tuple(tuple(x.detach() for x in row) for row in selected)
    old_uids = c.UIDS
    c.UIDS = [old_uids[3]]
    try:
        actual_metrics, tables = c.capture(selected_detached, [d], scales, SNAP/'actual_forward')
    finally:
        c.UIDS = old_uids
actual = tables[0]
repeat_deltas = {k: float(np.max(np.abs(actual[k].astype(np.float64)-archived[k].astype(np.float64))))
                 for k in ['mu','logvar','edge_logits','face_logits']}
assert all(np.array_equal(actual[k], archived[k]) for k in actual), repeat_deltas
assert np.array_equal(el, actual['edge_logits'])
actual_face_ids = actual['face_ids'][actual['face_actual']]
union_ids = np.unique(np.r_[train_ids, actual_face_ids, gt_ids])
union_triples = c.m.probe.triples(union_ids, n)
union_labels = np.isin(union_ids, gt_ids)
in_training = np.isin(union_ids, train_ids)
in_actual = np.isin(union_ids, actual_face_ids)
training_to_union = np.searchsorted(union_ids, train_ids)
actual_to_union = np.searchsorted(union_ids, actual['face_ids'])
with T.no_grad():
    union_logits = array(c.face_logits(rows[3][3].detach(), T.as_tensor(union_triples, device='cuda'), scales))
assert np.array_equal(union_logits[training_to_union], fl)
assert np.array_equal(union_logits[actual_to_union], actual['face_logits'])
occurrences = np.bincount(training_to_union, minlength=len(union_ids))
direct_gradient = np.zeros(len(union_ids), dtype=np.float64)
np.add.at(direct_gradient, training_to_union, fg.astype(np.float64))
# NaN means no direct objective term, not a supervised zero derivative.
direct_gradient[~in_training] = np.nan

def grad_summary(logits, labels, gradient):
    pred = logits>0
    gain = -(2*labels.astype(np.int8)-1)*gradient
    result = {}
    for group, mask in [('TP',pred&labels),('TN',~pred&~labels),('FP',pred&~labels),('FN',~pred&labels)]:
        g, q = gradient[mask], gain[mask]
        result[group] = dict(count=int(mask.sum()), favorable=int((q>0).sum()), unfavorable=int((q<0).sum()), exactly_zero=int((q==0).sum()),
            abs_gradient_le_1e_minus12=int((np.abs(g)<=1e-12).sum()),
            gradient_quantiles=None if not len(g) else dict(zip(['min','p01','median','p99','max'], map(float,np.quantile(g,[0,.01,.5,.99,1])))),
            margin_gain_per_unit_eta_quantiles=None if not len(q) else list(map(float,np.quantile(q,[0,.01,.5,.99,1]))))
    return result

def soft4_weights(logits, labels):
    with T.no_grad():
        lt=T.as_tensor(logits,device='cuda');yt=T.as_tensor(labels,device='cuda',dtype=T.float32)
        weights=b.full_weights(lt,yt)
        mass=weights.sum(1, dtype=T.float32)
        numerator=(weights*T.nn.functional.binary_cross_entropy_with_logits(lt,yt,reduction='none')).sum(1,dtype=T.float32)
        return array(weights), array(mass), array(numerator)
ew, emass, enum = soft4_weights(el,ey)
fw, fmass, fnum = soft4_weights(fl,train_labels)
np.savez_compressed(SNAP/'edge_all_pairs.npz',pairs=pairs,ids=edge_ids,labels=ey,logits=el,
    in_training=np.ones(len(el),bool),grad_loss_wrt_logit=eg,
    signed_margin=(2*ey.astype(np.int8)-1)*el,margin_gain_per_unit_eta=-(2*ey.astype(np.int8)-1)*eg,
    membership_TP_TN_FP_FN=ew,soft_group_mass=emass,soft_group_numerator=enum)
np.savez_compressed(SNAP/'face_training_rows.npz',triples=train_triples,ids=train_ids,labels=train_labels,logits=fl,
    grad_loss_wrt_logit=fg,union_row=training_to_union,
    signed_margin=(2*train_labels.astype(np.int8)-1)*fl,margin_gain_per_unit_eta=-(2*train_labels.astype(np.int8)-1)*fg,
    membership_TP_TN_FP_FN=fw,soft_group_mass=fmass,soft_group_numerator=fnum)
np.savez_compressed(SNAP/'face_union.npz',triples=union_triples,ids=union_ids,labels=union_labels,logits=union_logits,
    in_training_pool=in_training,in_actual_edge_candidates=in_actual,training_occurrences=occurrences,
    has_direct_loss_term=in_training,grad_loss_wrt_logit_sum=direct_gradient,
    signed_margin=(2*union_labels.astype(np.int8)-1)*union_logits,
    actual_predicted_face=in_actual&(union_logits>0),actual_gt_fn=union_labels&(~in_actual|(union_logits<=0)))

counts = [len(x) for x in rows[0]]
offset = sum(counts[:3])
sl = slice(offset,offset+n)
features = dict(local_vertex_id=np.arange(n),packed_vertex_row=np.arange(offset,offset+n),
    vertices=array(saved['vertex_xyz'][3]),gt_faces=d['positive'],face_centroids=array(saved['face_centroids'][3]),
    vertex_input_features=array(saved['vertex_projected'][3]),face_input_features=array(saved['face_projected'][3]),
    mu=array(rows[0][3]),logvar=array(rows[1][3]),z=array(zs[3]),
    edge_embedding_scoring=array(rows[2][3]),face_embedding_scoring=array(rows[3][3]),
    grad_edge_embedding_scoring=array(rows[2][3].grad),grad_face_embedding_scoring=array(rows[3][3].grad))
for key in ['mu_packed','decoder_hidden_before_final_ln','decoder_hidden_after_final_ln','edge_head_raw','face_head_raw']:
    t = saved[key][0]
    assert t.shape[0] == sum(counts)
    features[key.removesuffix('_packed')] = array(t[sl])
    assert t.grad is not None, key
    features['grad_'+key.removesuffix('_packed')] = array(t.grad[sl])
assert np.array_equal(features['mu'],array(saved['mu_packed'][0][sl]))
features['incidence_index_local_vertex_then_face'] = array(batch.incidence_index[3])
np.savez_compressed(SNAP/'representations_and_gradients.npz',**features)
head_weights = {name.replace('.','__'):array(value) for name,value in a.state_dict().items()
                if name.startswith(('vertex_input.','face_input.','encoder_output_norm.','mu.','log_variance.','decoder_output_norm.','edge_embedding.','face_embedding.'))}
np.savez_compressed(SNAP/'input_posterior_output_head_weights.npz',**head_weights)
shutil.copy2(REF/'nexus_2k_001333_pool.npz', SNAP/'original_training_pool_804.npz')

# A readable, executable export of the formulas actually called, including chunk order.
topology = c.m.topology
exports = ['import math\nimport torch\nimport numpy as np\nfrom torch import Tensor\nfrom torch.nn import functional as F\nfrom types import SimpleNamespace\n',
           "EPS=1e-8\nGROUPS=('tp','tn','fp','fn')\nPAIR_CHUNK="+str(c.PAIR_CHUNK)+'\n']
for name in ['_require_positive_finite','_split_spacetime','_squared_parallelogram_area','first_order_interval','second_order_interval','face_interval_logits']:
    exports.append(source(getattr(topology,name)))
exports.append(source(old_pairs))
exports.append(b.src)  # Exact replacement source compiled by backend00, no detach.
exports.append('teacher=SimpleNamespace(_all_pair_chunks=_all_pair_chunks)\nprobe=SimpleNamespace(first_order_interval=first_order_interval)\n')
exports.append(source(c.m.h.soft4_loss))
exports.append(b.src_weights)
exports.append(source(c.soft))
exports.append(source(c.edge_logits).replace('m.probe.first_order_interval','first_order_interval'))
exports.append(source(c.face_logits).replace('m.topology.face_interval_logits','face_interval_logits'))
exports.append(source(c.objective).replace('m.h.soft4_loss','soft4_loss'))
exports.append('def loss_mu804(edge, face, pool, scales):\n    value, parts, saved = objective(((), (), (edge,), (face,)), [pool], scales)\n    return value/4, parts, saved\n')
standalone='\n\n'.join(exports)
(CODE/'effective_loss_and_scoring.py').write_text(standalone)
ns={}
exec(compile(standalone,str(CODE/'effective_loss_and_scoring.py'),'exec'),ns)
with T.no_grad():
    standalone_loss,standalone_parts,_ = ns['loss_mu804'](rows[2][3].detach(),rows[3][3].detach(),d,scales)
assert float(standalone_loss) == loss_value and standalone_parts == parts
for name,fn in [('math00_attention_wrapper.py',b.unified_attention),('math00_attention_core.py',b.core),
                ('graph_gather_backward.py',b.bw.DeterministicGather),('soft4_runtime_dispatch.py',b.full_objective)]:
    (CODE/name).write_text(source(fn))
(CODE/'encoder_decoder_classes.py').write_text('\n\n'.join(source(getattr(topology,k)) for k in ['MeanSAGEConv','GraphTransformerBlock','AttentionBlock','TopologyAutoencoder']))
write(CODE/'runtime_contract.json',dict(scales=scales,logvar_clamp=[-20,10],beta=0,threshold=0,coefficient=.25,
    edge_chunk=c.PAIR_CHUNK,edge_diagnostic_chunk=65536,face_chunk='all training rows in one scoring call',
    soft4_epsilon=1e-8,tau=1,membership_detach=False,soft_group_order=['TP','TN','FP','FN'],
    normalize_spacetime_embeddings=a.normalize_spacetime_embeddings,output_centering='per mesh, per channel mean subtracted in FP32',
    output_clamp='no embedding/logit clamp; area helper follows exported code',
    layernorm='Encoder terminal before mu/logvar; Decoder terminal before both linear heads; also internal block norms',
    decoder_xyz=False,input='actual normalized dataset xyz and face centroids; separate Linear(3,512)',
    shape={k:list(v.shape) for k,v in features.items()},packed_uids=c.UIDS,packed_vertex_counts=counts,
    optimizer_created=False,optimizer_updates=0,latent='mu',training_scope='only804 reconstruction divided by4; logvar frozen'))

# Copy imported project sources with their paths, plus effective exports. No old run data.
sources = set(meta['source_sha256']) | {str(REF/'backend00.py'),str(BRANCH/'effective_run.py')}
for module in list(sys.modules.values())+[b,c,c.m,c.m.h,b.backend,b.bw,b.backend.b]:
    f=getattr(module,'__file__',None)
    if f and str(f).startswith(str(BASE)) and str(f).endswith('.py'):
        sources.add(str(Path(f)))
source_index=[]
for f in sorted(sources):
    p=Path(f)
    if not p.is_file():continue
    dst=SNAP/'source_closure'/p.relative_to(BASE)
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    source_index.append(dict(original=str(p),bundle=str(dst.relative_to(OUT)),sha256=digest(p)))
write(SNAP/'source_index.json',source_index)
actual_fp = in_actual & ~union_labels & (union_logits>0)
summary=dict(checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,additional_updates=1000,cumulative_updates=4400,
    captured_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),uid=c.UIDS[3],vertices=n,
    loss=loss_value,parts=parts[0],objective='(EdgeSoft4_804(mu)+FaceSoft4_804(mu))/4',
    edge_pairs=len(pairs),face_training_rows=len(fl),face_training_unique_ids=int(len(np.unique(train_ids))),
    face_union_rows=len(union_ids),actual_face_candidates=int(in_actual.sum()),gt_faces=len(gt_ids),
    actual_face_fp_with_training=int((actual_fp&in_training).sum()),actual_face_fp_without_training=int((actual_fp&~in_training).sum()),
    gt_missing_actual=int((union_labels&~in_actual).sum()),
    logit_gradient_statistics={'edge':grad_summary(el,ey,eg),'face_training_rows':grad_summary(fl,train_labels,fg)},
    actual_reconstruction=actual_metrics[0],archived_forward_max_abs_deltas=repeat_deltas,
    no_direct_loss_semantics='face_union.grad is NaN when absent from training; training zero derivatives remain numeric zero',
    head_gradients={k:float(np.linalg.norm(v.astype(np.float64))) for k,v in features.items() if k.startswith('grad_')},
    precision={'torch':T.__version__,'cuda':T.version.cuda,'gpu':T.cuda.get_device_name(0),'attention':'explicit FP32 SDPA MATH','tf32':False,'autocast':False},
    standalone_effective_loss_equal=True)
write(SNAP/'summary.json',summary)
assert all(p._version==versions[n] and tensor_digest(p)==state_hashes[n] for n,p in model.named_parameters())
assert T.equal(rng0,rng.get_state()) and T.equal(cuda_rng0,T.cuda.get_rng_state())
assert all(digest(p)==sha for p,sha in meta['source_sha256'].items())
assert digest(REF/'backend00.py')==meta['backend_sha']
write(SNAP/'verification.json',dict(complete=True,parameters_unchanged=True,training_rng_unchanged=True,
    cuda_rng_unchanged=True,no_optimizer_created=True,optimizer_updates=0,source_hashes_unchanged=True,
    baseline_matches_saved_mu_snapshot_bitwise=True,all_pairs_exactly_once=True,face_labels_match_gt=True,
    training_union_and_actual_logits_match_bitwise=True,standalone_effective_loss_matches_bitwise=True,
    training_logit_gradient_includes_full_pool_and_outer_divisor4=True,
    soft_membership_requires_grad=True,logvar_parameter_grad_is_none=True,
    loss=loss_value,original_saved_parts_match=parts[0]==json.loads((BRANCH/'mu_step1000.json').read_text())['parts'][3]))
print('SNAPSHOT_COMPLETE',json.dumps({k:summary[k] for k in ['edge_pairs','face_training_rows','face_union_rows','actual_face_fp_without_training','loss']}),flush=True)
