"""Short diagnostic interventions on two meshes, isolated from production code."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch
from scipy.sparse import csr_matrix

BASE = Path('/guohaoran/nexus_fast_track')
SOURCE = BASE / 'diagnostics/layernorm_no_rms_20260907_0824/variant_project'
CHECKPOINT = BASE / 'diagnostics/layernorm_no_rms_20260907_0824/run/checkpoint-0001000.pt'
PREVIOUS = BASE / 'diagnostics/matched_reassessment_20260907_results'
UIDS = ['nexus_2k_000387', 'nexus_2k_001849']
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / 'scripts'))
from mini_nexus.topology import first_order_interval, face_interval_logits
from mini_nexus.topology_checkpoint import load_topology_system_from_checkpoint
from mini_nexus.training_2k import topology_autoencoder_loss_with_face_negatives
from mini_nexus.data_2k import Nexus2KManifestDataset
from mini_nexus.packed_topology import collate_packed_topology
from train_topology_ae_overfit_packed import stable_uid_seed


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False, default=str) + '\n')


def counts(tp, fp, fn):
    tp, fp, fn = int(tp), int(fp), int(fn)
    return dict(tp=tp, fp=fp, fn=fn, precision=tp/max(tp+fp, 1),
                recall=tp/max(tp+fn, 1), f1=2*tp/max(2*tp+fp+fn, 1))


def ranking(pos, neg, total_positive=None):
    p = len(pos); total = p if total_positive is None else total_positive
    s = np.r_[pos, neg]; y = np.arange(len(s)) < p
    assert np.isfinite(s).all()
    tp = int(((s > 0) & y).sum()); fp = int(((s > 0) & ~y).sum())
    out = counts(tp, fp, total-tp)
    out.update(positives_scored=p, negatives_scored=len(neg), negative_rejection=float((neg <= 0).mean()) if len(neg) else None)
    if not len(s) or not total:
        return out
    order = np.argsort(-s, kind='stable'); ss = s[order]; yy = y[order]
    ends = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(ss)-1]
    t = np.cumsum(yy)[ends]; precision = t/(ends+1); recall = t/total
    f1 = 2*t/(ends+1+total); best = int(f1.argmax())
    out.update(ap=float((np.diff(np.r_[0., recall])*precision).sum()),
               oracle_best_f1=float(f1[best]), oracle_threshold=float(ss[ends[best]]),
               oracle_counts=counts(t[best], ends[best]+1-t[best], total-t[best]),
               recall_at_precision_99=float(recall[precision >= .99].max()) if (precision >= .99).any() else 0.)
    return out


def keys(tri, n):
    t = np.sort(np.asarray(tri, dtype=np.int64).reshape(-1, 3), axis=1)
    return (t[:, 0]*n+t[:, 1])*n+t[:, 2]


def triples(k, n):
    return np.stack([k//(n*n), k//n % n, k % n], axis=1)


@torch.no_grad()
def score_faces(f, tri, scales):
    out = np.empty(len(tri), dtype=np.float32)
    for start in range(0, len(tri), 65536):
        t = torch.as_tensor(tri[start:start+65536], device='cuda', dtype=torch.long)
        out[start:start+len(t)] = face_interval_logits(*(f[t[:, j]] for j in range(3)),
            logit_scale=scales['face_logit_scale'], area_factor=scales['face_interval_factor']).cpu().numpy()
    return out


@torch.no_grad()
def graph(e, gt_edges, scales):
    n = len(e); left = []; right = []; ps = []; ns = []
    truth = np.zeros((n, n), dtype=bool)
    g = gt_edges.T if gt_edges.shape[0] == 2 else gt_edges
    truth[g[:, 0], g[:, 1]] = True
    for start in range(0, n, 128):
        ii = torch.arange(start, min(start+128, n), device='cuda'); jj = torch.arange(n, device='cuda')
        d = (first_order_interval(e[ii, None], e[None])*scales['edge_logit_scale']).cpu().numpy()
        valid = np.arange(start, start+len(ii))[:, None] < np.arange(n)[None]
        y = truth[start:start+len(ii)]; ps.append(d[valid & y]); ns.append(d[valid & ~y])
        a, b = np.nonzero((d > 0) & valid); left.extend(a+start); right.extend(b)
    upper = csr_matrix((np.ones(len(left), dtype=np.int32), (left, right)), shape=(n, n))
    total = int((upper@upper).multiply(upper).sum())
    edge = ranking(np.concatenate(ps), np.concatenate(ns))
    if total > 5_000_000:
        return None, edge, total
    adj = [set(upper.indices[upper.indptr[i]:upper.indptr[i+1]].tolist()) for i in range(n)]
    k = np.empty(total, dtype=np.int64); at = 0
    for i, neigh in enumerate(adj):
        for j in sorted(neigh):
            for t in sorted(neigh.intersection(adj[j])):
                k[at] = (i*n+j)*n+t; at += 1
    assert at == total
    return np.sort(k), edge, total


def setup_model(checkpoint=CHECKPOINT):
    cp = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    model = load_topology_system_from_checkpoint(cp, device='cuda')
    ds = Nexus2KManifestDataset(Path(cp['args']['manifest']), 'train')
    samples = [ds[ds.index_for_uid(u)] for u in UIDS]
    batch = collate_packed_topology(samples).to('cuda')
    return cp, model, batch


def get_rows(model, batch, mode, seedstep=1000):
    model.train(mode != 'mu')
    seeds = tuple(stable_uid_seed(20260901, seedstep, u) for u in UIDS)
    return model.topology_embedding_rows(batch, sample_seeds=seeds)


@torch.no_grad()
def evaluate(root, label, outputs, scales):
    results = []
    for mode, rows in outputs.items():
        for i, uid in enumerate(UIDS):
            c = np.load(root/(uid+'_pool.npz')); pos = c['positive']; n = len(c['vertices'])
            e, f = rows[2][i], rows[3][i]
            pscore = score_faces(f, pos, scales)
            fixed = score_faces(f, c['original'], scales)
            hard = score_faces(f, c['heldout'], scales)
            allpool = score_faces(f, c['pool'], scales)
            k, edge, nc = graph(e, c['edges'], scales)
            result = dict(uid=uid, mode=mode, edge=edge, fixed=ranking(pscore, fixed),
                heldout=ranking(pscore, hard), frozen_prediction_pool=ranking(pscore, allpool),
                candidate_count=nc, recovery_complete=k is not None)
            if k is not None:
                cand_score = score_faces(f, triples(k, n), scales)
                y = np.isin(k, keys(pos, n))
                result['face'] = ranking(cand_score[y], cand_score[~y], total_positive=len(pos))
            results.append(result)
            print(json.dumps(dict(event='eval', label=label, uid=uid, mode=mode,
                edge_f1=edge['f1'], face=result.get('face'), hard_ap=result['heldout'].get('ap'))), flush=True)
            np.savez_compressed(root/(label+'_'+uid+'_'+mode+'.npz'), edge=e.cpu().numpy(), face=f.cpu().numpy())
    write(root/(label+'_evaluation.json'), dict(label=label, rows=results, zero_threshold=0.,
        threshold_oracles_use_gt=True, checkpoint_origin=str(CHECKPOINT), host=socket.gethostname()))
    return results


@torch.no_grad()
def prepare(root):
    cp, model, batch = setup_model(); sc = model.scoring_contract()
    original_report = json.loads((PREVIOUS/'matched.json').read_text())
    outputs = {mode: get_rows(model, batch, mode, 1000) for mode in ['mu', 'sample0']}
    pool_records = []
    for i, uid in enumerate(UIDS):
        c = np.load(PREVIOUS/(uid+'_candidates.npz')); n = len(c['vertices']); pos = c['positives']; orig = c['negatives']
        assert np.array_equal(batch.face_set[i].cpu().numpy(), pos)
        assert np.array_equal(batch.vertices[i, :n].cpu().numpy(), c['vertices'])
        pred = []; hard = []
        for mode, row in outputs.items():
            k, _, total = graph(row[2][i], c['edges'], sc)
            if k is None:
                raise RuntimeError(f'{uid}/{mode}: {total} exceeds preparation bound')
            false = k[~np.isin(k, keys(pos, n))]
            score = score_faces(row[3][i], triples(false, n), sc)
            pred.append(false); hard.append(false[score > 0])
        pool = np.unique(np.concatenate(pred)); hard = np.unique(np.concatenate(hard))
        hard = hard[~np.isin(hard, keys(orig, n))]
        rng = np.random.default_rng(stable_uid_seed(20260907, 1000, uid)); hard = rng.permutation(hard)
        heldout = hard[:len(hard)//5]; eligible = hard[len(hard)//5:]
        previous = next(r for r in original_report['rows'] if r['uid'] == uid and r['mode'] == 'mu')
        cycle = previous['source']['cycle']['count']; rest = len(orig)-cycle
        replace = min(rest//2, len(eligible)); chosen = eligible[:replace]
        keep = rng.permutation(np.arange(cycle, len(orig)))[:rest-replace]
        mixed = np.concatenate([orig[:cycle], orig[keep], triples(chosen, n)])
        assert len(mixed) == len(orig) and len(np.unique(keys(mixed, n))) == len(orig)
        assert not np.intersect1d(keys(mixed, n), keys(pos, n)).size
        assert not np.intersect1d(keys(mixed, n), heldout).size
        np.savez_compressed(root/(uid+'_pool.npz'), vertices=c['vertices'], positive=pos,
            edges=c['edges'], original=orig, mixed=mixed, pool=triples(pool, n), heldout=triples(heldout, n))
        row = dict(uid=uid, vertices=n, positive_faces=len(pos), original_negatives=len(orig),
            gt_false_cycles=cycle, predicted_false_union=len(pool), eligible_hard_outside_original=len(hard),
            heldout_hard=len(heldout), replaced=replace, mixed_negatives=len(mixed))
        pool_records.append(row); print(json.dumps(dict(event='pool', **row)), flush=True)
    meta = dict(host=socket.gethostname(), checkpoint_sha256=digest(CHECKPOINT), checkpoint=str(CHECKPOINT),
        uids=UIDS, args=cp['args'], spacetime_scoring=sc, source=str(SOURCE),
        source_hashes={str(p):digest(p) for p in [SOURCE/'mini_nexus/topology.py', SOURCE/'mini_nexus/training_2k.py', SOURCE/'mini_nexus/flash_varlen_topology.py']},
        optimizer_groups=[{k:v for k,v in g.items() if k != 'params'} for g in cp['optimizer']['param_groups']],
        optimizer_states=len(cp['optimizer']['state']), pool_records=pool_records,
        model_shapes={k:list(v.shape) for k,v in cp['model'].items() if isinstance(v, torch.Tensor) and any(t in k for t in ['mu.weight','latent_input.weight','edge_embedding.weight','face_embedding.weight'])})
    write(root/'preparation.json', meta)
    evaluate(root, 'initial', outputs, sc)


def install_label_loss(root):
    """Diagnostic: fixed positive/negative weighting, preserving all pairs/counts."""
    import inspect
    import mini_nexus.training_2k as training
    import mini_nexus.topology as topology
    old = 'loss = (group_sums[non_empty] / group_counts[non_empty]).mean()'
    new = ('loss = 0.5 * ((group_sums[0] + group_sums[3]) / (group_counts[0] + group_counts[3])'
           ' + (group_sums[1] + group_sums[2]) / (group_counts[1] + group_counts[2]))')
    edge_source = inspect.getsource(training.paper_edge_loss_all_pairs)
    assert edge_source.count(old) == 1
    edge_source = edge_source.replace(old, new)
    exec(compile(edge_source, '<diagnostic-label-edge>', 'exec'), training.__dict__)
    old = 'group_losses = [losses[mask].mean() for mask in masks.values() if mask.any()]'
    new = 'group_losses = [losses[labels > 0.5].mean(), losses[labels <= 0.5].mean()]'
    face_source = inspect.getsource(topology.paper_balanced_binary_loss)
    assert face_source.count(old) == 1
    face_source = face_source.replace(old, new)
    exec(compile(face_source, '<diagnostic-label-face>', 'exec'), topology.__dict__)
    training.paper_balanced_binary_loss = topology.paper_balanced_binary_loss
    (root/'label_balance_implementation.txt').write_text(edge_source+'\n'+face_source)


def install_ffn(model, optimizer, root):
    """Add zero-output 4x GELU FFNs, initially preserving the checkpoint function."""
    import inspect
    import mini_nexus.flash_varlen_topology as flash
    added = []
    for block in model.autoencoder.decoder_blocks:
        width = block.attention.embed_dim
        block.diagnostic_ffn = torch.nn.Sequential(torch.nn.LayerNorm(width), torch.nn.Linear(width,4*width),
            torch.nn.GELU(), torch.nn.Linear(4*width,width)).cuda()
        torch.nn.init.zeros_(block.diagnostic_ffn[-1].weight)
        torch.nn.init.zeros_(block.diagnostic_ffn[-1].bias)
        added.extend(block.diagnostic_ffn.parameters())
    optimizer.add_param_group(dict(params=added, **{k:v for k,v in optimizer.param_groups[0].items() if k!='params'}))
    source = inspect.getsource(flash._flash_varlen_autoencoder_forward)
    old = '        hidden = hidden + update\n'
    assert source.count(old) == 1
    source = source.replace(old, old+'        hidden = hidden + block.diagnostic_ffn(hidden)\n')
    exec(compile(source, '<diagnostic-decoder-ffn>', 'exec'), flash.__dict__)
    (root/'ffn_forward_implementation.txt').write_text(source)
    write(root/'ffn_parameter_count.json',dict(added=sum(p.numel() for p in added), total=sum(p.numel() for p in model.parameters())))


def train(root, variant, steps):
    starting_checkpoint = root/'deterministic_label_balance_checkpoint.pt' if variant=='warmstart_fourgroup' else CHECKPOINT
    cp, model, batch = setup_model(starting_checkpoint); sc = model.scoring_contract()
    deterministic = variant in ['deterministic_no_kl', 'deterministic_label_balance', 'coordinate_supervision', 'warmstart_fourgroup']
    kl_weight = 0. if deterministic else 1e-4
    if variant == 'deterministic_label_balance':
        install_label_loss(root)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.load_state_dict(cp['optimizer'])
    for p, state in optimizer.state.items():
        if 'exp_avg' in state:
            assert state['exp_avg'].shape == p.shape
    if variant == 'hardneg_ffn':
        with torch.no_grad():
            repeat_a = get_rows(model, batch, 'mu')
            repeat_b = get_rows(model, batch, 'mu')
            repeat_errors = [float((a-b).abs().max()) for j in [0,2,3] for a,b in zip(repeat_a[j], repeat_b[j])]
            fixed_latent = []
            def hold_latent(module, inputs):
                if not fixed_latent:
                    fixed_latent.append(inputs[0].detach().clone())
                return (fixed_latent[0],)
            handle = model.autoencoder.latent_input.register_forward_pre_hook(hold_latent)
            before = get_rows(model, batch, 'mu')
            install_ffn(model, optimizer, root)
            after = get_rows(model, batch, 'mu')
            errors = [float((a-b).abs().max()) for j in [2,3] for a,b in zip(after[j], before[j])]
            handle.remove()
            write(root/'ffn_initial_equivalence.json',dict(fixed_latent_decoder_max_abs_errors=errors, original_repeated_forward_errors=repeat_errors))
            assert max(errors) == 0., errors
    data = [np.load(root/(u+'_pool.npz')) for u in UIDS]
    negatives = [torch.as_tensor(d['original' if variant == 'original' else 'mixed'], device='cuda') for d in data]
    targets = []
    if variant == 'coordinate_supervision':
        for u in UIDS:
            target = np.load(root/('free32_'+u+'_direct.npz'))
            tensors = [torch.as_tensor(target[k], device='cuda') for k in ['edge','face']]
            targets.append([t-t.mean(dim=0, keepdim=True) for t in tensors])
    trace = []; started = time.monotonic()
    for step in range(1, steps+1):
        optimizer.zero_grad(set_to_none=True)
        rows = get_rows(model, batch, 'mu' if deterministic else 'sample0', 1000+step)
        losses = []; components = []
        for i in range(len(UIDS)):
            if variant == 'coordinate_supervision':
                terms = [((rows[j+2][i]-targets[i][j]).square().mean() / targets[i][j].square().mean()) for j in range(2)]
                losses.append(sum(terms)); components.append(dict(edge=terms[0], face=terms[1]))
                continue
            loss, comp = topology_autoencoder_loss_with_face_negatives(batch.edge_index[i], batch.face_set[i], negatives[i],
                rows[0][i], rows[1][i], rows[2][i], rows[3][i], kl_weight=kl_weight,
                pair_chunk_size=cp['args']['pair_chunk_size'], edge_logit_scale=sc['edge_logit_scale'],
                face_logit_scale=sc['face_logit_scale'], face_interval_factor=sc['face_interval_factor'])
            losses.append(loss); components.append(comp)
        loss = torch.stack(losses).mean(); loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 10 == 0:
            record = dict(step=step, variant=variant, loss=float(loss.detach()), gradient_norm=float(grad),
                seconds=time.monotonic()-started, components=[{k:float(v.detach()) for k,v in c.items()} for c in components])
            trace.append(record); print(json.dumps(dict(event='train', **record)), flush=True)
            write(root/(variant+'_trace.json'), trace)
        if step == steps//2:
            with torch.no_grad():
                mid = get_rows(model, batch, 'mu')
                quick = []
                for i, u in enumerate(UIDS):
                    ps=score_faces(mid[3][i], data[i]['positive'], sc); ns=score_faces(mid[3][i], data[i]['heldout'], sc)
                    quick.append(dict(uid=u, heldout=ranking(ps, ns)))
                write(root/(variant+'_midpoint.json'), quick)
                print(json.dumps(dict(event='midpoint', variant=variant, rows=quick)), flush=True)
    cp['model'] = model.state_dict(); cp['optimizer'] = optimizer.state_dict()
    cp['diagnostic_intervention'] = dict(variant=variant, steps=steps, selected_uids=UIDS, kl_weight=kl_weight,
        original_checkpoint_step=1000, training_objective_mode='mu' if deterministic else 'sample',
        starting_checkpoint=str(starting_checkpoint),
        note='Two-mesh diagnostic continuation; not a full-20 training run.')
    torch.save(cp, root/(variant+'_checkpoint.pt'))
    with torch.no_grad():
        outputs = {mode:get_rows(model, batch, mode, 1000+(1 if mode=='sample1' else 0)) for mode in ['mu','sample0','sample1']}
        evaluate(root, variant, outputs, sc)
    write(root/(variant+'_complete.json'), dict(steps=steps, seconds=time.monotonic()-started,
        max_gpu_gb=torch.cuda.max_memory_allocated()/2**30, intervention=cp['diagnostic_intervention']))


def free(root, steps):
    prep=json.loads((root/'preparation.json').read_text());sc=prep['spacetime_scoring']
    data=[np.load(root/(u+'_pool.npz')) for u in UIDS]
    edge=[];face=[]
    for uid in UIDS:
        x=np.load(root/('initial_'+uid+'_mu.npz'))
        edge.append(torch.nn.Parameter(torch.from_numpy(x['edge']).cuda()))
        face.append(torch.nn.Parameter(torch.from_numpy(x['face']).cuda()))
    params=edge+face;optimizer=torch.optim.Adam(params,lr=.01);trace=[];started=time.monotonic()
    for step in range(1,steps+1):
        optimizer.zero_grad(set_to_none=True);losses=[];components=[]
        for i,d in enumerate(data):
            zero=torch.zeros((len(edge[i]),64),device='cuda')
            loss,comp=topology_autoencoder_loss_with_face_negatives(torch.as_tensor(d['edges'],device='cuda'),
                torch.as_tensor(d['positive'],device='cuda'),torch.as_tensor(d['mixed'],device='cuda'),
                zero,zero,edge[i],face[i],kl_weight=0.,pair_chunk_size=1000000,
                edge_logit_scale=sc['edge_logit_scale'],face_logit_scale=sc['face_logit_scale'],face_interval_factor=sc['face_interval_factor'])
            losses.append(loss);components.append(comp)
        loss=torch.stack(losses).mean();loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
        if step==1 or step%50==0:
            record=dict(step=step,loss=float(loss.detach()),gradient_norm=float(grad),seconds=time.monotonic()-started,
                components=[{k:float(v.detach()) for k,v in c.items()} for c in components])
            trace.append(record);write(root/'free32_trace.json',trace);print(json.dumps(dict(event='free',**record)),flush=True)
    evaluate(root,'free32',{'direct':(None,None,edge,face)},sc)
    write(root/'free32_complete.json',dict(steps=steps,seconds=time.monotonic()-started,dimension=32,
        initialization='theta1000 mu decoder coordinates',optimizer='Adam lr=.01',
        interpretation='Direct per-mesh coordinates; no encoder, decoder, latent or KL; finite-pool optimization feasibility only.'))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train','free']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--variant',choices=['original','hardneg','deterministic_no_kl','deterministic_label_balance','coordinate_supervision','hardneg_ffn','warmstart_fourgroup']);p.add_argument('--steps',type=int,default=200)
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(20260907);torch.cuda.set_device(0)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    print(json.dumps(dict(event='start',host=socket.gethostname(),action=a.action,variant=a.variant,source=str(SOURCE))),flush=True)
    if a.action=='prepare':prepare(a.root)
    elif a.action=='train':train(a.root,a.variant,a.steps)
    else:free(a.root,a.steps)
