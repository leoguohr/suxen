"""Matched Edge+Face / Edge-only continuation from the same warm checkpoint."""
import argparse
import json
from pathlib import Path
import shutil
import socket
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent/'aggressive_topology_probe_20260907'
sys.path.insert(0, str(PREVIOUS))
import experiment as probe

CHECKPOINT = PREVIOUS/'deterministic_label_balance_checkpoint.pt'
probe.CHECKPOINT = CHECKPOINT


@torch.no_grad()
def evaluate_mu(model, batch, data, scales, variant, step):
    embeddings = probe.get_rows(model, batch, 'mu')
    results = []
    for i, uid in enumerate(probe.UIDS):
        e, f = embeddings[2][i], embeddings[3][i]
        candidates, edge, candidate_count = probe.graph(e, data[i]['edges'], scales)
        record = dict(uid=uid, step=step, mode='mu', edge=edge,
                      edge_gate=edge['precision'] >= .99 and edge['recall'] >= .99,
                      candidate_count=candidate_count, recovery_complete=candidates is not None)
        if candidates is not None:
            scores = probe.score_faces(f, probe.triples(candidates, len(e)), scales)
            truth = np.isin(candidates, probe.keys(data[i]['positive'], len(e)))
            record['face'] = probe.ranking(scores[truth], scores[~truth], len(data[i]['positive']))
        results.append(record)
        np.savez_compressed(ROOT/f'{variant}_step{step:04d}_{uid}_mu.npz', edge=e.cpu().numpy(), face=f.cpu().numpy())
    print(json.dumps(dict(event='evaluation', variant=variant, step=step, rows=results)), flush=True)
    return dict(step=step, rows=results, both_meshes_pass=all(r['edge_gate'] for r in results))


def run(variant, steps):
    torch.set_num_threads(1)
    torch.manual_seed(20260907)
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cp, model, batch = probe.setup_model(CHECKPOINT)
    assert cp['diagnostic_intervention']['variant']=='deterministic_label_balance'
    assert cp['diagnostic_intervention']['steps']==200
    assert tuple(cp['diagnostic_intervention']['selected_uids'])==tuple(probe.UIDS)
    assert model.autoencoder.mu.out_features==64
    scales = model.scoring_contract()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.load_state_dict(cp['optimizer'])
    for p, state in optimizer.state.items():
        if 'exp_avg' in state:
            assert state['exp_avg'].shape==p.shape
    data = [np.load(PREVIOUS/(u+'_pool.npz')) for u in probe.UIDS]
    negatives = [torch.as_tensor(d['mixed'], device='cuda') for d in data]
    for uid, d, i in zip(probe.UIDS, data, range(len(data))):
        assert np.array_equal(d['positive'], batch.face_set[i].cpu().numpy())
        assert np.array_equal(d['vertices'], batch.vertices[i,:len(d['vertices'])].cpu().numpy())
        if not (ROOT/(uid+'_pool.npz')).exists():
            shutil.copy2(PREVIOUS/(uid+'_pool.npz'), ROOT/(uid+'_pool.npz'))
    weight = 1. if variant=='joint' else 0.
    meta = dict(variant=variant, face_weight=weight, kl_weight=0., training_mode='mu',
                edge_loss='original nonempty TP/TN/FP/FN group means', face_loss='same four-group loss',
                steps=steps, evaluation_period=50, edge_gate=dict(precision=.99,recall=.99),
                gate_interpretation='both thresholds on both meshes; stability assessed across evaluations',
                checkpoint=str(CHECKPOINT), checkpoint_sha256=probe.digest(CHECKPOINT),
                checkpoint_intervention=cp['diagnostic_intervention'], source=str(probe.SOURCE),
                host=socket.gethostname(), uids=probe.UIDS, scoring=scales,
                optimizer_groups=[{k:v for k,v in g.items() if k!='params'} for g in optimizer.param_groups],
                source_sha256={str(p):probe.digest(p) for p in [Path(__file__), PREVIOUS/'experiment.py',
                    probe.SOURCE/'mini_nexus/topology.py', probe.SOURCE/'mini_nexus/training_2k.py',
                    probe.SOURCE/'mini_nexus/flash_varlen_topology.py']},
                pool_sha256={u:probe.digest(PREVIOUS/(u+'_pool.npz')) for u in probe.UIDS})
    probe.write(ROOT/(variant+'_provenance.json'), meta)
    print(json.dumps(dict(event='start', **meta)), flush=True)
    started=time.monotonic(); trace=[]
    evaluations=[evaluate_mu(model,batch,data,scales,variant,0)]
    probe.write(ROOT/(variant+'_evaluations.json'), evaluations)
    for step in range(1,steps+1):
        optimizer.zero_grad(set_to_none=True)
        rows=probe.get_rows(model,batch,'mu')
        losses=[]; components=[]
        for i in range(len(probe.UIDS)):
            _, comp=probe.topology_autoencoder_loss_with_face_negatives(batch.edge_index[i], batch.face_set[i], negatives[i],
                rows[0][i], rows[1][i], rows[2][i], rows[3][i], kl_weight=0.,
                pair_chunk_size=cp['args']['pair_chunk_size'], edge_logit_scale=scales['edge_logit_scale'],
                face_logit_scale=scales['face_logit_scale'], face_interval_factor=scales['face_interval_factor'])
            # Both groups execute identical objectives; the Face coefficient is the only intervention.
            losses.append(comp['edge'] + weight*comp['face'] + 0.*comp['kl'])
            components.append(comp)
        loss=torch.stack(losses).mean(); loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if step==1 or step%10==0:
            trace.append(dict(step=step, loss=float(loss.detach()), gradient_norm=float(grad),
                seconds=time.monotonic()-started, components=[{k:float(v.detach()) for k,v in c.items()} for c in components]))
            probe.write(ROOT/(variant+'_trace.json'),trace)
            print(json.dumps(dict(event='train',variant=variant,**trace[-1])),flush=True)
        if step%50==0 or step==steps:
            evaluations.append(evaluate_mu(model,batch,data,scales,variant,step))
            probe.write(ROOT/(variant+'_evaluations.json'),evaluations)
    cp['model']=model.state_dict();cp['optimizer']=optimizer.state_dict()
    cp['diagnostic_intervention']=dict(variant=variant,steps=steps,face_weight=weight,kl_weight=0.,training_mode='mu',
        starting_checkpoint=str(CHECKPOINT),starting_checkpoint_sha256=meta['checkpoint_sha256'],
        selected_uids=probe.UIDS,note='Two-mesh diagnostic; original step metadata is not a full-dataset training step.')
    torch.save(cp,ROOT/(variant+'_checkpoint.pt'))
    with torch.no_grad():
        outputs={mode:probe.get_rows(model,batch,mode,1000+(mode=='sample1')) for mode in ['mu','sample0','sample1']}
        probe.evaluate(ROOT,variant+'_final',outputs,scales)
    probe.write(ROOT/(variant+'_complete.json'),dict(steps=steps,seconds=time.monotonic()-started,
        max_gpu_gb=torch.cuda.max_memory_allocated()/2**30,passed_steps=[r['step'] for r in evaluations if r['both_meshes_pass']],
        final_two_evaluations_pass=all(r['both_meshes_pass'] for r in evaluations[-2:]),intervention=cp['diagnostic_intervention']))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=['joint','edge_only'],required=True)
    parser.add_argument('--steps',type=int,default=500)
    args=parser.parse_args()
    run(args.variant,args.steps)
