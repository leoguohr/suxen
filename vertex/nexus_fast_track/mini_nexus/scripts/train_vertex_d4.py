"""Four-object D4 continuation from verified D2; preserve the D2 baseline."""
import argparse
import hashlib
from itertools import combinations
import json
import os
import subprocess
from pathlib import Path
import signal
import sys
import time
import numpy as np
import torch


def pair_for(update, micro):
    k = ((update - 1) * 8 + micro) % 36
    return k // 9, k % 9 + 1


def all_splits(samples):
    from scripts.train_vertex_d2 import first_split
    assert len(samples) == 4
    rows = []
    for i, j in combinations(range(4), 2):
        depth, parents, _ = first_split([samples[i], samples[j]])
        rows.append({'indices': [i, j], 'uids': [samples[i].uid, samples[j].uid],
                     'depth': depth, 'common_parents': parents.tolist()})
    return rows


def with_f1(metrics):
    tp = metrics['target_count'] - metrics['missing_count']
    denominator = 2 * tp + metrics['extra_count'] + metrics['missing_count']
    metrics['occupancy_f1'] = 2 * tp / denominator if denominator else 1.
    return metrics


@torch.no_grad()
def condition_group(model, contexts, samples, seed, folder):
    from scripts.train_vertex_d2 import first_split, cross_matches
    from scripts.train_vertex_c import sample_tree, occupancy_for
    from scripts.train_vertex_staged import noise_for
    from scripts.evaluate_vertex_b2_frozen import occupancy_metrics
    from mini_nexus.vertex_evaluation import sample_level
    from mini_nexus.octree import expand_occupied_children
    assert len(samples) == len(contexts) == 4
    folder.mkdir(parents=True, exist_ok=False)
    full, matrix, roots = [], [], []
    for index, (context, sample) in enumerate(zip(contexts, samples)):
        path = folder / f'full-{index}'
        trajectory = sample_tree(model, context, sample, seed, path)
        for row in trajectory['levels']: with_f1(row)
        if trajectory.get('capacity_abort_depth'):
            matches = [False] * 4
        else:
            with np.load(path / 'depth-9.npz') as saved:
                predicted = torch.from_numpy(saved['predicted_cells'])
                matches = cross_matches(predicted, [s.quantized_vertices for s in samples])
        with np.load(path / 'depth-1.npz') as saved:
            roots.append((saved['parents'], saved['noise']))
        trajectory.update(condition_uid=sample.uid, matches_gt_uids=[s.uid for s in samples], matches=matches)
        full.append(trajectory); matrix.append(matches)
    for parents, noise in roots[1:]:
        assert np.array_equal(parents, roots[0][0]) and np.array_equal(noise, roots[0][1])
    common_pairs = []
    for pair_index, (i, j) in enumerate(combinations(range(4), 2)):
        depth, common, targets = first_split([samples[i], samples[j]])
        common = common.to(contexts[0].device); targets = [x.to(common.device) for x in targets]
        noise_seed = seed + 500_000 + pair_index * 1000
        noise = noise_for(torch.empty((1, len(common), 8), device=common.device), noise_seed)
        shared_sha = hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest()
        path = folder / f'common-{i}-{j}'; path.mkdir()
        np.savez_compressed(path / 'input.npz', parents=common.cpu().numpy(), noise=noise.cpu().numpy(), depth=depth)
        switches, common_matrix = [], []
        for local_index, index in enumerate([i, j]):
            value = sample_level(model, contexts[index], common, depth, noise.clone(), steps=20)
            predicted = expand_occupied_children(common, value[0] >= .5)
            target = occupancy_for(common, targets[local_index])
            matches = cross_matches(predicted, targets)
            np.savez_compressed(path / f'condition-{index}.npz', parents=common.cpu().numpy(), noise=noise.cpu().numpy(),
                                estimate=value.cpu().numpy(), target_occupancy=target.cpu().numpy(),
                                predicted_cells=predicted.cpu().numpy(), target_cells=targets[local_index].cpu().numpy())
            metrics = with_f1(occupancy_metrics(value, target, predicted, targets[local_index]))
            switches.append({'condition_uid': samples[index].uid, 'depth': depth, 'matches': matches, **metrics})
            common_matrix.append(matches)
        common_pairs.append({'indices':[i,j], 'uids':[samples[i].uid,samples[j].uid], 'depth':depth,
                             'parent_count':len(common), 'noise_seed':noise_seed, 'noise_sha256':shared_sha,
                             'conditions':switches, 'match_matrix':common_matrix,
                             'common_pair_correct':common_matrix == [[True,False],[False,True]]})
    return {'seed':seed, 'uid_order':[s.uid for s in samples], 'full':full, 'full_match_matrix':matrix,
            'full_group_correct':matrix == np.eye(4,dtype=bool).tolist(),
            'common_parent_pairs':common_pairs, 'same_root_actual_noise':True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['code-root', 'checkpoint', 'manifest', 'selection', 'output', 'durable-dir', 'provenance']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--updates', type=int, default=7200)
    parser.add_argument('--eval-every', type=int, default=400)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--resume-update', type=int, default=0)
    parser.add_argument('--resume-evaluation', type=Path)
    parser.add_argument('--resume-training-log', type=Path)
    parser.add_argument('--activation-checkpointing', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--evaluation-gpu', help='Physical UUID of the second GPU for disjoint evaluation seeds')
    args = parser.parse_args()
    assert args.updates > 0 and args.eval_every > 0
    sys.path.insert(0, str(args.code_root.resolve()))
    from mini_nexus.data_2k import Nexus2KManifestDataset, collate_nexus2k_samples
    from scripts.evaluate_vertex_b1_frozen import file_sha
    from scripts.evaluate_vertex_b2_frozen import base_config
    from scripts.train_vertex_c import gt_at_depth, occupancy_for
    from scripts.train_vertex_b2 import verify_restored_optimizer
    from scripts.train_vertex_a100_b1 import make_model
    from scripts.train_vertex_staged import autocast
    import shutil
    from scripts.train_vertex_d2 import Evidence
    torch.set_num_threads(8)
    evidence = Evidence(args.output, args.durable_dir)
    evidence.write('status.json', {'state': 'loading', 'update': args.resume_update, 'training_complete': False, 'final_evaluation_complete': False})
    shutil.copyfile(args.provenance, args.output / 'provenance.tar.gz')
    provenance = evidence.publish('provenance.tar.gz')
    sha = file_sha(args.checkpoint); assert sha == args.expected_sha
    state = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    source = state['config']; original = base_config(source)
    if args.resume_update:
        assert 0 < args.resume_update < args.updates
        assert source['phase'] == 'D4' and state['d4_update'] == args.resume_update
        base_step = source['base_d2_step']
        assert state['step'] == base_step + args.resume_update
        assert source['updates'] == args.updates and source['eval_every'] == args.eval_every
        assert args.resume_evaluation and args.resume_training_log
    else:
        assert state['step'] == 7400 and state['d2_update'] == 3600 and source['phase'] == 'D2'
        base_step = state['step']
    selection = json.loads(args.selection.read_text())
    if args.resume_update:
        assert selection == source['selection']
    assert source['lr'] == 1e-5 and source['weight_decay'] == 0 and source['clip'] == 1 and source['accumulation'] == 8
    if not args.resume_update:
        assert selection['uids'][:2] == source['uids']
        assert selection['records'][:2] == source['selection']['records']
    data = Nexus2KManifestDataset(args.manifest, 'train'); assert len(data) == 4
    samples = [data[i] for i in range(4)]
    assert [s.uid for s in samples] == selection['uids'] and samples[0].uid == 'nexus_2k_000105'
    for sample, record in zip(samples, selection['records']):
        assert sample.uid == record['uid'] and len(sample.quantized_vertices) == record['vertices']
        assert hashlib.sha256(sample.condition.numpy().tobytes()).hexdigest() == record['condition_sha256']
        assert hashlib.sha256(sample.quantized_vertices.numpy().tobytes()).hexdigest() == record['quantized_vertices_sha256']
        for depth, level in enumerate(sample.octree_levels, 1):
            assert torch.equal(occupancy_for(level.parent_codes, gt_at_depth(sample.quantized_vertices, depth))[0], level.target)
    assert len({r['condition_sha256'] for r in selection['records']}) == 4
    assert len({r['quantized_vertices_sha256'] for r in selection['records']}) == 4
    split_records = all_splits(samples)
    assert split_records == selection['pair_splits']
    model = make_model('R1', original['seed'], original['smoke_model'])
    model.load_state_dict(state['model'], strict=True)
    model.to(args.device).train().requires_grad_(True)
    model.condition_encoder.use_checkpoint = args.activation_checkpointing
    model.flow.use_checkpoint = args.activation_checkpointing
    assert all(p.requires_grad for p in model.parameters())
    assert sum(p.numel() for p in model.parameters()) == source['parameter_count']
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0., foreach=False)
    optimizer.load_state_dict(state['optimizer'])
    compared = verify_restored_optimizer(optimizer, state['optimizer'])
    torch.set_rng_state(state['torch_rng'])
    if args.device == 'cuda': torch.cuda.set_rng_state_all(state['cuda_rng'])
    assert torch.equal(torch.get_rng_state(), state['torch_rng'])
    if args.device == 'cuda': assert all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),state['cuda_rng']))
    args.precision = source['precision']
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    config.update(phase='D4', variant='R1', source_step=base_step+args.resume_update,
                  source_d2_update=3600, source_config=source,
                  base_d2_step=base_step, resumed_d4_update=args.resume_update,
                  source_checkpoint_sha256=sha, uids=selection['uids'], selection=selection,
                  parameter_count=source['parameter_count'], smoke_model=original['smoke_model'],
                  lr=1e-5, weight_decay=0., clip=1., accumulation=8, new_warmup=False,
                  depth_object_schedule='k=((update-1)*8+micro)%36; mesh=k//9; depth=k%9+1',
                  development_seeds=list(range(29_000_000,29_000_004)), final_seeds=list(range(30_000_000,30_000_016)),
                  full_tree_step_count=20, threshold=.5, pair_splits=split_records,
                  common_switch_is_diagnostic=True, trainable_vecset=True, provenance=provenance, entry_sha256=file_sha(Path(__file__)))
    evidence.write('config.json', config); evidence.write('selection.json', selection)
    del state
    batches = [collate_nexus2k_samples([s]).to(args.device) for s in samples]
    from mini_nexus.vertex import farthest_point_sample
    fixed_fps_indices = [farthest_point_sample(b.condition[..., :3], model.condition_encoder.num_tokens) for b in batches]
    evidence.write('fixed_fps_indices.json', {'per_uid': {
        sample.uid: indices.cpu().tolist() for sample, indices in zip(samples, fixed_fps_indices)},
        'condition_features_cached': False})
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    def evaluate(update, seeds, name):
        model.eval()
        cpu = torch.get_rng_state(); cuda = torch.cuda.get_rng_state_all() if args.device=='cuda' else []
        rows = []; manifest = {}
        worker = None
        local_seeds = seeds
        if args.evaluation_gpu:
            local_seeds, worker_seeds = seeds[:len(seeds)//2], seeds[len(seeds)//2:]
            request = dict(code_root=str(args.code_root), manifest=str(args.manifest),
                           checkpoint=str(args.checkpoint if update == args.resume_update else args.output/'checkpoint-last.pt'),
                           output=str(args.output/f'{name}-worker'), durable=str(args.durable_dir/f'{name}-worker'),
                           device=args.device, precision=args.precision, update=update, cumulative_step=base_step+update,
                           uids=config['uids'], seeds=worker_seeds, name=name)
            evidence.write(f'{name}-worker-request.json', request)
            worker_log = (args.output/f'{name}-worker.log').open('w')
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.evaluation_gpu)
            worker = subprocess.Popen([sys.executable, str(args.code_root/'scripts/evaluate_vertex_d4_worker.py'),
                                       str(args.output/f'{name}-worker-request.json')],
                                      env=environment, stdout=worker_log, stderr=subprocess.STDOUT)
        with torch.no_grad(), autocast(args):
            contexts = [model.condition_encoder(batch.condition, fps_indices=indices)
                        for batch, indices in zip(batches, fixed_fps_indices)]
            for seed in local_seeds:
                evidence.append('evaluation_ledger.jsonl', {'event':'seed_started','group':name,'update':update,'seed':seed})
                folder = args.output/name/f'seed-{seed}'
                row = condition_group(model, contexts, samples, seed, folder); rows.append(row)
                manifest.update(evidence.publish_folder(folder))
                evidence.write(f'{name}/progress.json', {'update':update,'completed_groups':len(rows),'rows':rows})
                evidence.append('evaluation_ledger.jsonl', {'event':'seed_completed','group':name,'update':update,'seed':seed})
        if worker is not None:
            returncode = worker.wait()
            worker_log.close()
            evidence.publish(f'{name}-worker.log')
            assert returncode == 0, f'Evaluation worker failed: {name}-worker.log'
            worker_root = Path(request['output'])
            result = json.loads((worker_root/'result.json').read_text())
            assert result['update'] == update and result['cumulative_step'] == base_step+update
            assert [r['seed'] for r in result['rows']] == worker_seeds
            for row in result['rows']:
                folder = args.output/name/f"seed-{row['seed']}"
                (worker_root/f"seed-{row['seed']}").rename(folder)
                manifest.update(evidence.publish_folder(folder))
                rows.append(row)
            for line in (worker_root/'ledger.jsonl').read_text().splitlines():
                evidence.append('evaluation_ledger.jsonl', json.loads(line))
        assert [r['seed'] for r in rows] == list(seeds)
        report = {'phase':'D4','update':update,'cumulative_step':base_step+update,'uid_order':config['uids'],
                  'groups':rows,'full_groups_correct':sum(r['full_group_correct'] for r in rows),'group_count':len(rows),
                  'full_trees_correct':sum(r['full_match_matrix'][i][i] for r in rows for i in range(4)),
                  'full_condition_switch_passed':all(r['full_group_correct'] for r in rows),
                  'common_parent_pairs_correct':sum(p['common_pair_correct'] for r in rows for p in r['common_parent_pairs']),
                  'common_parent_pair_count':6*len(rows),
                  'common_switch_is_diagnostic':True,'array_sha256':manifest}
        evidence.write(f'{name}.json',report)
        assert torch.equal(cpu,torch.get_rng_state())
        if args.device=='cuda': assert all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all()))
        model.train(); return report
    evidence.write('status.json',{'state':'initial_evaluation','update':args.resume_update})
    initial_name = f'evaluation-{args.resume_update:06d}'
    initial_report = evaluate(args.resume_update,config['development_seeds'],initial_name)
    reference_rows = {}
    compared_arrays = 0
    if args.resume_update:
        expected_report = json.loads(args.resume_evaluation.read_text())
        assert expected_report['update'] == args.resume_update
        assert initial_report['uid_order'] == expected_report['uid_order']
        for relative, expected_sha in expected_report['array_sha256'].items():
            old_path = args.resume_evaluation.parent / relative
            assert file_sha(old_path) == expected_sha
            old, new = np.load(old_path), np.load(args.output / relative)
            assert set(old.files) == set(new.files)
            for key in old.files:
                if key == 'estimate':
                    assert np.allclose(old[key],new[key],rtol=1e-5,atol=1e-6), (relative,key)
                else:
                    assert np.array_equal(old[key],new[key]), (relative,key)
            compared_arrays += 1
        for line in args.resume_training_log.read_text().splitlines():
            row = json.loads(line)
            if row['update'] > args.resume_update: reference_rows[row['update']] = row
        evidence.write('recovery_verification.json',{'resumed_d4_update':args.resume_update,
            'compared_prior_evaluation_arrays':compared_arrays,'replay_reference_updates':sorted(reference_rows),
            'evaluation_is_seen_recovery_check':True,'remaining_updates':args.updates-args.resume_update})
    evidence.write('resume_verification.json',{'source_step':base_step+args.resume_update,'resumed_d4_update':args.resume_update,'checkpoint_sha256':sha,
        'model_loaded_strictly':True,'optimizer_tensors_equal':compared,'optimizer_groups_equal':True,
        'rng_restored_and_preserved_through_initial_evaluation':True,'all_parameters_trainable':True,
        'vecset_training':model.condition_encoder.training,'dit_training':model.flow.training,'first_lr':1e-5,'new_warmup':False})
    tracked = {'vecset_point_embedding':model.condition_encoder.point_embedding.weight,
               'dit_input':model.flow.data_embedding.weight,'dit_output':model.flow.output.weight}
    enc_calls = 0
    def count_forward(module, inputs):
        nonlocal enc_calls
        enc_calls += 1
    handle = model.condition_encoder.register_forward_pre_hook(count_forward)
    step = args.resume_update
    for step in range(args.resume_update+1,args.updates+1):
        if stopped: step -= 1; break
        start = time.monotonic(); optimizer.zero_grad(set_to_none=True); consumed=[]; before_calls=enc_calls
        assert model.training and model.condition_encoder.training and all(p.requires_grad for p in model.parameters())
        for micro in range(8):
            index, depth = pair_for(step,micro); batch=batches[index];level=batch.octree_levels[depth-1]
            assert batch.uids==(samples[index].uid,)
            times=torch.rand((1,),device=args.device);noise=torch.randn_like(level.target)
            if step in reference_rows:
                previous = reference_rows[step]['microbatches'][micro]
                assert (previous['uid'],previous['depth'],previous['time']) == (samples[index].uid,depth,float(times))
                assert previous['noise_sha256'] == hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest()
            with autocast(args): loss=model(batch.condition,level,noise=noise,time=times,
                                           condition_fps_indices=fixed_fps_indices[index])
            assert torch.isfinite(loss); (loss/8).backward()
            consumed.append({'uid':samples[index].uid,'mesh_index':index,'depth':depth,'time':float(times),
                'noise_sha256':hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest(),'loss':float(loss)})
        assert enc_calls-before_calls==8,'VecSet must execute for every training microbatch'
        gradients={k:None if p.grad is None else float(p.grad.norm()) for k,p in tracked.items()}
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm)
        before={k:p.detach().clone() for k,p in tracked.items()};optimizer.step()
        updates={k:float((p.detach()-before[k]).norm()) for k,p in tracked.items()};del before
        row={'update':step,'cumulative_step':base_step+step,'microbatches':consumed,'loss':sum(x['loss'] for x in consumed)/8,
             'lr':optimizer.param_groups[0]['lr'],'weight_decay':0.,'accumulation':8,'gradient_norm_before_clip':float(norm),
             'component_gradients':gradients,'component_update_norms':updates,'vecset_forward_calls':enc_calls-before_calls,
             'seconds':time.monotonic()-start,'matches_original_replayed_input':step in reference_rows}
        evidence.append('train.jsonl',row);print(json.dumps(row),flush=True)
        evidence.write('status.json',{'state':'training','update':step,'training_complete':step==args.updates})
        if step==args.updates:evidence.write('training_complete.json',{'update':step,'cumulative_step':base_step+step,'complete':True})
        if step%args.eval_every==0 or step==args.updates or stopped:
            optimizer.zero_grad(set_to_none=True)
            evidence.write('status.json',{'state':'saving_checkpoint','update':step})
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'config':config,'step':base_step+step,
                'd4_update':step,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if args.device=='cuda' else None},args.output/'checkpoint-last.pt.tmp')
            (args.output/'checkpoint-last.pt.tmp').replace(args.output/'checkpoint-last.pt')
            saved=evidence.publish('checkpoint-last.pt');saved.update(phase='D4',d4_update=step,cumulative_step=base_step+step)
            evidence.write('checkpoint_verified.json',saved)
            evidence.write('status.json',{'state':'evaluating','update':step,'checkpoint_saved':True})
            evaluate(step,config['development_seeds'],f'evaluation-{step:06d}')
    handle.remove()
    final=None
    if step==args.updates and not stopped:
        evidence.write('final_evaluation_started.json',{'update':step,'seeds':config['final_seeds'],'no_further_parameter_updates':True})
        final=evaluate(step,config['final_seeds'],'final_once')
        evidence.write('final_evaluation_complete.json',{'update':step,'completed':True,'groups':16,'full_condition_switch_passed':final['full_condition_switch_passed']})
    result={'phase':'D4','updates':step,'training_complete':step==args.updates,'stopped':stopped,'D10_started':False,
            'final_evaluation_complete':final is not None,'full_condition_switch_passed':bool(final and final['full_condition_switch_passed']),
            'full_groups_correct':final['full_groups_correct'] if final else None,
            'full_trees_correct':final['full_trees_correct'] if final else None,
            'common_parent_pairs_correct':final['common_parent_pairs_correct'] if final else None}
    evidence.write('result.json',result);evidence.write('status.json',{'state':'stopped' if stopped else 'complete',**result})


if __name__=='__main__': main()
