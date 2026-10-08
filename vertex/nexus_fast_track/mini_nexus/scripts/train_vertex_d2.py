"""Two-object, nine-depth continuation with paired condition tests and durable evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import numpy as np
import torch


def pair_for(update, micro):
    k = ((update - 1) * 8 + micro) % 18
    return k // 9, k % 9 + 1


def cells_set(value):
    return set(map(tuple, value.cpu().tolist()))


def cross_matches(cells, targets):
    return [cells_set(cells) == cells_set(gt) for gt in targets]


def first_split(samples):
    from scripts.train_vertex_c import gt_at_depth
    for depth in range(1, 10):
        targets = [gt_at_depth(s.quantized_vertices, depth) for s in samples]
        if cells_set(targets[0]) != cells_set(targets[1]):
            parents = [torch.unique(s.quantized_vertices // 2 ** (10 - depth), dim=0) for s in samples]
            assert torch.equal(parents[0], parents[1])
            return depth, parents[0], targets
    raise AssertionError('Different UIDs must have different quantized targets')


class Evidence:
    def __init__(self, output, durable):
        output.mkdir(parents=True, exist_ok=False)
        durable.mkdir(parents=True, exist_ok=False)
        self.output, self.durable = output, durable

    def publish(self, relative):
        source, target = self.output / relative, self.durable / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.partial')
        expected = hashlib.sha256()
        with source.open('rb') as reader, temporary.open('wb') as writer:
            for block in iter(lambda: reader.read(16 * 1024**2), b''):
                writer.write(block); expected.update(block)
            writer.flush(); os.fsync(writer.fileno())
        actual = hashlib.sha256()
        with temporary.open('rb') as reader:
            for block in iter(lambda: reader.read(16 * 1024**2), b''): actual.update(block)
        assert actual.hexdigest() == expected.hexdigest()
        temporary.replace(target)
        return {'path': str(target), 'bytes': target.stat().st_size, 'sha256': actual.hexdigest(), 'verified_readback': True}

    def write(self, relative, value):
        path = self.output / relative; path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + '.tmp'); tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(path)
        return self.publish(relative)

    def append(self, relative, row):
        line = json.dumps(row) + '\n'
        for root in [self.output, self.durable]:
            with (root / relative).open('a') as writer:
                writer.write(line); writer.flush(); os.fsync(writer.fileno())

    def publish_folder(self, folder):
        return {str(p.relative_to(self.output)): self.publish(p.relative_to(self.output))['sha256']
                for p in sorted(folder.rglob('*')) if p.is_file()}


@torch.no_grad()
def condition_pair(model, contexts, samples, seed, folder):
    from scripts.train_vertex_c import sample_tree, occupancy_for
    from scripts.train_vertex_staged import noise_for
    from scripts.evaluate_vertex_b2_frozen import occupancy_metrics
    from mini_nexus.vertex_evaluation import sample_level
    from mini_nexus.octree import expand_occupied_children
    folder.mkdir(parents=True, exist_ok=False)
    full, full_matrix = [], []
    targets = [s.quantized_vertices.to(contexts[0].device) for s in samples]
    for index, (context, sample) in enumerate(zip(contexts, samples)):
        path = folder / f'full-{index}'
        trajectory = sample_tree(model, context, sample, seed, path)
        if trajectory.get('capacity_abort_depth'):
            matches = [False, False]
        else:
            saved = np.load(path / 'depth-9.npz')
            predicted = torch.from_numpy(saved['predicted_cells'])
            matches = cross_matches(predicted, [s.quantized_vertices for s in samples])
        trajectory.update(condition_uid=sample.uid, matches_gt_uids=[s.uid for s in samples], matches=matches)
        full.append(trajectory); full_matrix.append(matches)
    # The same root/noise must be used by both full rollouts; later parents are model-generated.
    a, b = [np.load(folder / f'full-{i}/depth-1.npz') for i in range(2)]
    assert np.array_equal(a['parents'], b['parents']) and np.array_equal(a['noise'], b['noise'])
    depth, common, layer_targets = first_split(samples)
    common = common.to(contexts[0].device)
    layer_targets = [x.to(common.device) for x in layer_targets]
    noise = noise_for(torch.empty((1, len(common), 8), device=common.device), seed + 500_000)
    shared_sha = hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest()
    # Actual shared input is saved once before either condition is sampled.
    np.savez_compressed(folder / 'common-input.npz', parents=common.cpu().numpy(), noise=noise.cpu().numpy(), depth=depth)
    switches, switch_matrix = [], []
    for index, context in enumerate(contexts):
        value = sample_level(model, context, common, depth, noise.clone(), steps=20)
        predicted = expand_occupied_children(common, value[0] >= .5)
        target = occupancy_for(common, layer_targets[index])
        matches = cross_matches(predicted, layer_targets)
        np.savez_compressed(folder / f'common-{index}.npz', parents=common.cpu().numpy(), noise=noise.cpu().numpy(),
                            estimate=value.cpu().numpy(), target_occupancy=target.cpu().numpy(),
                            predicted_cells=predicted.cpu().numpy(), target_cells=layer_targets[index].cpu().numpy())
        switches.append({'condition_uid': samples[index].uid, 'depth': depth, 'noise_sha256': shared_sha,
                         'matches': matches, **occupancy_metrics(value, target, predicted, layer_targets[index])})
        switch_matrix.append(matches)
    return {'seed': seed, 'uid_order': [s.uid for s in samples], 'full': full, 'full_match_matrix': full_matrix,
            'full_pair_correct': full_matrix == [[True, False], [False, True]],
            'common_parent_switch': switches, 'common_match_matrix': switch_matrix,
            'common_pair_correct': switch_matrix == [[True, False], [False, True]],
            'same_root_actual_noise': True, 'common_noise_sha256': shared_sha}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['code-root', 'checkpoint', 'manifest', 'selection', 'output', 'durable-dir', 'provenance']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--updates', type=int, default=3600)
    parser.add_argument('--eval-every', type=int, default=200)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--resume-update', type=int, default=0)
    parser.add_argument('--resume-evaluation', type=Path)
    parser.add_argument('--resume-training-log', type=Path)
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
        assert source['phase'] == 'D2' and state['d2_update'] == args.resume_update
        assert state['step'] == 3800 + args.resume_update
        assert source['updates'] == args.updates and source['eval_every'] == args.eval_every
        assert args.resume_evaluation and args.resume_training_log
    else:
        assert state['step'] == 3800 and state['c_update'] == 1800 and source['phase'] == 'C'
    selection = json.loads(args.selection.read_text())
    if args.resume_update:
        assert selection == source['selection']
        assert source['lr'] == 1e-5 and source['weight_decay'] == 0 and source['clip'] == 1 and source['accumulation'] == 8
    data = Nexus2KManifestDataset(args.manifest, 'train'); assert len(data) == 2
    samples = [data[i] for i in range(2)]
    assert [s.uid for s in samples] == selection['uids'] and samples[0].uid == 'nexus_2k_000105'
    for sample, record in zip(samples, selection['records']):
        assert sample.uid == record['uid'] and len(sample.quantized_vertices) == record['vertices']
        assert hashlib.sha256(sample.condition.numpy().tobytes()).hexdigest() == record['condition_sha256']
        assert hashlib.sha256(sample.quantized_vertices.numpy().tobytes()).hexdigest() == record['quantized_vertices_sha256']
        for depth, level in enumerate(sample.octree_levels, 1):
            assert torch.equal(occupancy_for(level.parent_codes, gt_at_depth(sample.quantized_vertices, depth))[0], level.target)
    assert selection['records'][0]['condition_sha256'] != selection['records'][1]['condition_sha256']
    assert selection['records'][0]['condition_sha256'] == original['condition_sha256'][samples[0].uid]
    depth, common, _ = first_split(samples)
    assert depth == selection['first_different_depth'] and common.tolist() == selection['common_parents']
    model = make_model('R1', original['seed'], original['smoke_model'])
    model.load_state_dict(state['model'], strict=True)
    model.to(args.device).train().requires_grad_(True)
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
    config.update(phase='D2', variant='R1', source_step=3800+args.resume_update,
                  source_c_update=1800 if not args.resume_update else None, source_config=source,
                  base_c_step=3800, resumed_d2_update=args.resume_update,
                  source_checkpoint_sha256=sha, uids=selection['uids'], selection=selection,
                  parameter_count=source['parameter_count'], smoke_model=original['smoke_model'],
                  lr=1e-5, weight_decay=0., clip=1., accumulation=8, new_warmup=False,
                  depth_object_schedule='k=((update-1)*8+micro)%18; mesh=k//9; depth=k%9+1',
                  development_seeds=list(range(27_000_000,27_000_004)), final_seeds=list(range(28_000_000,28_000_016)),
                  full_tree_step_count=20, threshold=.5, common_parent_depth=depth,
                  common_switch_is_diagnostic=True, trainable_vecset=True, provenance=provenance, entry_sha256=file_sha(Path(__file__)))
    evidence.write('config.json', config); evidence.write('selection.json', selection)
    del state
    batches = [collate_nexus2k_samples([s]).to(args.device) for s in samples]
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    def evaluate(update, seeds, name):
        model.eval()
        cpu = torch.get_rng_state(); cuda = torch.cuda.get_rng_state_all() if args.device=='cuda' else []
        rows = []; manifest = {}
        with torch.no_grad(), autocast(args):
            contexts = [model.condition_encoder(batch.condition) for batch in batches]
            for seed in seeds:
                evidence.append('evaluation_ledger.jsonl', {'event':'seed_started','group':name,'update':update,'seed':seed})
                folder = args.output/name/f'pair-{seed}'
                row = condition_pair(model, contexts, samples, seed, folder); rows.append(row)
                manifest.update(evidence.publish_folder(folder))
                evidence.write(f'{name}/progress.json', {'update':update,'completed_pairs':len(rows),'rows':rows})
                evidence.append('evaluation_ledger.jsonl', {'event':'seed_completed','group':name,'update':update,'seed':seed})
        report = {'phase':'D2','update':update,'cumulative_step':3800+update,'uid_order':config['uids'],
                  'pairs':rows,'full_pairs_correct':sum(r['full_pair_correct'] for r in rows),'pair_count':len(rows),
                  'full_trees_correct':sum(r['full_match_matrix'][i][i] for r in rows for i in range(2)),
                  'full_condition_switch_passed':all(r['full_pair_correct'] for r in rows),
                  'common_parent_pairs_correct':sum(r['common_pair_correct'] for r in rows),
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
        evidence.write('recovery_verification.json',{'resumed_d2_update':args.resume_update,
            'compared_prior_evaluation_arrays':compared_arrays,'replay_reference_updates':sorted(reference_rows),
            'evaluation_is_seen_recovery_check':True,'remaining_updates':args.updates-args.resume_update})
    evidence.write('resume_verification.json',{'source_step':3800+args.resume_update,'resumed_d2_update':args.resume_update,'checkpoint_sha256':sha,
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
            with autocast(args): loss=model(batch.condition,level,noise=noise,time=times)
            assert torch.isfinite(loss); (loss/8).backward()
            consumed.append({'uid':samples[index].uid,'mesh_index':index,'depth':depth,'time':float(times),
                'noise_sha256':hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest(),'loss':float(loss)})
        assert enc_calls-before_calls==8,'VecSet must execute for every training microbatch'
        gradients={k:None if p.grad is None else float(p.grad.norm()) for k,p in tracked.items()}
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm)
        before={k:p.detach().clone() for k,p in tracked.items()};optimizer.step()
        updates={k:float((p.detach()-before[k]).norm()) for k,p in tracked.items()};del before
        row={'update':step,'cumulative_step':3800+step,'microbatches':consumed,'loss':sum(x['loss'] for x in consumed)/8,
             'lr':optimizer.param_groups[0]['lr'],'weight_decay':0.,'accumulation':8,'gradient_norm_before_clip':float(norm),
             'component_gradients':gradients,'component_update_norms':updates,'vecset_forward_calls':enc_calls-before_calls,
             'seconds':time.monotonic()-start,'matches_original_replayed_input':step in reference_rows}
        evidence.append('train.jsonl',row);print(json.dumps(row),flush=True)
        evidence.write('status.json',{'state':'training','update':step,'training_complete':step==args.updates})
        if step==args.updates:evidence.write('training_complete.json',{'update':step,'cumulative_step':3800+step,'complete':True})
        if step%args.eval_every==0 or step==args.updates or stopped:
            optimizer.zero_grad(set_to_none=True)
            evidence.write('status.json',{'state':'saving_checkpoint','update':step})
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'config':config,'step':3800+step,
                'd2_update':step,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if args.device=='cuda' else None},args.output/'checkpoint-last.pt.tmp')
            (args.output/'checkpoint-last.pt.tmp').replace(args.output/'checkpoint-last.pt')
            saved=evidence.publish('checkpoint-last.pt');saved.update(phase='D2',d2_update=step,cumulative_step=3800+step)
            evidence.write('checkpoint_verified.json',saved)
            evidence.write('status.json',{'state':'evaluating','update':step,'checkpoint_saved':True})
            evaluate(step,config['development_seeds'],f'evaluation-{step:06d}')
    handle.remove()
    final=None
    if step==args.updates and not stopped:
        evidence.write('final_evaluation_started.json',{'update':step,'seeds':config['final_seeds'],'no_further_parameter_updates':True})
        final=evaluate(step,config['final_seeds'],'final_once')
        evidence.write('final_evaluation_complete.json',{'update':step,'completed':True,'pairs':16,'full_condition_switch_passed':final['full_condition_switch_passed']})
    result={'phase':'D2','updates':step,'training_complete':step==args.updates,'stopped':stopped,'D4_started':False,
            'final_evaluation_complete':final is not None,'full_condition_switch_passed':bool(final and final['full_condition_switch_passed']),
            'full_pairs_correct':final['full_pairs_correct'] if final else None,
            'common_parent_pairs_correct':final['common_parent_pairs_correct'] if final else None}
    evidence.write('result.json',result);evidence.write('status.json',{'state':'stopped' if stopped else 'complete',**result})


if __name__=='__main__': main()
