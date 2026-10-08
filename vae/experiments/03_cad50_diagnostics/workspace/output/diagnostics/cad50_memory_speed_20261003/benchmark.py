"""Read-only five-mesh forward/backward benchmark. Never construct an optimizer."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import torch


def main(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    assert not (output/'results.json').exists(), 'Refuse duplicate benchmark'
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source))
    from support import configure, sha, tensor_hash, write
    from native_models import Config, NativeTopologyAE, Graph
    from data_objective import load_dataset, epoch_batches, negative_faces, objective
    active = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
    assert not active.strip(), 'Assigned GPU is in use; do not interfere'
    configure(0)
    assert torch.cuda.device_count() == 1
    items, manifest = load_dataset(Path('/guohaoran/nexus_fast_track/diagnostics/own512_v2_recipe_pair_20260923_resume_01')/'data',
                                   Path('/guohaoran/nexus_fast_track/diagnostics/own512_v2_recipe_pair_20260923_resume_01')/'pools')
    batches = epoch_batches(manifest['uids'], 0)
    largest = max(items, key=lambda u: len(items[u]['vertices']))
    largest_batch = next(b for b in batches if largest in b)
    representative = batches[0]
    gpu_items = {u: {k: v.cuda() if torch.is_tensor(v) else v for k,v in item.items()} for u,item in items.items()}
    graphs = {u: Graph.from_faces(x['faces'],len(x['vertices'])) for u,x in gpu_items.items()}
    negatives = {u: negative_faces(x,0)[0] for u,x in items.items()}
    evidence = dict(scope='zero optimizer updates; real five-mesh accumulation forward/backward only',
        optimizer_updates=0, torch=torch.__version__, cuda=torch.version.cuda,
        gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'], largest_uid=largest, largest_vertices=len(items[largest]['vertices']),
        representative_batch=representative, largest_batch=largest_batch, arms={})
    for arm in ('V2_control','Graph_LN_pre','Graph_SiLU','Encoder_ReLU'):
        path = source/'runs'/arm/'checkpoint-02000.pt'
        digest = sha(path)
        checkpoint = torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        assert checkpoint['completed_updates'] == 2000
        model = NativeTopologyAE(Config(**checkpoint['model_config'])).cuda().float().train()
        model.load_state_dict(checkpoint['model'],strict=True)
        original_hash = tensor_hash(model.state_dict())
        parameters = [p for p in model.parameters() if p.requires_grad]
        adam_moments_bytes = 2*sum(p.numel()*p.element_size() for p in parameters)
        records = {}
        reference_gradients = reference_losses = None

        def run_batch(batch):
            model.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            losses = []
            for uid in batch:
                item = gpu_items[uid]
                values = model(item['vertices'],item['faces'],sample_latent=False,graph=graphs[uid])
                assert values['latent'] is values['mu']
                loss,_ = objective(values,item,negatives[uid])
                (loss/5).backward()
                losses.append(float(loss.detach()))
                del values,loss
            torch.cuda.synchronize()
            result = dict(seconds=time.perf_counter()-start,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(),losses=losses)
            return result

        for recompute in (True,False):
            model.cfg = replace(model.cfg,activation_checkpointing=recompute)
            run_batch(largest_batch)  # kernel warmup only; no optimizer
            check = run_batch(largest_batch)
            gradients = [p.grad.detach().cpu().clone() for p in parameters]
            assert all(torch.isfinite(g).all() for g in gradients)
            assert all(p.grad is None for p in model.log_variance.parameters())
            if recompute:
                reference_gradients, reference_losses = gradients, check['losses']
            else:
                assert reference_losses == check['losses'], 'Forward objective changed'
                equal = all(torch.equal(x,y) for x,y in zip(reference_gradients,gradients))
                error = sum((x.double()-y.double()).square().sum().item() for x,y in zip(reference_gradients,gradients))
                norm = sum(x.double().square().sum().item() for x in reference_gradients)
                relative = (error/max(norm,1e-300))**.5
                assert relative <= 1e-6, 'Parameter gradients materially changed'
                records['equivalence'] = dict(forward_losses_bitwise_equal=True,
                    parameter_gradients_bitwise_equal=equal,gradient_relative_l2_error=relative)
                del gradients
            times = [run_batch(representative) for _ in range(3)]
            maximum = [run_batch(largest_batch) for _ in range(3)]
            records['recompute_on' if recompute else 'recompute_off'] = dict(
                representative=times,largest_batch=maximum,
                representative_median_seconds=statistics.median(x['seconds'] for x in times),
                largest_batch_median_seconds=statistics.median(x['seconds'] for x in maximum),
                max_allocated_bytes=max(x['peak_allocated_bytes'] for x in times+maximum),
                max_reserved_bytes=max(x['peak_reserved_bytes'] for x in times+maximum),
                hypothetical_Adam_moments_bytes_not_allocated=adam_moments_bytes)
        del reference_gradients
        assert original_hash == tensor_hash(model.state_dict())
        assert digest == sha(path), 'Source checkpoint changed'
        records['checkpoint_sha256'] = digest
        records['parameters_and_buffers_unchanged'] = True
        records['representative_speedup'] = records['recompute_on']['representative_median_seconds']/records['recompute_off']['representative_median_seconds']
        records['largest_batch_speedup'] = records['recompute_on']['largest_batch_median_seconds']/records['recompute_off']['largest_batch_median_seconds']
        evidence['arms'][arm] = records
        write(output/'results.json',evidence)
        print(arm,'SPEEDUP',records['representative_speedup'],records['largest_batch_speedup'],
              'MEMORY',records['recompute_on']['max_allocated_bytes'],records['recompute_off']['max_allocated_bytes'],
              'GRADIENT_EQUIVALENCE',records['equivalence'],flush=True)
        del model, parameters, checkpoint
        torch.cuda.empty_cache()
    evidence['complete'] = True
    evidence['limitation'] = 'No optimizer.step and no four-process throughput test; full-training peak includes Adam, clipping and step temporaries.'
    write(output/'results.json',evidence)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    main(args.source,args.output)
