"""Read-only GPU equivalence and timing; no optimizer steps or source-state mutation."""
import argparse
from dataclasses import replace
from pathlib import Path
import time
from support import *
from native_models import Config, NativeTopologyAE, Graph
from data_objective import load_dataset, epoch_batches, negative_faces, objective


def main(source):
    configure(0)
    root = Path(__file__).resolve().parent
    items, manifest = load_dataset(Path(source)/'data', Path(source)/'pools')
    boundary = read(root/'performance/BOUNDARY.json')
    initial = read(root/'INITIALIZATION.json')['checkpoint']
    cases = []
    for arm, info, next_update in [('XYZ_LN_post', boundary['checkpoint'], boundary['completed_updates']+1),
                                   ('XYZ_LN_pre', initial, 1)]:
        assert sha(info['path']) == info['sha256']
        cp = torch.load(info['path'], map_location='cpu', mmap=True, weights_only=False)
        cfg = Config(model_variant='B_v2_teacher_blocks', **MODES[arm])
        model = NativeTopologyAE(cfg).cuda().float().train()
        model.load_state_dict(cp['model'], strict=True)
        params = [p for p in model.parameters() if p.requires_grad]
        epoch, cursor = divmod(next_update-1, 10)
        batches = epoch_batches(manifest['uids'], epoch)
        largest = max(batches, key=lambda b: max(len(items[u]['vertices']) for u in b))
        for label, uids in [('next_actual_batch', batches[cursor]), ('maximum_mesh_batch', largest)]:
            prepared = []
            for uid in uids:
                cpu = items[uid]
                item = {k:v.cuda() if torch.is_tensor(v) else v for k,v in cpu.items()}
                neg, _ = negative_faces(cpu, epoch)
                prepared.append((item, Graph.from_faces(item['faces'], len(item['vertices'])), neg))
            hashes, timings, peaks, losses = {}, {}, {}, {}
            for recompute in (True, False):
                model.cfg = replace(cfg, activation_checkpointing=recompute)
                key = 'recompute' if recompute else 'retain'
                seconds = []
                for repeat in range(3):
                    model.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.monotonic()
                    total = 0.
                    for item, graph, neg in prepared:
                        out = model(item['vertices'], item['faces'], graph=graph, sample_latent=False)
                        loss, _ = objective(out, item, neg)
                        total += float(loss.detach())/5
                        (loss/5).backward()
                        del out, loss
                    torch.cuda.synchronize()
                    seconds.append(time.monotonic()-start)
                hashes[key] = tensor_hash({str(i):p.grad for i,p in enumerate(params)})
                timings[key] = sum(seconds[1:])/2
                peaks[key] = torch.cuda.max_memory_allocated()
                losses[key] = total
            assert hashes['recompute'] == hashes['retain'], (arm, label, 'gradient mismatch')
            assert losses['recompute'] == losses['retain'], (arm, label, 'loss mismatch')
            case = dict(arm=arm, batch=label, uids=uids, gradient_hash=hashes['retain'],
                        seconds=timings, peak_allocated_bytes=peaks, loss=losses['retain'])
            cases.append(case)
            print(case, flush=True)
        del model, params, prepared, cp
        torch.cuda.empty_cache()
    write(root/'performance/EQUIVALENCE.json', dict(all_gradients_bitwise_equal=True,
          optimizer_updates=0, cases=cases, scope='real five-mesh forward/backward; excludes clip/Adam/I/O',
          performance_only_change='retain activation tensors instead of recomputing blocks'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    main(parser.parse_args().source)
