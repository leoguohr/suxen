"""Run one disjoint D4 evaluation shard on a second GPU."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    sys.path.insert(0, request['code_root'])
    from mini_nexus.data_2k import Nexus2KManifestDataset, collate_nexus2k_samples
    from mini_nexus.vertex import farthest_point_sample
    from scripts.evaluate_vertex_b2_frozen import base_config
    from scripts.train_vertex_a100_b1 import make_model
    from scripts.train_vertex_d2 import Evidence
    from scripts.train_vertex_d4 import condition_group
    from scripts.train_vertex_staged import autocast

    torch.set_num_threads(8)
    evidence = Evidence(Path(request['output']), Path(request['durable']))
    state = torch.load(request['checkpoint'], map_location='cpu', mmap=True, weights_only=False)
    assert state['step'] == request['cumulative_step'] and state.get('d4_update', 0) == request['update']
    source = base_config(state['config'])
    model = make_model('R1', source['seed'], source['smoke_model'])
    model.load_state_dict(state['model'], strict=True)
    model.to(request['device']).eval().requires_grad_(False)
    del state
    data = Nexus2KManifestDataset(request['manifest'], 'train')
    samples = [data[i] for i in range(len(data))]
    assert [s.uid for s in samples] == request['uids']
    batches = [collate_nexus2k_samples([s]).to(request['device']) for s in samples]
    indices = [farthest_point_sample(b.condition[..., :3], model.condition_encoder.num_tokens) for b in batches]
    rows = []
    with torch.no_grad(), autocast(SimpleNamespace(device=request['device'], precision=request['precision'])):
        contexts = [model.condition_encoder(b.condition, fps_indices=i) for b, i in zip(batches, indices)]
        for seed in request['seeds']:
            evidence.append('ledger.jsonl', {'event': 'seed_started', 'group': request['name'], 'update': request['update'], 'seed': seed})
            folder = evidence.output / f'seed-{seed}'
            rows.append(condition_group(model, contexts, samples, seed, folder))
            evidence.publish_folder(folder)
            evidence.write('progress.json', {'completed_groups': len(rows), 'rows': rows})
            evidence.append('ledger.jsonl', {'event': 'seed_completed', 'group': request['name'], 'update': request['update'], 'seed': seed})
    evidence.write('result.json', {'update': request['update'], 'cumulative_step': request['cumulative_step'], 'rows': rows})


if __name__ == '__main__':
    main()
