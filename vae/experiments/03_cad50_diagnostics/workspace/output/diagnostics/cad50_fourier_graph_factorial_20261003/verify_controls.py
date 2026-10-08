"""Verify two historical step0 controls on this GPU before reusing their results."""
import argparse
from pathlib import Path
from support import *
from native_models import Config, NativeTopologyAE
from data_objective import load_dataset
from evaluation import evaluate


def main(old_code, source):
    configure(0)
    root, old = Path(__file__).resolve().parent, Path(old_code)
    manifest = read(root/'INITIALIZATION.json')
    checkpoint = manifest['checkpoint']
    assert sha(checkpoint['path']) == checkpoint['sha256']
    initial = torch.load(checkpoint['path'], map_location='cpu', mmap=True, weights_only=False)
    assert tensor_hash(initial['model']) == manifest['tensor_hash']
    items, data = load_dataset(Path(source)/'data', Path(source)/'pools')
    reference = read(root/'reference/config.json')
    assert data == reference['data']
    assert torch.__version__ == reference['torch'] and torch.version.cuda == reference['cuda']
    assert torch.cuda.get_device_name(0) == reference['gpu']
    results = []
    for arm, position in [('V2_control', 'post_projection'), ('Graph_LN_pre', 'pre_aggregation')]:
        model = NativeTopologyAE(Config(model_variant='B_v2_teacher_blocks', graph_norm_position=position)).cuda()
        model.load_state_dict(initial['model'], strict=True)
        actual = evaluate(model, items, root/'control_checks'/arm, checkpoint)
        expected = read(old/f'runs/{arm}/evaluations/step-00000/evaluation.json')
        for a, b in zip(actual['meshes'], expected['meshes']):
            for key in ('uid', 'vertices', 'edge', 'face', 'face_fn_missing', 'face_fn_present',
                        'actual_face_candidates', 'joint_strict', 'complete'):
                assert a[key] == b[key], (arm, a['uid'], key, a[key], b[key])
        results.append(dict(arm=arm, all_50_step0_counts_match=True, counts=actual['counts']))
        del model
        torch.cuda.empty_cache()
    write(root/'CONTROL_GATE.json', dict(passed=True, initial_sha256=checkpoint['sha256'],
          gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'], results=results,
          limitation='Step0 inference parity does not guarantee bitwise long training trajectory on another GPU.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--old-code', required=True)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    main(args.old_code, args.source)
