"""Synthetic CPU checks of the two factors; no optimizer update or GPU call."""
import argparse
import importlib.util
import sys
from pathlib import Path
import torch
from support import configure, tensor_hash, write
from native_models import Config, NativeTopologyAE, Graph
from data_objective import load_dataset, epoch_batches, negative_faces, objective


def main(old_code, source):
    configure(0)
    assert not torch.cuda.is_available(), 'Run with CUDA_VISIBLE_DEVICES empty'
    spec = importlib.util.spec_from_file_location('prior_native', Path(old_code)/'native_models.py')
    prior = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = prior
    spec.loader.exec_module(prior)
    dims = dict(model_variant='B_v2_teacher_blocks', encoder_width=16, latent_width=16,
                decoder_width=32, encoder_composite_blocks=2, decoder_blocks=2, heads=4)
    original = prior.NativeTopologyAE(prior.Config(**dims))
    vertices = torch.randn(7, 3)
    faces = torch.tensor([[0, 1, 2], [2, 3, 4], [4, 5, 6]])
    pairs = torch.triu_indices(7, 7, 1).T
    item = dict(uid='synthetic', vertices=vertices, faces=faces, gt_faces=faces,
                pairs=pairs, edge_labels=torch.arange(len(pairs))%2 == 0)
    graph = Graph.from_faces(faces, len(vertices))
    negatives, _ = negative_faces(item, 0)
    checked = []
    for position in ('post_projection', 'pre_aggregation'):
        for encoding in ('fourier', 'xyz_only'):
            cfg = Config(**dims, graph_norm_position=position, coordinate_encoding=encoding)
            model = NativeTopologyAE(cfg)
            model.load_state_dict(original.state_dict(), strict=True)
            assert tensor_hash(model.state_dict()) == tensor_hash(original.state_dict())
            outputs = model(vertices, faces, graph=graph)
            loss, _ = objective(outputs, item, negatives, pair_chunk=5, face_chunk=3)
            loss.backward()
            assert torch.isfinite(loss)
            assert outputs['latent'] is outputs['mu']
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in model.parameters() if p.requires_grad)
            assert all(p.grad is None for p in model.log_variance.parameters())
            if encoding == 'xyz_only':
                features = model.position_features(vertices)
                assert torch.equal(features[:, :3], vertices) and torch.count_nonzero(features[:, 3:]) == 0
                compact = torch.nn.functional.linear(vertices, model.vertex_input.weight[:, :3],
                                                     model.vertex_input.bias)
                assert torch.allclose(model.vertex_input(features), compact, atol=1e-7, rtol=1e-6)
                for projection in (model.vertex_input, model.face_input):
                    assert torch.count_nonzero(projection.weight.grad[:, 3:]) == 0
            else:
                reference = prior.NativeTopologyAE(prior.Config(**dims, graph_norm_position=position))
                reference.load_state_dict(original.state_dict())
                expected = reference(vertices, faces, graph=graph)
                assert all(torch.equal(outputs[k], expected[k]) for k in outputs)
                objective(expected, item, negatives, pair_chunk=5, face_chunk=3)[0].backward()
                for (name, param), (other_name, other) in zip(model.named_parameters(), reference.named_parameters()):
                    assert name == other_name
                    assert ((param.grad is None and other.grad is None) or
                            (param.grad is not None and other.grad is not None and torch.equal(param.grad, other.grad)))
            checked.append(dict(encoding=encoding, norm_position=position, finite_loss=float(loss.detach())))
    items, manifest = load_dataset(Path(source)/'data', Path(source)/'pools')
    for epoch in range(200):
        batches = epoch_batches(manifest['uids'], epoch)
        assert len(batches) == 10 and sorted(sum(batches, [])) == manifest['uids']
    result = dict(scope='synthetic CPU forward/backward plus real CAD50 data integrity; zero optimizer updates',
                  torch=torch.__version__, same_initial_tensors=True,
                  Fourier_controls_forward_and_gradients_bitwise_equal=True,
                  XYZ_only_equivalent_to_compact_linear=True, logvar_frozen=True,
                  real_data_totals=manifest['totals'], all_200_epochs_cover_50_once=True, cases=checked)
    write(Path(__file__).parent/'CPU_TESTS.json', result)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--old-code', required=True)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    main(args.old_code, args.source)
