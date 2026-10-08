"""Create exactly one fresh random V2 state, shared by all four arms."""
import argparse
from pathlib import Path
from support import configure, save_torch, tensor_hash, rng_state, write
from native_models import Config, NativeTopologyAE
from data_objective import load_dataset


def main(source):
    root = Path(__file__).resolve().parent
    assert not (root/'initial.pt').exists(), 'Initialization already exists'
    configure(0)
    items, manifest = load_dataset(Path(source)/'data', Path(source)/'pools')
    cfg = Config(model_variant='B_v2_teacher_blocks')
    model = NativeTopologyAE(cfg)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 246575680
    entry = save_torch(root/'initial.pt', dict(model=model.state_dict(), model_config=cfg.to_dict(),
        seed=0, rng=rng_state(), completed_updates=0, origin='fresh random initialization, no learned weights'))
    write(root/'INITIALIZATION.json', dict(checkpoint=entry, tensor_hash=tensor_hash(model.state_dict()),
        data=manifest, all_arms_use_identical_parameter_tensors=True, optimizer='fresh AdamW in each arm'))
    print('INITIALIZATION_COMPLETE', entry['sha256'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    main(p.parse_args().source)
