"""Create fresh paired random models; reads no trained checkpoints."""
from run_support import ROOT, configure, write, save_torch, code_hashes, tensor_hash
from native_models import Config, NativeTopologyAE, VARIANTS


def main():
    destination = ROOT/'initialization'
    assert not destination.exists()
    destination.mkdir()
    configure(0)
    a = NativeTopologyAE(Config(model_variant=VARIANTS[0]))
    source = a.state_dict()
    first = save_torch(destination/(VARIANTS[0]+'.pt'),dict(model=source,model_config=a.cfg.to_dict(),
        initialization='random seed0',teacher_weights_used=False,trained_weights_used=False))
    configure(0)
    b = NativeTopologyAE(Config(model_variant=VARIANTS[1]))
    target = b.state_dict()
    shared = [n for n in source if n in target and source[n].shape == target[n].shape]
    for name in shared:target[name].copy_(source[name])
    b.load_state_dict(target,strict=True)
    second = save_torch(destination/(VARIANTS[1]+'.pt'),dict(model=b.state_dict(),model_config=b.cfg.to_dict(),
        initialization='random seed0 plus common name/shape random A tensors',teacher_weights_used=False,trained_weights_used=False))
    records = dict(passed=True,seed=0,teacher_weights_used=False,old_checkpoints_used=False,
        files={VARIANTS[0]:first,VARIANTS[1]:second},shared_names=shared,
        common_tensor_sha256=tensor_hash({n:source[n] for n in shared}),
        counts={m.cfg.model_variant:sum(p.numel() for p in m.parameters() if p.requires_grad) for m in [a,b]},
        code_sha256=code_hashes(),optimizer='not created until training; no historical state')
    write(ROOT/'repro_outputs/INITIALIZATION.json',records)
    print(records['counts'],flush=True)


if __name__=='__main__':main()
