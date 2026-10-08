"""Cold-load the final resumable state and repeat complete real-network evaluation."""
import argparse
from pathlib import Path
import torch
from support import configure, read, tensor_hash, write
from native_models import Config, NativeTopologyAE
from data_objective import load_dataset
from evaluation import evaluate


def main(arm,source):
    configure(0)
    root=Path(__file__).resolve().parent
    folder=root/'runs'/arm
    entry=read(folder/'checkpoint-02000.json')
    from support import sha
    assert sha(entry['path'])==entry['sha256']
    cp=torch.load(entry['path'],map_location='cpu',mmap=True,weights_only=False)
    assert cp['completed_updates']==2000
    model=NativeTopologyAE(Config(**cp['model_config'])).cuda().float().eval()
    model.load_state_dict(cp['model'],strict=True)
    assert tensor_hash(model.state_dict())==tensor_hash(cp['model'])
    items,_=load_dataset(Path(source)/'data',Path(source)/'pools')
    actual=evaluate(model,items,folder/'cold_final',entry)
    expected=read(folder/'evaluations/step-02000/evaluation.json')
    for key in ('counts','joint_strict_uids','joint_strict','edge_strict','face_fn_missing','face_fn_present'):
        assert actual[key]==expected[key],key
    assert all(a==b for a,b in zip(actual['meshes'],expected['meshes']))
    write(folder/'COLD_FINAL_VERIFIED.json',dict(passed=True,checkpoint=entry,
        optimizer_updates=0,counts=actual['counts'],joint_strict=actual['joint_strict'],
        scope='fresh process full real Encoder -> mu -> Decoder; complete actual Face enumeration'))
    print('COLD_FINAL_VERIFIED',arm,actual['joint_strict'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--source',required=True)
    args=p.parse_args();main(args.arm,args.source)
