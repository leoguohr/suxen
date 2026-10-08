"""One independent, complete fixed100 inference evaluation; no optimizer exists."""
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'code'))
from run_support import torch, configure, read, write, sha, tensor_hash, code_hashes
from native_models import NativeTopologyAE, Config
from data_objective import load_dataset
from runtime_fixed100 import lock_mainline, verify_gpu
from evaluate_checkpoint import evaluate


class EvaluationStop:
    def stop(self):
        return (ROOT/'run'/'STOP').exists()


def main():
    cfg = read(ROOT/'eval_config.json')
    run = ROOT/'run'; run.mkdir(exist_ok=True)
    lock = lock_mainline(run/'evaluation.lock')
    hardware = verify_gpu(cfg['gpu_uuid'])
    entry = cfg['checkpoint']
    evaluation_dir = f"update-{entry['completed_updates']:08d}"
    started = time.time()
    write(run/'status.json',dict(state='loading_frozen_checkpoint',checkpoint=entry,pid=os.getpid()))
    try:
        assert sha(entry['path']) == entry['sha256']
        cp = torch.load(entry['path'],map_location='cpu',weights_only=False)
        assert cp['completed_updates'] == entry['completed_updates']
        assert cp['source_sha256'] == code_hashes()
        assert tensor_hash(cp['model']) == entry['model_state_sha256']
        items,data = load_dataset(cp['config']['data_source'])
        assert data == cp['data']
        configure(cp['config']['seed']); torch.cuda.set_device(0)
        model = NativeTopologyAE(Config(model_variant='B_v2_teacher_blocks'))
        assert model.cfg.to_dict() == cp['model_config']
        model.load_state_dict(cp['model'],strict=True)
        model = model.cuda().float().eval()
        del cp
        write(run/'runtime.json',dict(pid=os.getpid(),hardware=hardware,torch=torch.__version__,
            cuda=torch.version.cuda,checkpoint=entry,source_sha256=code_hashes(),
            source_paths={k:str(ROOT/'code'/k) for k in code_hashes()},data=data,
            model_config=model.cfg.to_dict(),optimizer_updates=0,
            note='True Encoder-mu-Decoder inference. Face candidates fully enumerated from predicted edges. Independent process and device; training remains untouched.'))
        write(run/'status.json',dict(state='evaluating',pid=os.getpid(),checkpoint=entry,
            progress_file=str(run/'evaluations'/evaluation_dir/'progress.json')))
        result = evaluate(model,items,data,entry,run,EvaluationStop())
        summary = dict(state='complete' if result['complete'] else 'incomplete',
            pid=os.getpid(),checkpoint=entry,elapsed_seconds=time.time()-started,
            optimizer_updates=0,result_file=str(run/'evaluations'/evaluation_dir/('complete.json' if result['complete'] else 'incomplete.json')))
        if result['complete']:
            summary.update(edge=result['edge'],face=result['face'],joint_perfect=result['joint_perfect'],
                edge_perfect=len(result['edge_perfect_uids']),face_perfect=len(result['face_perfect_uids']),
                face_fn_missing=result['face_fn_missing'],face_fn_present=result['face_fn_present'])
        write(run/'status.json',summary)
        if result['complete']: write(run/'complete.json',summary)
        print(__import__('json').dumps(summary,indent=2),flush=True)
    except BaseException as error:
        write(run/'failure.json',dict(error=str(error),traceback=traceback.format_exc(),
            checkpoint=entry,optimizer_updates=0,elapsed_seconds=time.time()-started))
        raise
    finally:
        lock.close()


if __name__ == '__main__': main()
