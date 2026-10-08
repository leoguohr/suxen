"""Launch only this experiment on one freshly checked allocated GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
JOBS={
    'A':('GPU-0ae7719a-74e5-08ae-75de-0a6205381365','A_v1_recipe_control'),
    'B':('GPU-7ea0dcc5-8893-8144-31c6-df3c9b25b351','B_v2_teacher_blocks'),
    'eval':('GPU-6a420b8b-6596-aba3-657a-02729d35afa3',None),
}


def main(role):
    uuid,variant=JOBS[role]
    out=ROOT/'repro_outputs';record=out/f'launch-{role}.json'
    assert not record.exists(),'Refuse duplicate launch'
    assert json.loads((out/'CORE_TESTS.json').read_text())['passed']
    assert json.loads((out/'INITIALIZATION.json').read_text())['passed']
    text=subprocess.check_output(['nvidia-smi','--id='+uuid,'--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    fields=[x.strip() for x in text.split(',')]
    assert fields[0]==uuid and int(fields[1])==0 and int(fields[2])==0,text
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
    assert uuid not in processes
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        PYTHONDONTWRITEBYTECODE='1',RIGORPILOT_LESSONS='0')
    if variant:
        runner=Path('/guohaoran/nexus_fast_track/diagnostics/decoder_last3_trainability_pair_A500_20260921/_rigorpilot/skills/run-train/scripts/run_training.py')
        command=f'/opt/conda/bin/python -u {ROOT}/train.py --variant {variant}'
        argv=['/opt/conda/bin/python',str(runner),'--repo',str(ROOT),'--command',command,
            '--run-mode','full_kickoff','--lane','trusted','--timeout','90000','--max-steps','20000',
            '--dataset','fixed CAD50, shuffled full epochs, five meshes per optimizer update',
            '--checkpoint-source','fresh random seed0; no trained weights',
            '--runtime-root',str(out/'_runtime'/variant)]
    else:argv=['/opt/conda/bin/python','-u',str(ROOT/'eval_worker.py')]
    with (out/f'launch-{role}.log').open('x') as log:
        process=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record.write_text(json.dumps(dict(role=role,pid=process.pid,argv=argv,gpu_uuid=uuid,
        gpu_before=text,processes_before=processes,started=time.time(),hostname=os.uname().nodename),indent=2)+'\n')
    print(record.read_text())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--role',choices=list(JOBS),required=True)
    main(parser.parse_args().role)
