import hashlib
import json
from pathlib import Path
import subprocess
import sys
import torch
from test_vertex_b2_frozen import make_checkpoint


def test_c_recovery_is_frozen_and_does_not_claim_unseen_seeds(tmp_path):
    cp, manifest = make_checkpoint(tmp_path)
    state = torch.load(cp, weights_only=False)
    state['step'] = 3800; state['c_update'] = 1800
    state['config'].update(phase='C', precision='fp32', development_seeds=list(range(15000000,15000004)), final_seeds=list(range(16000000,16000008)))
    state['model']['flow.output.bias'].fill_(-20)
    torch.save(state, cp)
    sha = hashlib.sha256(cp.read_bytes()).hexdigest()
    root = Path(__file__).resolve().parents[1]; out = tmp_path/'out'
    p = subprocess.run([sys.executable, str(root/'scripts/evaluate_vertex_c_checkpoint.py'), '--code-root',str(root),
        '--checkpoint',str(cp),'--manifest',str(manifest),'--output',str(out),'--expected-sha',sha,'--device','cpu'],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    result=json.loads((out/'result.json').read_text())
    assert result['cumulative_step']==3800 and not result['parameters_updated']
    assert result['original_final_execution_unknown'] and not result['unseen_final_test_claimed']
    assert result['final_recovery_full_exact']==0
    assert len(json.loads((out/'final_recovery.json').read_text())['full_generated_parents'])==8
    assert hashlib.sha256(cp.read_bytes()).hexdigest()==sha
