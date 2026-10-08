"""Cross-link already completed CPU and independent GPU cold-verify reports."""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--experiment-root',type=Path,required=True)
root=p.parse_args().experiment_root
out=root/'repro_outputs'
audit=json.loads((out/'TREATMENT_FINAL_AUDIT.json').read_text())
cold_path=out/'COLD_VERIFY.json'
cold=json.loads(cold_path.read_text())
assert audit['passed'] and cold['passed'] and cold['optimizer_updates']==0
assert cold['model_adam_rng_restored_exactly'] and cold['all50_arrays_bitwise_equal']
assert len(cold['meshes'])==50 and all(x['all_arrays_bitwise_equal'] for x in cold['meshes'])
assert cold['adam_steps']==dict(encoder_mu=2600,decoder=2600,edge_head=2600,face_head=2600,terminal_ffn=100)
final=audit['trajectory'][-1]
assert cold['counts']==final['treatment']['counts'] and cold['joint_perfect']==final['treatment']['joint']
assert cold['checkpoint_sha256']==audit['checkpoint_identity_checks'][-1]['sha256']
assert [x['uid'] for x in cold['meshes']]==[x['uid'] for x in audit['per_mesh'] if x['step']==100]
audit['independent_cold_verify']=dict(passed=True,path=str(cold_path),
    sha256=hashlib.sha256(cold_path.read_bytes()).hexdigest(),optimizer_updates=0,
    all50_arrays_bitwise_equal=True,model_adam_rng_restored_exactly=True,
    evidence_attached_by_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out/'TREATMENT_FINAL_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
review=out/'TREATMENT_FINAL_REVIEW.md'
text=review.read_text()
marker='\n## 冷加载证据与结构结论\n'
assert marker not in text
text+=marker+'''
已读取并交叉核对根任务完成的 `COLD_VERIFY.json`：fresh process 从最终 checkpoint 恢复完整 model/Adam/RNG，重新安装 FFN prehook 后，全部 50 条预测数组逐字节相同；旧组 Adam2600、新 FFN100，optimizer updates=0。冷验 checkpoint SHA、counts 与本 CPU 审计一致。冷验不是本 CPU 审计运行的 forward。

本次末端 FFN 没有改善终点结果：Face F1 H=0.3525331725，S=0.3539593250；严格成功 H32、S33。H 在25步曾达到33条，但末尾回到父32条；两支在较大16条上均未严格成功。中间25/50/75步 H 的 Face F1 高于对应 S，这不构成终点或收敛上限改善的证据。结果不支持“仅加这个末端 FFN、沿用当前优化状态和100步预算便可解决困难样本”；仍不能据此排除更早层的逐点变换、不同初始化/更长训练或其他结构机制。

审计执行说明：首个 CPU 审计脚本在对 state_dict 的非 Tensor `_extra_state` 元数据执行 isfinite 时发生 TypeError；已改为 Tensor 检查 finite、元数据与父精确比较，完整重跑通过。仅修改审计脚本，未修改训练源码、状态或数据；首个脚本错误记录另存 TREATMENT_AUDIT_SCRIPT_ATTEMPT1.json。
'''
review.write_text(text)
print('CPU_AND_COLD_EVIDENCE_LINK_PASS')
