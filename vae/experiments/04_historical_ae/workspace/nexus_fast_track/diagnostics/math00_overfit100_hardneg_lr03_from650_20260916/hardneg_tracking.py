"""Save the fixed mined negatives' logits from the unchanged training-pool scoring call."""
from pathlib import Path
import numpy as np

def record(folder,epoch,uid,pool,mining_record,saved):
    count=mining_record['selected']
    ids=pool['mixed'][-count:] if count else pool['mixed'][:0]
    logits=saved['face_train_logits'][-count:] if count else saved['face_train_logits'][:0]
    assert len(ids)==len(logits)==count and np.isfinite(logits).all()
    out=Path(folder)/f'epoch{epoch}';out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/f'{uid}.npz',ids=ids,logits=logits)
    return dict(count=count,pool_fp=int((logits>0).sum()),pool_tn=int((logits<=0).sum()),
                min_logit=float(logits.min()) if count else None,max_logit=float(logits.max()) if count else None,
                definition='Fixed mined GT negatives, scored in full augmented training pool; actual Face acceptance remains separate')
