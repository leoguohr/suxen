"""Run only the 3x LR branch using the frozen paired-branch runner."""
import hashlib
from pathlib import Path


base = Path(__file__).resolve().parent
source_path = base / "run.py"
source = source_path.read_text()
expected_sha256 = "f4cc9bdb5a3b36f4ffdeee5c0bfaf72b574f9bd723850df239d531ac450beb62"
assert hashlib.sha256(source.encode()).hexdigest() == expected_sha256

replacements = {
    "BASE=Path(__file__).resolve().parent": "BASE=Path(__file__).resolve().parent.parent",
    "choices=['control','high10x']": "choices=['mid3x']",
    "NOISE_CHECKS={0,200,400}": "NOISE_CHECKS={0,100,200,400}",
    "if args.branch=='high10x':\n    for group in optimizer.param_groups:\n        if group['name']!='logvar':group['lr']*=10":
        "if args.branch=='mid3x':\n    for group in optimizer.param_groups:\n        if group['name']!='logvar':group['lr']*=3",
    "lr_change='none' if args.branch=='control' else '10x encoder_mu, decoder_body, edge_head, face_head; logvar unchanged'":
        "lr_change='3x encoder_mu, decoder_body, edge_head, face_head; logvar unchanged'",
}
for old, new in replacements.items():
    assert source.count(old) == 1, old
    source = source.replace(old, new)

target = base / "mid3x" / "effective_run.py"
target.parent.mkdir(exist_ok=True)
target.write_text(source)
namespace = {"__file__": str(target), "__name__": "__main__"}
exec(compile(source, str(target), "exec"), namespace)
