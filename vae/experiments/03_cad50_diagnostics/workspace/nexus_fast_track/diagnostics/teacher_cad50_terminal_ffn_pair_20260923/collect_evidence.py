"""Collect small server artifacts after the bounded run and cold reload pass."""
import hashlib
import json
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    done = json.loads((ROOT / 'H_terminal_ffn/complete.json').read_text())
    cold = json.loads((ROOT / 'repro_outputs/COLD_VERIFY.json').read_text())
    assert done['new_updates'] == 100 and done['stopped_at_budget']
    assert cold['passed']
    target = ROOT / 'repro_outputs/result_small_download.zip'
    records = []
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for directory, directories, names in os.walk(ROOT, followlinks=False):
            directories[:] = sorted(d for d in directories if d not in
                {'.git', '__pycache__', 'data', 'pools', 'representations', 'Control_reused'}
                and not (Path(directory) / d).is_symlink())
            for name in sorted(names):
                path = Path(directory) / name
                relative = path.relative_to(ROOT)
                if path.is_symlink() or path.suffix not in {'.py', '.json', '.jsonl', '.md', '.txt', '.log', '.npz', '.csv'}:
                    continue
                if name == 'RESULT_DOWNLOAD.json':
                    continue
                if path.suffix == '.npz' and not any(p.startswith('predictions-new') for p in relative.parts):
                    continue
                data = path.read_bytes()
                archive.writestr(relative.as_posix(), data)
                records.append(dict(path=relative.as_posix(), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
    receipt = dict(zip=str(target), bytes=target.stat().st_size,
        sha256=hashlib.sha256(target.read_bytes()).hexdigest(), files=records)
    (ROOT / 'repro_outputs/RESULT_DOWNLOAD.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k:v for k,v in receipt.items() if k != 'files'}))
    print('files', len(records))


if __name__ == '__main__':
    main()
