#!/usr/bin/env python3
"""Read-only audit of the single explicitly authorized original teacher ZIP.

Author: Codex gpt-6-astra, reasoning effort xhigh, 2026-09-23.
Does not execute or extract archive contents. Stdlib only.
"""
import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

SOURCE = Path('/Users/luthier/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_lmth97zoyepq22_50c3/msg/file/2026-09/nexus_overfit_data_results_no_code.zip')
OUT = Path(__file__).resolve().parent
PREVIOUS_SHA256 = '982686feb58c207932d9311786cabb3ffd7a1397627b83c44df3d5c8ab2189e5'

def digest_stream(stream):
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(chunk)
    return h.hexdigest()

def main():
    with SOURCE.open('rb') as f:
        archive_hash = digest_stream(f)
    entries, unsafe, duplicates = [], [], []
    seen = set()
    with zipfile.ZipFile(SOURCE) as z:
        for info in z.infolist():
            p = PurePosixPath(info.filename)
            mode = info.external_attr >> 16
            reasons = []
            if p.is_absolute() or '..' in p.parts or '\\' in info.filename or (p.parts and ':' in p.parts[0]):
                reasons.append('unsafe_path')
            if stat.S_ISLNK(mode):
                reasons.append('symlink')
            if info.filename in seen:
                duplicates.append(info.filename)
            seen.add(info.filename)
            if reasons:
                unsafe.append({'path': info.filename, 'reasons': reasons})
            with z.open(info, 'r') as f:
                member_hash = digest_stream(f)  # ZipExtFile checks CRC at EOF.
            entries.append({'path': info.filename, 'bytes': info.file_size,
                            'compressed_bytes': info.compress_size,
                            'crc32': f'{info.CRC:08x}', 'crc_verified': True,
                            'sha256': member_hash, 'unix_mode': oct(mode),
                            'is_dir': info.is_dir()})
    result = {'audit_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'author_model': 'gpt-6-astra', 'reasoning_effort': 'xhigh',
              'source': str(SOURCE), 'archive_bytes': SOURCE.stat().st_size,
              'archive_sha256': archive_hash, 'previous_sha256': PREVIOUS_SHA256,
              'same_bytes_as_previous_archive': archive_hash == PREVIOUS_SHA256,
              'member_count': len(entries), 'total_uncompressed_bytes': sum(x['bytes'] for x in entries),
              'all_member_crcs_verified': True, 'unsafe_entries': unsafe,
              'duplicate_names': duplicates, 'entries': entries}
    (OUT / 'FILE_MANIFEST.json').write_text(json.dumps(result, indent=2) + '\n')
    summary = {k: v for k, v in result.items() if k != 'entries'}
    print(json.dumps(summary, indent=2))
    (OUT / 'zip_audit_command.log').write_text(
        '$ /Users/luthier/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 audit_original_zip.py\n'
        + json.dumps(summary, indent=2) + '\n')

if __name__ == '__main__':
    main()
