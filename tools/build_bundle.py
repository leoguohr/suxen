#!/usr/bin/env python3
"""Build local, source-preserving Nexus evidence archives. No network operations."""
import argparse
import hashlib
import io
import json
import os
import re
import shutil
import stat
import tarfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

BASE = Path('/Users/luthier/Documents/sophomore')
NEXUS = BASE / 'nexus_fast_track'
OUT = BASE / 'output/github_nexus_20261008'
TERMS = ('vertex', 'r1_d15', 'point_native', 'nexus_phase', 'phase3',
         'nexus_overfit50', 'nexus_d9', 'nexus_e0')
SKIP_DIRS = {'.git', '__pycache__', '.pytest_cache', '.ruff_cache', '.mypy_cache',
             '.venv', 'venv', 'env', 'node_modules', 'site-packages', 'local_deps', 'wheelhouse'}
WEIGHTS = {'.pt', '.pth', '.ckpt'}
CODE_EXTS = {'.py', '.sh', '.tex', '.toml', '.yml', '.yaml', '.cfg', '.ini', '.txt'}
TEXT_EXTS = CODE_EXTS | {'.md', '.json', '.jsonl', '.csv', '.log', '.diff', '.patch',
                        '.sha256', '.stderr', '.html', '.js', '.xml'}
SECRET_RULES = {
    'sshpass_reference': re.compile(r'\bsshpass\b', re.I),
    'private_key_marker': re.compile(r'-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----'),
    'known_token_shape': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,}|glpat-[A-Za-z0-9_-]{16,}|sk-[A-Za-z0-9_-]{24,})\b'),
    'credential_assignment': re.compile(r'''\b(?:password|passwd|api[_-]?key|access[_-]?token|secret[_-]?key|client[_-]?secret|auth[_-]?token)\b\s*[=:]\s*["'][^"'\r\n]{5,}["']''', re.I),
    'url_embedded_credentials': re.compile(r'\b(?:https?|socks5?|ssh)://[^\s/:]+:[^\s/@]+@', re.I),
    'bearer_literal': re.compile(r'\bBearer\s+[A-Za-z0-9._~-]{20,}', re.I),
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def source_label(path):
    if path.is_relative_to(BASE):
        return path.relative_to(BASE).as_posix()
    return 'downloads/' + path.relative_to(Path('/Users/luthier/Downloads')).as_posix()


def is_archive(name):
    return name.lower().endswith(('.zip', '.tar', '.tar.gz', '.tgz'))


def exclusion(parts, group, mini=False):
    lowered = [p.lower() for p in parts]
    name = lowered[-1]
    if any(p in SKIP_DIRS for p in lowered):
        return 'cache_git_or_dependency_environment'
    if name == '.ds_store' or name.endswith(('.pyc', '.pyo')):
        return 'generated_cache'
    if name == '.env' or name.startswith('.env.') or name in {'id_rsa', 'id_ed25519', 'credentials.json', 'credentials', '.netrc'} or name.endswith(('.key', '.pem', '.p12', '.pfx')):
        return 'credential_or_private_key_filename_not_read'
    if group == 'vertex' and Path(name).suffix in WEIGHTS:
        return 'vertex_checkpoint_weights_excluded'
    if mini:
        if 'data' in lowered:
            return 'mini_nexus_dataset_excluded'
        if 'outputs' in lowered:
            pos = lowered.index('outputs')
            if pos + 1 < len(lowered) and 'vertex' not in lowered[pos + 1]:
                return 'mini_nexus_non_vertex_output_outside_scope'
        if name.endswith('.md') and ('topology' in name or name.startswith(('packed_', 'bf16_'))):
            return 'mini_nexus_non_vertex_document_outside_scope'
    return None


def safe_member(name):
    normal = name.replace('\\', '/')
    p = PurePosixPath(normal)
    if normal.startswith('/') or re.match(r'^[A-Za-z]:', normal) or '..' in p.parts:
        return None
    parts = tuple(x for x in p.parts if x not in ('', '.'))
    return parts or None


class Bundle:
    def __init__(self, group, name, roots):
        self.group, self.name, self.roots = group, name, roots
        self.root = OUT / 'stage' / name
        if self.root.exists():
            raise RuntimeError(f'Refusing to overwrite existing stage: {self.root}')
        self.root.mkdir(parents=True)
        self.files, self.excluded, self.archive_audit, self.origins = [], [], [], []
        self.by_hash, self.pending = {}, []

    def omit(self, origin, size, reason):
        self.excluded.append({'original_path': origin, 'size': size, 'reason': reason})

    def store(self, origin, rel, stream, size, dedup=False):
        destination = self.root / rel
        if destination.exists():
            rel += '.archive_member_' + str(len(self.origins))
            destination = self.root / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256()
        actual = 0
        with destination.open('wb') as out:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                actual += len(chunk)
                h.update(chunk)
                out.write(chunk)
        if actual != size:
            raise RuntimeError(f'Size changed during copy: {origin}')
        digest = h.hexdigest()
        previous = self.by_hash.get((size, digest)) if dedup else None
        if previous:
            destination.unlink()
            mapped = previous
        else:
            mapped = rel
            self.files.append({'original_path': origin, 'archive_path': rel, 'size': size, 'sha256': digest})
            self.by_hash.setdefault((size, digest), rel)
        self.origins.append({'original_path': origin, 'archive_path': mapped, 'size': size, 'sha256': digest,
                             'storage': 'same_content_existing_file' if previous else 'stored_file'})
        return mapped, bool(previous), digest

    def collect(self):
        for root in self.roots:
            if not root.exists():
                raise FileNotFoundError(root)
            mini = root == NEXUS / 'mini_nexus'
            if root.is_file():
                entries = [root]
            else:
                entries = []
                for directory, dirs, names in os.walk(root, followlinks=False):
                    for d in list(dirs):
                        p = Path(directory) / d
                        if p.is_symlink():
                            self.omit(str(p), p.lstat().st_size, 'symlink_not_followed')
                            dirs.remove(d)
                    entries.extend(Path(directory) / n for n in names)
            for p in sorted(entries):
                size = p.lstat().st_size
                origin = str(p)
                if p.is_symlink():
                    self.omit(origin, size, 'symlink_not_followed')
                    continue
                parts = p.relative_to(root).parts if root.is_dir() else (p.name,)
                reason = exclusion(parts, self.group, mini)
                if reason:
                    self.omit(origin, size, reason)
                elif is_archive(p.name):
                    self.pending.append((p, source_label(p)))
                else:
                    with p.open('rb') as f:
                        _, duplicate, _ = self.store(origin, 'payload/' + source_label(p), f, size,
                                                     dedup=self.group == 'vertex' and p.suffix.lower() in {'.npz', '.npy'})
                    if duplicate:
                        self.omit(origin, size, 'same_content_array_deduplicated_original_mapped')
        for p, label in self.pending:
            self.unpack(p, str(p), 'payload/expanded_archives/' + label, 0)

    def unpack(self, obj, origin, target, depth):
        if depth > 8:
            raise RuntimeError(f'Archive nesting exceeds 8: {origin}')
        container_size = obj.stat().st_size if isinstance(obj, Path) else len(obj.getbuffer())
        self.omit(origin, container_size, 'archive_container_replaced_by_audited_member_mapping')
        archive = zipfile.ZipFile(obj) if origin.lower().endswith('.zip') else tarfile.open(fileobj=obj if not isinstance(obj, Path) else None, name=str(obj) if isinstance(obj, Path) else None, mode='r:*')
        with archive:
            members = archive.infolist() if isinstance(archive, zipfile.ZipFile) else archive.getmembers()
            for member in members:
                is_zip = isinstance(archive, zipfile.ZipFile)
                name = member.filename if is_zip else member.name
                size = member.file_size if is_zip else member.size
                if member.is_dir() if is_zip else member.isdir():
                    continue
                child_origin = origin + '::' + name
                parts = safe_member(name)
                special = stat.S_ISLNK(member.external_attr >> 16) if is_zip else not member.isfile()
                reason = 'unsafe_archive_member_path' if parts is None else ('archive_link_or_special_file' if special else exclusion(parts, self.group))
                if reason:
                    self.omit(child_origin, size, reason)
                    self.archive_audit.append({'original_path': child_origin, 'size': size, 'status': 'excluded', 'reason': reason})
                    continue
                stream = archive.open(member) if is_zip else archive.extractfile(member)
                with stream:
                    if is_archive(name):
                        self.unpack(io.BytesIO(stream.read()), child_origin, target + '/' + '/'.join(parts), depth + 1)
                        continue
                    mapped, duplicate, digest = self.store(child_origin, target + '/' + '/'.join(parts), stream, size, dedup=True)
                self.archive_audit.append({'original_path': child_origin, 'archive_path': mapped, 'size': size, 'sha256': digest,
                                           'status': 'content_duplicate_mapped' if duplicate else 'unique_member_preserved'})

    def finish(self):
        self.files.sort(key=lambda x: x['archive_path'])
        self.origins.sort(key=lambda x: x['original_path'])
        self.excluded.sort(key=lambda x: x['original_path'])
        write_jsonl(self.root / 'FILES_MANIFEST.jsonl', self.files)
        write_jsonl(self.root / 'ORIGIN_PATH_MAP.jsonl', self.origins)
        write_jsonl(self.root / 'EXCLUSIONS.jsonl', self.excluded)
        write_jsonl(self.root / 'NESTED_ARCHIVE_AUDIT.jsonl', self.archive_audit)
        write_json(self.root / 'SOURCE_ROOTS.json', [str(p) for p in self.roots])
        counts = Counter(x['reason'] for x in self.excluded)
        summary = {'name': self.name, 'group': self.group, 'source_roots': len(self.roots),
                   'payload_files': len(self.files), 'payload_bytes': sum(x['size'] for x in self.files),
                   'mapped_origins': len(self.origins), 'excluded_items': len(self.excluded),
                   'excluded_bytes': sum(x['size'] for x in self.excluded), 'exclusions_by_reason': dict(counts),
                   'nested_archive_members': len(self.archive_audit),
                   'archive_member_statuses': dict(Counter(x['status'] for x in self.archive_audit)),
                   'created_at': datetime.now(timezone.utc).isoformat(),
                   'verification_scope': 'File preservation, SHA256, safe archive expansion, and ZIP CRC. No model/result evaluation.'}
        write_json(self.root / 'BUNDLE_SUMMARY.json', summary)
        (self.root / 'ARCHIVE_SCOPE.md').write_text(
            '# Local evidence archive\n\nThis bundle preserves the explicitly selected local files. '
            'It does not establish model quality, training completion, or faithful paper reproduction.\n\n'
            '`FILES_MANIFEST.jsonl` contains one SHA256 for each stored payload file. '
            '`ORIGIN_PATH_MAP.jsonl` additionally maps deduplicated archive members to their stored equivalent. '
            'Identical Vertex .npz/.npy arrays are stored once by SHA256; every original path and hash remains in the origin map. '
            '`EXCLUSIONS.jsonl` records all omitted files and container archives with sizes and reasons. '
            '`NESTED_ARCHIVE_AUDIT.jsonl` records unique, duplicate, and rejected members of ZIP/TAR inputs. '
            'Source files were not modified. Teacher original ZIP and new server evidence are separate assets.\n\n'
            '`SHA256SUMS.txt` covers every bundled file except itself; the external verification record hashes this checksum file and the final ZIP.\n')
        return summary


def scan_text(root, output):
    findings, scanned = [], []
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_EXTS:
            with path.open('rb') as probe:
                head = probe.read(8192)
            if b'\0' in head:
                continue
            try:
                head.decode('utf-8')
            except UnicodeDecodeError:
                continue
        rel = path.relative_to(root).as_posix()
        scanned.append(rel)
        with path.open('r', encoding='utf-8', errors='replace') as f:
            for number, line in enumerate(f, 1):
                for category, pattern in SECRET_RULES.items():
                    if pattern.search(line):
                        findings.append({'path': rel, 'line': number, 'category': category})
    write_jsonl(output, findings)
    return {'text_files_scanned': len(scanned), 'findings': len(findings),
            'findings_by_category': dict(Counter(x['category'] for x in findings)),
            'report': str(output), 'report_contains': 'paths, line numbers and categories only; no matched values'}


def make_repo(bundles):
    repo = OUT / 'repo'
    rows = []
    for bundle in bundles:
        for item in bundle.files:
            rel = item['archive_path']
            p = bundle.root / rel
            suffix = p.suffix.lower()
            mini = '/mini_nexus/' in rel and '/outputs/' not in rel
            research = '/research/' in rel
            selected = suffix == '.md' or (bundle.group == 'teacher' and suffix in CODE_EXTS) or ((mini or research) and suffix in CODE_EXTS)
            if not selected:
                continue
            dest_rel = bundle.group + '/' + rel.removeprefix('payload/')
            dest = repo / dest_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
            rows.append({'original_path': item['original_path'], 'repo_path': dest_rel,
                         'release_asset': bundle.name + '.zip', 'archive_path': rel,
                         'size': item['size'], 'sha256': item['sha256']})
        for name in ['SOURCE_ROOTS.json', 'BUNDLE_SUMMARY.json', 'ARCHIVE_SCOPE.md']:
            dest = repo / 'manifests' / bundle.group / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(bundle.root / name, dest)
    write_jsonl(repo / 'manifests/REPO_SOURCE_MAP.jsonl', rows)
    return {'browse_files': len(rows), 'browse_bytes': sum(x['size'] for x in rows),
            'source_map': str(repo / 'manifests/REPO_SOURCE_MAP.jsonl')}


def checksum_and_zip(bundle):
    checksum = bundle.root / 'SHA256SUMS.txt'
    files = sorted(p for p in bundle.root.rglob('*') if p.is_file() and p != checksum)
    with checksum.open('w') as f:
        for path in files:
            f.write(sha256(path) + '  ' + path.relative_to(bundle.root).as_posix() + '\n')
    asset = OUT / 'assets' / (bundle.name + '.zip')
    asset.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(asset, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for path in files + [checksum]:
            archive.write(path, bundle.name + '/' + path.relative_to(bundle.root).as_posix())
    record = verify_bundle(bundle.root, asset)
    if record['zip_bytes'] >= 2 * 1024 ** 3:
        raise RuntimeError(f'Asset exceeds the required strict 2GiB maximum: {asset}')
    write_json(OUT / 'assets' / (bundle.name + '.verification.json'), record)
    (OUT / 'assets' / (bundle.name + '.zip.sha256')).write_text(record['zip_sha256'] + '  ' + asset.name + '\n')
    return record


def verify_bundle(root, asset):
    checksums = root / 'SHA256SUMS.txt'
    checked = 0
    for line in checksums.read_text().splitlines():
        digest, name = line.split('  ', 1)
        if sha256(root / name) != digest:
            raise RuntimeError(f'Checksum mismatch: {root / name}')
        checked += 1
    manifest = root / 'FILES_MANIFEST.jsonl'
    payload = {item['archive_path']: item for item in map(json.loads, manifest.read_text().splitlines())}
    origin_rows = list(map(json.loads, (root / 'ORIGIN_PATH_MAP.jsonl').read_text().splitlines()))
    for item in origin_rows:
        stored = payload.get(item['archive_path'])
        if stored is None or (stored['size'], stored['sha256']) != (item['size'], item['sha256']):
            raise RuntimeError(f'Origin mapping mismatch: {item["original_path"]}')
    payload_checked = 0
    with zipfile.ZipFile(asset) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f'ZIP CRC failure: {bad}')
        for line in manifest.read_text().splitlines():
            item = json.loads(line)
            h = hashlib.sha256()
            with archive.open(root.name + '/' + item['archive_path']) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(chunk)
            if h.hexdigest() != item['sha256']:
                raise RuntimeError(f'ZIP payload SHA256 mismatch: {item["archive_path"]}')
            payload_checked += 1
        entries = len(archive.infolist())
    return {'asset': str(asset), 'zip_bytes': asset.stat().st_size, 'zip_sha256': sha256(asset),
            'testzip': 'passed', 'stage_checksum_files_verified': checked,
            'zip_payload_sha256_files_verified': payload_checked, 'zip_entries': entries,
            'origin_path_mappings_verified': len(origin_rows),
            'manifest_sha256': sha256(manifest), 'checksum_manifest_sha256': sha256(checksums),
            'verified_at': datetime.now(timezone.utc).isoformat(),
            'verification_scope': 'File integrity only; no generation score validation.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    names = ['Nexus_Vertex_Historical_Evidence_20261008', 'Nexus_Teacher_Reconstruction_20261008']
    if args.verify_only:
        for name in names:
            print(json.dumps(verify_bundle(OUT / 'stage' / name, OUT / 'assets' / (name + '.zip'))))
        return
    vertex = sorted(p for p in (NEXUS / 'diagnostics').iterdir() if p.is_dir() and any(t in p.name.lower() for t in TERMS))
    vertex.extend(NEXUS / p for p in ['research/vertex_diffusion_architecture_20260907',
                  'research/vertex_structure_options_20261005', 'research/vertex_structure_sweep_20261006',
                  'research/NEXUS_VERTEX_EXPERIMENT_LOG.md', 'tmp/vertex_interp_review_20261003',
                  'server_snapshots/vertex_overfit20_20260908_31548_v1', 'mini_nexus'])
    teacher = [Path('/Users/luthier/Downloads/Nexus_teacher_full_reconstruction'),
               NEXUS / 'diagnostics/teacher_reverse_audit_20260922',
               NEXUS / 'diagnostics/teacher_original_zip_reaudit_20260923']
    bundles = [Bundle('vertex', names[0], vertex), Bundle('teacher', names[1], teacher)]
    summaries = []
    for bundle in bundles:
        print(f'Collecting {bundle.group}', flush=True)
        bundle.collect()
        summaries.append(bundle.finish())
        print(f'Collected {bundle.group}: {len(bundle.files)} files', flush=True)
    repo = make_repo(bundles)
    scan = {}
    for bundle in bundles:
        scan[bundle.group] = scan_text(bundle.root, OUT / 'stage' / (bundle.group + '_potential_secrets.jsonl'))
    scan['repo'] = scan_text(OUT / 'repo', OUT / 'stage/repo_potential_secrets.jsonl')
    write_json(OUT / 'stage/TEXT_SCAN_SUMMARY.json', scan)
    checks = []
    for bundle in bundles:
        print(f'Compressing and verifying {bundle.group}', flush=True)
        checks.append(checksum_and_zip(bundle))
    final = {'bundles': summaries, 'repo': repo, 'text_scan': scan, 'verification': checks}
    write_json(OUT / 'stage/BUILD_REPORT.json', final)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
