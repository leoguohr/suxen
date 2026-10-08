#!/usr/bin/env python3
"""Isolated publisher for the authorized private Topology Flow archive.

clone creates the fixed branch from remote main. Stage and commit archive files
separately, then push; this helper never commits, force-pushes, edits a release,
deletes an asset, or pushes main/tags. Credentials exist only in process memory
and the authenticated git child's environment, never URLs/config files/receipts.
"""
import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'repo'
RECEIPTS = ROOT / 'github_receipts'
REPO = 'leoguohr/nexus'
BRANCH = 'archive/topology-flow-20261008'
TAG = 'topology-flow-evidence-2026-10-08'
REMOTE = 'https://github.com/' + REPO + '.git'
PREFIX = '/repos/' + REPO


class PublishError(RuntimeError):
    pass


class RequestError(PublishError):
    def __init__(self, status=None):
        self.status = status
        super().__init__('GitHub request failed' + (f' (HTTP {status})' if status else ' (transport error)'))


def require(condition, message):
    if not condition:
        raise PublishError(message)


def token():
    result = subprocess.run(['security', 'find-generic-password', '-s', 'gh:github.com',
        '-a', 'leoguohr', '-w'], capture_output=True, text=True)
    require(result.returncode == 0, 'Cannot read the existing GitHub Keychain credential')
    value = result.stdout.strip()
    if value.startswith('go-keyring-base64:'):
        value = base64.b64decode(value.split(':', 1)[1], validate=True).decode()
    require(bool(value) and '\n' not in value and '\r' not in value, 'Invalid Keychain credential')
    return value


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


def request_json(url, credential, data=None, method=None, timeout=120, size=None):
    parsed = urllib.parse.urlsplit(url)
    require(parsed.scheme == 'https' and parsed.hostname in ('api.github.com', 'uploads.github.com')
        and parsed.port in (None, 443) and not parsed.username and not parsed.password, 'Unexpected API host')
    headers = {'Authorization': 'Bearer ' + credential, 'Accept': 'application/vnd.github+json',
        'User-Agent': 'nexus-topology-flow-archive', 'X-GitHub-Api-Version': '2022-11-28'}
    if size is None:
        data = None if data is None else json.dumps(data).encode()
        headers['Content-Type'] = 'application/json'
    else:
        headers.update({'Content-Type': 'application/octet-stream', 'Content-Length': str(size)})
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RequestError(error.code) from None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        raise RequestError() from None


def api(path, credential, data=None, method=None):
    require(path.startswith(PREFIX + '/') or path == PREFIX, 'Unexpected repository API path')
    return request_json('https://api.github.com' + path, credential, data, method)


def optional(path, credential):
    try:
        return api(path, credential)
    except RequestError as error:
        if error.status == 404:
            return None
        raise


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def save(name, value):
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    destination = RECEIPTS / (name + '.json')
    temporary = destination.with_name(destination.name + '.tmp-' + uuid.uuid4().hex)
    with temporary.open('x') as stream:
        os.chmod(temporary, 0o600)
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    directory_fd = os.open(RECEIPTS, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def receipt(name):
    path = RECEIPTS / (name + '.json')
    return json.loads(path.read_text()) if path.exists() else None


def git_environment(credential=None):
    env = {key: value for key, value in os.environ.items()
        if not key.startswith(('GIT_', 'GH_', 'GITHUB_'))}
    env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT='0')
    config = [('http.version', 'HTTP/1.1'), ('credential.helper', ''), ('core.hooksPath', os.devnull),
        ('http.followRedirects', 'false'), ('push.followTags', 'false')]
    if credential is not None:
        header = base64.b64encode(('x-access-token:' + credential).encode()).decode()
        config.append(('http.https://github.com/.extraheader', 'AUTHORIZATION: basic ' + header))
    env['GIT_CONFIG_COUNT'] = str(len(config))
    for index, (key, value) in enumerate(config):
        env[f'GIT_CONFIG_KEY_{index}'] = key
        env[f'GIT_CONFIG_VALUE_{index}'] = value
    return env


def git(*args, credential=None, folder=FOLDER):
    command = ['git'] + ([] if folder is None else ['-C', str(folder)]) + list(args)
    result = subprocess.run(command, env=git_environment(credential), capture_output=True, text=True)
    require(result.returncode == 0, 'Git operation failed; no credential-bearing output was recorded')
    return result.stdout.strip()


def private_repository(credential):
    identity = api(PREFIX, credential)
    require(identity.get('private') is True and identity.get('full_name') == REPO,
        'Refusing to publish: repository identity/private status mismatch')
    return identity


def session():
    value = receipt('clone')
    require(value and value.get('repo') == REPO and value.get('branch') == BRANCH
        and value.get('tag') == TAG and value.get('run_id'), 'No matching isolated-clone receipt')
    return value


def local_head():
    session()
    require(FOLDER.is_dir() and not FOLDER.is_symlink() and (FOLDER / '.git').is_dir(), 'Invalid isolated checkout')
    require(Path(git('rev-parse', '--show-toplevel')).resolve() == FOLDER.resolve(), 'Checkout escaped isolated directory')
    require(git('remote', 'get-url', 'origin') == REMOTE, 'Unexpected origin URL')
    require(git('symbolic-ref', '--short', 'HEAD') == BRANCH, 'Only the dedicated Topology Flow branch is allowed')
    require(not git('status', '--porcelain'), 'Commit archive changes before publishing')
    head = git('rev-parse', 'HEAD')
    require(re.fullmatch('[0-9a-f]{40}', head), 'Invalid commit SHA')
    return head


def remote_branch(credential):
    value = optional(PREFIX + '/git/ref/heads/' + BRANCH, credential)
    return value['object']['sha'] if value else None


def remote_tag(credential):
    value = optional(PREFIX + '/git/ref/tags/' + TAG, credential)
    if value is None:
        return None
    obj = value['object']
    for _ in range(5):
        if obj['type'] == 'commit':
            return obj['sha']
        require(obj['type'] == 'tag', 'Unexpected tag object')
        obj = api(PREFIX + '/git/tags/' + obj['sha'], credential)['object']
    raise PublishError('Unexpected nested tag chain')


def clone(credential):
    require(not FOLDER.exists(), 'Isolated checkout already exists; refusing to replace it')
    require(not receipt('clone'), 'Clone receipt already exists; refusing a different run')
    require(remote_branch(credential) is None, 'Dedicated branch already exists without this run receipt')
    require(remote_tag(credential) is None and optional(PREFIX + '/releases/tags/' + TAG, credential) is None,
        'Dedicated tag/release already exists without this run receipt')
    base = api(PREFIX + '/git/ref/heads/main', credential)['object']['sha']
    value = {'repo': REPO, 'branch': BRANCH, 'tag': TAG, 'run_id': uuid.uuid4().hex, 'base_main': base}
    save('clone_intent', value)
    git('clone', '--single-branch', '--branch', 'main', '--no-tags', REMOTE, str(FOLDER), credential=credential, folder=None)
    require(git('rev-parse', 'HEAD') == base, 'Remote main changed during clone; inspect before publishing')
    git('switch', '--create', BRANCH)
    save('clone', value)
    return value


def push(credential):
    head, owner = local_head(), session()
    previous, pending = receipt('push'), receipt('push_intent')
    remote = remote_branch(credential)
    known = [r for r in (previous, pending) if r and r.get('run_id') == owner['run_id']]
    require(remote is None or any(r.get('commit') == remote for r in known), 'Remote branch is not owned by this run receipt')
    value = {'repo': REPO, 'branch': BRANCH, 'run_id': owner['run_id'], 'commit': head}
    save('push_intent', value)
    if remote != head:
        git('push', '--no-force', '--no-follow-tags', '--recurse-submodules=no', REMOTE,
            'HEAD:refs/heads/' + BRANCH, credential=credential)
    require(remote_branch(credential) == head, 'Remote branch does not match exact local HEAD')
    save('push', value)
    return value


def owned_release(credential, head=None):
    owner = session()
    value = optional(PREFIX + '/releases/tags/' + TAG, credential)
    if value is None:
        return None
    known, intent = receipt('release'), receipt('release_intent')
    matching = known or intent
    require(matching and matching.get('run_id') == owner['run_id'] and matching.get('tag') == TAG,
        'Release exists without this run receipt; refusing to modify or reuse it')
    marker = '<!-- topology-flow-publish-run:' + owner['run_id'] + ' -->'
    require(marker in value.get('body', '') and value.get('tag_name') == TAG and value.get('draft') is False,
        'Release ownership marker/tag/publication status mismatch')
    if known:
        require(known['id'] == value['id'], 'Release ID changed')
    target = remote_tag(credential)
    require(target == matching['commit'] and (head is None or target == head), 'Release tag does not match pinned commit')
    if not known:
        save('release', {**intent, 'id': value['id'], 'html_url': value['html_url']})
    return value


def release(credential, body_path):
    head, owner = local_head(), session()
    published = receipt('push')
    require(published and published.get('commit') == head and published.get('run_id') == owner['run_id']
        and remote_branch(credential) == head, 'Push and verify this exact HEAD before creating the release')
    existing = owned_release(credential, head)
    if existing:
        return {'id': existing['id'], 'html_url': existing['html_url'], 'reused': True}
    require(remote_tag(credential) is None, 'Tag exists without a matching completed release receipt; refusing to overwrite it')
    body = body_path.read_text() + '\n\n<!-- topology-flow-publish-run:' + owner['run_id'] + ' -->\n'
    intent = {'repo': REPO, 'tag': TAG, 'run_id': owner['run_id'], 'commit': head,
        'body_sha256': hashlib.sha256(body.encode()).hexdigest(), 'make_latest': 'false'}
    previous = receipt('release_intent')
    require(previous is None or previous == intent, 'Release intent changed; refusing a different creation request')
    save('release_intent', intent)
    try:
        api(PREFIX + '/releases', credential, {'tag_name': TAG, 'target_commitish': head,
            'name': 'NEXUS Topology Flow evidence — 2026-10-08', 'body': body,
            'draft': False, 'prerelease': False, 'make_latest': 'false'}, 'POST')
    except RequestError:
        # An uncertain POST outcome is queried before any later retry.
        existing = owned_release(credential, head)
        if existing is None:
            raise
    existing = owned_release(credential, head)
    require(existing is not None, 'Release creation is unconfirmed; rerun after inspecting receipts')
    return {'id': existing['id'], 'html_url': existing['html_url'], 'commit': head}


def assets(credential, release_id):
    result = []
    page = 1
    while True:
        batch = api(PREFIX + f'/releases/{release_id}/assets?per_page=100&page={page}', credential)
        result.extend(batch)
        if len(batch) < 100:
            return result
        page += 1


def matching_asset(credential, release_id, name, size, digest):
    same = [a for a in assets(credential, release_id) if a['name'] == name]
    require(len(same) <= 1, 'Duplicate remote asset name: ' + name)
    if not same:
        return None
    value = same[0]
    require(value.get('state') == 'uploaded' and value.get('size') == size
        and value.get('digest') == 'sha256:' + digest,
        'Existing asset differs or is incomplete; refusing deletion/overwrite: ' + name)
    return value


def asset_receipt(path, release_id, asset, digest):
    value = {key: asset[key] for key in ('id', 'name', 'size', 'state', 'digest', 'browser_download_url')}
    value.update(local_path=str(path.resolve()), local_sha256=digest, release_id=release_id,
        tag=TAG, run_id=session()['run_id'])
    save('asset-' + hashlib.sha256(path.name.encode()).hexdigest(), value)
    return value


def transfer(credential, files, upload=False):
    head = local_head()
    require(remote_branch(credential) == head, 'Local and remote archive branch differ')
    published = owned_release(credential, head)
    require(published is not None, 'Create the owned release first')
    if not files:
        files = [Path(json.loads(p.read_text())['local_path']) for p in sorted(RECEIPTS.glob('asset-*.json'))]
    require(bool(files), 'Supply local files to upload or verify')
    require(len({p.name for p in files}) == len(files), 'Local asset filenames must be unique')
    records = []
    for path in files:  # Deliberately serial uploads.
        require(path.is_file() and 0 < path.stat().st_size < 2**31, 'Asset must be a nonempty file below 2 GiB')
        size, digest = path.stat().st_size, sha(path)
        value = matching_asset(credential, published['id'], path.name, size, digest)
        if value is None and upload:
            url = published['upload_url'].split('{', 1)[0]
            parsed = urllib.parse.urlsplit(url)
            require(parsed.scheme == 'https' and parsed.netloc == 'uploads.github.com'
                and parsed.path == PREFIX + f"/releases/{published['id']}/assets", 'Unexpected asset upload URL')
            url += '?name=' + urllib.parse.quote(path.name, safe='')
            for attempt in range(3):
                failure = None
                try:
                    with path.open('rb') as stream:
                        request_json(url, credential, stream, 'POST', timeout=600, size=size)
                except RequestError as error:
                    failure = error
                # Query before retry, including failed/unknown upload outcomes.
                value = matching_asset(credential, published['id'], path.name, size, digest)
                if value:
                    break
                if failure and failure.status not in (None, 408, 429, 500, 502, 503, 504):
                    raise failure
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
            require(value is not None, 'Upload unconfirmed after three attempts; no asset was deleted')
        require(value is not None, 'Missing remote asset: ' + path.name)
        require(path.stat().st_size == size and sha(path) == digest, 'Local asset changed while verifying')
        records.append(asset_receipt(path, published['id'], value, digest))
    result = {'repo': REPO, 'private': True, 'branch': BRANCH, 'commit': head, 'tag': TAG,
        'release': published['html_url'], 'assets': records}
    save('upload' if upload else 'verify', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    for name in ('inspect', 'clone', 'push'):
        commands.add_parser(name)
    create = commands.add_parser('release')
    create.add_argument('--body', type=Path, default=ROOT / 'release_body.md')
    for name in ('upload', 'verify'):
        commands.add_parser(name).add_argument('files', nargs='*', type=Path)
    args = parser.parse_args()
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    with (RECEIPTS / '.publisher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        credential = token()
        identity = private_repository(credential)
        if args.action == 'inspect':
            existing = optional(PREFIX + '/releases/tags/' + TAG, credential)
            result = {'repo': REPO, 'private': True, 'html_url': identity['html_url'],
                'main': api(PREFIX + '/git/ref/heads/main', credential)['object']['sha'],
                'branch': BRANCH, 'branch_commit': remote_branch(credential), 'tag': TAG,
                'tag_commit': remote_tag(credential), 'release_id': existing['id'] if existing else None}
            save('inspect', result)
        elif args.action == 'clone':
            result = clone(credential)
        elif args.action == 'push':
            result = push(credential)
        elif args.action == 'release':
            result = release(credential, args.body)
        else:
            result = transfer(credential, args.files, upload=args.action == 'upload')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (PublishError, OSError, ValueError) as error:
        # No subprocess/API response bodies or credential-bearing request repr.
        print('ERROR:', str(error), file=sys.stderr)
        sys.exit(1)
