"""Private VAE archive transport; credentials stay in memory."""
import base64
import json
import os
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = 'leoguohr/nexus'
BRANCH = 'vae/archive-20261008'
TAG = 'vae-evidence-2026-10-08'


def credential():
    result = subprocess.run(['security', 'find-generic-password', '-s', 'gh:github.com',
                             '-a', 'leoguohr', '-w'], capture_output=True, text=True, check=True)
    value = result.stdout.strip()
    if value.startswith('go-keyring-base64:'):
        value = base64.b64decode(value.split(':', 1)[1]).decode()
    return value


def api(path, token, data=None, method=None):
    request = urllib.request.Request('https://api.github.com' + path,
        data=None if data is None else json.dumps(data).encode(), method=method,
        headers={'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json',
                 'Content-Type': 'application/json', 'User-Agent': 'nexus-vae-archive'})
    with urllib.request.urlopen(request, timeout=120) as stream:
        return json.load(stream)


def git(args, token=None, cwd=None):
    env = os.environ.copy()
    for key in ('GIT_TRACE', 'GIT_TRACE_CURL', 'GIT_CURL_VERBOSE'):
        env.pop(key, None)
    if token:
        header = base64.b64encode(('x-access-token:' + token).encode()).decode()
        env.update(GIT_CONFIG_COUNT='1', GIT_CONFIG_KEY_0='http.https://github.com/.extraheader',
                   GIT_CONFIG_VALUE_0='AUTHORIZATION: basic ' + header, GIT_TERMINAL_PROMPT='0')
    return subprocess.run(['git', *args], cwd=cwd, env=env, check=True,
                          capture_output=True, text=True).stdout.strip()


if __name__ == '__main__':
    token = credential()
    info = api('/repos/' + REPO, token)
    assert info['private'] and info['default_branch'] == 'main'
    destination = ROOT / 'repo'
    assert not destination.exists(), 'Use this clone only once'
    git(['clone', '--depth', '1', 'https://github.com/' + REPO + '.git', str(destination)], token)
    base = git(['rev-parse', 'HEAD'], cwd=destination)
    git(['switch', '-c', BRANCH], cwd=destination)
    assert not (destination / 'vae').exists(), 'Namespace collision'
    (ROOT / 'evidence' / 'clone.json').write_text(json.dumps({
        'repository': REPO, 'private': True, 'base_commit': base,
        'branch': BRANCH, 'release_tag': TAG, 'working_directory': str(destination)
    }, indent=2) + '\n')
    print('Independent branch created:', BRANCH, base)
