#!/usr/bin/env python3
"""Restricted, CPU/stdlib-only metadata reader for the original torch ZIPs.

No torch import, GPU use, training, extraction or arbitrary pickle globals.
Tensor rebuilds produce inert descriptors; only scalar storage is decoded.
Every other GLOBAL is rejected. Author: gpt-6-astra/xhigh, 2026-09-23.
"""
import collections
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import pickletools
import struct
import zipfile

OUT = Path(__file__).resolve().parent

class FloatStorage: pass
class LongStorage: pass
class ByteStorage: pass

DTYPES = {FloatStorage: ('float32', 'f', 4), LongStorage: ('int64', 'q', 8),
          ByteStorage: ('uint8', 'B', 1)}

class Storage:
    def __init__(self, kind, key, location, count, archive, prefix):
        self.kind, self.key, self.location, self.count = kind, key, location, count
        self.archive, self.prefix = archive, prefix

class Tensor:
    def __init__(self, storage, offset, size, stride, requires_grad, hooks, metadata=None):
        self.storage, self.offset, self.size, self.stride = storage, offset, size, stride
        self.requires_grad = requires_grad

    def descriptor(self):
        dtype, fmt, width = DTYPES[self.storage.kind]
        result = {'tensor_shape': list(self.size), 'tensor_stride': list(self.stride),
                  'storage_offset': self.offset, 'dtype': dtype,
                  'numel': math.prod(self.size), 'storage_device': self.storage.location,
                  'requires_grad_in_serialized_tensor': self.requires_grad}
        if result['numel'] == 1:
            raw = self.storage.archive.read(f'{self.storage.prefix}/data/{self.storage.key}')
            result['scalar'] = struct.unpack('<' + fmt, raw[self.offset*width:(self.offset+1)*width])[0]
        return result

    def content_hash(self):
        # The checkpoint tensors audited here are contiguous, offset-zero.
        if self.offset != 0 or math.prod(self.size) != self.storage.count:
            return None
        contiguous_stride = []
        stride = 1
        for length in reversed(self.size):
            contiguous_stride.insert(0, stride)
            stride *= length
        if tuple(contiguous_stride) != tuple(self.stride):
            return None
        raw = self.storage.archive.read(f'{self.storage.prefix}/data/{self.storage.key}')
        return hashlib.sha256(raw).hexdigest()

class RestrictedReader(pickle.Unpickler):
    def __init__(self, stream, archive, prefix):
        super().__init__(stream)
        self.archive, self.prefix = archive, prefix

    def find_class(self, module, name):
        allowed = {('collections', 'OrderedDict'): collections.OrderedDict,
                   ('torch', 'FloatStorage'): FloatStorage,
                   ('torch', 'LongStorage'): LongStorage,
                   ('torch', 'ByteStorage'): ByteStorage,
                   ('torch._utils', '_rebuild_tensor_v2'): Tensor}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'Rejected GLOBAL: {module}.{name}')
        return allowed[module, name]

    def persistent_load(self, pid):
        if not (isinstance(pid, tuple) and len(pid) == 5 and pid[0] == 'storage' and pid[1] in DTYPES):
            raise pickle.UnpicklingError('Rejected persistent id')
        return Storage(*pid[1:], self.archive, self.prefix)

def simplify(x, tensor_hash=False):
    if isinstance(x, Tensor):
        result = x.descriptor()
        if tensor_hash: result['contiguous_storage_sha256'] = x.content_hash()
        return result
    if isinstance(x, dict): return {str(k): simplify(v, tensor_hash) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [simplify(v, tensor_hash) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None: return x
    raise TypeError(f'Unexpected object {type(x)}')

def interesting_paths(x, path=''):
    result = []
    if isinstance(x, dict):
        for k, v in x.items():
            p = f'{path}/{k}'
            if any(w in str(k).lower() for w in ('optim', 'rng', 'random', 'scheduler', 'scaler')):
                result.append(p)
            result.extend(interesting_paths(v, p))
    elif isinstance(x, (tuple, list)):
        for i, v in enumerate(x): result.extend(interesting_paths(v, f'{path}/{i}'))
    return result

def main():
    manifest = json.loads((OUT/'FILE_MANIFEST.json').read_text())
    checkpoints = {}
    with zipfile.ZipFile(manifest['source']) as outer:
        for entry in manifest['entries']:
            name = entry['path']
            if not name.endswith('.pt'): continue
            with zipfile.ZipFile(io.BytesIO(outer.read(name))) as inner:
                data_name = next(n for n in inner.namelist() if n.endswith('/data.pkl'))
                data = inner.read(data_name)
                payload = RestrictedReader(io.BytesIO(data), inner, data_name.rsplit('/',1)[0]).load()
                globals_used = sorted(set(arg for op,arg,pos in pickletools.genops(data) if op.name=='GLOBAL'))
                compact = simplify(payload, tensor_hash=True)
                checkpoints[name] = {'original_member_sha256':entry['sha256'],
                    'pickle_globals': globals_used, 'top_level_keys': list(payload) if isinstance(payload,dict) else None,
                    'training_state_paths': interesting_paths(payload), 'payload': compact}
                print(json.dumps({'path':name, 'top_level_keys': checkpoints[name]['top_level_keys'],
                                  'training_state_paths':checkpoints[name]['training_state_paths']}, ensure_ascii=False))
    (OUT/'ORIGINAL_CHECKPOINT_METADATA.json').write_text(json.dumps(checkpoints,indent=2,ensure_ascii=False)+'\n')

if __name__ == '__main__': main()
