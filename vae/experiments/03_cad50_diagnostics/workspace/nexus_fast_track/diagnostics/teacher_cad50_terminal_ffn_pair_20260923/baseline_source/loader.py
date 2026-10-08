"""Float-mesh adapter. No quantization, vertex reorder, welding, or stage fields."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import torch

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()

@dataclass
class FloatMeshBatch:
    vertices: torch.Tensor
    vertex_mask: torch.Tensor
    faces: tuple
    incidence_index: tuple

    def to(self,device):
        return FloatMeshBatch(self.vertices.to(device),self.vertex_mask.to(device),
            tuple(f.to(device) for f in self.faces),tuple(i.to(device) for i in self.incidence_index))

def collate(samples):
    maximum=max(len(s['vertices']) for s in samples)
    vertices=torch.zeros((len(samples),maximum,3),dtype=torch.float32)
    mask=torch.zeros((len(samples),maximum),dtype=torch.bool)
    faces=[];incidence=[]
    for k,s in enumerate(samples):
        n=len(s['vertices']);vertices[k,:n]=torch.from_numpy(s['vertices']);mask[k,:n]=True
        faces.append(torch.from_numpy(s['faces']))
        incidence.append(torch.from_numpy(s['incidence_index']))
    return FloatMeshBatch(vertices,mask,tuple(faces),tuple(incidence))

def load_all(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text());result={}
    assert manifest['uids']==[f'teacher_cad50_{i:02d}' for i in range(50)]
    for row in manifest['meshes']:
        p=root/row['path'];assert sha(p)==row['sha256']
        with np.load(p,allow_pickle=False) as f:result[row['uid']]={k:f[k].copy() for k in f.files}
    return manifest,result
