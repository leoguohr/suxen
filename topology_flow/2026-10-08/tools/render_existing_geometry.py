"""CPU-only render of committed GT/predicted OBJ files; never invokes a model."""
import json
from pathlib import Path
import tarfile
from visualization_reference import read_json, sha, source_identity, collect_variant, contact_sheets

EXPORT=Path('/guohaoran/nexus_fast_track/diagnostics/topology_flow_export_20261008')
INITIAL=Path('/ssdwork/guohaoran/nexus_fast_track/diagnostics/topology_flow_user50_20261002')
C0=Path('/guohaoran/nexus_fast_track/diagnostics/topology_flow_user50_continue_20261006')
CAND=Path('/guohaoran/nexus_fast_track/diagnostics/topology_flow_candidates_20261007')


def main():
    output=EXPORT/'visuals';output.mkdir(exist_ok=False)
    cache=read_json(INITIAL/'cache/manifest.json');digest=sha(INITIAL/'cache/manifest.json')
    variants=[]
    identity=source_identity(INITIAL/'vae_baseline')
    for name in ('mu','posterior_seed0','posterior_seed1'):
        variants.append(collect_variant(INITIAL/'vae_baseline'/name,'vae_'+name,'vae_reconstruction',cache,digest,dict(identity,variant=name)))
    for name,path in [('c0_step500',C0/'evaluations/step500_seed0'),('c0_step1000',C0/'evaluations/step1000_seed0'),
            ('c1_step500',CAND/'c1/eval500'),('c1_step904_partial',CAND/'c1/eval904'),('c2_step500_partial',CAND/'c2/eval500')]:
        variants.append(collect_variant(path,name,'flow_generation',cache,digest))
    geometry={};truth={};manifest=[]
    for v in variants:
        sheets=contact_sheets(v,variants,INITIAL/'cache',cache,output,geometry,truth)
        manifest.append(dict(label=v['label'],identity=v['identity'],aggregate=v['aggregate'],sheets=sheets))
        print('RENDERED',v['label'],len(sheets),flush=True)
    report=dict(gpu_used=False,new_generation=False,source_cache_sha256=digest,variants=manifest,
        note='Existing committed OBJ only. Every edge and actual face drawn; fixed orthographic view; unscored/missing UIDs labeled. Source visualization SHA, GT vertex identity and predicted edge/face counts checked.',
        files=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(output.glob('*.png'))])
    (output/'VISUAL_MANIFEST.json').write_text(json.dumps(report,indent=2)+'\n')
    archive=EXPORT/'TopologyFlow_visualizations_20261008.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for p in sorted(output.iterdir()):tar.add(p,arcname='visuals/'+p.name,recursive=False)
    (EXPORT/'VISUAL_ASSET.json').write_text(json.dumps(dict(name=archive.name,bytes=archive.stat().st_size,sha256=sha(archive)),indent=2)+'\n')
    print('VISUALS_COMPLETE',flush=True)


if __name__=='__main__':main()
