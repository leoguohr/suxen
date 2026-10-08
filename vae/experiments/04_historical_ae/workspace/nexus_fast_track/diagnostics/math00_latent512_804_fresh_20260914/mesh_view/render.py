from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

root=Path(__file__).resolve().parent
p=np.load(root.parent/'nexus_2k_001333_pool.npz')
v=p['vertices'].astype(float); f=p['positive'].astype(int)
all_e=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
e,counts=np.unique(all_e,axis=0,return_counts=True)
adj=[[] for _ in v]
for i,j in e:adj[i].append(j);adj[j].append(i)
unseen=set(range(len(v)));comp=0
while unseen:
    todo=[unseen.pop()];comp+=1
    while todo:
        for j in adj[todo.pop()]:
            if j in unseen:unseen.remove(j);todo.append(j)
lengths=np.linalg.norm(v[e[:,0]]-v[e[:,1]],axis=1)
areas=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)/2
stat=dict(uid='nexus_2k_001333',vertices=len(v),edges=len(e),faces=len(f),connected_components=comp,
          boundary_edges=int((counts==1).sum()),edges_shared_by_more_than_two_faces=int((counts>2).sum()),
          zero_area_faces=int((areas==0).sum()),euler_characteristic=len(v)-len(e)+len(f),
          degree_min=int(np.bincount(e.ravel()).min()),degree_median=float(np.median(np.bincount(e.ravel()))),degree_max=int(np.bincount(e.ravel()).max()),
          all_pairs=len(v)*(len(v)-1)//2,positive_edge_fraction=len(e)/(len(v)*(len(v)-1)/2),
          edge_length_quantiles=np.quantile(lengths,[0,.5,.95,1]).tolist(),bounds=np.ptp(v,axis=0).tolist())
(root/'mesh_statistics.json').write_text(json.dumps(stat,indent=2)+'\n')
with (root/'nexus_2k_001333_GT.obj').open('w') as out:
    out.write('# Original GT mesh; 804 vertices, 1604 faces\n')
    for x in v:out.write('v '+' '.join(f'{a:.9g}' for a in x)+'\n')
    for t in f:out.write('f '+' '.join(str(int(a)+1) for a in t)+'\n')
tri=v[f]
normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-20)
light=np.array([.3,-.4,.85]);light/=np.linalg.norm(light)
shade=.65+.35*np.abs(normals@light)
colors=np.column_stack([.35*shade,.68*shade,.87*shade,np.ones(len(f))])
fig=plt.figure(figsize=(15,6),facecolor='#f7f9fc')
views=[('Surface + triangulation',15,30),('Front (XY)',0,0),('Reverse view',15,210)]
for index,(title,elev,azim) in enumerate(views,1):
    ax=fig.add_subplot(1,3,index,projection='3d',computed_zorder=False)
    ax.set_facecolor('#f7f9fc')
    mesh=Poly3DCollection(tri,facecolors=colors,edgecolors=(.03,.12,.18,.65),linewidths=.34,antialiased=True)
    ax.add_collection3d(mesh)
    ax.set(xlim=(-1.08,1.08),ylim=(-1.08,1.08),zlim=(-1.08,1.08))
    ax.set_box_aspect((1,1,1),zoom=1.4);ax.view_init(elev=elev,azim=azim,vertical_axis='y');ax.set_proj_type('ortho');ax.set_axis_off()
    ax.set_title(title,fontsize=14,pad=-14)
fig.suptitle('Actual training mesh: nexus_2k_001333 (ground truth)',fontsize=19,y=.98)
fig.text(.5,.09,'804 vertices  |  2,406 edges  |  1,604 triangles  |  1 connected component',ha='center',fontsize=14)
fig.text(.5,.035,'Same geometry in all views; original scale proportions preserved; wire lines are GT edges.',ha='center',fontsize=11,color='#526477')
fig.subplots_adjust(left=.005,right=.995,top=.84,bottom=.15,wspace=-.12)
fig.savefig(root/'804_gt_three_views.png',dpi=180,facecolor=fig.get_facecolor());plt.close(fig)
print(json.dumps(stat,indent=2))
