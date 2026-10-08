"""Original tail forward and unchanged Edge/Face Soft4, equal weight per mesh."""
from runtime import T
import prior_core as edge
import face_core as face


def score(tail,d,edge_scale,sc,scales):
    h,ev=edge.score(tail,d['x'],d['pairs'],d['labels'],edge_scale,sc)
    raw,embedding=face.center(tail['face'],h)
    loss,logits=face.objective(embedding,d['face_tris'],d['face_labels'],scales)
    return h,ev,(raw,embedding,logits,loss)


def cycle(tail,data,edge_scale,sc,scales,backward):
    rows=[]
    for d in data:
        with T.enable_grad():
            h,ev,fv=score(tail,d,edge_scale,sc,scales)
            row=edge.metrics(d,ev);row['face_soft4']=float(fv[3].detach());rows.append(row)
            if backward:((ev[3]+fv[3])/100).backward()
        del h,ev,fv
    return rows


def summary(rows,step):
    result=edge.summary(rows,step)
    result['edge_objective']=result['objective']
    result['face_objective']=sum(r['face_soft4'] for r in rows)/100
    result['objective']=result['edge_objective']+result['face_objective']
    return result
