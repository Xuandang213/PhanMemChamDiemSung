"""Consistent old/new labels on original images and confidence tables."""
import cv2
import numpy as np
from core import transform


def is_old(hole):
    return hole.get('classification', 'old' if hole.get('matched_before') is not None else 'new')=='old'


def is_visible(hole):
    return not hole.get('excluded',False) and (hole.get('selected',True) or (is_old(hole) and not hole.get('excluded',hole.get('review_edited',False))))


def hole_labels(holes):
    labels=[];new=0;old=0
    for hole in holes:
        if not is_visible(hole):labels.append('—');continue
        if not is_old(hole):
            new+=1;labels.append(f'M{new}')
        else:old+=1;labels.append(f'C{old}')
    return labels


def markers(result,kind):
    holes=result.get('before_detections',[]) if kind=='before' else result['detections']
    labels=[f'C{i+1}' for i in range(len(holes))] if kind=='before' else hole_labels(holes)
    if kind=='before':
        after=result['detections'];after_labels=hole_labels(after)
        linked={h['matched_before']:after_labels[i] for i,h in enumerate(after) if h.get('matched_before') is not None and is_old(h) and is_visible(h)}
        next_old=sum(is_old(h) and is_visible(h) for h in after)
        for i in range(len(holes)):
            if i in linked:labels[i]=linked[i]
            else:next_old+=1;labels[i]=f'C{next_old}'
    inverse=np.linalg.inv(np.array(result['homography_'+kind]))
    points=transform([h['template_point'] for h in holes],inverse)
    return [dict(point=p,label=label if kind=='before' or is_old(h) else f"{label}:{h['score']}",old=kind=='before' or is_old(h),
                 index=i,confidence=h.get('confidence',0),radius=h.get('manual_radius_original',12)) for i,(h,p,label) in enumerate(zip(holes,points,labels))
            if kind=='before' or is_visible(h)]


def draw_markers(image,items,selected=None):
    out=image.copy();h,w=out.shape[:2]
    radius=max(8,round(min(h,w)*.012));scale=max(.5,min(h,w)/1300)
    for item in items:
        radius=max(1,round(item.get('radius',radius)))
        x,y=map(lambda v:int(round(v)),item['point'])
        color=(0,0,255) if item['old'] else (0,190,0)
        cv2.circle(out,(x,y),radius,color,4 if item['index']==selected else 2,cv2.LINE_AA)
        label=item['label'];(tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,scale,2)
        pos=(max(0,min(w-tw-2,x+radius+3)),max(th+2,min(h-3,y-radius)))
        cv2.putText(out,label,pos,cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),4,cv2.LINE_AA)
        cv2.putText(out,label,pos,cv2.FONT_HERSHEY_SIMPLEX,scale,color,2,cv2.LINE_AA)
    return out
