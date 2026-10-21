"""Torch-only RetinaNet matching, losses, decoding and deterministic NMS."""
import torch
import torch.nn.functional as F

from topologies.models.retinanet import box_iou, decode_boxes, encode_boxes


def match_anchors(anchors, boxes, labels, positive_iou=.5, negative_iou=.4):
    """Labels are zero-based foreground classes; returns -1 ignore, 0 negative and 1..K positive."""
    if not 0 <= negative_iou < positive_iou <= 1: raise ValueError("thresholds require 0 <= negative_iou < positive_iou <= 1")
    if boxes.ndim!=2 or boxes.shape[-1]!=4 or labels.ndim!=1 or labels.shape[0]!=boxes.shape[0]: raise ValueError("boxes [N,4] and labels [N] must align")
    if anchors.ndim!=2 or anchors.shape[-1]!=4: raise ValueError("anchors must have shape [A,4]")
    if boxes.numel()==0: return torch.zeros(anchors.shape[0],dtype=torch.long,device=anchors.device),torch.zeros_like(anchors)
    if not torch.isfinite(boxes).all() or torch.any(boxes[:,2:]<=boxes[:,:2]): raise ValueError("ground-truth boxes must be finite, nondegenerate xyxy")
    if torch.any(labels<0): raise ValueError("labels must be zero-based nonnegative class ids")
    iou=box_iou(anchors,boxes); best,idx=iou.max(1); matched=labels[idx].long()+1; out=torch.full_like(matched,-1); out[best<negative_iou]=0; pos=best>=positive_iou; out[pos]=matched[pos]
    return out,encode_boxes(anchors,boxes[idx])

def sigmoid_focal_loss(logits, targets, alpha=.25, gamma=2.):
    ce=F.binary_cross_entropy_with_logits(logits,targets,reduction="none"); p=logits.sigmoid(); pt=p*targets+(1-p)*(1-targets); at=alpha*targets+(1-alpha)*(1-targets)
    return (at*(1-pt).pow(gamma)*ce).sum()

def detection_loss(class_logits,box_deltas,anchors,boxes,labels,num_classes,positive_iou=.5,negative_iou=.4):
    if class_logits.ndim!=2 or class_logits.shape[-1]!=num_classes: raise ValueError("class logits must be [anchors,num_classes]")
    if boxes.shape[0] and labels.max()>=num_classes: raise ValueError("ground-truth label exceeds num_classes")
    matched,reg=match_anchors(anchors,boxes,labels,positive_iou,negative_iou); valid=matched>=0; positive=matched>0; target=torch.zeros_like(class_logits); rows=positive.nonzero().flatten()
    if rows.numel(): target[rows,matched[rows]-1]=1
    cls=sigmoid_focal_loss(class_logits[valid],target[valid])/max(int(positive.sum()),1)
    box=F.smooth_l1_loss(box_deltas[positive],reg[positive],reduction="sum")/max(int(positive.sum()),1) if positive.any() else box_deltas.sum()*0
    return cls+box,{"classification_loss":cls.detach(),"box_loss":box.detach()}

def nms(boxes,scores,iou_threshold=.5):
    """Greedy score-descending NMS with stable original-index tie breaking."""
    if not 0 <= iou_threshold <= 1: raise ValueError("iou_threshold must be in [0,1]")
    if boxes.ndim!=2 or boxes.shape[-1]!=4 or scores.ndim!=1 or len(scores)!=len(boxes): raise ValueError("NMS expects boxes [N,4] and scores [N]")
    order=sorted(range(len(scores)),key=lambda i:(-float(scores[i]),i)); kept=[]
    while order:
        i=order.pop(0); kept.append(i)
        if order:
            rest=torch.tensor(order,device=boxes.device,dtype=torch.long); overlap=box_iou(boxes[i:i+1],boxes[rest]).squeeze(0)
            order=[j for j,v in zip(order,overlap) if float(v)<=iou_threshold]
    return torch.tensor(kept,device=boxes.device,dtype=torch.long)

def undo_resize_pad(boxes,scale,pad_xy=(0.,0.),original_size=None):
    """Map padded resized xyxy pixel coordinates to original-image coordinates."""
    sx,sy=(scale,scale) if isinstance(scale,(int,float)) else scale; px,py=pad_xy
    if sx<=0 or sy<=0: raise ValueError("resize scales must be positive")
    out=boxes.clone()
    out[:,[0,2]]=(out[:,[0,2]]-px)/sx; out[:,[1,3]]=(out[:,[1,3]]-py)/sy
    if original_size is not None:
        h,w=original_size; out[:,[0,2]]=out[:,[0,2]].clamp(0,w); out[:,[1,3]]=out[:,[1,3]].clamp(0,h)
    return out

def postprocess_detections(class_logits,box_deltas,anchors,score_threshold=.05,nms_threshold=.5,max_detections=100):
    scores=class_logits.sigmoid(); results=[]
    for cls in range(scores.shape[-1]):
        vals=scores[:,cls]; ids=(vals>=score_threshold).nonzero().flatten()
        if not len(ids): continue
        boxes=decode_boxes(anchors[ids],box_deltas[ids]); keep=nms(boxes,vals[ids],nms_threshold)
        results.extend((boxes[keep[k]],vals[ids][keep[k]],cls) for k in range(len(keep)))
    results.sort(key=lambda x:-float(x[1])); results=results[:max_detections]
    if not results: return {"boxes":anchors.new_empty((0,4)),"scores":anchors.new_empty(0),"labels":torch.empty(0,device=anchors.device,dtype=torch.long)}
    return {"boxes":torch.stack([r[0] for r in results]),"scores":torch.stack([r[1] for r in results]),"labels":torch.tensor([r[2] for r in results],device=anchors.device)}
